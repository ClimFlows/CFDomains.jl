module CFDomainsEnzymeExt

using Base: @propagate_inbounds as @prop

using Enzyme: Active, Const, Duplicated, EnzymeRules, MixedDuplicated, ReverseWithPrimal, autodiff
using CFDomains.LazyExpressions: LazyExpression, cartesian_indices, get
import CFDomains.LazyExpressions: lazy_cotangent
import CFDomains.LazyOperators as LazyOps
import ForwardDiff as FD

"""
    LazyCotangent

A write-through view of input cotangents for a lazy expression. It lets
operator code accumulate directly into the Enzyme shadow input arrays without
allocating a full-sized cotangent array. Lazy-expression parameters are
constants and therefore have no cotangent storage.
"""
struct LazyCotangent{T,N,E,I} <: AbstractArray{T,N}
    expr::E
    dinputs::I
end

function lazy_cotangent(expr::LazyExpression{T,N}, shadow::LazyExpression{T,N}) where {T,N}
    size(expr) == size(shadow) || throw(DimensionMismatch("lazy expression and shadow sizes differ"))
    length(expr.inputs) == length(shadow.inputs) ||
        throw(DimensionMismatch("lazy expression and shadow input counts differ"))
    length(expr.params) == length(shadow.params) ||
        throw(DimensionMismatch("lazy expression and shadow parameter counts differ"))
    return LazyCotangent{T,N,typeof(expr),typeof(shadow.inputs)}(expr, shadow.inputs)
end

lazy_cotangent(input::Duplicated{<:LazyExpression}) =
    lazy_cotangent(input.val, input.dval)
lazy_cotangent(input::MixedDuplicated{<:LazyExpression}) =
    lazy_cotangent(input.val, input.dval[])

# The custom rule differentiates homogeneous real floating-point input arrays.
# LazyExpression params are constants (as in Mooncake), including array-valued
# params. Complex/non-floating and mixed-precision expressions, LazyExpression-
# valued params, and differentiable arrays captured by `fun` are out of scope.
const FloatingLazyExpression = LazyExpression{T,N,F,I,P} where {
    T<:AbstractFloat,N,F,I<:Tuple{Vararg{AbstractArray{T}}},P,
}
# Enzyme wrappers that provide a primal expression and its input shadow.
const LazyExpressionInput = Union{
    Duplicated{<:FloatingLazyExpression},
    MixedDuplicated{<:FloatingLazyExpression},
}

"""
    LazyReadTape

Forward-pass data for one scalar read. `input_partials` stores the local
partial for each explicit input value, and `children` stores tapes for nested
lazy inputs in the same order. Reverse mode uses these cached partials instead
of rereading primal arrays that may have changed since the read. Parameters
are constants and have no partials in the tape.
"""
struct LazyReadTape{I,C}
    input_partials::I
    children::C
end

is_differentiable_scalar(x) = x isa AbstractFloat

# Use Enzyme for the rare callable that rejects ForwardDiff.Dual arguments.
# The activity mask keeps constants (including LazyExpression params) unseeded.
function enzyme_value_and_partials(fun, values::Tuple, active::Tuple)
    annotations = ntuple(length(values)) do i
        active[i] ? Active(values[i]) : Const(values[i])
    end
    (partials, value) = autodiff(ReverseWithPrimal, fun, Active, annotations...)
    partials = ntuple(i -> partials[i] === nothing ? zero(value) : partials[i], length(partials))
    return value, partials
end

# Compute the local scalar pullback data once in the primal pass. The mask
# marks only active input values; params are appended as constants. Cache the
# resulting input partials in LazyReadTape for the matching reverse read.
function scalar_value_and_partials(fun, values::Tuple, active::Tuple=map(is_differentiable_scalar, values))
    active = ntuple(i -> active[i] && is_differentiable_scalar(values[i]), length(values))
    nactive = count(identity, active)
    if iszero(nactive)
        return fun(values...), ntuple(_ -> 0, length(values))
    end

    active_values = ntuple(i -> active[i] ? values[i] : nothing, length(values))
    value_type = promote_type(map(typeof, filter(!isnothing, active_values))...)
    positions = ntuple(length(values)) do i
        active[i] ? count(identity, active[1:i]) : 0
    end
    dual_values = ntuple(length(values)) do i
        position = positions[i]
        if iszero(position)
            values[i]
        else
            seed_partials = ntuple(j -> j == position ? one(value_type) : zero(value_type), nactive)
            FD.Dual{typeof(fun)}(values[i], seed_partials)
        end
    end

    result = try
        fun(dual_values...)
    catch err
        if err isa MethodError
            return enzyme_value_and_partials(fun, values, active)
        end
        rethrow()
    end
    if result isa FD.Dual
        output_partials = ntuple(
            i -> iszero(positions[i]) ? zero(value_type) : result.partials.values[positions[i]],
            length(values),
        )
        return FD.value(result), output_partials
    else
        return result, ntuple(_ -> zero(value_type), length(values))
    end
end

@prop function read_with_tape(expr::LazyExpression, indices::Tuple, dinputs::Tuple)
    @boundscheck foreach(input -> checkbounds(input, indices...), expr.inputs)
    reads = map((input, dinput) -> read_input_with_tape(input, dinput, indices), expr.inputs, dinputs)
    inputs = map(first, reads)
    children = map(last, reads)
    params = map(param -> get(param, indices...), expr.params)
    active_inputs = map(is_differentiable_scalar, inputs)
    inactive_params = ntuple(_ -> false, length(params))
    fun(args...) = expr.fun(args...)
    value, partials = scalar_value_and_partials(fun, (inputs..., params...), (active_inputs..., inactive_params...))
    input_partials = ntuple(i -> partials[i], length(inputs))
    return value, LazyReadTape(input_partials, children)
end

# Input activity depends on the input scalar type, not on runtime aliasing
# between primal and shadow arrays. This keeps the partial tuple concrete.
@prop read_input_with_tape(input::AbstractArray, dinput, indices::Tuple) =
    (@inbounds input[indices...], nothing)
@prop read_input_with_tape(input::LazyExpression, dinput::LazyExpression, indices::Tuple) =
    read_with_tape(input, indices, dinput.inputs)

@prop function accumulate_taped!(cotangent::LazyCotangent, tape::LazyReadTape, value, indices...)
    accumulate_taped_inputs!(cotangent.expr.inputs, cotangent.dinputs,
                             tape.input_partials, tape.children, value, indices...)
    return cotangent
end

@prop accumulate_taped_inputs!(::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, value, indices...) = nothing

@prop function accumulate_taped_inputs!((input, inputs...), (dinput, dinputs...),
                                        (partial, partials...), (child, children...),
                                        value, indices...)
    contribution = value * partial
    if input isa LazyExpression
        accumulate_taped!(lazy_cotangent(input, dinput), child, contribution, indices...)
    elseif eltype(input) <: Integer || dinput === input
        nothing
    else
        dinput[indices...] += contribution
    end
    accumulate_taped_inputs!(inputs, dinputs, partials, children, value, indices...)
end

@prop function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(Base.getindex)},
    ::Type{<:Active},
    input::LazyExpressionInput,
    indices::Vararg{Const},
)
    cartesian = cartesian_indices(input.val, map(index -> index.val, indices))
    cotangent = lazy_cotangent(input)
    value, tape = read_with_tape(input.val, cartesian, cotangent.dinputs)
    primal = EnzymeRules.needs_primal(config) ? value : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, tape)
end

@prop function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(Base.getindex)},
    derivative::Active,
    tape::LazyReadTape,
    input::LazyExpressionInput,
    indices::Vararg{Const},
)
    cartesian = cartesian_indices(input.val, map(index -> index.val, indices))
    accumulate_taped!(lazy_cotangent(input), tape, derivative.val, cartesian...)
    return ntuple(_ -> nothing, length(indices) + 1)
end

Base.size(cotangent::LazyCotangent) = size(cotangent.expr)
Base.axes(cotangent::LazyCotangent) = axes(cotangent.expr)
Base.eachindex(cotangent::LazyCotangent) = eachindex(cotangent.expr)
Base.IndexStyle(::Type{<:LazyCotangent}) = IndexCartesian()

@prop LazyOps.addto!(cotangent::LazyCotangent, value, indices...) =
    accumulate!(cotangent, value, indices...)
@prop LazyOps.subfrom!(cotangent::LazyCotangent, value, indices...) =
    accumulate!(cotangent, -value, indices...)

@prop function accumulate!(cotangent::LazyCotangent, value, indices...)
    expr = cotangent.expr
    cartesian = cartesian_indices(expr, indices)
    inputs = map(input -> get(input, cartesian...), expr.inputs)
    params = map(param -> get(param, cartesian...), expr.params)
    fun(args...) = expr.fun(args...)
    active_inputs = map(is_differentiable_scalar, inputs)
    inactive_params = ntuple(_ -> false, length(params))
    _, partials = scalar_value_and_partials(fun, (inputs..., params...), (active_inputs..., inactive_params...))
    input_partials = ntuple(i -> partials[i], length(inputs))
    accumulate_inputs!(expr.inputs, cotangent.dinputs, input_partials, value, cartesian...)
    return cotangent
end

@prop accumulate_inputs!(::Tuple{}, ::Tuple{}, ::Tuple{}, value, indices...) = nothing

@prop function accumulate_inputs!((input, inputs...), (dinput, dinputs...), (partial, partials...), value, indices...)
    contribution = value * partial
    if input isa LazyExpression
        cotangent = lazy_cotangent(input, dinput)
        accumulate!(cotangent, contribution, indices...)
    elseif eltype(input) <: Integer || dinput === input
        nothing
    else
        dinput[indices...] += contribution
    end
    accumulate_inputs!(inputs, dinputs, partials, value, indices...)
end

end
