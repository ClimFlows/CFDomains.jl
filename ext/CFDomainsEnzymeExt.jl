module CFDomainsEnzymeExt

using Base: @propagate_inbounds as @prop

using Enzyme: Active, Const, Duplicated, EnzymeRules, MixedDuplicated
using CFDomains.LazyExpressions: LazyExpression, cartesian_indices, get, parameter_indices
import CFDomains.LazyExpressions: lazy_cotangent
import CFDomains.LazyOperators as LazyOps
import ForwardDiff as FD

"A write-through tangent view for a lazy expression."
struct LazyCotangent{T,N,E,I,P} <: AbstractArray{T,N}
    expr::E
    dinputs::I
    dparams::P
end

function lazy_cotangent(expr::LazyExpression{T,N}, shadow::LazyExpression{T,N}) where {T,N}
    size(expr) == size(shadow) || throw(DimensionMismatch("lazy expression and shadow sizes differ"))
    length(expr.inputs) == length(shadow.inputs) ||
        throw(DimensionMismatch("lazy expression and shadow input counts differ"))
    length(expr.params) == length(shadow.params) ||
        throw(DimensionMismatch("lazy expression and shadow parameter counts differ"))
    return LazyCotangent{T,N,typeof(expr),typeof(shadow.inputs),typeof(shadow.params)}(
        expr, shadow.inputs, shadow.params,
    )
end

lazy_cotangent(input::Duplicated{<:LazyExpression}) =
    lazy_cotangent(input.val, input.dval)
lazy_cotangent(input::MixedDuplicated{<:LazyExpression}) =
    lazy_cotangent(input.val, input.dval[])

# Direct scalar reads support homogeneous real floating-point expressions.
# Complex/non-floating and mixed-precision expressions, scalar active
# parameters, and differentiable arrays captured by `fun` are out of scope.
const FloatingLazyExpression = LazyExpression{T,N,F,I,P} where {T<:AbstractFloat,N,F,I,P}
const LazyExpressionInput = Union{
    Duplicated{<:FloatingLazyExpression},
    MixedDuplicated{<:FloatingLazyExpression},
}

struct LazyReadTape{I,P,C,V}
    input_partials::I
    param_partials::P
    children::C
    values::V
end

is_differentiable_scalar(x) = x isa AbstractFloat

function scalar_value_and_partials(fun, values::Tuple)
    nactive = count(is_differentiable_scalar, values)
    if iszero(nactive)
        return fun(values...), ntuple(_ -> 0, length(values))
    end

    active_values = filter(is_differentiable_scalar, values)
    value_type = promote_type(map(typeof, active_values)...)
    positions = ntuple(length(values)) do i
        is_differentiable_scalar(values[i]) ?
        count(j -> is_differentiable_scalar(values[j]), 1:i) : 0
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

    result = fun(dual_values...)
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

@prop function read_with_tape(expr::LazyExpression, indices::Tuple)
    @boundscheck foreach(input -> checkbounds(input, indices...), expr.inputs)
    reads = map(input -> read_input_with_tape(input, indices), expr.inputs)
    inputs = map(first, reads)
    children = map(last, reads)
    params = map(param -> get(param, indices...), expr.params)
    fun(args...) = expr.fun(args...)
    value, partials = scalar_value_and_partials(fun, (inputs..., params...))
    ninputs = length(inputs)
    input_partials = ntuple(i -> partials[i], ninputs)
    param_partials = ntuple(i -> partials[ninputs + i], length(params))
    return value, LazyReadTape(input_partials, param_partials, children, (inputs..., params...))
end

@prop read_input_with_tape(input::AbstractArray, indices::Tuple) =
    (@inbounds input[indices...], nothing)
@prop read_input_with_tape(input::LazyExpression, indices::Tuple) = read_with_tape(input, indices)

@prop function accumulate_taped!(cotangent::LazyCotangent, tape::LazyReadTape, value, indices...)
    accumulate_taped_inputs!(cotangent.expr.inputs, cotangent.dinputs,
                             tape.input_partials, tape.children, value, indices...)
    accumulate_taped_params!(cotangent.expr.params, cotangent.dparams,
                             tape.param_partials, value, indices...)
    return cotangent
end

@prop accumulate_taped_params!(::Tuple{}, ::Tuple{}, ::Tuple{}, value, indices...) = nothing

@prop function accumulate_taped_params!((param, params...), (dparam, dparams...),
                                        (partial, partials...), value, indices...)
    if param isa AbstractArray && !(eltype(param) <: Integer)
        param_indices = parameter_indices(Val(length(indices)), Val(ndims(dparam)), indices)
        dparam[param_indices...] += value * partial
    end
    accumulate_taped_params!(params, dparams, partials, value, indices...)
end

@prop accumulate_taped_inputs!(::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, value, indices...) = nothing

@prop function accumulate_taped_inputs!((input, inputs...), (dinput, dinputs...),
                                        (partial, partials...), (child, children...),
                                        value, indices...)
    contribution = value * partial
    if input isa LazyExpression
        accumulate_taped!(lazy_cotangent(input, dinput), child, contribution, indices...)
    elseif eltype(input) <: Integer
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
    value, tape = read_with_tape(input.val, cartesian)
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
    _, partials = scalar_value_and_partials(fun, (inputs..., params...))
    ninputs = length(expr.inputs)
    input_partials = ntuple(i -> partials[i], ninputs)
    param_partials = ntuple(i -> partials[ninputs + i], length(expr.params))
    accumulate_inputs!(expr.inputs, cotangent.dinputs, input_partials, value, cartesian...)
    accumulate_taped_params!(expr.params, cotangent.dparams, param_partials, value, cartesian...)
    return cotangent
end

@prop accumulate_inputs!(::Tuple{}, ::Tuple{}, ::Tuple{}, value, indices...) = nothing

@prop function accumulate_inputs!((input, inputs...), (dinput, dinputs...), (partial, partials...), value, indices...)
    contribution = value * partial
    if input isa LazyExpression
        cotangent = lazy_cotangent(input, dinput)
        accumulate!(cotangent, contribution, indices...)
    elseif eltype(input) <: Integer
        nothing
    else
        dinput[indices...] += contribution
    end
    accumulate_inputs!(inputs, dinputs, partials, value, indices...)
end

end
