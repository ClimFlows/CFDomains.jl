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
    any(input -> input isa LazyExpression, expr.inputs) &&
        throw(ArgumentError("LazyExpression-valued inputs are unsupported by the Enzyme rule"))
    return LazyCotangent{T,N,typeof(expr),typeof(shadow.inputs)}(expr, shadow.inputs)
end

lazy_cotangent(input::Duplicated{<:LazyExpression}) =
    lazy_cotangent(input.val, input.dval)
lazy_cotangent(input::MixedDuplicated{<:LazyExpression}) =
    lazy_cotangent(input.val, input.dval[])

# The custom rule differentiates homogeneous real floating-point input arrays.
# LazyExpression params are constants (as in Mooncake), including array-valued
# params. Nested lazy inputs, complex/mixed-precision expressions, LazyExpression-
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
partial for each explicit array input. Reverse mode uses these cached partials
instead of rereading primal arrays that may have changed since the read.
Parameters are constants and have no partials in the tape.
"""
struct LazyReadTape{I}
    input_partials::I
end

# Enzyme fallback for scalar callables whose method signature rejects Duals.
# Parameters and non-floating inputs are annotated Const; inactive derivatives
# returned as `nothing` become numeric zero partials.
function enzyme_value_and_partials(fun, values::Tuple, active::Tuple)
    annotations = ntuple(length(values)) do i
        active[i] ? Active(values[i]) : Const(values[i])
    end
    (partials, value) = autodiff(ReverseWithPrimal, fun, Active, annotations...)
    partials = ntuple(i -> partials[i] === nothing ? zero(value) : partials[i], length(partials))
    return value, partials
end

# Differentiate only floating-point scalar values read from explicit inputs.
# Type-directed tuple recursion keeps seed positions and the returned partial
# tuple concrete; all params are passed to the primal callable as constants.
nactive_inputs(::Tuple{}) = 0
nactive_inputs(values::Tuple{<:AbstractFloat,Vararg}) =
    1 + nactive_inputs(Base.tail(values))
nactive_inputs(values::Tuple{Any,Vararg{Any}}) =
    nactive_inputs(Base.tail(values))

input_activity(::Tuple{}) = ()
input_activity(values::Tuple{<:AbstractFloat,Vararg}) =
    (true, input_activity(Base.tail(values))...)
input_activity(values::Tuple{Any,Vararg{Any}}) =
    (false, input_activity(Base.tail(values))...)

function seed_input_values(::Tuple{}, ::Val{N}, ::Val{I}, ::Type{Tag}) where {N,I,Tag}
    ()
end
function seed_input_values(values::Tuple{<:AbstractFloat,Vararg},
                           ::Val{N}, ::Val{I}, ::Type{Tag}) where {N,I,Tag}
    value = first(values)
    seed_partials = ntuple(j -> j == I ? one(value) : zero(value), N)
    dual = FD.Dual{Tag}(value, seed_partials)
    return (dual, seed_input_values(Base.tail(values), Val(N), Val(I + 1), Tag)...)
end
function seed_input_values(values::Tuple{Any,Vararg{Any}},
                           ::Val{N}, ::Val{I}, ::Type{Tag}) where {N,I,Tag}
    return (first(values), seed_input_values(Base.tail(values), Val(N), Val(I), Tag)...)
end

function input_partials(::Tuple{}, result::FD.Dual, ::Val{I}, zero_value) where {I}
    ()
end
function input_partials(values::Tuple{<:AbstractFloat,Vararg}, result::FD.Dual,
                        ::Val{I}, zero_value) where {I}
    partial = result.partials.values[I]
    return (partial, input_partials(Base.tail(values), result, Val(I + 1), zero_value)...)
end
function input_partials(values::Tuple{Any,Vararg{Any}}, result::FD.Dual,
                        ::Val{I}, zero_value) where {I}
    return (zero_value, input_partials(Base.tail(values), result, Val(I), zero_value)...)
end

function scalar_value_and_partials(fun, inputs::Tuple, params::Tuple, fallback_fun=fun)
    nactive = nactive_inputs(inputs)
    if iszero(nactive)
        value = fun(inputs..., params...)
        return value, ntuple(_ -> zero(value), length(inputs))
    end

    dual_inputs = seed_input_values(inputs, Val(nactive), Val(1), typeof(fun))
    dual_args = (dual_inputs..., params...)

    # Select Enzyme before calling the scalar function when its method signature
    # rejects Duals. Do not catch an internal MethodError and retry: a callable
    # may have side effects before reaching the incompatible helper.
    if !applicable(fallback_fun, dual_args...)
        active = (input_activity(inputs)..., ntuple(_ -> false, length(params))...)
        return enzyme_value_and_partials(fallback_fun, (inputs..., params...), active)
    end

    result = fun(dual_args...)
    if result isa FD.Dual
        value = FD.value(result)
        partials = input_partials(inputs, result, Val(1), zero(value))
        return value, partials
    else
        return result, ntuple(_ -> zero(result), length(inputs))
    end
end

@prop function read_with_tape(expr::LazyExpression{T}, indices::Tuple) where {T}
    @boundscheck foreach(input -> checkbounds(input, indices...), expr.inputs)
    inputs = map(input -> (@inbounds input[indices...]), expr.inputs)
    params = map(param -> get(param, indices...), expr.params)
    callable = expr.fun
    fun(args...) = callable(args...)
    value, input_partials = scalar_value_and_partials(fun, inputs, params, callable)
    value = convert(T, value)
    input_partials = map(partial -> convert(T, partial), input_partials)
    return value, LazyReadTape(input_partials)
end

@prop function accumulate_taped!(cotangent::LazyCotangent, tape::LazyReadTape, value, indices...)
    accumulate_taped_inputs!(cotangent.expr.inputs, cotangent.dinputs, tape.input_partials, value, indices...)
    return cotangent
end

@prop accumulate_taped_inputs!(::Tuple{}, ::Tuple{}, ::Tuple{}, value, indices...) = nothing

@prop function accumulate_taped_inputs!((input, inputs...), (dinput, dinputs...),
                                        (partial, partials...), value, indices...)
    contribution = value * partial
    if eltype(input) <: Integer || dinput === input
        nothing
    else
        dinput[indices...] += contribution
    end
    accumulate_taped_inputs!(inputs, dinputs, partials, value, indices...)
end

@inline expand_cartesian_indices(::Tuple{}) = ()
@inline expand_cartesian_indices(indices::Tuple{CartesianIndex{M},Vararg{Any}}) where {M} =
    (Tuple(first(indices))..., expand_cartesian_indices(Base.tail(indices))...)
@inline expand_cartesian_indices(indices::Tuple{Any,Vararg{Any}}) =
    (first(indices), expand_cartesian_indices(Base.tail(indices))...)

@inline scalar_cartesian_indices(expr, indices::Tuple) =
    cartesian_indices(expr, expand_cartesian_indices(indices))

@prop function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(Base.getindex)},
    ::Type{<:Active},
    input::LazyExpressionInput,
    indices::Vararg{Const},
)
    cartesian = scalar_cartesian_indices(input.val, map(index -> index.val, indices))
    cotangent = lazy_cotangent(input)
    value, tape = read_with_tape(input.val, cartesian)
    value = value::eltype(input.val)
    if EnzymeRules.needs_primal(config)
        return EnzymeRules.AugmentedReturn{typeof(value),Nothing,typeof(tape)}(value, nothing, tape)
    else
        return EnzymeRules.AugmentedReturn{Nothing,Nothing,typeof(tape)}(nothing, nothing, tape)
    end
end

@prop function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(Base.getindex)},
    derivative::Active,
    tape,
    input::LazyExpressionInput,
    indices::Vararg{Const},
)
    cartesian = scalar_cartesian_indices(input.val, map(index -> index.val, indices))
    tape = tape::LazyReadTape
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
    callable = expr.fun
    fun(args...) = callable(args...)
    _, input_partials = scalar_value_and_partials(fun, inputs, params, callable)
    accumulate_inputs!(expr.inputs, cotangent.dinputs, input_partials, value, cartesian...)
    return cotangent
end

@prop accumulate_inputs!(::Tuple{}, ::Tuple{}, ::Tuple{}, value, indices...) = nothing

@prop function accumulate_inputs!((input, inputs...), (dinput, dinputs...),
                                  (partial, partials...), value, indices...)
    contribution = value * partial
    if eltype(input) <: Integer || dinput === input
        nothing
    else
        dinput[indices...] += contribution
    end
    accumulate_inputs!(inputs, dinputs, partials, value, indices...)
end

end
