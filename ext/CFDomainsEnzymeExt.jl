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

# The @lazy contract guarantees homogeneous floating-point positional inputs.
function scalar_value_and_partials(fun, inputs::Tuple, params::Tuple)
    ninputs = length(inputs)
    dual_inputs = ntuple(i -> FD.Dual{typeof(fun)}(
        inputs[i], ntuple(j -> i == j ? one(inputs[i]) : zero(inputs[i]), ninputs),
    ), ninputs)
    result = fun(dual_inputs..., params...)
    if result isa FD.Dual
        value = FD.value(result)
        return value, result.partials.values
    else
        return result, ntuple(_ -> zero(result), ninputs)
    end
end

@prop function read_with_tape(expr::LazyExpression{T}, indices::Tuple) where {T}
    @boundscheck foreach(input -> checkbounds(input, indices...), expr.inputs)
    inputs = map(input -> (@inbounds input[indices...]), expr.inputs)
    params = map(param -> get(param, indices...), expr.params)
    callable = expr.fun
    fun(args...) = callable(args...)
    value, input_partials = scalar_value_and_partials(fun, inputs, params)
    value = convert(T, value)
    input_partials = map(partial -> convert(T, partial), input_partials)
    return value, LazyReadTape(input_partials)
end

@inline expand_cartesian_indices(::Tuple{}) = ()
@inline expand_cartesian_indices(indices::Tuple{CartesianIndex{M},Vararg{Any}}) where {M} =
    (Tuple(first(indices))..., expand_cartesian_indices(Base.tail(indices))...)
@inline expand_cartesian_indices(indices::Tuple{Any,Vararg{Any}}) =
    (first(indices), expand_cartesian_indices(Base.tail(indices))...)

@prop function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(Base.getindex)},
    ::Type{<:Active},
    input::LazyExpressionInput,
    indices::Vararg{Const},
)
    cartesian = cartesian_indices(input.val, expand_cartesian_indices(map(index -> index.val, indices)))
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
    cotangent = lazy_cotangent(input)
    accumulate_inputs!(cotangent.expr.inputs, cotangent.dinputs, tape.input_partials,
                       derivative.val, cartesian...)
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
    _, input_partials = scalar_value_and_partials(fun, inputs, params)
    accumulate_inputs!(expr.inputs, cotangent.dinputs, input_partials, value, cartesian...)
    return cotangent
end

@prop accumulate_inputs!(::Tuple{}, ::Tuple{}, ::Tuple{}, value, indices...) = nothing

@prop function accumulate_inputs!((input, inputs...), (dinput, dinputs...),
                                  (partial, partials...), value, indices...)
    contribution = value * partial
    if dinput === input
        nothing
    else
        dinput[indices...] += contribution
    end
    accumulate_inputs!(inputs, dinputs, partials, value, indices...)
end

end
