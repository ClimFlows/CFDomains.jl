module CFDomainsEnzymeExt

using Base: @propagate_inbounds as @prop

using Enzyme: Active, Const, Duplicated, EnzymeRules, MixedDuplicated
using CFDomains.LazyExpressions: LazyExpression, cartesian_indices, get, pdv
import CFDomains.LazyExpressions: lazy_cotangent
import CFDomains.LazyOperators as LazyOps

"A write-through tangent view for a lazy expression."
struct LazyCotangent{T,N,E,I} <: AbstractArray{T,N}
    expr::E
    dinputs::I
end

function lazy_cotangent(expr::LazyExpression{T,N}, shadow::LazyExpression{T,N}) where {T,N}
    size(expr) == size(shadow) || throw(DimensionMismatch("lazy expression and shadow sizes differ"))
    length(expr.inputs) == length(shadow.inputs) ||
        throw(DimensionMismatch("lazy expression and shadow input counts differ"))
    return LazyCotangent{T,N,typeof(expr),typeof(shadow.inputs)}(expr, shadow.inputs)
end

lazy_cotangent(input::Duplicated{<:LazyExpression}) =
    lazy_cotangent(input.val, input.dval)
lazy_cotangent(input::MixedDuplicated{<:LazyExpression}) =
    lazy_cotangent(input.val, input.dval[])

const LazyExpression1 = LazyExpression{T,N,F,Tuple{A},P} where {T,N,F,A,P}
const LazyExpression2 = LazyExpression{T,N,F,Tuple{A,B},P} where {T,N,F,A,B,P}
const LazyExpression3 = LazyExpression{T,N,F,Tuple{A,B,C},P} where {T,N,F,A,B,C,P}
const SupportedLazyExpression = Union{LazyExpression1, LazyExpression2, LazyExpression3}
const LazyExpressionInput = Union{
    Duplicated{<:SupportedLazyExpression},
    MixedDuplicated{<:SupportedLazyExpression},
}

struct LazyReadTape{P,C}
    partials::P
    children::C
end

@prop function read_with_tape(expr::LazyExpression, indices::Tuple)
    reads = map(input -> read_input_with_tape(input, indices), expr.inputs)
    inputs = map(first, reads)
    children = map(last, reads)
    params = map(param -> get(param, indices...), expr.params)
    fun(args...) = expr.fun(args..., params...)
    partials = pdv(fun, inputs...)
    partials = partials isa Tuple ? partials : (partials,)
    return fun(inputs...), LazyReadTape(partials, children)
end

@prop read_input_with_tape(input::AbstractArray, indices::Tuple) =
    (@inbounds input[indices...], nothing)
@prop read_input_with_tape(input::LazyExpression, indices::Tuple) = read_with_tape(input, indices)

@prop function accumulate_taped!(cotangent::LazyCotangent, tape::LazyReadTape, value, indices...)
    accumulate_taped_inputs!(cotangent.expr.inputs, cotangent.dinputs,
                             tape.partials, tape.children, value, indices...)
    return cotangent
end

@prop accumulate_taped_inputs!(::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, value, indices...) = nothing

@prop function accumulate_taped_inputs!((input, inputs...), (dinput, dinputs...),
                                        (partial, partials...), (child, children...),
                                        value, indices...)
    contribution = value * partial
    if eltype(input) <: Integer
        nothing
    elseif input isa LazyExpression
        accumulate_taped!(lazy_cotangent(input, dinput), child, contribution, indices...)
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
    fun(args...) = expr.fun(args..., params...)
    partials = pdv(fun, inputs...)
    partials = partials isa Tuple ? partials : (partials,)
    accumulate_inputs!(expr.inputs, cotangent.dinputs, partials, value, cartesian...)
    return cotangent
end

@prop accumulate_inputs!(::Tuple{}, ::Tuple{}, ::Tuple{}, value, indices...) = nothing

@prop function accumulate_inputs!((input, inputs...), (dinput, dinputs...), (partial, partials...), value, indices...)
    contribution = value * partial
    if eltype(input) <: Integer
        nothing
    elseif input isa LazyExpression
        cotangent = lazy_cotangent(input, dinput)
        accumulate!(cotangent, contribution, indices...)
    else
        dinput[indices...] += contribution
    end
    accumulate_inputs!(inputs, dinputs, partials, value, indices...)
end

end
