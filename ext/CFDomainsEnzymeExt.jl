module CFDomainsEnzymeExt

using Base: @propagate_inbounds as @prop

using Enzyme: MixedDuplicated
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

lazy_cotangent(input::MixedDuplicated{<:LazyExpression}) =
    lazy_cotangent(input.val, input.dval[])

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
    accumulate_inputs!(cotangent.dinputs, partials, value, cartesian...)
    return cotangent
end

@prop accumulate_inputs!(::Tuple{}, ::Tuple{}, value, indices...) = nothing

@prop function accumulate_inputs!((dinput, dinputs...), (partial, partials...), value, indices...)
    dinput[indices...] += value * partial
    accumulate_inputs!(dinputs, partials, value, indices...)
end

end
