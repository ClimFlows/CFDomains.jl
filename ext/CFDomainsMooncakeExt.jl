module CFDomainsMooncakeExt

using Base: @propagate_inbounds as @prop

using CFDomains.LazyExpressions: LazyExpression, lazy_expr, cartesian_indices, get, pdv
using CFDomains.LazyOperators: LazyDiagonalOp, WritableDVP
import CFDomains.LazyOperators as Ops

import Mooncake
using Mooncake: CoDual, NoTangent, NoPullback, NoFData, NoRData
using Mooncake: zero_fcodual, primal, tangent, lgetfield

const CoVector{F} = CoDual{<:AbstractVector{F}, <:AbstractVector{F}}
const CoArray{F} = CoDual{<:AbstractArray{F}, <:AbstractArray{F}}
const CoNumber{F} = CoDual{F,NoFData}
CoFunction(f) = CoDual{typeof(f), NoFData}

# `y = Diag(x)` where `Diag` is a `LazyDiagonalOp` is a WritableDVP
# (diagonal-vector-product), a write-only AbstractArray
# to be passed to a VoronoiOperator `op` as an output argument.
# We want the adjoint of the VoronoiOperator to read from
# the tangent `∂y` of `y`. The latter is a ReadableCDP (covector-diagonal product)
# which reads from the tangent `∂x` of `x`. 
# For this we need the `rrule!!` for `Diag` to return `∂y` as FData
# which is then passed to the `rrule!!` for `op`.

# ∂y[i] == diag[i] * ∂x[i]
struct ReadableCDP{N,T,D<:AbstractVector,V<:AbstractArray{T,N}} <: AbstractArray{T,N}
    diag::D
    ∂x::V
end
Base.eachindex(∂y::ReadableCDP) = eachindex(∂y.∂x)
Base.axes(∂y::ReadableCDP) = axes(∂y.∂x)

@prop Base.getindex(∂y::ReadableCDP{1}, i)    = ∂y.diag[i]*∂y.∂x[i]
@prop Base.getindex(∂y::ReadableCDP{2}, k, i) = ∂y.diag[i]*∂y.∂x[k,i]

@prop Ops.setzero!(∂y::ReadableCDP, i) = ∂y.∂x[i]=0

Mooncake.tangent_type(::Type{<:WritableDVP{N,T,D,V}}) where {N,T,D,V} = ReadableCDP{N,T,D,V}
Mooncake.rdata_type(::Type{<:ReadableCDP}) = NoRData
Mooncake.fdata_type(::Type{T}) where {T<:ReadableCDP} = T

# reverse rule for (::LazyDiagonalOp)(args...)
Mooncake.tangent_type(::Type{<:LazyDiagonalOp}) = NoTangent
const CoLazyDiagonalOp{V} = CoDual{LazyDiagonalOp{V}, NoFData}
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{LazyDiagonalOp, Vararg}
function Mooncake.rrule!!(op::CoLazyDiagonalOp, field::CoArray)
    diag, x, ∂x = primal(op).diag, primal(field), tangent(field)
    return CoDual(WritableDVP(diag, x), ReadableCDP(diag, ∂x)), NoPullback(op, field)
end

# With 
#   @lazy c(a,b) = a*b
# `c` is a `LazyExpression`, a read-only AbstractArray
# which can be passed to a VoronoiOperator `op` as an input argument.
# We want the adjoint of the VoronoiOperator to write to
# the tangent `∂c` of `c`. The latter is a TLazyExpression
# which writes to the tangents `∂a`, `∂b` of `a`, `b`.
# For this we need the `rrule!!` for `LazyExpression` to return `∂y` as FData
# which is then passed to the `rrule!!` for `op`.

struct TLazyExpression{T, N, Fun, Inputs, Params, DInputs} <: AbstractArray{T,N}
    fun :: Fun
    inputs :: Inputs
    params :: Params
    ∂inputs :: DInputs
end
tlazy_expr(expr::LazyExpression{T,N,F,I,P}, ∂inputs::DI) where {T,N,F,I,P,DI} =
    TLazyExpression{T,N,F,I,P,DI}(expr.fun, expr.inputs, expr.params, ∂inputs)
Base.size(tlazy::TLazyExpression) = size(tlazy.inputs[1])
Base.axes(tlazy::TLazyExpression) = axes(tlazy.inputs[1])
Base.IndexStyle(::Type{<:TLazyExpression}) = IndexCartesian()

Mooncake.tangent_type(::Type{<:LazyExpression{T,N,F,I,P}}) where {T,N,F,I,P} =
    TLazyExpression{T,N,F,I,P,Mooncake.tangent_type(I)}
Mooncake.rdata_type(::Type{<:TLazyExpression}) = NoRData
Mooncake.fdata_type(::Type{T}) where {T<:TLazyExpression} = T

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(lazy_expr), Any, Any, Any}

function Mooncake.rrule!!(colazy::CoFunction(lazy_expr), cofun::CoDual, coinputs::CoDual, coparams::CoDual)
    fun, inputs, params, ∂inputs = primal(cofun), primal(coinputs), primal(coparams), tangent(coinputs)
    lazy = lazy_expr(fun, inputs, params)
    ∂lazy = tlazy_expr(lazy, ∂inputs)
    return CoDual(lazy, ∂lazy), NoPullback(colazy, cofun, coinputs, coparams)
end

@prop Ops.subfrom!(out::TLazyExpression, v, indices...) = Ops.addto!(out, -v, indices...)

@prop function Ops.addto!(out::TLazyExpression, v, indices...)
    expr = lazy_expr(out.fun, out.inputs, out.params)
    cartesian = cartesian_indices(expr, indices)
    params = get(out.params, cartesian...)
    inputs = get(out.inputs, cartesian...)
    fun(ins...) = out.fun(ins..., params...)
    _, derivatives = pdv(fun, inputs...)
    partials = derivatives isa Tuple ? derivatives : (derivatives,)
    addto_lazy!(out.∂inputs, partials, v, cartesian...)
    return out
end

@prop addto_lazy!(::Tuple{}, ::Tuple{}, v, indices...) = nothing

@prop function addto_lazy!((∂input, ∂inputs...), (partial, partials...), v, indices...)
    contribution = v * partial
    if ∂input isa TLazyExpression
        Ops.addto!(∂input, contribution, indices...)
    else
        ∂input[indices...] += contribution
    end
    addto_lazy!(∂inputs, partials, v, indices...)
end

end
