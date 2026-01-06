module CFDomainsMooncakeExt

using Base: @propagate_inbounds as @prop

using CFDomains.LazyExpressions: LazyExpression, lazy_expr
import CFDomains.VoronoiOperators as Ops
using CFDomains.VoronoiOperators: apply!, apply_adj!, apply_internal!, VoronoiOperator
using CFDomains.VoronoiOperators: LazyDiagonalOp, WritableDVP

import Mooncake
using Mooncake: CoDual, NoTangent, NoPullback, NoFData, NoRData
using Mooncake: zero_fcodual, primal, tangent, lgetfield

Mooncake.tangent_type(::Type{<:VoronoiOperator}) = NoTangent

const CoVector{F} = CoDual{<:AbstractVector{F}, <:AbstractVector{F}}
const CoArray{F} = CoDual{<:AbstractArray{F}, <:AbstractArray{F}}
const CoNumber{F} = CoDual{F,NoFData}
const CoOperator{A,B} = CoDual{<:VoronoiOperator{A,B}, NoFData}
CoFunction(f) = CoDual{typeof(f), NoFData}

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(apply!), Vararg}
Mooncake.rrule!!(::CoFunction(apply!), fx::Vararg) = apply!_rrule!!(fx...)

# Keep a copy of output argument x.
archive(x) = copy(x)
archive(y::WritableDVP) = archive(y.x)
# Restore the archived value of output argument x
restore!(x,x0) = copy!(x, x0)
restore!(y::WritableDVP, x0) = restore!(y.x, x0)

function apply!_rrule!!(foutput::CoArray{F}, fmgr::CoDual, op::CoOperator{1,1}, finput::CoArray{F}) where F
    # @info "apply!_rrule!!" typeof(foutput) typeof(op) typeof(finput)
    output, mgr, stencil, input = primal(foutput), primal(fmgr), primal(op), primal(finput)
    output0 = archive(output)    
    dout, din = tangent(foutput), tangent(finput)
    extras = apply_internal!(output, mgr, stencil, input) # inputs needed by pullback, if any
    function apply!_pullback!!(::NoRData)
        restore!(output, output0) # undo mutation
        apply_adj!(dout, mgr, stencil, din, extras)
        # rdata for (apply!, output, mgr, op, input)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(nothing), apply!_pullback!!
end

function apply!_rrule!!(foutput::CoArray{F}, fmgr::CoDual, op::CoOperator{1,2}, finput1::CoArray{F}, finput2::CoArray{F}) where F
#    @info "apply!_rrule!!" typeof(foutput) typeof(op) typeof(finput1) typeof(finput2)
    output, mgr, stencil, input1, input2 = map(primal, (foutput, fmgr, op, finput1, finput2))
    output0 = archive(output)
    ∂out, ∂in1,  ∂in2 = tangent(foutput), tangent(finput1), tangent(finput2)
    extras = apply_internal!(output, mgr, stencil, input1, input2) # inputs needed by pullback, if any
    function apply!_pullback!!(::NoRData)
        restore!(output, output0) # undo mutation
        apply_adj!(∂out, mgr, stencil, ∂in1, ∂in2, extras)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData() # rdata for (apply!, output, op, input)
    end
    return zero_fcodual(nothing), apply!_pullback!!
end

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

struct TLazyExpression{T, N, Fun, Inputs, Params} <: AbstractArray{T,N}
    fun :: Fun
    inputs :: Inputs
    params :: Params
    ∂inputs :: Inputs
end
tlazy_expr(expr::LazyExpression{T,N,F,I,P}, ∂inputs::I) where {T,N,F,I,P} = TLazyExpression{T,N,F,I,P}(expr.fun, expr.inputs, expr.params, ∂inputs)
Base.eachindex(tlazy::TLazyExpression) = eachindex(tlazy.∂inputs...)

Mooncake.tangent_type(::Type{<:LazyExpression{T,N,F,I,P}}) where {T,N,F,I,P} = TLazyExpression{T,N,F,I,P}
Mooncake.rdata_type(::Type{<:TLazyExpression}) = NoRData
Mooncake.fdata_type(::Type{T}) where {T<:TLazyExpression} = T

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(lazy_expr), Any, Any, Any}

function Mooncake.rrule!!(colazy::CoFunction(lazy_expr), cofun::CoDual, coinputs::CoDual, coparams::CoDual)
    fun, inputs, params, ∂inputs = primal(cofun), primal(coinputs), primal(coparams), tangent(coinputs)
    lazy = lazy_expr(fun, inputs, params)
    ∂lazy = tlazy_expr(lazy, ∂inputs)
    return CoDual(lazy, ∂lazy), NoPullback(colazy, cofun, coinputs, coparams)
end

@prop Ops.subfrom!(out::TLazyExpression, v, i)  = Ops.addto!(out, -v, i)

@prop function Ops.addto!(out::TLazyExpression, v, i)
    params = get_tuple(out.params, i)
    inputs = get_tuple(out.inputs, i)
    fun(ins...) = out.fun(ins..., params...)
    derivatives = Ops.pdv(fun, inputs...)
    addto_lazy(i, v, out.∂inputs, derivatives)
end

@prop geti(a::AbstractVector, i) = a[i]
geti(a::Number, _) = a

@prop get_tuple((a,)::Tuple{Any}, i) = (geti(a,i), )
@prop get_tuple((a,b)::Tuple{Any,Any}, i) = (geti(a,i), geti(b,i))
@prop get_tuple((a,b,c)::Tuple{Any,Any,Any}, i) = (geti(a,i), geti(b,i), geti(c,i))

@prop function addto_lazy(i, v, (∂a,)::Tuple{Any}, da)
    ∂a[i] += v*da
end

@prop function addto_lazy(i, v, (∂a,∂b)::Tuple{Any,Any}, (da,db))
    ∂a[i] += v*da
    ∂b[i] += v*db
end

end
