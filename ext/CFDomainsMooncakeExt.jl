module CFDomainsMooncakeExt

using Base: @propagate_inbounds as @prop

import CFDomains.VoronoiOperators as Ops
using CFDomains.VoronoiOperators: apply!, apply_adj!, apply_internal!, VoronoiOperator
using CFDomains.VoronoiOperators: LazyDiagonalOp, WritableDVP

import Mooncake
using Mooncake: CoDual, NoTangent, NoPullback, NoFData, NoRData, zero_fcodual, primal, tangent

Mooncake.tangent_type(::Type{<:VoronoiOperator}) = NoTangent

const CoVector{F} = CoDual{<:AbstractVector{F}, <:AbstractVector{F}}
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

function apply!_rrule!!(foutput::CoVector{F}, op::CoOperator{1,1}, finput::CoVector{F}) where F
    # @info "apply!_rrule!!" typeof(foutput) typeof(op) typeof(finput)
    output, stencil, input = primal(foutput), primal(op), primal(finput)
    output0 = archive(output)    
    dout, din = tangent(foutput), tangent(finput)
    extras = apply_internal!(output, stencil, input) # inputs needed by pullback, if any
    function apply!_pullback!!(::NoRData)
        restore!(output, output0) # undo mutation
        apply_adj!(dout, stencil, din, extras)
        return NoRData(), NoRData(), NoRData(), NoRData() # rdata for (apply!, output, op, input)
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
struct ReadableCDP{T, V<:AbstractVector{T}} <: AbstractVector{T}
    diag::V
    ∂x::V
end
@prop Base.getindex(∂y::ReadableCDP, i) = ∂y.diag[i]*∂y.∂x[i]
@prop Ops.setzero!(∂y::ReadableCDP, i) = ∂y.∂x[i]=0
Base.eachindex(∂y::ReadableCDP) = eachindex(∂y.∂x)

Mooncake.tangent_type(::Type{<:WritableDVP{T,D,V}}) where {T,D,V} = ReadableCDP{T, V}
Mooncake.rdata_type(::Type{<:ReadableCDP}) = NoRData
Mooncake.fdata_type(::Type{T}) where {T<:ReadableCDP} = T

# reverse rule for (::LazyDiagonalOp)(args...)
Mooncake.tangent_type(::Type{<:LazyDiagonalOp}) = NoTangent
const CoLazyDiagonalOp{V} = CoDual{LazyDiagonalOp{V}, NoFData}
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{LazyDiagonalOp, Vararg}
function Mooncake.rrule!!(op::CoLazyDiagonalOp, field::CoVector)
    diag, x, ∂x = primal(op).diag, primal(field), tangent(field)
    return CoDual(WritableDVP(diag, x), ReadableCDP(diag, ∂x)), NoPullback(op, field)
end

end
