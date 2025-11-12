module CFDomainsMooncakeExt

# codual = reverse codual = primal + rdata
# fcodual = forward codual = primal + fdata

import Mooncake
using Mooncake: CoDual, NoTangent, NoPullBack, NoFData, NoRData, zero_fcodual, primal, tangent
using CFDomains.VoronoiOperators: apply!, apply_adj!, apply_internal!, VoronoiOperator

using CFDomains.VoronoiOperators: LazyDiagonalOp, LazyDVP, LazyVDP

const CoVector{F} = CoDual{<:AbstractVector{F}, <:AbstractVector{F}}
const CoNumber{F} = CoDual{F,NoFData}
const CoOperator{A,B} = CoDual{<:VoronoiOperator{A,B}, NoFData}
CoFunction(f) = CoDual{typeof(f), NoFData}

Mooncake.tangent_type(::Type{<:VoronoiOperator}) = NoTangent

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(apply!), Vararg}
Mooncake.rrule!!(::CoFunction(apply!), fx::Vararg) = apply!_rrule!!(fx...)

function apply!_rrule!!(foutput::CoVector{F}, op::CoOperator{1,1}, finput::CoVector{F}) where F
    output, stencil, input = primal(foutput), primal(op), primal(finput)
    output_ = copy(output) # to undo mutation during backward pass    
    dout, din = tangent(foutput), tangent(finput)
    function apply!_pullback!!(::NoRData)
        copy!(output, output_) # undo mutation
        apply_adj!(dout, stencil, din)
        return NoRData(), NoRData(), NoRData(), NoRData() # rdata for (apply!, output, op, input)
    end
    apply_internal!(output, stencil, input)
    return zero_fcodual(nothing), apply!_pullback!!
end

# `y = Diag(x)` where `Diag` is a `LazyDiagonalOp` is a LazyDVP
# (diagonal-vector-product), a write-only AbstractArray
# to be passed to a VoronoiOperator `op` as an output argument.
# We want the adjoint of the VoronoiOperator to read from
# the tangent `∂y` of `y`. The latter is a LazyVDP (vector-diagonal product)
# which reads from the tangent `∂x` of `x`. 
# For this we need the `rrule!!` for `Diag` to return `∂y` as FData
# which is then passed to the `rruel!!` for `op`.

Mooncake.tangent_type(::Type{<:LazyDiagonalOp}) = NoTangent
Mooncake.tangent_type(::Type{<:LazyDVP{V}}) where V = LazyVDP{V}

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{LazyDiagonalOp, Vararg}

const CoLazyDiagonalOp{V} = CoDual{LazyDiagonalOp{V}, NoFData}

function Mooncake.rrule!!(op::CoLazyDiagonalOp, field::CoVector)
    diag, x, ∂x = primal(op).diag, primal(field), tangent(field)
    return CoDual(LazyDVP(diag, x), LazyVDP(diag, ∂x)), NoPullBack(op, field)
end

# ∂y[i] == diag[i] * ∂x[i]
struct LazyVDP{T, V<:AbstractVector{T}} <: AbstractVector{T}
    diag::V
    ∂x::V
end
@prop Base.getindex(∂y::LazyVDP, i) = ∂y.diag[i]*∂y.∂x[i]

end
