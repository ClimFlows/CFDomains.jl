module VoronoiOperators

using Base: @propagate_inbounds as @prop
using ManagedLoops: @unroll

import CFDomains.Stencils

abstract type VoronoiOperator{In,Out} end

#================== lazy diagonal operator ===============#

struct LazyDiagonalOp{V<:AbstractVector}
    diag::V
end
struct WritableDVP{T, D<:AbstractVector, V<:AbstractVector{T}} <: AbstractVector{T}
    diag::D
    x::V
end

"""
    as_density = AsDensity(vsphere) # a `LazyDiagonalOp`
    density = as_density(scalar)    # a `WritableDVP` (diagonal-vector-product)
    op!(density, ...)               # pass `density as *output* argument

Given a zero-form `scalar`, `as_density` returns the equivalent two-form
as a lazy, write-only `AbstractArray` to be passed to a VoronoiOperator `op!` 
as an *output* argument.
"""
AsDensity(vsphere) = LazyDiagonalOp(vsphere.inv_Ai)
(op::LazyDiagonalOp)(field) = WritableDVP(op.diag, field)

# x[i] == diag[i] * y[i]
Base.eachindex(y::WritableDVP) = eachindex(y.x)
@prop Base.setindex!(y::WritableDVP, v, i) = y.x[i] = y.diag[i]*v
@prop addto!(y::WritableDVP, v, i) = y.x[i] += y.diag[i]*v
@prop subfrom!(y::WritableDVP, v, i) = y.x[i] -= y.diag[i]*v

#================ actions: what to do on the output of operators ================#

@prop set!(out, v, i)      = out[i] = v
@prop setminus!(out, v, i) = out[i] = -v
@prop addto!(out, v, i)    = out[i] += v
@prop subfrom!(out, v, i)  = out[i] -= v
@prop setzero!(out, i)     = out[i] = 0
@prop unchanged!(out, i)   = nothing

# (out, in) := (op(in), in) => (∂out, ∂in) := (0, ∂in + opᵀ(∂out))
adj_action_in(::typeof(set!)) = addto!
adj_action_out(::typeof(set!)) = setzero!

# (out, in) := (-op(in), in) => (∂out, ∂in) := (0, ∂in - opᵀ(∂out))
adj_action_in(::typeof(setminus!)) = subfrom!
adj_action_out(::typeof(setminus!)) = setzero!

# (out, in) := (out + op(in), in) => (∂out, ∂in) := (∂out, ∂in + opᵀ(∂out))
adj_action_in(::typeof(addto!)) = addto!
adj_action_out(::typeof(addto!)) = unchanged!

# (out, in) := (out - op(in), in)  => (∂out, ∂in) := (∂out, ∂in - opᵀ(∂out))
adj_action_in(::typeof(subfrom!), ∂in, i, ∂in_i) = subfrom!
adj_action_out(::typeof(subfrom!), ∂out, i) = unchanged!

#==================== VoronoiOperator{1,1} ==============#

(op::VoronoiOperator{1,1})(output, input) = apply!(output, op, input)

function apply!(output, stencil::VoronoiOperator{1,1}, input) 
    apply_internal!(output, stencil, input)
    return nothing
end

function apply_adj!(∂out, op::VoronoiOperator{1,1}, ∂in) 
    apply_adj_internal!(∂out, op, ∂in)
    action! = adj_action_out(op.action!)
    @inbounds for i in eachindex(∂out)
        action!(∂out, i)
    end
end

#========== gradient operator and its adjoint ===========#

struct Gradient{Action, F<:AbstractFloat} <: VoronoiOperator{1,1}
    action!::Action # how to combine op(input) with output
    edge_left_right::Matrix{Int32}
    # for the adjoint
    primal_deg::Vector{Int32}
    primal_edge::Matrix{Int32}
    primal_ne::Matrix{F}
end
Gradient(sph, action! = set!) = Gradient(action!, sph.edge_left_right, sph.primal_deg, sph.primal_edge, sph.primal_ne)

@inline function apply_internal!(output, op::Gradient, input)
    loop_simple(op.action!, op, output, Stencils.gradient, input)
end

@inline function apply_adj_internal!(∂out, op::Gradient, ∂in)
    loop_cell(flip(adj_action_in(op.action!)), op, ∂in, Stencils.div_form, ∂out)
end

#========== divergence operator and its adjoint ===========#

struct Divergence{Action, F<:AbstractFloat} <: VoronoiOperator{1,1}
    action!::Action # how to combine op(input) with output
    primal_deg::Vector{Int32}
    primal_edge::Matrix{Int32}
    primal_ne::Matrix{F}
    # for the adjoint
    edge_left_right::Matrix{Int32}
end
Divergence(sph, action! = set!) = Divergence(action!, sph.primal_deg, sph.primal_edge, sph.primal_ne, sph.edge_left_right)

@inline function apply_internal!(output, op::Divergence, input)
    loop_cell(op.action!, op, output, Stencils.div_form, input)
end

@inline function apply_adj_internal!(∂out, op::Divergence, ∂in)
    loop_simple(flip(adj_action_in(op.action!)), op, ∂in, Stencils.gradient, ∂out)
end

#========== TriSK operator and its adjoint ===========#

struct TRiSK{Action, F<:AbstractFloat} <: VoronoiOperator{1,1}
    action!::Action # how to combine op(input) with output
    trisk_deg::Vector{Int32}
    trisk::Matrix{Int32}
    wee::Matrix{F}
end
TRiSK(sph, action! = set!) = TRiSK(action!, sph.trisk_deg, sph.trisk, sph.wee)

@inline function apply_internal!(output, op::TRiSK, input)
    loop_trisk(op.action!, op, output, Stencils.TRiSK, input)
end

@inline function apply_adj_internal!(∂out, op::TRiSK, ∂in)
    loop_trisk(flip(adj_action_in(op.action!)), op, ∂in, Stencils.TRiSK, ∂out)
end

#============ Loop styles ==========#

@inline function loop_simple(action!, op, output, stencil, input)
    @inbounds for i in eachindex(output)
        st = stencil(op, i)
        action!(output, st(input), i)
    end
    return nothing
end

@inline function loop_cell(action!, op, output, stencil, input)
    @inbounds for cell in eachindex(output)
        deg = op.primal_deg[cell]
        @unroll deg in 5:7 begin
            st = stencil(op, cell, Val(deg))
            action!(output, st(input), cell)
        end
    end
    return nothing
end

@inline function loop_trisk(action!, op, output, stencil, input)
    @inbounds for edge in eachindex(output)
        deg = op.trisk_deg[edge]
        @unroll deg in 9:11 begin
            st = stencil(op, edge, Val(deg))
            action!(output, st(input), edge)
        end
    end
    return nothing
end

flip(::typeof(addto!)) = subfrom!
flip(::typeof(subfrom!)) = addto!

#===================== automatic partial derivatives =================#

"""
    fa = pdv(fun, a)
    fa, fb = pdv(fun, a, b)
    fa, fb, fc = pdv(fun, a, b, c)

Return the partial derivatives of scalar function `fun` evaluated at input `a, ...`.
*This function is implemented only when the package ForwardDiff is loaded*
either directly from the main program or via some dependency.
"""
function pdv end

end
