module VoronoiOperators

using Base: @propagate_inbounds as @prop
using ManagedLoops: @unroll

import CFDomains.Stencils

abstract type VoronoiOperator{In,Out} end

@inline function (T::Type{<:VoronoiOperator})(sph, action! = set!)
    _, names... = fieldnames(T)
    fields = map(name->getproperty(sph, name), names)
    return T(action!, fields...)
end

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

#========== actions: what to do on the output of operators ===========#

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

#================================================================#
#===================== VoronoiOperator{1,1} =====================#
#================================================================#

(op::VoronoiOperator{1,1})(output, input) = apply!(output, op, input)

function apply!(output, stencil::VoronoiOperator{1,1}, input) 
    apply_internal!(output, stencil, input)
    return nothing
end

function apply_adj!(∂out, op::VoronoiOperator{1,1}, ∂in, extras)
    apply_adj_internal!(∂out, op, ∂in, extras)
    action! = adj_action_out(op.action!)
    @inbounds for i in eachindex(∂out)
        action!(∂out, i)
    end
end

#========== primal => dual ==========#

struct DualFromPrimal{Action, F<:AbstractFloat} <: VoronoiOperator{1,1}
    action!::Action # how to combine op(input) with output
    dual_vertex::Matrix{Int32}
    Avi::Matrix{F}
    # for the adjoint
    primal_deg::Vector{Int32}
    primal_vertex::Matrix{Int32}
    Aiv::Matrix{F}
end

@inline function apply_internal!(output, op::DualFromPrimal, input)
    loop_simple(op.action!, op, output, Stencils.average_iv_form, input)
    return nothing
end

@inline function apply_adj_internal!(∂out, op::DualFromPrimal, ∂in, ::Nothing)
    loop_cell(adj_action_in(op.action!), op, ∂in, Stencils.average_vi_form, ∂out)
end

#========== gradient ===========#

struct Gradient{Action, F<:AbstractFloat} <: VoronoiOperator{1,1}
    action!::Action # how to combine op(input) with output
    edge_left_right::Matrix{Int32}
    # for the adjoint
    primal_deg::Vector{Int32}
    primal_edge::Matrix{Int32}
    primal_ne::Matrix{F}
end

@inline function apply_internal!(output, op::Gradient, input)
    loop_simple(op.action!, op, output, Stencils.gradient, input)
    return nothing
end

@inline function apply_adj_internal!(∂out, op::Gradient, ∂in, ::Nothing)
    loop_cell(flip(adj_action_in(op.action!)), op, ∂in, Stencils.div_form, ∂out)
end

#========== divergence ===========#

struct Divergence{Action, F<:AbstractFloat} <: VoronoiOperator{1,1}
    action!::Action # how to combine op(input) with output
    primal_deg::Vector{Int32}
    primal_edge::Matrix{Int32}
    primal_ne::Matrix{F}
    # for the adjoint
    edge_left_right::Matrix{Int32}
end

@inline function apply_internal!(output, op::Divergence, input)
    loop_cell(op.action!, op, output, Stencils.div_form, input)
    return nothing
end

@inline function apply_adj_internal!(∂out, op::Divergence, ∂in, ::Nothing)
    loop_simple(flip(adj_action_in(op.action!)), op, ∂in, Stencils.gradient, ∂out)
end

#========== curl ===========#

struct Curl{Action, F<:AbstractFloat} <: VoronoiOperator{1,1}
    action!::Action # how to combine op(input) with output
    dual_edge::Matrix{Int32}
    dual_ne::Matrix{F}
    edge_down_up::Matrix{Int32} # for gradperp
end

@inline function apply_internal!(output, op::Curl, input)
    loop_simple(op.action!, op, output, Stencils.curl, input)
    return nothing
end

@inline function apply_adj_internal!(∂out, op::Curl, ∂in, ::Nothing)
    loop_simple(adj_action_in(op.action!), op, ∂in, Stencils.gradperp, ∂out)
end

#========== TriSK ===========#

struct TRiSK{Action, F<:AbstractFloat} <: VoronoiOperator{1,1}
    action!::Action # how to combine op(input) with output
    trisk_deg::Vector{Int32}
    trisk::Matrix{Int32}
    wee::Matrix{F}
end

@inline function apply_internal!(output, op::TRiSK, input)
    loop_trisk(op.action!, op, output, Stencils.TRiSK, input)
    return nothing
end

@inline function apply_adj_internal!(∂out, op::TRiSK, ∂in, ::Nothing)
    loop_trisk(flip(adj_action_in(op.action!)), op, ∂in, Stencils.TRiSK, ∂out)
end

#========== Squared covector ===========#

struct SquaredCovector{Action, F} <: VoronoiOperator{1,1}
    action!::Action # how to combine op(input) with output
    primal_deg::Vector{Int32}
    primal_edge::Matrix{Int32}
    le_de::Vector{F}
    # for the adjoint
    edge_left_right::Matrix{Int32}
end

@inline function apply_internal!(output, op::SquaredCovector, input)
    loop_cell(op.action!, op, output, Stencils.squared_covector, input)
    return input # will be needed by adjoint
end

@inline @inbounds function stencil_squared_adj(op, edge)
    left = op.edge_left_right[1, edge] 
    right = op.edge_left_right[2, edge] 
    hodge = op.le_de[edge]
    @inline value(∂K, ucov) = @inbounds hodge*ucov[edge]*(∂K[left]+∂K[right])
    @inline value(∂K, ucov, k) = @inbounds hodge*ucov[k,edge]*(∂K[k, left]+∂K[k, right])
    return value
end

@inline function apply_adj_internal!(∂K, op::SquaredCovector, ∂ucov, ucov)
    loop_simple(adj_action_in(op.action!), op, ∂ucov, stencil_squared_adj, ∂K, ucov)
end

#================================================================#
#===================== VoronoiOperator{1,2} =====================#
#================================================================#

(op::VoronoiOperator{1,2})(output, in1, in2) = apply!(output, op, in1, in2)

function apply!(output, stencil::VoronoiOperator{1,2}, in1, in2) 
    apply_internal!(output, stencil, in1, in2)
    return nothing
end

function apply_adj!(∂out, op::VoronoiOperator{1,2}, ∂in1, ∂in2, extras)
    apply_adj_internal!(∂out, op, ∂in1, ∂in2, extras)
    action! = adj_action_out(op.action!)
    @inbounds for i in eachindex(∂out)
        action!(∂out, i)
    end
end

#========== Centered flux ===========#

struct CenteredFlux{Action, F} <: VoronoiOperator{1,2}
    action!::Action # how to combine op(input) with output
    le_de::Vector{F}
    edge_left_right::Matrix{Int32}
    # for adjoint
    primal_deg::Vector{Int32}
    primal_edge::Matrix{Int32}
end

@inline function apply_internal!(output, op::CenteredFlux, m, ucov)
    loop_simple(op.action!, op, output, Stencils.centered_flux, m, ucov)
    return m, ucov # needed by adjoint
end

@inline function apply_adj_internal!(∂F, op::CenteredFlux, ∂m, ∂ucov, (m, ucov))
    loop_cell(adj_action_in(op.action!), op, ∂m, Stencils.dot_product_form, ucov, ∂F)
    loop_simple(adj_action_in(op.action!), op, ∂ucov, Stencils.centered_flux, m, ∂F)
end

#========== Energy-conserving TRiSK ===========#

struct EnergyTRiSK{Action, F} <: VoronoiOperator{1,2}
    action!::Action # how to combine op(input) with output
    trisk_deg::Vector{Int32}
    trisk::Matrix{Int32}
    wee::Matrix{F}
end

@inline function apply_internal!(ucov, op::EnergyTRiSK, U, q)
    loop_trisk(op.action!, op, ucov, Stencils.TRiSK, U, q)
    return U, q
end

@inline function apply_adj_internal!(∂ucov, op::EnergyTRiSK, ∂U, ∂q, (U,q))
    loop_trisk(flip(adj_action_in(op.action!)), op, ∂U, Stencils.TRiSK, ∂ucov, q)
    loop_trisk(adj_action_in(op.action!), op, ∂q, Stencils.cross_product, U, ∂ucov)
end

#========== Centered flux divergence ===========#

struct DivCenteredFlux{Action, F<:AbstractFloat} <: VoronoiOperator{1,2}
    action!::Action # how to combine op(input) with output
    primal_deg::Vector{Int32}
    primal_edge::Matrix{Int32}
    primal_neighbour::Matrix{Int32}
    primal_ne::Matrix{F}
    # for the adjoint
    edge_left_right::Matrix{Int32}
end

@inline function apply_internal!(divqF, op::DivCenteredFlux, q, F)
    loop_cell(op.action!, op, divqF, Stencils.div_centered_flux, q, F)
    return q, F
end

@inline function apply_adj_internal!(∂divqF, op::DivCenteredFlux, ∂q, ∂F, (q, F))
    action! = flip(adj_action_in(op.action!))
    loop_simple(action!, op, ∂F, Stencils.mul_grad, q, ∂divqF)
    loop_cell(action!, op, ∂q, Stencils.dot_grad, F, ∂divqF)
end

#=======================================================#
#===================== Loop styles =====================#
#=======================================================#

@inline function loop_simple(action!, op, output, stencil, inputs...)
    @inbounds for i in eachindex(output)
        st = stencil(op, i)
        action!(output, st(inputs...), i)
    end
    return nothing
end

@inline function loop_cell(action!, op, output, stencil, inputs...)
    @inbounds for cell in eachindex(output)
        deg = op.primal_deg[cell]
        @unroll deg in 5:7 begin
            st = stencil(op, cell, Val(deg))
            action!(output, st(inputs...), cell)
        end
    end
    return nothing
end

@inline function loop_trisk(action!, op, output, stencil, inputs...)
    @inbounds for edge in eachindex(output)
        deg = op.trisk_deg[edge]
        @unroll deg in 9:11 begin
            st = stencil(op, edge, Val(deg))
            action!(output, st(inputs...), edge)
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

#=
Shallow-water tendencies Variant 1: m=gh is a zero-form, ucov a 1-form

# allocators
on_cells(tmp) = similar!(m, tmp)
on_edges(tmp) = similar!(ucov, tmp)
on_duals(tmp) = similar!(sphere.Av, tmp)

# constant inputs to lazy arrays must be given names
metric = model.planet.radius^-2 # constant contravariant metric
(; inv_Ai, fcov) = sphere

# operators
cflux! = CenteredFlux(sphere)
square! = SquaredCoVector(sphere) # 1-form -> 2-form
minus_grad! = Gradient(sphere, setminus!) # 0-form -> 1-form
curl! = Curl(sphere) # 1-form -> 2-form
primal_to_dual! = Average_iv(sphere) # 0-form -> 2-form
dual_to_edge! = Average_ve(sphere) # 0-form -> 0-form
substract_trisk! = TriskEnergy(sphere, subfrom!) # (2-form, 0-form at edges) -> 1-form
as_two_form = AsTwoForm(sphere)

# compute temporaries
u2 = on_cells(tmp.K)
U, qe = on_edges(tmp.U), on_edges(tmp.qe)
zeta, mv = on_duals(tmp.zeta), on_duals(tmp.mv)

@lazy ucontra(ucov) = ucov*radius_m2
cflux!(mgr, U, ucontra, m)
square!(mgr, u2, ucov)
curl!(mgr, zetav, ucov)
primal_to_dual!(mgr, mv, m)
@lazy qv(zetav, mv ; fcov) = (zeta+fcov)/mv
dual_to_edge!(mgr, qe, qv)
@lazy B(m, u2 ; inv_Ai) = metric*(m + inv_Ai*u2/2)

# compute tendencies
ducov, dm = on_edges(dstate.ucov), on_cells(dstate.m)
minus_grad!(mgr, ducov, B)
substract_trisk!(mgr, ducov, U, qe)
minus_div!(mgr, as_two_form(dm), U)

=#

#=

Shallow-water tendencies Variant 2: m=gh is a two-form, ucov a 1-form

# operators
cflux = CenteredFlux(sphere)
square = SquaredCoVector(sphere) # 1-form -> 2-form
minus_grad = Gradient(sphere, setminus!) # 0-form -> 1-form
curl = Curl(sphere) # 1-form -> 2-form
primal_to_dual = Average_iv(sphere) # 0-form -> 2-form
dual_to_edge = Average_ve(sphere) # 0-form -> 0-form
substract_trisk = TriskEnergy(sphere, subfrom!) # (2-form, 0-form at edges) -> 1-form

# constant inputs to lazy arrays must be given names
metric = model.planet.radius^-2 # constant contravariant metric
(; inv_Ai, fcov) = sphere

# allocate temporaries
on_cells(tmp) = similar!(tmp, m)
on_edges(tmp) = similar!(tmp, ucov)
on_duals(tmp) = similar!(tmp, sphere.Av)
u2 = on_cells(tmp.K)
U, qe = on_edges(tmp.U), on_edges(tmp.qe)
zeta, mv = on_duals(tmp.zeta), on_duals(tmp.mv)

# compute temporaries

@lazy ucontra(ucov) = metric*ucov
@lazy m0(m ; inv_Ai) = inv_Ai*m0
cflux!(U, ucontra, m0)
square!(u2, ucov)
curl!(zetav, ucov)
primal_to_dual!(mv, m)
@lazy qv(zetav, mv ; fcov) = (zeta+fcov)/mv
dual_to_edge!(qe, qv)
@lazy B(m, u2 ; inv_Ai) = (metric*inv_Ai)*(m + u2/2)

# compute tendencies
ducov = on_edges(dstate.ucov)
dm = on_cells(dstate.m)
minus_grad!(ducov, B)
substract_trisk!(ducov, U, qe)
minus_div!(dm, U)

=#
