module VoronoiOperators

using Base: @propagate_inbounds as @prop
# using Base: @inbounds as @prop

using ManagedLoops: @unroll, @vec, @with

import CFDomains.Stencils

macro inb(expr)
    esc(:(@inbounds $expr))
#    esc(expr)
end

abstract type VoronoiOperator{In,Out} end

@inline (::Type{T})(sph) where { T<:VoronoiOperator } = T(sph, set!)

@inline @generated function (::Type{T})(sph, action!::Action) where { T<:VoronoiOperator, Action }
    fields = [ :( getproperty(sph, $(QuoteNode(name)))) for name in fieldnames(T)]
    Expr(:call, T, :action!, fields[2:end]...)
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

@prop set!(out, v, k, i)   = out[k, i] = v

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

(op::VoronoiOperator{1,1})(output, mgr, input) = apply!(output, mgr, op, input)

@inline function apply!(output, mgr, stencil::VoronoiOperator{1,1}, input) 
    apply_internal!(output, mgr, stencil, input)
    return nothing
end

function apply_adj!(∂out, mgr, op::VoronoiOperator{1,1}, ∂in, extras)
    apply_adj_internal!(∂out, mgr, op, ∂in, extras)
    action! = adj_action_out(op.action!)
    @inb for i in eachindex(∂out)
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

@inline function apply_internal!(output, mgr, op::DualFromPrimal, input)
    loop_simple(output, mgr, op.action!, op, Stencils.average_iv_form, input)
    return nothing
end

@inline function apply_adj_internal!(∂out, mgr, op::DualFromPrimal, ∂in, ::Nothing)
    loop_cell(∂in, mgr, adj_action_in(op.action!), op, Stencils.average_vi_form, ∂out)
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

@inline function apply_internal!(output, mgr, op::Gradient, input)
    loop_simple(output, mgr, op.action!, op, Stencils.gradient, input)
    return nothing
end

@inline function apply_adj_internal!(∂out, mgr, op::Gradient, ∂in, ::Nothing)
    loop_cell(∂in, mgr, flip(adj_action_in(op.action!)), op, Stencils.div_form, ∂out)
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

@inline function apply_internal!(output, mgr, op::Divergence, input)
    loop_cell(output, mgr, op.action!, op, Stencils.div_form, input)
    return nothing
end

@inline function apply_adj_internal!(∂out, mgr, op::Divergence, ∂in, ::Nothing)
    loop_simple(∂in, mgr, flip(adj_action_in(op.action!)), op, Stencils.gradient, ∂out)
end

#========== curl ===========#

struct Curl{Action, F<:AbstractFloat} <: VoronoiOperator{1,1}
    action!::Action # how to combine op(input) with output
    dual_edge::Matrix{Int32}
    dual_ne::Matrix{F}
    edge_down_up::Matrix{Int32} # for gradperp
end

@inline function apply_internal!(output, mgr, op::Curl, input)
    loop_simple(output, mgr, op.action!, op, Stencils.curl, input)
    return nothing
end

@inline function apply_adj_internal!(∂out, mgr, op::Curl, ∂in, ::Nothing)
    loop_simple(∂in, mgr, adj_action_in(op.action!), op, Stencils.gradperp, ∂out)
end

#========== TriSK ===========#

struct TRiSK{Action, F<:AbstractFloat} <: VoronoiOperator{1,1}
    action!::Action # how to combine op(input) with output
    trisk_deg::Vector{Int32}
    trisk::Matrix{Int32}
    wee::Matrix{F}
end

@inline function apply_internal!(output, mgr, op::TRiSK, input)
    loop_trisk(output, mgr, op.action!, op, Stencils.TRiSK, input)
    return nothing
end

@inline function apply_adj_internal!(∂out, mgr, op::TRiSK, ∂in, ::Nothing)
    loop_trisk(∂in, mgr, flip(adj_action_in(op.action!)), op, Stencils.TRiSK, ∂out)
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

@inline function apply_internal!(output, mgr, op::SquaredCovector, input)
    loop_cell(output, mgr, op.action!, op, Stencils.squared_covector, input)
    return input # will be needed by adjoint
end

@inline @inb function stencil_squared_adj(op, edge)
    left = op.edge_left_right[1, edge] 
    right = op.edge_left_right[2, edge] 
    hodge = op.le_de[edge]
    @inline value(∂K, ucov) = @inb hodge*ucov[edge]*(∂K[left]+∂K[right])
    @inline value(∂K, ucov, k) = @inb hodge*ucov[k,edge]*(∂K[k, left]+∂K[k, right])
    return value
end

@inline function apply_adj_internal!(∂K, mgr, op::SquaredCovector, ∂ucov, ucov)
    loop_simple(∂ucov, mgr, adj_action_in(op.action!), op, stencil_squared_adj, ∂K, ucov)
end

#================================================================#
#===================== VoronoiOperator{1,2} =====================#
#================================================================#

(op::VoronoiOperator{1,2})(output, mgr, in1, in2) = apply!(output, mgr, op, in1, in2)

function apply!(output, mgr, stencil::VoronoiOperator{1,2}, in1, in2) 
    apply_internal!(output, mgr, stencil, in1, in2)
    return nothing
end

function apply_adj!(∂out, mgr, op::VoronoiOperator{1,2}, ∂in1, ∂in2, extras)
    apply_adj_internal!(∂out, mgr, op, ∂in1, ∂in2, extras)
    action! = adj_action_out(op.action!)
    @inb for i in eachindex(∂out)
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

@inline function apply_internal!(output, mgr, op::CenteredFlux, m, ucov)
    loop_simple(output, mgr, op.action!, op, Stencils.centered_flux, m, ucov)
    return m, ucov # needed by adjoint
end

@inline function apply_adj_internal!(∂F, mgr, op::CenteredFlux, ∂m, ∂ucov, (m, ucov))
    loop_cell(∂m, mgr, adj_action_in(op.action!), op, Stencils.dot_product_form, ucov, ∂F)
    loop_simple(∂ucov, mgr, adj_action_in(op.action!), op, Stencils.centered_flux, m, ∂F)
end

#========== Energy-conserving TRiSK ===========#

struct EnergyTRiSK{Action, F} <: VoronoiOperator{1,2}
    action!::Action # how to combine op(input) with output
    trisk_deg::Vector{Int32}
    trisk::Matrix{Int32}
    wee::Matrix{F}
end

@inline function apply_internal!(ucov, mgr, op::EnergyTRiSK, U, q)
    loop_trisk(ucov, mgr, op.action!, op, Stencils.TRiSK, U, q)
    return U, q
end

@inline function apply_adj_internal!(∂ucov, mgr, op::EnergyTRiSK, ∂U, ∂q, (U,q))
    loop_trisk(∂U, mgr, flip(adj_action_in(op.action!)), op, Stencils.TRiSK, ∂ucov, q)
    loop_trisk(∂q, mgr, adj_action_in(op.action!), op, Stencils.cross_product, U, ∂ucov)
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

@inline function apply_internal!(divqF, mgr, op::DivCenteredFlux, q, F)
    loop_cell(divqF, mgr, op.action!, op, Stencils.div_centered_flux, q, F)
    return q, F
end

@inline function apply_adj_internal!(∂divqF, mgr, op::DivCenteredFlux, ∂q, ∂F, (q, F))
    action! = flip(adj_action_in(op.action!))
    loop_simple(∂F, mgr, action!, op, Stencils.mul_grad, q, ∂divqF)
    loop_cell(∂q, mgr, action!, op, Stencils.dot_grad, F, ∂divqF)
end

#========== Multiplied gradient ===========#

struct MulGradient{Action, F<:AbstractFloat} <: VoronoiOperator{1,2}
    action!::Action # how to combine op(input) with output
    edge_left_right::Matrix{Int32}
    # for the adjoint
    primal_deg::Vector{Int32}
    primal_edge::Matrix{Int32}
    primal_neighbour::Matrix{Int32}
    primal_ne::Matrix{F}
end

@inline function apply_internal!(u, mgr, op::MulGradient, a, b)
    loop_simple(u, mgr, op.action!, op, Stencils.mul_grad, a, b)
    return a, b
end

@inline function apply_adj_internal!(∂u, mgr, op::MulGradient, ∂a, ∂b, (a,b))
    action! = adj_action_in(op.action!)
    loop_cell(∂a, mgr, action!, op, Stencils.dot_grad, ∂u, b)
    loop_cell(∂b, mgr, flip(action!), op, Stencils.div_centered_flux, a, ∂u)
end

#=======================================================#
#===================== Loop styles =====================#
#=======================================================#


rank(::AbstractArray{T,N}) where {T,N} = Val(N) # to dispatch to the adequate loop

# for batched operations

struct MergedIndex{I}
    k::I # could be a SIMD.VecRange
    stride::Int32
end
Base.@propagate_inbounds Base.getindex(a::DenseArray{T,3}, k::MergedIndex, ij) where T = getindex(a, k.k + (ij-1)*k.stride)
Base.@propagate_inbounds Base.setindex!(a::DenseArray{T,3}, v, k::MergedIndex, ij) where T = setindex!(a, v, k.k + (ij-1)*k.stride)

@inline loop_simple(output::AbstractArray, args...) = loop_simple(rank(output), output, args...)

@inline function loop_simple(::Val{1}, output, mgr, action!, op, stencil, inputs...)
    @with mgr, 
    let irange = eachindex(output)
        @inb for i in irange
            st = stencil(op, i)
            @inbounds action!(output, st(inputs...), i) # FIXME
        end
    end
    return nothing
end

@inline function loop_simple(::Val{2}, output, mgr, action!, op, stencil, inputs...)
    @with mgr, 
    let (krange, irange) = axes(output)
        @inb for i in irange
            st = stencil(op, i)
            @vec for k in krange
#            @simd ivdep for k in krange
                action!(output, st(inputs..., k), k, i)
            end
        end
    end
    return nothing
end

@inline function loop_simple(::Val{3}, output, mgr, action!, op, stencil, inputs...)
    nl = Int32(size(output,1)*size(output,2)) # merged axis
    @with mgr, 
    let (lrange, irange) = (1:nl, axes(output, 3))
        @inb for i in irange
            st = stencil(op, i)
            @vec for l in lrange
#            @simd ivdep for l in lrange
                k = MergedIndex(l, nl)
                action!(output, st(inputs..., k), k, i)
            end
        end
    end
    return nothing
end

loop_cell(output::AbstractArray, args...) = loop_cell(rank(output), output, args...)

@inline function loop_cell(::Val{1}, output, mgr, action!, op, stencil, inputs...)
    @inb for cell in eachindex(output)
        deg = op.primal_deg[cell]
        @unroll deg in 5:7 begin
            st = stencil(op, cell, Val(deg))
            action!(output, st(inputs...), cell)
        end
    end
    return nothing
end

@inline function loop_cell(::Val{2}, output, mgr, action!, op, stencil, inputs...)
    @with mgr, 
    let (krange, irange) = axes(output)
        @inb for cell in irange
            deg = op.primal_deg[cell]
            @unroll deg in 5:7 begin
                st = stencil(op, cell, Val(deg))
                @vec for k in krange
#                @simd ivdep for k in krange
                    action!(output, st(inputs..., k), k, cell)
                end
            end
        end
    end
    return nothing
end

@inline function loop_cell(::Val{3}, output, mgr, action!, op, stencil, inputs...)
    nl = Int32(size(output,1)*size(output,2)) # merged axis
    @with mgr, 
    let (lrange, irange) = (1:nl, axes(output, 3))
        @inb for cell in irange
            deg = op.primal_deg[cell]
            @unroll deg in 5:7 begin
                st = stencil(op, cell, Val(deg))
                @vec for l in lrange
    #            @simd ivdep for l in lrange
                    k = MergedIndex(l, nl)
                    action!(output, st(inputs..., k), k, cell)
                end
            end
        end
    end
    return nothing
end

loop_trisk(output::AbstractArray, args...) = loop_trisk(rank(output), output, args...)

@inline function loop_trisk(::Val{1}, output, mgr, action!, op, stencil, inputs...)
    @with mgr,
    let irange = eachindex(output)
        @inb for edge in irange
            deg = op.trisk_deg[edge]
            @unroll deg in 9:11 begin
                st = stencil(op, edge, Val(deg))
                action!(output, st(inputs...), edge)
            end
        end
    end
    return nothing
end

@inline function loop_trisk(::Val{2}, output, mgr, action!, op, stencil, inputs...)
    @with mgr, 
    let (krange, irange) = axes(output)
        @inb for edge in irange
            deg = op.trisk_deg[edge]
            @unroll deg in 9:11 begin
                st = stencil(op, edge, Val(deg))
                @vec for k in krange
#                @simd ivdep for k in krange
                    action!(output, st(inputs..., k), k, edge)
                end
            end
        end
    end
    return nothing
end

@inline function loop_trisk(::Val{3}, output, mgr, action!, op, stencil, inputs...)
    nl = Int32(size(output,1)*size(output,2)) # merged axis
    @with mgr, 
    let (lrange, irange) = (1:nl, axes(output, 3))
        @inb for edge in irange
            deg = op.trisk_deg[edge]
            @unroll deg in 9:11 begin
                st = stencil(op, edge, Val(deg))
                @vec for l in lrange
    #            @simd ivdep for l in lrange
                    k = MergedIndex(l, nl)
                    action!(output, st(inputs..., k), k, edge)
                end
            end
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
