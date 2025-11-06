module VoronoiOperators

using ManagedLoops: @unroll
import CFDomains.VornoiStencils as Stencils

abstract type VoronoiOperator{In,Out} end

#====================== actions =====================#

ignore(_, y) = y     # output := op(input)  (default action)
minus(_, y) = -y     # output := -op(input)
addto(x, y) = x + y  # output := output + op(input)
subfrom(x, y) = x-y  # output := output - op(input)

# (out, in) := (op(in), in) => (∂out, ∂in) := (0, ∂in + opᵀ(∂out))
adj_action_in(::typeof(ignore), ∂in, op_∂out) = ∂in + op_∂out
Base.@propagate_inbounds adj_action_out!(::typeof(ignore), i, ∂out) = ∂out[i] = 0

# (out, in) := (-op(in), in) => (∂out, ∂in) := (0, ∂in - opᵀ(∂out))
adj_action_in(::typeof(minus), ∂in, op_∂out) = ∂in - op_∂out
Base.@propagate_inbounds adj_action_out!(::typeof(minus), i, ∂out) = ∂out[i] = 0

# (out, in) := (out + op(in), in) => (∂out, ∂in) := (∂out, ∂in + opᵀ(∂out))
adj_action_in(::typeof(addto), ∂in, op_∂out) = ∂in + op_∂out
adj_action_out!(::typeof(addto), i, ∂out) = nothing

# (out, in) := (out - op(in), in)  => (∂out, ∂in) := (∂out, ∂in - opᵀ(∂out))
adj_action_in(::typeof(subfrom), ∂in, op_∂out) = ∂in - op_∂out
adj_action_out!(::typeof(subfrom), i, ∂out) = nothing

#==================== VoronoiOperator{1,1} ==============#

(op::VoronoiOperator{1,1})(input, output) = apply!(output, op, input)

function apply!(output, stencil::VoronoiOperator{1,1}, input) 
    apply_internal!(output, stencil, input)
    return nothing
end

function apply_adj!(dout, op::VoronoiOperator{1,1}, din) 
    apply_adj_internal!(dout, op, din)
    @inbounds for i in eachindex(dout)
        adj_action_out!(op.action, i, dout)
    end
end

#========== gradient operator and its adjoint ===========#

struct Gradient{Action} <: VoronoiOperator{1,1}
    action::Action # how to combine op(input) with output
    edge_left_right::Matrix{Int32}
    # for the adjoint
    primal_deg::Vector{Int32}
    primal_edge::Matrix{Int32}
    primal_ne::Matrix{Int32}
end
Gradient(sph, action=ignore) = Gradient(action, sph.edge_left_right, sph.primal_deg, sph.primal_edge, sph.primal_ne)

@inline function apply_internal!(output, op::Gradient, input)
    @inbounds for edge in eachindex(output)
        grad = Stencils.gradient(op, edge)
        output[edge] = op.action(output[edge], grad(input))
    end
end

@inline function apply_adj_internal!(dout, op::Gradient, din)
    @inbounds for cell in eachindex(din)
        deg = op.primal_deg[cell]
        @unroll deg in 5:7 begin
            dvg = Stencils.div_form(op, cell, Val(deg))
            din[cell] = adj_action_in(op.action, din[cell], -dvg(dout))
        end
    end
end

#========== divergence operator and its adjoint ===========#

struct Divergence{Action, F<:AbstractFloat} <: VoronoiOperator{1,1}
    action::Action # how to combine op(input) with output
    Ai::Vector{F}
    primal_deg::Vector{Int32}
    primal_edge::Matrix{Int32}
    primal_ne::Matrix{Int32}
    # for the adjoint
    edge_left_right::Matrix{Int32}
end
Divergence(sph, action=ignore) = Divergence(action, sph.Ai, sph.primal_deg, sph.primal_edge, sph.primal_ne, sph.edge_left_right)

@inline function apply_internal!(output, op::Divergence, input)
    @inbounds for cell in eachindex(output)
        deg = op.primal_deg[cell]
        @unroll deg in 5:7 begin
            dvg = Stencils.divergence(op, cell, Val(deg))
            output[cell] = op.action(output[cell], dvg(input))
        end
    end
end

@inline function apply_adj_internal!(dout, op::Divergence, din)
    @inbounds for cell in eachindex(din)
        grad = Stencils.grad_form(op, edge) # gradient of a 2-form
        din[edge] = adj_action_in(op.action, din[cell], -grad(dout))
    end
end

#========== TriSK operator and its adjoint ===========#

struct TRiSK{Action, F<:AbstractFloat} <: VoronoiOperator{1,1}
    action::Action # how to combine op(input) with output
    trisk_deg::Vector{Int32}
    trisk::Matrix{Int32}
    wee::Matrix{F}
end
TRiSK(sph, action=ignore) = TRiSK(action, sph.trisk_deg, sph.trisk, sph.wee)

@inline function apply_internal!(output, op::TRiSK, input)
    (; action) = op
    for edge in eachindex(output)
        deg = op.trisk_deg[edge]
        @unroll deg in 9:11 begin
            trsk = Stencils.TRiSK(op, edge, Val(deg))
            output[edge] = action(output[edge], trsk(input))
        end
    end
    return nothing
end

@inline function apply_adj_internal!(dout, op::TRiSK, din)
    for edge in eachindex(din)
        deg = op.trisk_deg[edge]
        @unroll deg in 9:11 begin
            trsk = Stencils.TRiSK(op, edge, Val(deg))
            din[edge] = adj_action_in(op.action, din[edge], -trsk(dout))
        end
    end
end

end
