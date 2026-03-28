module LazyOperators

using Base: @propagate_inbounds as @prop

#========== actions: what to do on the output of operators ===========#

@prop set!(out, v, i...)      = out[i...] = v
@prop setminus!(out, v, i...) = out[i...] = -v
@prop addto!(out, v, i...)    = out[i...] += v
@prop subfrom!(out, v, i...)  = out[i...] -= v
@prop setzero!(out, i)     = out[i] = 0
@prop unchanged!(_, i)     = nothing

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

flip(::typeof(addto!)) = subfrom!
flip(::typeof(subfrom!)) = addto!

#================== lazy diagonal operator ===============#

struct LazyDiagonalOp{V<:AbstractVector}
    diag::V
end
struct WritableDVP{N, T, D<:AbstractVector, V<:AbstractArray{T,N}} <: AbstractArray{T,N}
    diag::D
    x::V
end
(op::LazyDiagonalOp)(field) = WritableDVP(op.diag, field)

Base.eachindex(y::WritableDVP) = eachindex(y.x)
Base.axes(y::WritableDVP) = axes(y.x)

# x[i] == diag[i] * y[i]
@prop Base.setindex!(y::WritableDVP, v, i...) = y.x[i...] =  v*getdiag(y, i...)
@prop addto!(y::WritableDVP, v, i...)         = y.x[i...] += v*getdiag(y, i...)
@prop subfrom!(y::WritableDVP, v, i...)       = y.x[i...] -= v*getdiag(y, i...)
@prop getdiag(d::WritableDVP{1}, i) = d.diag[i]
@prop getdiag(d::WritableDVP{2}, _, i) = d.diag[i]

# Used by adjoints, see VoronoiSpheresMooncakeExt
# Keep a copy of output argument x.
archive(x) = copy(x)
archive(y::WritableDVP) = archive(y.x)
# Restore the archived value of output argument x
restore!(x,x0) = copy!(x, x0)
restore!(y::WritableDVP, x0) = restore!(y.x, x0)

end # module
