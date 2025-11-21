macro inl(expr)
    esc(:(Base.@propagate_inbounds $expr))
#    esc(expr)
end

macro gen(expr)
    esc(:(Base.@propagate_inbounds @generated $expr))
#    esc(:(@generated $expr))
end

macro lhs(x::Expr) # in assignment 'a=b', returns 'a' instead of 'b'
    @assert x.head == :(=)
    a, b = x.args
    return esc( :( $a=$b ; $a))
end

"""
    g = Fix(f, args)
Return callable `g` such that `g(x,...)` calls `f` by prepending `args...` before `x...`:

    g(x...) == f(args..., x...)
This is similar to `Base.Fix1`, with several arguments.
"""
struct Fix{Fun,Coefs}
    fun::Fun # operator to call
    coefs::Coefs # local mesh information
end
@inl (st::Fix)(args...) = st.fun(st.coefs..., args...)

struct Get{N}
    ij::Int
    Get(ij, ::Val{N}) where N = new{N}(ij)
    Get(ij, N::Int) = new{N}(ij)
end
(getter::Get{N})(stencil) where N = get_stencil(Val{N}(), getter.ij, stencil)
(getter::Get)(s1, s2) where N = getter(s1), getter(s2)
Fix(fun, getter::Get, a, b) = Fix(fun, getter(a, b))

@gen get_stencil(::Val{N}, ij, stencil) where {N} = quote
    @unroll (stencil[n, ij] for n = 1:$N)
end

# for docstrings
const OPTIONAL = "optional, returns only relevant fields as a named tuple"
const WRT = "with respect to the unit sphere"
const SPH = "`vsphere::VoronoiSphere`"
const CELL = "`cell::Int`"
const EDGE = "`edge::Int`"
const DUAL = "`dual_cell::Int`"
const NEDGE = "`N=sphere.primal_deg[cell]` is the number of cell edges and must be provided as a compile-time constant for performance. This may be done via the macro `@unroll` from `ManagedLoops`. "
INB(a,b) = "`@inbounds` propagates into `$(string(a))` and `$(string(b))`."
SINGLE(u) = "single-layer, $(string(u))::AbstractVector"
MULTI(u) = "multi-layer, $(string(u))::AbstractMatrix"
SCALAR(q) = "`$(string(q))` is a scalar field known at *primal* cells."
DUALSCALAR(q) = "`$(string(q))` is a scalar field known at *dual* cells."
EDGESCALAR(q) = "`$(string(q))` is a scalar field known at *edges*."
DUAL2FORM(q) = "`$(string(q))` is a *density* (two-form) over *dual* cells. To obtain a scalar, divide `$(string(q))` by the dual cell area `vsphere.Av`"
EDGE2FORM(q) = "`$(string(q))` is a *density* (two-form) over *edges*. To obtain a scalar, divide `$(string(q))` by the edge cell area `vsphere.Ae`"
TWOFORM(q) = "`$(string(q))` is a *density* (two-form) over *primal* cells. To obtain a scalar, divide `$(string(q))` by the primal cell area `vsphere.Ai`"

COV(q) = "`$(string(q))` is a *covariant* vector field known at edges."
CONTRA(q) = "`$(string(q))` is a *contravariant* vector field known at edges."

SINGLE(u,v) = "single-layer, `$(string(u))` and `$(string(v))` are ::AbstractVector"
MULTI(u,v) = "multi-layer, `$(string(u))` and `$(string(v))` are ::AbstractMatrix"
COV(u,v) = "`$(string(u))` and `$(string(v))` are *covariant* vector fields known at edges."
CONTRA(u,v) = "`$(string(u))` and `$(string(v))` are *contravariant* vector fields known at edges."
