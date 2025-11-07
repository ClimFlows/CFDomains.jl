module Stencils

using CFDomains: HVLayout, VHLayout

using ManagedLoops: @unroll

macro gen(expr)
    esc(:(Base.@propagate_inbounds @generated $expr))
end

macro inl(expr)
    esc(:(Base.@propagate_inbounds $expr))
end

macro lhs(x::Expr) # in assignment 'a=b', returns 'a' instead of 'b'
    @assert x.head == :(=)
    a, b = x.args
    return esc( :( $a=$b ; $a))
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
TWOFORM(q) = "`$(string(q))` is a *density* (two-form) over *primal* cells. To obtain a scalar, divide `$(string(q))` by the primal cell area `vsphere.Ai`"

COV(q) = "`$(string(q))` is a *covariant* vector field known at edges."
CONTRA(q) = "`$(string(q))` is a *contravariant* vector field known at edges."

SINGLE(u,v) = "single-layer, `$(string(u))` and `$(string(v))` are ::AbstractVector"
MULTI(u,v) = "multi-layer, `$(string(u))` and `$(string(v))` are ::AbstractMatrix"
COV(u,v) = "`$(string(u))` and `$(string(v))` are *covariant* vector fields known at edges."
CONTRA(u,v) = "`$(string(u))` and `$(string(v))` are *contravariant* vector fields known at edges."

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

#======================== averaging =======================#

"""
    vsphere = average_ie(vsphere) # $OPTIONAL
    avg = average_ie(vsphere, edge)
    qe[edge] = avg(qi)         # $(SINGLE(:qi))
    qe[k, edge] = avg(qi, k)   # $(MULTI(:qi))

Interpolate scalar field at $EDGE of $SPH by a centered average (second-order accurate).

$(SCALAR(:qi))
$(EDGESCALAR(:qe))

$(INB(:average_ie, :avg))
"""
@inl average_ie(vsphere) = @lhs (; edge_left_right) = vsphere
@inl average_ie(vsphere, ij) =
    Fix(get_average_ie, (vsphere.edge_left_right[1, ij], vsphere.edge_left_right[2, ij]))

@inl get_average_ie(left, right, mass) = (mass[left] + mass[right]) / 2
@inl get_average_ie(left, right, mass, k) = (mass[k, left] + mass[k, right]) / 2

"""
    vsphere = average_iv(vsphere) # $OPTIONAL
    avg = average_iv(vsphere, dual_cell)
    qv[dual_cell] = avg(qi)         # $(SINGLE(:qi))
    qv[k, dual_cell] = avg(qi, k)   # $(MULTI(:qi))

Interpolate scalar field at $DUAL of $SPH by an area-weighted average (first-order accurate).

$(SCALAR(:qi))
$(DUALSCALAR(:qv))

$(INB(:average_iv, :avg))
"""
@inl average_iv(vsphere) = @lhs (; dual_vertex, Riv2) = vsphere

@inl function average_iv(vsphere, ij::Int)
    cells = @unroll (vsphere.dual_vertex[e, ij] for e = 1:3)
    weights = @unroll (vsphere.Riv2[e, ij] for e = 1:3)
    return Fix(get_average_iv, (cells, weights))
end

@inl get_average_iv(cells, weights, mass) =
    @unroll sum(weights[e] * mass[cells[e]] for e = 1:3)

@inl get_average_iv(cells, weights, mass, k) =
    @unroll sum(weights[e] * mass[k, cells[e]] for e = 1:3)

"""
    vsphere = average_ve(vsphere) # $OPTIONAL
    avg = average_ve(vsphere, edge)
    qe[edge] = avg(qv)         # $(SINGLE(:qv))
    qe[k, edge] = avg(qv, k)   # $(MULTI(:qv))

Interpolate scalar field at $EDGE of $SPH by a centered average (first-order accurate).

$(DUALSCALAR(:qv))
$(EDGESCALAR(:qe))

$(INB(:average_ve, :avg))
"""
@inl average_ve(vsphere) = @lhs (; edge_down_up) = vsphere
@inl average_ve(vsphere, ij::Int) =
    Fix(get_average_ve, (vsphere.edge_down_up[1, ij], vsphere.edge_down_up[2, ij]))

@inl get_average_ve(up, down, qv) = (qv[down] + qv[up]) / 2

@inl get_average_ve(up, down, qv, k) = (qv[k, down] + qv[k, up]) / 2

#========================= divergence =======================#

"""
    vsphere = divergence(vsphere) # $OPTIONAL
    div = divergence(vsphere, cell, Val(N))
    dvg[cell] = div(flux) # $(SINGLE(:flux))
    dvg[k, cell] = div(flux, k)  # $(MULTI(:flux))

Compute divergence $WRT of `flux` at $CELL of $SPH.
$(CONTRA(:flux))
$(SCALAR(:dvg))

$NEDGE

$(INB(:divergence, :div))
"""
@inl divergence(vsphere) = @lhs (; Ai, primal_edge, primal_ne) = vsphere

@gen divergence(vsphere, ij::Int, v::Val{N}) where {N} = quote
    # signs include the inv_area factor
    inv_area = inv(vsphere.Ai[ij])
    edges = @unroll (vsphere.primal_edge[e, ij] for e = 1:$N)
    signs = @unroll (inv_area * vsphere.primal_ne[e, ij] for e = 1:$N)
    return Fix(get_divergence, (v, edges, signs))
end

@gen get_divergence(::Val{N}, edges, signs, flux) where {N} = quote
    @unroll sum(flux[edges[e]] * signs[e] for e = 1:$N)
end

@gen get_divergence(::Val{N}, edges, signs, flux, k) where {N} = quote
    @unroll sum(flux[k, edges[e]] * signs[e] for e = 1:$N)
end

#========================= divergence (2-form) =======================#

"""
    vsphere = div_form(vsphere) # $OPTIONAL
    divf = div_form(vsphere, cell, Val(N))
    dvg[cell] = divf(flux) # $(SINGLE(:flux))
    dvg[k, cell] = divf(flux, k)  # $(MULTI(:flux))

Compute divergence $WRT of `flux` at $CELL of $SPH.
$(CONTRA(:flux))
$(TWOFORM(:dvg))

$NEDGE

$(INB(:div_form, :divf))
"""
@inl div_form(vsphere) = @lhs (; primal_edge, primal_ne) = vsphere

@gen div_form(vsphere, ij::Int, v::Val{N}) where {N} = quote
    edges = @unroll (vsphere.primal_edge[e, ij] for e = 1:$N)
    signs = @unroll (vsphere.primal_ne[e, ij] for e = 1:$N)
    return Fix(get_divergence, (v, edges, signs))
end

#========================= curl =====================#

"""
    vsphere = curl(vsphere) # $OPTIONAL
    op = curl(vsphere, dual_cell)
    curlu[dual_cell] = op(ucov)         # $(SINGLE(:ucov))
    curlu[k, dual_cell] = op(ucov, k)   # $(MULTI(:ucov))

Compute curl of `ucov` at $DUAL of $SPH. 
$(COV(:ucov))
$(DUAL2FORM(:curlu))

$(INB(:curl, :op))
"""
@inl curl(vsphere) = @lhs (; Riv2, dual_edge, dual_ne) = vsphere

@inl function curl(vsphere, ij::Int)
    F = eltype(vsphere.Riv2)
    edges = @unroll (vsphere.dual_edge[e, ij] for e = 1:3)
    signs = @unroll (F(vsphere.dual_ne[e, ij]) for e = 1:3)
    return Fix(get_curl, (edges, signs))
end

@inl get_curl(edges, signs, ucov) = @unroll sum(ucov[edges[e]] * signs[e] for e = 1:3)

@inl get_curl(edges, signs, ucov, k) = @unroll sum(ucov[k, edges[e]] * signs[e] for e = 1:3)

#========================= gradient =====================#

"""
    vsphere = gradient(vsphere) # $OPTIONAL
    grad = gradient(vsphere, edge)
    gradcov[edge] = grad(q)         # $(SINGLE(:q))
    gradcov[k, edge] = grad(q, k)   # $(MULTI(:q))

Compute gradient of `q` at $EDGE of $SPH.

$(SCALAR(:q))
$(COV(:gradcov)) `gradcov` is numerically zero-curl.

$(INB(:gradient, :gradcov))
"""
@inl gradient(vsphere) = @lhs (; edge_left_right) = vsphere

@inl gradient(vsphere, ij::Int) =
    Fix(get_gradient, (vsphere.edge_left_right[1, ij], vsphere.edge_left_right[2, ij]))

@inl get_gradient(left, right, q) = q[right] - q[left]
@inl get_gradient(left, right, q, k) = q[k, right] - q[k, left]

"""
    vsphere = grad_form(vsphere) # $OPTIONAL
    grad = grad_form(vsphere, edge)
    gradcov[edge] = grad(Q)         # $(SINGLE(:Q))
    gradcov[k, edge] = grad(Q, k)   # $(MULTI(:Q))

Compute gradient of `Q` at $EDGE of $SPH.

$(TWOFORM(:Q))
$(COV(:gradcov)) `gradcov` is numerically zero-curl.

$(INB(:grad_form, :gradcov))
"""
@inl grad_form(vsphere) = @lhs (; Ai, edge_left_right) = vsphere
@inl function grad_form(vsphere, ij::Int)
    Ai, left, right = vsphere.Ai, vsphere.edge_left_right[1, ij], vsphere.edge_left_right[2, ij]
    Fix(get_grad_form, (left, right, inv(Ai[left]), inv(Ai[right])))
end
@inl get_grad_form(left, right, Xl, Xr, Q) = Xr*Q[right] - Xl*Q[left]
@inl get_grad_form(left, right, Xl, Xr, Q, k) = Xr*Q[k, right] - Xl*Q[k, left]

"""
    vsphere = gradperp(vsphere) # $OPTIONAL
    grad = gradperp(vsphere, edge)
    flux[edge] = grad(psi)         # $(SINGLE(:q))
    flux[k, edge] = grad(psi, k)   # $(MULTI(:q))

Compute grad⟂ of streamfunction `psi` at $EDGE of $SPH.

$(DUALSCALAR(:psi))
$(CONTRA(:flux)) `flux` is numerically non-divergent.

$(INB(:gradperp, :grad))
"""
@inl gradperp(vsphere) = @lhs (; edge_down_up) = vsphere
@inl gradperp(vsphere, ij::Int) =
    Fix(get_gradient, (vsphere.edge_down_up[1, ij], vsphere.edge_down_up[2, ij]))


"""
    vsphere = gradient3d(vsphere) # $OPTIONAL
    grad = gradient3d(vsphere, cell, Val(N))
    gradq[ij] = grad(q)         # $(SINGLE(:q))
    gradq[k, ij] = grad(q, k)   # $(MULTI(:q))

Compute 3D gradient of `q` at $CELL of $SPH.
$(SCALAR(:q))
`gradq` is a 3D vector field yielding a 3-uple at each primal cell.

$NEDGE

$(INB(:gradient3d, :grad))
"""
@inl gradient3d(vsphere) = @lhs (; primal_neighbour, primal_grad3d) = vsphere

@gen gradient3d(vsphere, cell, v::Val{deg}) where {deg} = quote
    neighbours = @unroll (vsphere.primal_neighbour[edge, cell] for edge = 1:$deg)
    grads = @unroll (vsphere.primal_grad3d[edge, cell] for edge = 1:$deg)
    return Fix(get_gradient3d, (v, cell, neighbours, grads))
end
@gen get_gradient3d(::Val{deg}, cell, neighbours, grads, q, k) where {deg} = quote
    dq = @unroll (q[k, neighbours[edge]] - q[k, cell] for edge = 1:$deg)
    @unroll (sum(dq[edge] * grads[edge][dim] for edge = 1:$deg) for dim = 1:3)
end
@gen get_gradient3d(::Val{deg}, cell, neighbours, grads, q) where {deg} = quote
    dq = @unroll (q[neighbours[edge]] - q[cell] for edge = 1:$deg)
    @unroll (sum(dq[edge] * grads[edge][dim] for edge = 1:$deg) for dim = 1:3)
end

# Kept for testing
@gen gradient3d(vsphere, layout, cell, dim, v::Val{deg}) where {deg} = quote
    neighbours = @unroll (vsphere.primal_neighbour[edge, cell] for edge = 1:$deg)
    grads = @unroll (vsphere.primal_grad3d[edge, cell][dim] for edge = 1:$deg)
    return Fix(get_gradient3d, (layout, v, cell, neighbours, grads))
end
@gen get_gradient3d(::HVLayout{1}, ::Val{deg}, cell, neighbours, grads, q, k) where {deg} =
    quote
        dq = @unroll (q[neighbours[edge], k] - q[cell, k] for edge = 1:$deg)
        @unroll sum(dq[edge] * grads[edge] for edge = 1:$deg)
    end

#================= dot product (covariant inputs) =================#

"""
    vsphere = dot_product(vsphere::VoronoiSphere) # $OPTIONAL
    dot_prod = dot_product(vsphere, cell::Int, v::Val{N})

    # $(SINGLE(:ucov, :vcov))
    dp[cell] = dot_prod(ucov, vcov) 

    # $(MULTI(:ucov, :vcov))
    dp[k, cell] = dot_prod(ucov, vcov, k)

Compute dot product $WRT of `ucov`, `vcov` at $CELL of $SPH. 
$(COV(:ucov, :vcov))

$NEDGE

$(INB(:dot_product, :dot_prod))
"""
@inl dot_product(vsphere) = @lhs (; Ai, primal_edge, le_de) = vsphere

@gen dot_product(vsphere, ij, v::Val{N}) where {N} = quote
    # the factor 1/2 for the Perot formula is incorporated into inv_area
    # inv_area is incorporated into hodges
    inv_area = inv(2 * vsphere.Ai[ij])
    edges = @unroll (vsphere.primal_edge[e, ij] for e = 1:$N)
    hodges = @unroll (inv_area * vsphere.le_de[edges[e]] for e = 1:$N)
    return Fix(get_dot_product, (v, edges, hodges))
end

@gen get_dot_product(::Val{N}, edges, hodges, ucov, vcov) where {N} = quote
    @unroll sum(hodges[e] * (ucov[edges[e]] * vcov[edges[e]]) for e = 1:$N)
end

@gen get_dot_product(::Val{N}, edges, hodges, ucov, vcov, k) where {N} = quote
    @unroll sum(hodges[e] * (ucov[k, edges[e]] * vcov[k, edges[e]]) for e = 1:$N)
end

#=============== dot product (contravariant inputs) ===============#

"""
    vsphere = dot_prod_contra(vsphere::VoronoiSphere) # $OPTIONAL
    dot_prod = dot_prod_contra(vsphere, cell::Int, v::Val{N})

    # $(SINGLE(:U, :V))
    dp[cell] = dot_prod(U, V) 

    # $(MULTI(:U, :V))
    dp[k, cell] = dot_prod(U, V, k)

Compute dot product $WRT of `U`, `V` at $CELL of $SPH. 
$(CONTRA(:U, :V))

$NEDGE

$(INB(:dot_prod_contra, :dot_prod))
"""
@inl dot_prod_contra(vsphere) = @lhs (; Ai, primal_edge, le_de) = vsphere

@gen dot_prod_contra(vsphere, ij, v::Val{N}) where {N} = quote
    # inv(2*area) is incorporated into hodges
    dbl_area = 2 * vsphere.Ai[ij]
    edges = @unroll (vsphere.primal_edge[e, ij] for e = 1:$N)
    hodges = @unroll (inv(dbl_area * vsphere.le_de[edges[e]]) for e = 1:$N)
    return Fix(get_dot_product, (v, edges, hodges))
end

#======================= contraction ======================#

"""
    vsphere = contraction(vsphere::VoronoiSphere) # $OPTIONAL
    contract = contraction(vsphere, cell::Int, v::Val{N})

    # $(SINGLE(:ucontra, :vcov))
    uv[cell] = contract(ucontra, vcov) 

    # $(MULTI(:ucontra, :vcov))
    uv[k, cell] = contract(ucontra, vcov, k)

Compute the contraction of `ucov` and `vcov` at $CELL of $SPH. 
$(CONTRA(:ucontra))
$(COV(:vcov))

$NEDGE

$(INB(:contraction, :contract))
"""
@inl contraction(vsphere) = @lhs (; Ai, primal_edge) = vsphere

@gen contraction(vsphere, ij, v::Val{N}) where {N} = quote
    # the factor 1/2 for the Perot formula is incorporated into inv_area
    inv_area = inv(2 * vsphere.Ai[ij])
    edges = @unroll (vsphere.primal_edge[e, ij] for e = 1:$N)
    return Fix(get_contraction, (v, edges, inv_area))
end

@gen get_contraction(::Val{N}, edges, inv_area, ucontra, vcov) where {N} = quote
    inv_area * @unroll sum(ucontra[edges[e]] * vcov[edges[e]] for e = 1:$N)
end

@gen get_contraction(::Val{N}, edges, inv_area, ucontra, vcov, k) where {N} = quote
    inv_area * @unroll sum(ucontra[k, edges[e]] * vcov[k, edges[e]] for e = 1:$N)
end

#======================= centered flux ======================#

"""
    vsphere = centered_flux(vsphere) # $OPTIONAL
    cflux = centered_flux(vsphere, edge)
    flux[edge] = cflux(mass, ucov)         # $(SINGLE(:ucov))
    flux[k, edge] = cflux(mass, ucov, k)   # $(MULTI(:ucov))

Compute centered `flux` at $EDGE of $SPH, $WRT. 

$(SCALAR(:mass))
$(COV(:ucov))
$(CONTRA(:flux)) 

If `ucov` is  defined with respect to a physical metric (e.g. in m²⋅s⁻¹) 
which is conformal, multiply `cflux` by the contravariant physical 
metric factor (in m⁻²). `mass` being e.g. in kg, on gets a `flux` 
in kg⋅s⁻¹ which can be fed into [`divergence`](@ref).

$(INB(:centered_flux, :cflux))
"""
@inl centered_flux(vsphere) = @lhs (; edge_left_right, le_de) = vsphere

@inl function centered_flux(vsphere, ij::Int)
    # le_de includes the factor 1/2 for the centered average
    left_right, le_de = vsphere.edge_left_right, vsphere.le_de
    Fix(get_centered_flux, (ij, left_right[1, ij], left_right[2, ij], le_de[ij] / 2))
end

# Makes sense for a conformal metric.
# It is the job of the caller to multiply the covariant velocity
# `ucov` (which has units m^2/s), or the flux, by the
# contravariant metric factor (which has units m^-2) so that,
# if mass is in kg, the flux and its divergence are in kg/s.
@inl get_centered_flux(ij, left, right, le_de, mass, ucov, k) =
    le_de * ucov[k, ij] * (mass[k, left] + mass[k, right])
@inl get_centered_flux(ij, left, right, le_de, mass, ucov) =
    le_de * ucov[ij] * (mass[left] + mass[right])

#=========================== TRiSK ======================#

"""
    vsphere = TRiSK(vsphere) # $OPTIONAL
    trisk = TRiSK(vsphere, edge, Val(N))
    U_perp[edge]    = trisk(U)        # linear, $(SINGLE(:U))
    U_perp[k, edge] = trisk(U, k)     # linear, $(MULTI(:U))
    qU[edge]        = trisk(U, q)     # nonlinear, single-layer
    qU[k, edge]     = trisk(U, q, k)  # nonlinear, multi-layer

Compute TRiSK operator U⟂ or q×U at $EDGE of $SPH.

$(CONTRA(:U))
$(COV(:U_perp))

`N=sphere.trisk_deg[edge]` is the number of edges involved in the TRiSK stencil
and must be provided as a compile-time constant for performance. 
This may be done via the macro `@unroll` from `ManagedLoops`.

$(INB(:TRiSK, :trisk))
"""
@inl TRiSK(vsphere) = @lhs (; trisk, wee) = vsphere

@gen TRiSK(vsphere, ij::Int, v::Val{N}) where {N} = quote
    trisk = @unroll (vsphere.trisk[edge, ij] for edge = 1:$N)
    wee = @unroll (vsphere.wee[edge, ij] for edge = 1:$N)
    Fix(get_TRiSK1, (ij, v, trisk, wee))
end

# single-layer, linear
@gen get_TRiSK1(ij, ::Val{N}, edge, weight, U) where {N} = quote
    @unroll sum((weight[e] * U[edge[e]]) for e = 1:$N)
end

# multi-layer, linear
@gen get_TRiSK1(ij, ::Val{N}, edge, weight, U, k) where {N} = quote
    @unroll sum((weight[e] * U[k, edge[e]]) for e = 1:$N)
end

# single-layer, non-linear
@gen get_TRiSK1(ij, ::Val{N}, edge, weight, U, qe::AbstractVector) where {N} = quote
    @unroll sum((weight[e] * U[edge[e]]) * (qe[ij] + qe[edge[e]]) for e = 1:$N) / 2
end

# multi-layer, non-linear
@gen get_TRiSK1(ij, ::Val{N}, edge, weight, U, qe, k) where {N} = quote
    @unroll sum((weight[e] * U[k, edge[e]]) * (qe[k, ij] + qe[k, edge[e]]) for e = 1:$N) / 2
end

# this implementation is less efficient but kept for benchmarking
# weight includes the factor 1/2 of the centered average of qe
@inl TRiSK(vsphere, ij::Int, edge::Int) =
    Fix(get_TRiSK2, (ij, vsphere.trisk[edge, ij], vsphere.wee[edge, ij] / 2))

# multi-layer, nonlinear
@inl get_TRiSK2(ij, edge, weight, du, U, qe, k) =
    muladd(weight * U[k, edge], qe[k, ij] + qe[k, edge], du[k, ij])


#=========================== perp ======================#

"""
    op = perp(vsphere) # $OPTIONAL
    op = perp(vsphere, ij)
    U_perp[ij] = op(U)              # $(SINGLE(:U))
    U_perp[k, ij] = op(U, k::Int)   # $(MULTI(:U))
Compute the perp operator U⟂ at $EDGE of $SPH.
Unlike the TRiSK operator, this operator is not antisymmetric but
it has a smaller stencil and is numerically consistent.

Array `U` represents a vector field U by its 
components *normal* to edges of *primal* cells.
`U_perp` represents similarly U⟂. Equivalently, it represents U  
by its components *normal* to edges of *dual* cells.

$(INB(:perp, :op))
"""
@inl perp(vsphere) = @lhs (; edge_kite, edge_perp) = vsphere

@inl perp(vsphere, edge) = perp(vsphere, VHLayout{1}(), edge)

# The layout arg is kept only for testing since HVLayout is inefficient
@inl perp(vsphere, layout, edge) = @unroll Fix(
    get_perp,
    (
        layout,
        (vsphere.edge_kite[ind, edge] for ind = 1:4),
        (vsphere.edge_perp[ind, edge] for ind = 1:4),
    ),
)

@inl get_perp(_, kite, wperp, un) = @unroll sum(un[kite[ind]] * wperp[ind] for ind = 1:4)
@inl get_perp(_, kite, wperp, un, k) =
    @unroll sum(un[k, kite[ind]] * wperp[ind] for ind = 1:4)

@inl get_perp(::HVLayout{1}, kite, wperp, un, k) =
    @unroll sum(un[kite[ind], k] * wperp[ind] for ind = 1:4)

#===================== helpers hiding @unroll ====================#

@inl function on_cell_edges(fun, deg)
    @unroll deg in 5:7 begin
        fun(Val(deg))
    end
end

@inl function on_trisk_edges(fun, deg)
    @unroll deg in 9:11 begin
        fun(Val(deg))
    end
end

end #===== module ====#
