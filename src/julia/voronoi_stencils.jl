module Stencils

using CFDomains: HVLayout, VHLayout
using ManagedLoops: @unroll

include("voronoi_stencils_helpers.jl")

# Each stencil operation is implemented in three steps:
# 1- Extract only those fields that are relevant for the stencil
#       sph = <stencil>(sph)
#    Returns a named tuple.
# 2- Extract mesh data for a given mesh element ij (cell, edge, dual)
#       op = <stencil>(sph, ij, [::Val{N}]) . 
#    N is the "degree" = number of connected mesh elements, if not known in advance.
#    op is of the form Fix(expr, data...) which is a callable object 
#    such that op(args...) = expr(data..., args...)
# 3- `expr` evaluates the stencil expression, e.g. `sum_weighted`, ...

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
    Fix(get_average, (vsphere.edge_left_right[1, ij], vsphere.edge_left_right[2, ij]))

"""
    vsphere = average_iv(vsphere) # $OPTIONAL
    avg = average_iv(vsphere, dual_cell)
    qv[dual_cell] = avg(qi)         # $(SINGLE(:qi))
    qv[k, dual_cell] = avg(qi, k)   # $(MULTI(:qi))

Estimate scalar field integrated over $DUAL of $SPH as an area-weighted sum of values sampled at primal cell centers.

$(SCALAR(:qi))
$(DUALSCALAR(:qv))

$(INB(:average_iv, :avg))
"""
@inl average_iv(vsphere) = @lhs (; dual_vertex, Riv2) = vsphere
@inl average_iv((; dual_vertex, Riv2), ij::Int) = Fix(sum_weighted, Get(ij, 3), dual_vertex, Riv2)

"""
    vsphere = average_iv_form(vsphere) # $OPTIONAL
    avg = average_iv_form(vsphere, dual_cell)
    qv[dual_cell] = avg(qi)         # $(SINGLE(:qi))
    qv[k, dual_cell] = avg(qi, k)   # $(MULTI(:qi))

Interpolate scalar field at $DUAL of $SPH by an area-weighted sum (first-order accurate).

$(SCALAR(:qi))
$(DUAL2FORM(:qv))

$(INB(:average_iv_form, :avg))
"""
@inl average_iv_form(vsphere) = @lhs (; dual_vertex, Avi) = vsphere
@inl average_iv_form((; dual_vertex, Avi), ij::Int) = Fix(sum_weighted, Get(ij, 3), dual_vertex, Avi)

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
    Fix(get_average, (vsphere.edge_down_up[1, ij], vsphere.edge_down_up[2, ij]))

"""
    vsphere = average_vi_form(vsphere) # $OPTIONAL
    avg = avg_vi_form(vsphere, cell, Val(N))
    qi[cell] = avg(qv) # $(SINGLE(:qv))
    qi[k, cell] = avg(qv, k)  # $(MULTI(:qv))

Estimate scalar field integrated over $CELL of $SPH as an area-weighted sum of values sampled at vertices.
$(DUALSCALAR(:qv))
$(TWOFORM(:qi))

$NEDGE

$(INB(:average_vi_form, :avg))
"""
@inl average_vi_form(vsphere) = @lhs (; Aiv, primal_vertex) = vsphere

@inl average_vi_form((; Aiv, primal_vertex), ij::Int, N::Val) =
    Fix(sum_weighted, Get(ij, N), primal_vertex, Aiv)

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
@inl divergence(vsphere) = @lhs (; inv_Ai, primal_edge, primal_ne) = vsphere

@gen divergence(vsphere, ij::Int, v::Val{N}) where N = quote
    # signs include the inv_area factor
    inv_area = vsphere.inv_Ai[ij]
    edges = @unroll (vsphere.primal_edge[e, ij] for e = 1:$N)
    signs = @unroll (inv_area * vsphere.primal_ne[e, ij] for e = 1:$N)
    return Fix(sum_weighted, (edges, signs))
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

@inl div_form((; primal_edge, primal_ne), ij::Int, N::Val) =    
    Fix(sum_weighted, Get(ij, N), primal_edge, primal_ne)

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
    F = eltype(vsphere.dual_ne)
    edges = @unroll (vsphere.dual_edge[e, ij] for e = 1:3)
    signs = @unroll (F(vsphere.dual_ne[e, ij]) for e = 1:3)
    return Fix(sum_weighted, (edges, signs))
end

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
    Fix(get_difference, (vsphere.edge_left_right[1, ij], vsphere.edge_left_right[2, ij]))

"""
    vsphere = mul_grad(vsphere) # $OPTIONAL
    mulgrad = mul_grad(vsphere, edge)
    a∇b[edge] = mulgrad(a,b)     # $(SINGLE(:a, :b))
    a∇b[k, edge] = mulgrad(a, b, k)   # $(MULTI(:a, :b))

Compute gradient of `b` at $EDGE of $SPH, multiplied by `a`.

$(SCALAR(:a, :b))
$(COV(:a∇b))

$(INB(:mul_grad, :mulgrad))
"""
@inl mul_grad(vsphere) = @lhs (; edge_left_right) = vsphere

@inl mul_grad((; edge_left_right), ij::Int) =
    Fix(get_mul_grad, (edge_left_right[1, ij], edge_left_right[2, ij]))

@inl get_mul_grad(left, right, a, q) = (a[right]+a[left])*(q[right] - q[left])/2
@inl get_mul_grad(left, right, a, q, k) = (a[k, right]+a[k, left])*(q[k, right] - q[k, left])/2

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
@inl grad_form(vsphere) = @lhs (; inv_Ai, edge_left_right) = vsphere
@inl function grad_form(vsphere, ij::Int)
    (; inv_Ai, edge_left_right) = vsphere
    left, right = edge_left_right[1, ij], edge_left_right[2, ij]
    Fix(get_grad_form, (left, right, inv_Ai[left], inv_Ai[right]))
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
    Fix(get_difference, (vsphere.edge_down_up[1, ij], vsphere.edge_down_up[2, ij]))

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

@inl function gradient3d((; primal_neighbour, primal_grad3d), cell, N::Val)
    get = Get(cell, N)
    neighbours, grads = get(primal_neighbour, primal_grad3d)
    return Fix(get_gradient3d, (cell, neighbours, grads))
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
    return Fix(sum_bilinear, (edges, hodges))
end

"""
    vsphere = dot_product_form(vsphere::VoronoiSphere) # $OPTIONAL
    dot_prod = dot_product_form(vsphere, cell::Int, v::Val{N})

    # $(SINGLE(:ucov, :vcov))
    dp[cell] = dot_prod(ucov, vcov) 

    # $(MULTI(:ucov, :vcov))
    dp[k, cell] = dot_prod(ucov, vcov, k)

Compute dot product $WRT of `ucov`, `vcov` at $CELL of $SPH. 
$(COV(:ucov, :vcov))
$(TWOFORM(:dp))

$NEDGE

$(INB(:dot_product_form, :dot_prod))
"""
@inl dot_product_form(vsphere) = @lhs (; primal_edge, le_de) = vsphere

@gen dot_product_form(vsphere, ij, v::Val{N}) where {N} = quote
    # the factor 1/2 for the Perot formula is incorporated into hodges
    edges = @unroll (vsphere.primal_edge[e, ij] for e = 1:$N)
    hodges = @unroll (vsphere.le_de[edges[e]]/2 for e = 1:$N)
    return Fix(sum_bilinear, (edges, hodges))
end

"""
    vsphere = squared_covector(vsphere::VoronoiSphere) # $OPTIONAL
    square = squared_covector(vsphere, cell::Int, v::Val{N})

    # $(SINGLE(:ucov))
    u_squared_form[cell] = square(ucov) 

    # $(MULTI(:ucov))
    u_squared[k, cell] = square(ucov, k)

Compute dot product $WRT of `ucov` and istelf at $CELL of $SPH. 
$(COV(:ucov))
$(TWOFORM(:u_squared))

$NEDGE

$(INB(:squared_covector, :square))
"""
@inl squared_covector(vsphere) = @lhs (; primal_edge, le_de) = vsphere

@gen squared_covector(vsphere, ij, v::Val{N}) where {N} = quote
    # the factor 1/2 for the Perot formula is incorporated into hodges
    edges = @unroll (vsphere.primal_edge[e, ij] for e = 1:$N)
    hodges = @unroll (vsphere.le_de[edges[e]]/2 for e = 1:$N)
    return Fix(sum_square, (edges, hodges))
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

@gen dot_prod_contra(vsphere, ij, ::Val{N}) where {N} = quote
    # inv(2*area) is incorporated into hodges
    dbl_area = 2 * vsphere.Ai[ij]
    edges = @unroll (vsphere.primal_edge[e, ij] for e = 1:$N)
    hodges = @unroll (inv(dbl_area * vsphere.le_de[edges[e]]) for e = 1:$N)
    return Fix(sum_bilinear, (edges, hodges))
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
@inl contraction(vsphere) = @lhs (; inv_Ai, primal_edge) = vsphere

@inl function contraction((; inv_Ai, primal_edge), ij, N::Val)
    # the factor 1/2 is for the Perot formula
    get = Get(ij, N)
    return Fix(get_contraction, (get(primal_edge), inv_Ai[ij]/2))
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

@inl function centered_flux((; edge_left_right, le_de), ij::Int)
    # factor 1/2 is for the centered average
    Fix(get_centered_flux, (ij, edge_left_right[1, ij], edge_left_right[2, ij], le_de[ij] / 2))
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

#============ centered flux divergence and related ======================#

"""
    vsphere = div_centered_flux(vsphere) # $OPTIONAL
    div_flux = div_centered_flux(vsphere, cell)
    divqF[cell] = div_flux(q, flux)         # $(SINGLE(:flux, :q))
    divqF[k, cell] = div_flux(q, flux, k)   # $(MULTI(:flux, :q))

Compute divergence of `q*flux` at $CELL of $SPH, $WRT. `q` is interpolated by a simple
centered average.

$(SCALAR(:q))
$(CONTRA(:flux))
$(TWOFORM(:divqF))

$(INB(:div_centered_flux, :div_flux))
"""
@inl div_centered_flux(vsphere) = @lhs (; primal_neighbour, primal_edge, primal_ne) = vsphere

@gen div_centered_flux(vsphere, cell::Int, ::Val{N}) where N = quote
    (; primal_neighbour, primal_edge, primal_ne) = vsphere
    cells = @unroll (primal_neighbour[e, cell] for e=1:$N)
    edges = @unroll (primal_edge[e, cell] for e=1:$N)
    signs = @unroll (primal_ne[e, cell]/2 for e=1:$N) # factor 1/2 is for centered average
    Fix(get_div_centered_flux, (cell, cells, edges, signs))    
end

"""
    vsphere = dot_grad(vsphere) # $OPTIONAL
    dotgrad = dot_grad(vsphere, cell)
    Fgradq[cell] = dotgrad(flux, q)         # $(CONTRA(:U))
    Fgradq[k, cell] = dotgrad(flux, q, k)   # $(MULTI(:flux))

At $CELL, compute the dot product of `flux` with the gradient of `q` $WRT .

$(SCALAR(:q))
$(CONTRA(:flux))
$(TWOFORM(:Fgradq))

$(INB(:dot_grad, :dotgrad))
"""
@inl dot_grad(vsphere) = @lhs (; primal_neighbour, primal_edge, primal_ne) = vsphere

@inl function dot_grad((; primal_neighbour, primal_edge, primal_ne), cell::Int, N::Val)
    get = Get(cell, N)
    Fix(get_dot_grad, (cell, get(primal_neighbour), get(primal_edge), get(primal_ne)))    
end

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

@inl TRiSK(vsphere, edge, deg) = Fix_TRiSK(sum_TRiSK1, vsphere, edge, deg)

"""
    vsphere = cross_product(vsphere) # $OPTIONAL
    cprod = cross_product(vsphere, edge, Val(N))
    q[edge]    = cprod(U,V)        # $(SINGLE(:U, :V))
    q[k, edge] = cprod(U, V, k)    # $(MULTI(:U, :V))

Compute the cross product U×V at $EDGE of $SPH.

$(CONTRA(:U, :V))
$(EDGE2FORM(:q))

`N=sphere.trisk_deg[edge]` is the number of edges involved in the TRiSK stencil
and must be provided as a compile-time constant for performance. 
This may be done via the macro `@unroll` from `ManagedLoops`.

$(INB(:cross_product, :cprod))
"""
@inl cross_product(vsphere) = @lhs (; trisk, wee) = vsphere

@inl cross_product(vsphere, edge, deg) = Fix_TRiSK(sum_antisym, vsphere, edge, deg)

@inl function Fix_TRiSK(fun::Fun, (; trisk, wee), edge::Int, deg::Val) where Fun
    get = Get(edge, deg)
    Fix(fun, (edge, get(trisk), get(wee)))
end

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

@inl function perp(vsphere, edge) 
    edges = @unroll (vsphere.edge_kite[ind, edge] for ind = 1:4)
    coefs = @unroll (vsphere.edge_perp[ind, edge] for ind = 1:4)
    return Fix(sum_weighted, (edges, coefs))
end

#==================== leaf expressions ======================#

const Ints{N} = NTuple{N, Int32}

@inl get_average(left, right, a) = (a[left] + a[right]) / 2
@inl get_average(left, right, a, k) = (a[k, left] + a[k, right]) / 2

@inl get_difference(left, right, q) = q[right] - q[left]
@inl get_difference(left, right, q, k) = q[k, right] - q[k, left]

@inl sum_TRiSK1(_, edges::Ints, weights, U) = sum_weighted(edges, weights, U)
@inl sum_TRiSK1(_, edges::Ints, weights, U, k) = sum_weighted(edges, weights, U, k)

@gen sum_weighted(cells::Ints{N}, weights, a) where N = quote
    @unroll sum(weights[e] * a[cells[e]] for e = 1:$N)
end

@gen sum_weighted(cells::Ints{N}, weights, a, k) where N = quote
    @unroll sum(weights[e] * a[k, cells[e]] for e = 1:$N)
end

@gen sum_square(edges::Ints{N}, hodges, a) where {N} = quote
    @unroll sum(hodges[e] * (a[edges[e]]^2) for e = 1:$N)
end

@gen sum_square(edges::Ints{N}, hodges, a, k) where {N} = quote
    @unroll sum(hodges[e] * (a[k, edges[e]]^2) for e = 1:$N)
end

@gen sum_bilinear(cells::Ints{N}, weights, a,b) where N = quote
    @unroll sum(weights[e] * a[cells[e]]*b[cells[e]] for e = 1:$N)
end

@gen sum_bilinear(cells::Ints{N}, weights, a, b, k) where N = quote
    @unroll sum(weights[e] * a[k, cells[e]]*b[k, cells[e]] for e = 1:$N)
end

@gen get_div_centered_flux(cell, cells::Ints{N}, edges, weights, q, flux) where N = quote
    @unroll sum( weights[e]*flux[edges[e]]*(q[cell]+q[cells[e]]) for e=1:$N)
end

@gen get_div_centered_flux(cell, cells::Ints{N}, edges, weights, q, flux, k) where N = quote
    @unroll sum( weights[e]*flux[k, edges[e]]*(q[k, cell]+q[k, cells[e]]) for e=1:$N)
end

@gen get_dot_grad(cell, cells::Ints{N}, edges, weights, flux, q) where N = quote
    @unroll sum( weights[e]*flux[edges[e]]*(q[cells[e]]-q[cell]) for e=1:$N)/2
end

@gen get_dot_grad(cell, cells::Ints{N}, edges, weights, flux, q, k) where N = quote
    @unroll sum( weights[e]*flux[k, edges[e]]*(q[k, cells[e]]-q[k, cell]) for e=1:$N)/2
end

@gen sum_antisym(edge, edges::Ints{N}, weight, U, V) where {N} = quote
    @unroll sum(weight[e]*(U[edges[e]]*V[edge]-V[edges[e]]*U[edge]) for e = 1:$N) / 2
end

@gen sum_antisym(edge, edges::Ints{N}, weight, U, V, k) where {N} = quote
    @unroll sum(weight[e]*(U[k, edges[e]]*V[k, edge]-V[k, edges[e]]*U[k, edge]) for e = 1:$N) / 2
end

@gen sum_TRiSK1(edge, edges::Ints{N}, weights, U, qe::AbstractVector) where {N} = quote
    @unroll sum((weights[e] * U[edges[e]]) * (qe[edge] + qe[edges[e]]) for e = 1:$N) / 2
end

@gen sum_TRiSK1(edge, edges::Ints{N}, weights, U, qe::AbstractMatrix, k) where {N} = quote
    @unroll sum((weights[e] * U[k, edges[e]]) * (qe[k, edge] + qe[k, edges[e]]) for e = 1:$N) / 2
end

@gen get_contraction(edges::Ints{N}, inv_area, ucontra, vcov) where {N} = quote
    inv_area * @unroll sum(ucontra[edges[e]] * vcov[edges[e]] for e = 1:$N)
end

@gen get_contraction(edges::Ints{N}, inv_area, ucontra, vcov, k) where {N} = quote
    inv_area * @unroll sum(ucontra[k, edges[e]] * vcov[k, edges[e]] for e = 1:$N)
end

@gen get_gradient3d(cell, neighbours::Ints{N}, grads, q, k) where {N} = quote
    dq = @unroll (q[k, neighbours[edge]] - q[k, cell] for edge = 1:$N)
    @unroll (sum(dq[edge] * grads[edge][dim] for edge = 1:$N) for dim = 1:3)
end
@gen get_gradient3d(cell, neighbours::Ints{N}, grads, q) where {N} = quote
    dq = @unroll (q[neighbours[edge]] - q[cell] for edge = 1:$N)
    @unroll (sum(dq[edge] * grads[edge][dim] for edge = 1:$N) for dim = 1:3)
end

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
