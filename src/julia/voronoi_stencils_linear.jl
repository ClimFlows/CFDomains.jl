#========================= averaging =======================#

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
average_ie(vsphere) = @lhs (; edge_left_right) = vsphere
@inl average_ie(vsphere, ij) =
    Fix(get_average, (vsphere.edge_left_right[1, ij], vsphere.edge_left_right[2, ij]))

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
average_iv_form(vsphere) = @lhs (; dual_vertex, Avi) = vsphere
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
average_ve(vsphere) = @lhs (; edge_down_up) = vsphere

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
average_vi_form(vsphere) = @lhs (; Aiv, primal_vertex) = vsphere

@inl average_vi_form((; Aiv, primal_vertex), ij::Int, N::Val) =
    Fix(sum_weighted, Get(ij, N), primal_vertex, Aiv)

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
div_form(vsphere) = @lhs (; primal_edge, primal_ne) = vsphere

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
curl(vsphere) = @lhs (; Riv2, dual_edge, dual_ne) = vsphere

@inl function curl(vsphere, ij::Int)
    edges = @unroll (vsphere.dual_edge[e, ij] for e = 1:3)
    signs = @unroll (vsphere.dual_ne[e, ij] for e = 1:3)
    return Fix(sum_weighted, (edges, signs))
end

#===================== grad =====================#

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
gradient(vsphere) = @lhs (; edge_left_right) = vsphere

@inl gradient(vsphere, ij::Int) =
    Fix(get_difference, (vsphere.edge_left_right[1, ij], vsphere.edge_left_right[2, ij]))

#===================== grad ⟂ =====================#

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
gradperp(vsphere) = @lhs (; edge_down_up) = vsphere
@inl gradperp(vsphere, ij::Int) =
    Fix(get_difference, (vsphere.edge_down_up[1, ij], vsphere.edge_down_up[2, ij]))

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
TRiSK(vsphere) = @lhs (; trisk, wee) = vsphere

@inl TRiSK(vsphere, edge, deg) = Fix_TRiSK(sum_TRiSK1, vsphere, edge, deg)

@inl function Fix_TRiSK(fun::Fun, (; trisk, wee), edge::Int, deg::Val) where Fun
    get = Get(edge, deg)
    Fix(fun, (edge, get(trisk), get(wee)))
end

#==================== leaf expressions ======================#

@inl get_average(left, right, a) = (a[left] + a[right]) / 2
@inl get_average(left, right, a, k) = (a[k, left] + a[k, right]) / 2

@inl get_difference(left, right, q) = q[right] - q[left]
@inl get_difference(left, right, q, k) = q[k, right] - q[k, left]

@gen sum_weighted(cells::Ints{N}, weights, a) where N = quote
    @unroll sum(weights[e] * a[cells[e]] for e = 1:$N)
end

@gen sum_weighted(cells::Ints{N}, weights, a, k) where N = quote
    @unroll sum(weights[e] * a[k, cells[e]] for e = 1:$N)
end

@inl sum_TRiSK1(_, edges::Ints, weights, U) = sum_weighted(edges, weights, U)
@inl sum_TRiSK1(_, edges::Ints, weights, U, k) = sum_weighted(edges, weights, U, k)
