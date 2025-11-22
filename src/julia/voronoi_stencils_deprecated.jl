#========================= averaging =======================#

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

#========================= gradient =====================#

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

#==================== leaf expressions ======================#

@inl get_grad_form(left, right, Xl, Xr, Q) = Xr*Q[right] - Xl*Q[left]
@inl get_grad_form(left, right, Xl, Xr, Q, k) = Xr*Q[k, right] - Xl*Q[k, left]

@gen get_contraction(edges::Ints{N}, inv_area, ucontra, vcov) where {N} = quote
    inv_area * @unroll sum(ucontra[edges[e]] * vcov[edges[e]] for e = 1:$N)
end

@gen get_contraction(edges::Ints{N}, inv_area, ucontra, vcov, k) where {N} = quote
    inv_area * @unroll sum(ucontra[k, edges[e]] * vcov[k, edges[e]] for e = 1:$N)
end
