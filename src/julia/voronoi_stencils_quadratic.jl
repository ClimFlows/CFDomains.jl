#=================== multiplied gradient =====================#

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
mul_grad(vsphere) = @lhs (; edge_left_right) = vsphere

@inl mul_grad((; edge_left_right), ij::Int) =
    Fix(get_mul_grad, (edge_left_right[1, ij], edge_left_right[2, ij]))

@inl get_mul_grad(left, right, a, q) = (a[right]+a[left])*(q[right] - q[left])/2
@inl get_mul_grad(left, right, a, q, k) = (a[k, right]+a[k, left])*(q[k, right] - q[k, left])/2

#======= u⋅v (covector,covector -> two-form) =========#

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
dot_product_form(vsphere) = @lhs (; primal_edge, le_de) = vsphere

@gen dot_product_form(vsphere, ij, v::Val{N}) where {N} = quote
    # the factor 1/2 for the Perot formula is incorporated into hodges
    edges = @unroll (vsphere.primal_edge[e, ij] for e = 1:$N)
    hodges = @unroll (vsphere.le_de[edges[e]]/2 for e = 1:$N)
    return Fix(sum_bilinear, (edges, hodges))
end

#======= u⋅u (covector -> two-form) =========#

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
squared_covector(vsphere) = @lhs (; primal_edge, le_de) = vsphere

@gen squared_covector(vsphere, ij, v::Val{N}) where {N} = quote
    # the factor 1/2 for the Perot formula is incorporated into hodges
    edges = @unroll (vsphere.primal_edge[e, ij] for e = 1:$N)
    hodges = @unroll (vsphere.le_de[edges[e]]/2 for e = 1:$N)
    return Fix(sum_square, (edges, hodges))
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
centered_flux(vsphere) = @lhs (; edge_left_right, le_de) = vsphere

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

#============== ∇⋅(qU) (scalar, vector -> two-form) ================#

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
div_centered_flux(vsphere) = @lhs (; primal_neighbour, primal_edge, primal_ne) = vsphere

@gen div_centered_flux(vsphere, cell::Int, ::Val{N}) where N = quote
    (; primal_neighbour, primal_edge, primal_ne) = vsphere
    cells = @unroll (primal_neighbour[e, cell] for e=1:$N)
    edges = @unroll (primal_edge[e, cell] for e=1:$N)
    signs = @unroll (primal_ne[e, cell]/2 for e=1:$N) # factor 1/2 is for centered average
    Fix(get_div_centered_flux, (cell, cells, edges, signs))    
end

#=============== U⋅∇q (scalar, vector -> two-form)================#

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
dot_grad(vsphere) = @lhs (; primal_neighbour, primal_edge, primal_ne) = vsphere

@inl function dot_grad((; primal_neighbour, primal_edge, primal_ne), cell::Int, N::Val)
    get = Get(cell, N)
    Fix(get_dot_grad, (cell, get(primal_neighbour), get(primal_edge), get(primal_ne)))    
end

#=============== u × v (vector, vector -> edge two-form) ==============#

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
cross_product(vsphere) = @lhs (; trisk, wee) = vsphere

@inl cross_product(vsphere, edge, deg) = Fix_TRiSK(sum_antisym, vsphere, edge, deg)

#==================== leaf expressions ======================#

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
