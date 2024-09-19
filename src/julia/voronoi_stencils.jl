module Stencils

using ManagedLoops: @unroll

macro gen(expr)
    esc(:(Base.@propagate_inbounds @generated $expr))
end

macro inl(expr)
    esc(:(Base.@propagate_inbounds $expr))
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

#======================== averaging =======================#

# cell -> edge
@inl average_ie(vsphere, ij) =
    Fix(get_average_ie, (vsphere.edge_left_right[1, ij], vsphere.edge_left_right[2, ij]))

@inl get_average_ie(left, right, mass, k) = (mass[k, left] + mass[k, right]) / 2

# cell -> vertex
@inl function average_iv(vsphere, ij::Int)
    cells = @unroll (vsphere.dual_vertex[e, ij] for e = 1:3)
    weights = @unroll (vsphere.Riv2[e, ij] for e = 1:3)
    return Fix(get_average_iv, (cells, weights))
end

@inl get_average_iv(cells, weights, mass) =
    @unroll sum(weights[e] * mass[cells[e]] for e = 1:3)

@inl get_average_iv(cells, weights, mass, k) =
    @unroll sum(weights[e] * mass[k, cells[e]] for e = 1:3)

# vertex -> edge
@inl average_ve(vsphere, ij::Int) =
    Fix(get_average_ve, (vsphere.edge_down_up[1, ij], vsphere.edge_down_up[2, ij]))

@inl get_average_ve(up, down, qv) = (qv[down] + qv[up]) / 2

@inl get_average_ve(up, down, qv, k) = (qv[k, down] + qv[k, up]) / 2

#========================= divergence =======================#

# flux must be a contravariant vector density = 2-form in 3D space
# in X/s for the flux of X

# signs include the inv_area factor
@gen divergence(vsphere, ij::Int, v::Val{N}) where {N} = quote
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

#========================= curl =====================#

@inl function curl(vsphere, ij::Int)
    F = eltype(vsphere.Riv2)
    edges = @unroll (vsphere.dual_edge[e, ij] for e = 1:3)
    signs = @unroll (F(vsphere.dual_ne[e, ij]) for e = 1:3)
    return Fix(get_curl, (edges, signs))
end

@inl get_curl(edges, signs, ucov) = @unroll sum(ucov[edges[e]] * signs[e] for e = 1:3)

@inl get_curl(edges, signs, ucov, k) = @unroll sum(ucov[k, edges[e]] * signs[e] for e = 1:3)

#========================= gradient =====================#

@inl gradient(vsphere, ij::Int) =
    Fix(get_gradient, (vsphere.edge_left_right[1, ij], vsphere.edge_left_right[2, ij]))

@inl get_gradient(left, right, q) = q[right] - q[left]

@inl get_gradient(left, right, q, k) = q[k, right] - q[k, left]


#======================= dot product ======================#

"""
    dot_prod = dot_product(vsphere::VoronoiSphere, ij, v::Val{N})

Return the callable `dot_prod` which knows how to compute a dot product
at primal cell `ij` of `sphere`. `N=sphere.primal_deg[ij]` is the number of edges
and must be provided as a compile-time constant for performance.
`dot_prod` is to be used as:

    dp_ij = dot_prod(ucov::V, vcov::V) # single layer, V<:AbstractVector
    dp_ijk = dot_prod(ucov::M, vcov::M, k) # multi-layer, M<:AbstractMatrix

The dot product is with respect to the unit sphere and `vcov, ucov`
represent covariant vector fields (1-forms).

@inbounds may be specified at either or both call sites, and will propagate, as in:
    # do not check bounds when accessing mesh data
    dot_prod = @inbounds dot_product(sphere::VoronoiSphere, ij, v::Val{N})
    # do not check bounds when accessing ucov, vcov
    dp_ij = @inbounds dot_prod(ucov::V, vcov::V)

"""
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

#======================= centered flux ======================#

"""
    cflux = centered_flux(vsphere, ij::Int)
    flux[ij] = flux(mass, ucov)         # single-layer
    flux[k, ij] = flux(mass, ucov, k)   # multi-layer

Two-step computation of centered flux at edge `ij` of `vsphere`.
"""
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

# one-layer
@gen TRiSK(vsphere, ij::Int, v::Val{N}) where N = quote
    trisk = @unroll (vsphere.trisk[edge, ij] for edge=1:$N)
    wee = @unroll (vsphere.wee[edge, ij] for edge=1:$N)
    Fix(get_TRiSK1, (ij, v, trisk, wee))
end

@gen get_TRiSK1(ij, ::Val{N}, edge, weight, U) where N = quote
    @unroll sum((weight[e] * U[edge[e]]) for e in 1:$N)
end

@gen get_TRiSK1(ij, ::Val{N}, edge, weight, U, k::Int) where N = quote
    @unroll sum((weight[e] * U[k, edge[e]]) for e in 1:$N)
end

@gen get_TRiSK1(ij, ::Val{N}, edge, weight, U, qe) where N = quote
    @unroll sum((weight[e] * U[edge[e]])*(qe[ij] + qe[edge[e]]) for e in 1:$N)/2
end

# multi-layer, non-linear
# weight includes the factor 1/2 of the centered average of qe
@inl TRiSK(vsphere, ij::Int, edge::Int) =
    Fix(get_TRiSK2, (ij, vsphere.trisk[edge, ij], vsphere.wee[edge, ij] / 2))

@inl get_TRiSK2(ij, edge, weight, du, U, qe, k) =
    muladd(weight * U[k, edge], qe[k, ij] + qe[k, edge], du[k, ij])

end #===== module ====#
