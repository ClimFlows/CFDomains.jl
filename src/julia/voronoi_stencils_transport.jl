#================= tangential component ================#

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
perp(vsphere) = @lhs (; edge_kite, edge_perp) = vsphere

@inl function perp(vsphere, edge) 
    edges = @unroll (vsphere.edge_kite[ind, edge] for ind = 1:4)
    coefs = @unroll (vsphere.edge_perp[ind, edge] for ind = 1:4)
    return Fix(sum_weighted, (edges, coefs))
end

#=================== 3D gradient =====================#

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
gradient3d(vsphere) = @lhs (; primal_neighbour, primal_grad3d) = vsphere

@inl function gradient3d((; primal_neighbour, primal_grad3d), cell, N::Val)
    get = Get(cell, N)
    neighbours, grads = get(primal_neighbour, primal_grad3d)
    return Fix(get_gradient3d, (cell, neighbours, grads))
end

#==================== leaf expressions ======================#

@gen get_gradient3d(cell, neighbours::Ints{N}, grads, q, k) where {N} = quote
    dq = @unroll (q[k, neighbours[edge]] - q[k, cell] for edge = 1:$N)
    @unroll (sum(dq[edge] * grads[edge][dim] for edge = 1:$N) for dim = 1:3)
end
@gen get_gradient3d(cell, neighbours::Ints{N}, grads, q) where {N} = quote
    dq = @unroll (q[neighbours[edge]] - q[cell] for edge = 1:$N)
    @unroll (sum(dq[edge] * grads[edge][dim] for edge = 1:$N) for dim = 1:3)
end

