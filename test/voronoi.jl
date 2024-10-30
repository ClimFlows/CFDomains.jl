# generic
apply_primal(op, sphere, args...) = [
    (@unroll N in 5:7 ret = op(sphere, cell, Val(N))(args...);
    ret) for (cell, N) in enumerate(sphere.primal_deg)
]
apply_trisk(op, sphere, args...) = [
    (@unroll N in 9:10 ret = op(sphere, edge, Val(N))(args...);
    ret) for (edge, N) in enumerate(sphere.trisk_deg)
]
apply(objects, op, sphere, args...) = [op(sphere, obj)(args...) for obj in objects(sphere)]

# mesh objects
cells(sphere) = eachindex(sphere.xyz_i)
duals(sphere) = eachindex(sphere.xyz_v)
edges(sphere) = eachindex(sphere.xyz_e)

# operators
gradient(sphere, qi) = apply(edges, Stencils.gradient, sphere, qi)
gradperp(sphere, qv) = apply(edges, Stencils.gradperp, sphere, qv)
perp(sphere, un) = apply(edges, Stencils.perp, sphere, un)
curl(sphere, ue) = apply(duals, Stencils.curl, sphere, ue)
average_iv(sphere, qi) = apply(duals, Stencils.average_iv, sphere, qi)
divergence(sphere, U) = apply_primal(Stencils.divergence, sphere, U)
gradient3d(sphere, U) = apply_primal(Stencils.gradient3d, sphere, U)
TRiSK(sphere, args...) = apply_trisk(Stencils.TRiSK, sphere, args...)

# error measures
Linf(x) = maximum(abs, x)
Linf(x,y) = maximum(abs(a-b) for (a,b) in zip(x,y))
maxeps(x) = Linf(x) * eps(eltype(x))

function test_curlgrad(sphere, qi)
    gradq = gradient(sphere, qi)
    curlgradq = curl(sphere, gradq)
    return Linf(curlgradq) < maxeps(gradq)
end

function test_divgradperp(sphere, psi)
    U = gradperp(sphere, psi)
    divU = divergence(sphere, U)
    return Linf(divU) * maximum(sphere.Ai) < 2maxeps(U)
end

function test_TRiSK(sphere, phi, psi, qe)
    U = gradperp(sphere, psi) + gradient(sphere, phi)
    Uperp = TRiSK(sphere, U, qe)
    sym = sum(u * up for (u, up) in zip(U, Uperp))
    return abs(sym) < Linf(Uperp) * maxeps(U) * sqrt(length(U))
end

function test_curlTRiSK(sphere, qi)
    U = gradient(sphere, qi) .* sphere.le_de # cov => contra
    Uperp = TRiSK(sphere, U)
    curlUperp = curl(sphere, Uperp) ./ sphere.Av
    divU = average_iv(sphere, divergence(sphere, U))
    return Linf(curlUperp + divU) < 1e-12 # can we get closer to eps(Float64) ?
end

function test_perp(tol, sphere)
    gradz_n = [z for (x, y, z) in sphere.normal_e] # ∇z, normal component
    gradz_t = [z for (x, y, z) in sphere.tangent_e] # ∇z, tangential component
    return Linf(perp(sphere, gradz_n), gradz_t) < Linf(gradz_n) * tol
end

function test_div(tol, sphere)
    curlz = [z * le for ((x, y, z), le) in zip(sphere.tangent_e, sphere.le)] # ∇z⟂, contravariant
    return Linf(divergence(sphere, curlz)) < tol
end

function test_gradient3d(tol, sphere, qi)
    # q=sinϕ, |∇q|²=cos²ϕ  ⇒  |∇q|²+q²-1 = 0
    gradq = gradient3d(sphere, qi)
    check = (dot(gq, gq) + q^2 - 1 for (q, gq) in zip(qi, gradq))
    return Linf(check) < tol
end

dot(a::NTuple{3,F}, b::NTuple{3,F}) where {F} = @unroll sum(a[i] * b[i] for i = 1:3)
