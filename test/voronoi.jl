# generic

apply_primal(op, sphere, arg::Vector, more...) = [
    (@unroll N in 5:7 op(op(sphere), cell, Val(N))(arg, more...)) for
    (cell, N) in enumerate(sphere.primal_deg)
]
apply_primal(op, sphere, arg::Matrix, more...) = [
    (@unroll N in 5:7 op(op(sphere), cell, Val(N))(arg, more..., k)) for
    k in axes(arg,1), (cell, N) in enumerate(sphere.primal_deg)
]

apply_trisk(op, sphere, arg::Vector, more...) = [
    (@unroll N in 9:10 op(op(sphere), edge, Val(N))(arg, more...)) for
    (edge, N) in enumerate(sphere.trisk_deg)
]

apply_trisk(op, sphere, arg::Matrix, more...) = [
    (@unroll N in 9:10 op(op(sphere), edge, Val(N))(arg, more..., k)) for
    k in axes(arg,1), (edge, N) in enumerate(sphere.trisk_deg)
]

apply(objects, op, sphere, arg::Vector, more...) = [op(op(sphere), obj)(arg, more...) for obj in objects(sphere)]
apply(objects, op, sphere, arg::Matrix, more...) = 
    [op(op(sphere), obj)(arg, more..., k) for k in axes(arg,1), obj in objects(sphere)]

# mesh objects
cells(sphere) = eachindex(sphere.xyz_i)
duals(sphere) = eachindex(sphere.xyz_v)
edges(sphere) = eachindex(sphere.xyz_e)

# operators
gradient(sphere, qi) = apply(edges, Stencils.gradient, sphere, qi)
gradperp(sphere, qv) = apply(edges, Stencils.gradperp, sphere, qv)
perp(sphere, un) = apply(edges, Stencils.perp, sphere, un)
centered_flux(sphere, qi, un) = apply(edges, Stencils.centered_flux, sphere, qi, un)
curl(sphere, ue) = apply(duals, Stencils.curl, sphere, ue)
average_iv(sphere, qi) = apply(duals, Stencils.average_iv, sphere, qi)
average_ie(sphere, qi) = apply(edges, Stencils.average_ie, sphere, qi)
average_ve(sphere, qv) = apply(edges, Stencils.average_ve, sphere, qv)
divergence(sphere, U) = apply_primal(Stencils.divergence, sphere, U)
gradient3d(sphere, U) = apply_primal(Stencils.gradient3d, sphere, U)
dot_product(sphere, U) = apply_primal(Stencils.dot_product, sphere, U, U)
contraction(sphere, U) = apply_primal(Stencils.contraction, sphere, U, U)
TRiSK(sphere, args...) = apply_trisk(Stencils.TRiSK, sphere, args...)

# error measures
Linf(x) = maximum(abs, x)
Linf(x, y) = maximum(abs(a - b) for (a, b) in zip(x, y))
maxeps(x) = Linf(x) * eps(eltype(x))

# check 3D operator
check_3D(op) = function(sphere, arg::Matrix, more...)
    ret = op(sphere, arg, more...)
    @test ret[1,:] == op(sphere, arg[1,:], map(x->x[1,:], more)...)
    return ret
end

function test_curlgrad(sphere, qi)
    gradq = check_3D(gradient)(sphere, qi)
    curlgradq = curl(sphere, gradq)
    @test Linf(curlgradq) < maxeps(gradq)
end

function test_divgradperp(sphere, psi)
    U = check_3D(gradperp)(sphere, psi)
    divU = check_3D(divergence)(sphere, U)
    @test Linf(divU) * maximum(sphere.Ai) < 2maxeps(U)
end

function test_TRiSK(sphere, phi, psi, qe)
    U = gradperp(sphere, psi) + gradient(sphere, phi)
    Uperp = check_3D(TRiSK)(sphere, U, qe)
    sym = sum(u * up for (u, up) in zip(U, Uperp))
    @test abs(sym) < 2Linf(Uperp) * maxeps(U) * sqrt(length(U))
end

function test_curlTRiSK(sphere, qi)
    U = check_3D(gradient)(sphere, qi) .* transpose(sphere.le_de) # cov => contra
    Uperp = TRiSK(sphere, U)
    curlUperp = check_3D(curl)(sphere, Uperp) ./ transpose(sphere.Av)
    divU = check_3D(average_iv)(sphere, divergence(sphere, U))
    @test Linf(curlUperp + divU) < 1e-12 # can we get closer to eps(Float64) ?
end

function test_perp(tol, sphere, levels)
    gradz_n = [z for k in levels, (x, y, z) in sphere.normal_e] # ∇z, normal component
    gradz_t = [z for k in levels, (x, y, z) in sphere.tangent_e] # ∇z, tangential component
    @test Linf(check_3D(perp)(sphere, gradz_n), gradz_t) < Linf(gradz_n) * tol
    # test dot_product with same data
    check_3D(dot_product)(sphere, gradz_n)    
    check_3D(contraction)(sphere, gradz_n)    
end

function test_div(tol, sphere, levels)
    curlz = [z * le for k in levels, ((x, y, z), le) in zip(sphere.tangent_e, sphere.le)] # ∇z⟂, contravariant
    divcurlz = check_3D(divergence)(sphere, curlz)
    @test  Linf(divcurlz) < tol
end

function test_gradient3d(tol, sphere, qi)
    # q=sinϕ, |∇q|²=cos²ϕ  ⇒  |∇q|²+q²-1 = 0
    gradq = check_3D(gradient3d)(sphere, qi)
    check = (dot(gq, gq) + q^2 - 1 for (q, gq) in zip(qi, gradq))
    return Linf(check) < tol
end

function test_average(tol, sphere, qi)
    qie = check_3D(average_ie)(sphere, qi)
    qiv = check_3D(average_iv)(sphere, qi)
    qve = check_3D(average_ve)(sphere, qiv)
    check_3D(centered_flux)(sphere, qi, qie)
    return Linf(qie, qve) < 2tol
end

dot(a::NTuple{3,F}, b::NTuple{3,F}) where {F} = @unroll sum(a[i] * b[i] for i = 1:3)
