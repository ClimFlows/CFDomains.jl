using ThreadPinning
pinthreads(:cores)
using NetCDF: ncread
using BenchmarkTools

using LoopManagers: VectorizedCPU, MultiThread
using ManagedLoops: @with, @vec, @unroll
using SHTnsSpheres: SHTnsSphere
using CFDomains: CFDomains, Stencils, VoronoiSphere
using ClimFlowsData: DYNAMICO_reader
using ClimFlowsPlots.SphericalInterpolations: lonlat_interp

using Test

const tol = Dict("uni.1deg.mesh.nc" => 0.001f0)

nlat = 16
sph = SHTnsSphere(nlat)
@info CFDomains.data_layout(sph)

choices = (precision = Float32, meshname = "uni.1deg.mesh.nc")
reader = DYNAMICO_reader(ncread, choices.meshname)
sphere = VoronoiSphere(reader; prec = choices.precision)
@info sphere

to_lonlat = let
    F = choices.precision
    lons, lats = F.(1:2:360), F.(-89:2:90)
    permute(data) = permutedims(data, (2, 3, 1))
    permute(data::Matrix) = data
    interp = lonlat_interp(sphere, lons, lats)
    permute ∘ interp ∘ Array
end

dot(a::NTuple{3,F}, b::NTuple{3,F}) where {F} = @unroll sum(a[i] * b[i] for i = 1:3)

cells(sphere) = eachindex(sphere.xyz_i)
duals(sphere) = eachindex(sphere.xyz_v)
edges(sphere) = eachindex(sphere.xyz_e)

function test_curlgrad(sphere) # check curl∘grad=0
    q = [xyz[3] for xyz in sphere.xyz_i]
    gradq = [Stencils.gradient(sphere, edge)(q) for edge in edges(sphere)]
    curlgradq = (Stencils.curl(sphere, dual)(gradq) for dual in duals(sphere))
    return maximum(abs, curlgradq) < maximum(abs, gradq) * eps(eltype(q))
end

function test_perp(meshname, sphere)
    gradz_n = [z for (x, y, z) in sphere.normal_e] # ∇z, normal component
    gradz_t = [z for (x, y, z) in sphere.tangent_e] # ∇z, tangential component
    err = (Stencils.perp(sphere, edge)(gradz_n) - gradz_t[edge] for edge in edges(sphere))
    return maximum(abs, err) < maximum(abs, gradz_n) * tol[meshname]
end

function test_div(meshname, sphere)
    curlz = [z * le for ((x, y, z), le) in zip(sphere.tangent_e, sphere.le)] # ∇z⟂, contravariant
    divcurlz = [
        (@unroll deg in 5:7 dvg = Stencils.divergence(sphere, cell, Val(deg))(curlz);
        dvg) for (cell, deg) in enumerate(sphere.primal_deg)
    ]
    return maximum(abs, divcurlz) < tol[meshname]
end

function test_gradient3d(meshname, sphere)
    maxerr = Dict("uni.1deg.mesh.nc" => 0.00054872036f0)
    q = [xyz[3] for xyz in sphere.xyz_i]
    gradq = [
        (@unroll deg in 5:7 gq = Stencils.gradient3d(sphere, cell, Val(deg))(q);
        gq) for (cell, deg) in enumerate(sphere.primal_deg)
    ]
    # q=sinϕ, |∇q|²=cos²ϕ  ⇒  |∇q|²+q²-1 = 0
    check = (dot(gq, gq) + q[cell]^2 - 1 for (cell, gq) in enumerate(gradq))
    return maximum(abs, check) ≈ maxerr[meshname]
end

@testset "VoronoiSphere" begin
    @test test_curlgrad(sphere)
    @test test_perp(choices.meshname, sphere)
    @test test_div(choices.meshname, sphere)
    @test test_gradient3d(choices.meshname, sphere)
end

# include("benchmark.jl")
