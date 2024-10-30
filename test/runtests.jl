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

include("voronoi.jl")

nlat = 16
sph = SHTnsSphere(nlat)
@info CFDomains.data_layout(sph)

choices = (precision = Float64, meshname = "uni.1deg.mesh.nc", tol=1e-3)

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

@testset "VoronoiSphere" begin
    qi = [z for (x,y,z) in sphere.xyz_i]
    qv = [z for (x,y,z) in sphere.xyz_v]
    qe = [z for (x,y,z) in sphere.xyz_e]

    @test test_curlgrad(sphere, qi) # curl∘grad == 0
    @test test_divgradperp(sphere, qv)  # div∘gradperp == 0
    @test test_TRiSK(sphere, qi, qv, qe)  # antisymmetry
    @test test_curlTRiSK(sphere, qi)  # curl∘TRiSK = average_iv∘div
    @test test_perp(choices.tol, sphere) # accuracy
    @test test_div(choices.tol, sphere) # accuracy
    @test test_gradient3d(choices.tol, sphere, qi)
end

# include("benchmark.jl")
