using ThreadPinning
pinthreads(:cores)
using NetCDF: ncread
using BenchmarkTools

using LoopManagers: VectorizedCPU, MultiThread
using ManagedLoops: @with, @vec, @unroll
using SHTnsSpheres: SHTnsSphere
using CFDomains: CFDomains, Stencils, VoronoiSphere
using ClimFlowsData: DYNAMICO_reader, DYNAMICO_meshfile
# using ClimFlowsPlots.SphericalInterpolations: lonlat_interp

using Test

include("voronoi.jl")

nlat = 16
sph = SHTnsSphere(nlat)
@info CFDomains.data_layout(sph)

choices = (precision = Float64, meshname = "uni.1deg.mesh.nc", tol=1e-3)

reader = DYNAMICO_reader(ncread, DYNAMICO_meshfile(choices.meshname))
sphere = VoronoiSphere(reader; prec = choices.precision)
@info sphere

#=
to_lonlat = let
    F = choices.precision
    lons, lats = F.(1:2:360), F.(-89:2:90)
    permute(data) = permutedims(data, (2, 3, 1))
    permute(data::Matrix) = data
    interp = lonlat_interp(sphere, lons, lats)
    permute ∘ interp ∘ Array
end
=#

@testset "VoronoiSphere" begin
    levels = 1:8
    qi = [z for k in levels, (x,y,z) in sphere.xyz_i]
    qv = [z for k in levels, (x,y,z) in sphere.xyz_v]
    qe = [z for k in levels, (x,y,z) in sphere.xyz_e]

    # check mimetic identities
    test_curlgrad(sphere, qi) # curl∘grad == 0
    test_divgradperp(sphere, qv)  # div∘gradperp == 0
    test_TRiSK(sphere, qi, qv, qe)  # antisymmetry
    test_curlTRiSK(sphere, qi)  # curl∘TRiSK = average_iv∘div
    # check accuracy
    test_perp(choices.tol, sphere, levels) # accuracy
    test_div(choices.tol, sphere, levels) # accuracy
    test_average(choices.tol, sphere, qi) 
    test_gradient3d(choices.tol, sphere, qi)
end

# include("benchmark.jl")
