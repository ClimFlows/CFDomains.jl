using ThreadPinning
pinthreads(:cores)
using NetCDF: ncread
using BenchmarkTools

using LoopManagers: VectorizedCPU, MultiThread
using ManagedLoops: @with, @vec, @unroll
using SHTnsSpheres: SHTnsSphere
using CFDomains: CFDomains, Stencils, VoronoiSphere
using ClimFlowsData: DYNAMICO_reader

using Test

@testset "CFDomains.jl" begin

    nlat = 16
    sph = SHTnsSphere(nlat)
    @info CFDomains.data_layout(sph)


    # Write your tests here.
end

include("benchmark.jl")
