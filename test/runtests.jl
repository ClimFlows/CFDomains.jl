using ThreadPinning
pinthreads(:cores)
using NetCDF: ncread
import LinearAlgebra as LinAlg
using BenchmarkTools
using InteractiveUtils

import Mooncake
import ForwardDiff
import DifferentiationInterface as DI
using DifferentiationInterface: Constant as Const

using LoopManagers: SIMD, VectorizedCPU, MultiThread
using ManagedLoops: @with, @vec, @unroll
using SHTnsSpheres: SHTnsSphere
using ClimFlowsData: DYNAMICO_reader, DYNAMICO_meshfile

using CFDomains: CFDomains, Stencils, VoronoiSphere, transpose!, void
using CFDomains.LazyExpressions: @lazy
import CFDomains.VoronoiOperators as Ops

# using ClimFlowsPlots.SphericalInterpolations: lonlat_interp

using Test

include("partial_derivative.jl")
include("voronoi_operators.jl")

include("zero_arrays.jl")
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

@testset "transpose!" begin
    x = randn(3,4)
    y = transpose!(void, nothing, x)
    @test y == transpose!(similar(y), void, x)
    @test y == x'
end

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

function f1(cc, a, g) 
    @lazy c(a ; g) = a+g/2
    for i in eachindex(cc)
        @inbounds cc[i] = c[i]
    end
    return c
end

function f2(cc, a, b) 
    @lazy c(a, b) = b*a^2
    for i in eachindex(cc)
        @inbounds cc[i] = c[i]
    end
    return c
end

function f3(d, op, a, b) 
    @lazy c(a ; b) = b*a^2
    op!(d, nothing, a)
    return sum(d)
end

@testset "LazyExpressions" begin
    F, ncell, nedge = choices.precision, length(sphere.lon_i), length(sphere.lon_e)
    a = randn(F, ncell);
    b = randn(F, ncell);
    c = randn(F, ncell);
    g = F(9.81)

    grad! = Ops.Gradient(sphere)
    ucov = randn(F, nedge)
    ucov2 = similar(ucov)

    grad!(ucov, nothing, f1(c, a, g))
    grad!(ucov2, nothing, c)
    @test ucov ≈ ucov2

    grad!(ucov, nothing, f1(c, a, b))
    grad!(ucov2, nothing, c)
    @test ucov ≈ ucov2

    c_ = f2(c, a, b)
    grad!(ucov, nothing, c_) # c_ is lazy
    grad!(ucov2, nothing, c)
    @test ucov ≈ ucov2

    @info "Gradient of concrete array"
    display(@benchmark $grad!($ucov2, nothing, $c) )
    @info "Gradient of lazy array"
    display(@benchmark $grad!($ucov2, nothing, $c_) )
#    display(@code_native grad!(ucov2, nothing, c_))
end

@testset "VoronoiOperators" begin
    q = randn(choices.precision, length(sphere.lon_i))
    r = randn(choices.precision, length(sphere.lon_i))
    qe = randn(choices.precision, length(sphere.lon_e))
    ucov = randn(choices.precision, length(sphere.lon_e))
    qv = randn(choices.precision, length(sphere.lon_v))
    tmp_i = similar(q)
    tmp_e = similar(q, length(sphere.lon_e))
    tmp_v = similar(q, length(sphere.lon_v))

    # Linear VoronoiOperator{1,1}
    test_op(q, tmp_v, Ops.DualFromPrimal(sphere))
    test_op(qv, tmp_e, Ops.EdgeFromDual(sphere))
    test_op(ucov, tmp_v, Ops.Curl(sphere))
    test_op(q, tmp_e, Ops.Gradient(sphere))
    test_op(ucov, tmp_i, Ops.Divergence(sphere))
    test_op(ucov, tmp_e, Ops.TRiSK(sphere))
    # Quadratic VoronoiOperator{1,1}
    test_op(ucov, tmp_i, Ops.SquaredCovector(sphere))
    # Bilinear VoronoiOperator{1,2}
    test_op(q, ucov, tmp_e, Ops.CenteredFlux(sphere))
    test_op(qe, ucov, tmp_e, Ops.EnergyTRiSK(sphere))
    test_op(q, ucov, tmp_i, Ops.DivCenteredFlux(sphere))
    test_op(q, r, tmp_e, Ops.MulGradient(sphere))
    # LazyDiagonalOp
    test_norm_div(ucov, tmp_i, sphere)
end

# include("benchmark.jl")
