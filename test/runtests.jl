using ThreadPinning
pinthreads(:cores)

using LoopManagers: SIMD, VectorizedCPU
using ManagedLoops: @with, @vec

using CFDomains: CFDomains, transpose!, void
using CFDomains.LazyExpressions: @lazy, pdv
using CFDomains.LazyOperators: adj_action_in, adj_action_out, subfrom!, unchanged!

using Test
using Adapt
import ForwardDiff

@testset "Subtracting operator output" begin
    @test adj_action_in(subfrom!) === subfrom!
    @test adj_action_out(subfrom!) === unchanged!
end
include("partial_derivative.jl")
include("zero_arrays.jl")

@testset "transpose!" begin
    x = randn(3,4)
    y = transpose!(void, nothing, x)
    @test y == transpose!(similar(y), void, x)
    @test y == x'
end

@testset "Multidimensional LazyExpressions" begin
    field = reshape(Float64.(1:6), 2, 3)
    factor = [10.0, 20.0, 30.0]
    @lazy scaled(field; factor) = factor * field

    @test size(scaled) == size(field)
    @test scaled[2, 3] == factor[3] * field[2, 3]
    @test scaled[UInt32(2), UInt32(3)] == factor[3] * field[2, 3]
    @test scaled[4] == factor[2] * field[2, 2]

    vector = [10.0, 20.0]
    @lazy vector_field(vector) = vector
    @test vector_field[2, 1] == vector_field[2]
    @test vector_field[2, 1, 1] == vector_field[2]

    @lazy identity_field(field) = field
    adapted = Adapt.adapt(Array{Float32}, identity_field)
    @test eltype(adapted) == Float32
    @test adapted[2, 3] == Float32(field[2, 3])
end
