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
import Enzyme

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

@testset "Enzyme LazyExpression cotangent" begin
    LazyExpressions = CFDomains.LazyExpressions
    LazyOperators = CFDomains.LazyOperators
    x = reshape(Float64.(1:6), 2, 3)
    y = reshape(Float64.(7:12), 2, 3)
    dx, dy = zeros(size(x)), zeros(size(y))
    factor = [10.0, 20.0, 30.0]
    @lazy product(x, y; factor) = factor * x * y
    expr = product
    shadow = LazyExpressions.lazy_expr(expr.fun, (dx, dy), expr.params)
    mixed = Enzyme.MixedDuplicated(expr, Ref(shadow))
    cotangent = LazyExpressions.lazy_cotangent(mixed)

    @test cotangent.dinputs === shadow.inputs
    LazyOperators.addto!(cotangent, 2.0, 2, 3)
    LazyOperators.subfrom!(cotangent, 0.5, 2, 3)
    weight = 1.5 * factor[3]
    @test dx[2, 3] == weight * y[2, 3]
    @test dy[2, 3] == weight * x[2, 3]
    @test count(!iszero, dx) == 1
    @test count(!iszero, dy) == 1

    LazyOperators.addto!(cotangent, 1.0, 2)
    LazyOperators.addto!(cotangent, 0.5, first(eachindex(cotangent)))
    @test dx[2, 1] == factor[1] * y[2, 1]
    @test dy[2, 1] == factor[1] * x[2, 1]
    @test dx[1, 1] == 0.5 * factor[1] * y[1, 1]
    @test dy[1, 1] == 0.5 * factor[1] * x[1, 1]
end

@testset "Nested Enzyme LazyExpression cotangent" begin
    LazyExpressions = CFDomains.LazyExpressions
    LazyOperators = CFDomains.LazyOperators
    x = reshape(Float64.(1:6), 2, 3)
    dx = zeros(size(x))
    @lazy square(x) = x * x
    @lazy double(x) = 2 * x

    inner = square
    outer = LazyExpressions.lazy_expr(double.fun, (inner,), double.params)
    inner_shadow = LazyExpressions.lazy_expr(inner.fun, (dx,), inner.params)
    outer_shadow = LazyExpressions.lazy_expr(double.fun, (inner_shadow,), double.params)
    mixed = Enzyme.MixedDuplicated(outer, Ref(outer_shadow))
    cotangent = LazyExpressions.lazy_cotangent(mixed)

    LazyOperators.addto!(cotangent, 1.0, 2, 3)
    @test dx[2, 3] == 4 * x[2, 3]
    @test count(!iszero, dx) == 1
end

@testset "Enzyme differentiates direct LazyExpression reads" begin
    LazyExpressions = CFDomains.LazyExpressions
    x = reshape([1.5, -2.0, 3.0, 0.5], 2, 2)
    dx = zeros(size(x))
    @lazy square(x) = x * x
    expr = square
    shadow = LazyExpressions.lazy_expr(expr.fun, (dx,), expr.params)

    read_sum(expr) = expr[1, 1] + expr[2, 2]
    Enzyme.autodiff(Enzyme.Reverse, read_sum, Enzyme.Active,
                    Enzyme.MixedDuplicated(expr, Ref(shadow)))
    @test dx == [2 * x[1, 1] 0; 0 2 * x[2, 2]]

    fill!(dx, 0)
    @lazy twice(x) = 2 * x
    nested = LazyExpressions.lazy_expr(twice.fun, (expr,), twice.params)
    nested_shadow = LazyExpressions.lazy_expr(twice.fun, (shadow,), twice.params)
    Enzyme.autodiff(Enzyme.Reverse, read_sum, Enzyme.Active,
                    Enzyme.MixedDuplicated(nested, Ref(nested_shadow)))
    @test dx == [4 * x[1, 1] 0; 0 4 * x[2, 2]]
end
