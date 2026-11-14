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
import Mooncake

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
    @test factor == [10.0, 20.0, 30.0]
end

@testset "Typed Enzyme LazyExpression cotangent with inactive input" begin
    LazyExpressions = CFDomains.LazyExpressions
    LazyOperators = CFDomains.LazyOperators
    x, mask = [1.5], [2]
    dx, dmask = zeros(1), zeros(Int, 1)
    expr = LazyExpressions.lazy_expr((x::Float64, mask::Int) -> x * mask, (x, mask), ())
    shadow = LazyExpressions.lazy_expr(expr.fun, (dx, dmask), ())
    cotangent = LazyExpressions.lazy_cotangent(Enzyme.MixedDuplicated(expr, Ref(shadow)))

    LazyOperators.addto!(cotangent, 1.0, 1)
    @test dx == [2.0]
    @test dmask == [0]
end


@testset "Nested Mooncake LazyExpression input" begin
    function nested_lazy_read(x)
        LazyExpressions = CFDomains.LazyExpressions
        @lazy inner(x) = x * x
        @lazy outer(x) = 2 * x
        nested = LazyExpressions.lazy_expr(outer.fun, (inner,), outer.params)
        nested[1]
    end

    x = [2.0]
    cache = Mooncake.prepare_gradient_cache(nested_lazy_read, x)
    value, gradient = Mooncake.value_and_gradient!!(cache, nested_lazy_read, x)
    @test value == 8.0
    @test gradient[2] == [8.0]
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
    LazyOperators = CFDomains.LazyOperators
    x = reshape([1.5, -2.0, 3.0, 0.5], 2, 2)
    dx = zeros(size(x))
    @lazy square(x) = x * x
    expr = square
    shadow = LazyExpressions.lazy_expr(expr.fun, (dx,), expr.params)

    read_sum(expr) = expr[1, 1] + expr[2, 2]
    Enzyme.autodiff(Enzyme.Reverse, read_sum, Enzyme.Active,
                    Enzyme.MixedDuplicated(expr, Ref(shadow)))
    @test dx == [2 * x[1, 1] 0; 0 2 * x[2, 2]]
    @test expr[CartesianIndex(1), 2] == x[1, 2]^2
    fill!(dx, 0)
    Enzyme.autodiff(Enzyme.Reverse, expr -> expr[1, CartesianIndex(2)], Enzyme.Active,
                    Enzyme.MixedDuplicated(expr, Ref(shadow)))
    @test dx == [0 2 * x[1, 2]; 0 0]
    fill!(dx, 0)
    Enzyme.autodiff(Enzyme.Reverse, expr -> expr[CartesianIndex(1), CartesianIndex(2)],
                    Enzyme.Active, Enzyme.MixedDuplicated(expr, Ref(shadow)))
    @test dx == [0 2 * x[1, 2]; 0 0]
    fill!(dx, 0)
    fill!(dx, 0)
    Enzyme.autodiff(Enzyme.Reverse, expr -> expr[CartesianIndex(1), 2], Enzyme.Active,
                    Enzyme.MixedDuplicated(expr, Ref(shadow)))
    @test dx == [0 2 * x[1, 2]; 0 0]
    fill!(dx, 0)
    @test_throws BoundsError Enzyme.autodiff(
        Enzyme.Reverse, expr -> expr[3, 1], Enzyme.Active,
        Enzyme.MixedDuplicated(expr, Ref(shadow)),
    )

    fill!(dx, 0)
    @lazy twice(x) = 2 * x
    nested = LazyExpressions.lazy_expr(twice.fun, (expr,), twice.params)
    nested_shadow = LazyExpressions.lazy_expr(twice.fun, (shadow,), twice.params)
    Enzyme.autodiff(Enzyme.Reverse, read_sum, Enzyme.Active,
                    Enzyme.MixedDuplicated(nested, Ref(nested_shadow)))
    @test dx == [4 * x[1, 1] 0; 0 4 * x[2, 2]]

    x_float = [1.5]
    mask = [2]
    dx_float = zeros(size(x_float))
    dmask = zeros(Int, size(mask))
    @lazy masked_product(x_float, mask) = x_float * mask
    expr = masked_product
    shadow = LazyExpressions.lazy_expr(expr.fun, (dx_float, dmask), expr.params)
    Enzyme.autodiff(Enzyme.Reverse, expr -> expr[1], Enzyme.Active,
                    Enzyme.MixedDuplicated(expr, Ref(shadow)))
    @test dx_float == [2.0]
    @test dmask == [0]

    x_powered = [-2.0]
    dx_powered = zeros(size(x_powered))
    exponent = [2.0]
    @lazy powered(x_powered; exponent) = x_powered ^ exponent
    shared_power_shadow = LazyExpressions.lazy_expr(powered.fun, (dx_powered,), powered.params)
    shared_power_cotangent = LazyExpressions.lazy_cotangent(
        Enzyme.MixedDuplicated(powered, Ref(shared_power_shadow)),
    )
    LazyOperators.addto!(shared_power_cotangent, 1.0, 1)
    @test dx_powered == [-4.0]
    @test exponent == [2.0]

    typed_x = [2.0]
    typed_dx = zeros(size(typed_x))
    typed_expr = LazyExpressions.lazy_expr((x::Float64) -> x * x, (typed_x,), ())
    typed_shadow = LazyExpressions.lazy_expr(typed_expr.fun, (typed_dx,), ())
    Enzyme.autodiff(Enzyme.Reverse, expr -> expr[1], Enzyme.Active,
                    Enzyme.MixedDuplicated(typed_expr, Ref(typed_shadow)))
    @test typed_dx == [4.0]

    x_powered = [-2.0]
    dx_powered = zeros(size(x_powered))
    exponent = [2.0]
    @lazy powered(x_powered; exponent) = x_powered ^ exponent
    powered_shadow = LazyExpressions.lazy_expr(powered.fun, (dx_powered,), powered.params)
    Enzyme.autodiff(Enzyme.Reverse, expr -> expr[1], Enzyme.Active,
                    Enzyme.MixedDuplicated(powered, Ref(powered_shadow)))
    @test dx_powered == [-4.0]
    @test exponent == [2.0]

    x_scaled = [2.0]
    dx_scaled = zeros(size(x_scaled))
    factor = [3.0]
    @lazy scaled(x_scaled; factor) = x_scaled * factor
    scaled_shadow = LazyExpressions.lazy_expr(scaled.fun, (dx_scaled,), scaled.params)
    Enzyme.autodiff(Enzyme.Reverse, expr -> expr[1], Enzyme.Active,
                    Enzyme.MixedDuplicated(scaled, Ref(scaled_shadow)))
    @test dx_scaled == [3.0]
    @test factor == [3.0]

    dx_shared = zeros(size(x_scaled))
    shared_shadow = LazyExpressions.lazy_expr(scaled.fun, (dx_shared,), scaled.params)
    shared_cotangent = LazyExpressions.lazy_cotangent(
        Enzyme.MixedDuplicated(scaled, Ref(shared_shadow)),
    )
    LazyOperators.addto!(shared_cotangent, 1.0, 1)
    @test dx_shared == [3.0]
    @test factor == [3.0]

    x_negative = [-1.0]
    dx_negative = zeros(size(x_negative))
    @lazy positive(x_negative) = x_negative > 0 ? x_negative : 0.0
    positive_shadow = LazyExpressions.lazy_expr(positive.fun, (dx_negative,), positive.params)
    Enzyme.autodiff(Enzyme.Reverse, expr -> expr[1], Enzyme.Active,
                    Enzyme.MixedDuplicated(positive, Ref(positive_shadow)))
    @test dx_negative == [0.0]

    x_gated = [2.0]
    dx_gated = zeros(size(x_gated))
    flag = [true]
    dflag = [false]
    @lazy gated(x_gated, flag) = flag ? x_gated : -x_gated
    gated_shadow = LazyExpressions.lazy_expr(gated.fun, (dx_gated, dflag), gated.params)
    Enzyme.autodiff(Enzyme.Reverse, expr -> expr[1], Enzyme.Active,
                    Enzyme.MixedDuplicated(gated, Ref(gated_shadow)))
    @test dx_gated == [1.0]
    @test dflag == [false]

    x32 = Float32[2]
    dx32 = zeros(Float32, 1)
    @lazy square32(x32) = x32 * x32
    square32_shadow = LazyExpressions.lazy_expr(square32.fun, (dx32,), square32.params)
    Enzyme.autodiff(Enzyme.Reverse, expr -> expr[1]^2, Enzyme.Active,
                    Enzyme.MixedDuplicated(square32, Ref(square32_shadow)))
    @test dx32 == [32.0f0]

    counter = Ref(0)
    x_counted = [2.0]
    dx_counted = zeros(size(x_counted))
    @lazy counted(x_counted) = (counter[] += 1; x_counted * counter[])
    counted_shadow = LazyExpressions.lazy_expr(counted.fun, (dx_counted,), counted.params)
    Enzyme.autodiff(Enzyme.Reverse, expr -> expr[1], Enzyme.Active,
                    Enzyme.MixedDuplicated(counted, Ref(counted_shadow)))
    @test counter[] == 1
    @test dx_counted == [1.0]

    x_original = [1.25, -0.75]
    expected_dx = 2 * x_original[1]
    dx_original = zeros(size(x_original))
    @lazy square_to_mutate(x_original) = x_original * x_original
    mutating_expr = square_to_mutate
    mutating_shadow = LazyExpressions.lazy_expr(mutating_expr.fun, (dx_original,), mutating_expr.params)
    read_then_mutate(expr) = begin
        value = expr[1]
        expr.inputs[1][1] = 0.0
        value
    end
    Enzyme.autodiff(Enzyme.Reverse, read_then_mutate, Enzyme.Active,
                    Enzyme.MixedDuplicated(mutating_expr, Ref(mutating_shadow)))
    @test dx_original[1] == expected_dx

    arrays = ntuple(_ -> [1.0, 2.0], 4)
    shadows = ntuple(_ -> zeros(2), 4)
    a, b, c, d = arrays
    @lazy four_input_sum(a, b, c, d) = a + b + c + d
    four_input_expr = LazyExpressions.lazy_expr(four_input_sum.fun, arrays, four_input_sum.params)
    four_input_shadow = LazyExpressions.lazy_expr(four_input_sum.fun, shadows, four_input_sum.params)
    Enzyme.autodiff(Enzyme.Reverse, expr -> expr[1], Enzyme.Active,
                    Enzyme.MixedDuplicated(four_input_expr, Ref(four_input_shadow)))
    @test shadows == ([1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0])

    nested_four = LazyExpressions.lazy_expr(twice.fun, (four_input_expr,), twice.params)
    nested_four_shadow = LazyExpressions.lazy_expr(twice.fun, (four_input_shadow,), twice.params)
    Enzyme.autodiff(Enzyme.Reverse, expr -> expr[1], Enzyme.Active,
                    Enzyme.MixedDuplicated(nested_four, Ref(nested_four_shadow)))
    @test shadows == ([3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0])
end
