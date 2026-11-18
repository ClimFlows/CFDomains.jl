@testset "Mooncake lazy cotangents" begin
    LazyExpressions = CFDomains.LazyExpressions
    LazyOperators = CFDomains.LazyOperators
    MooncakeExt = Base.get_extension(CFDomains, :CFDomainsMooncakeExt)
    @test MooncakeExt !== nothing

    x = reshape(Float64.(1:6), 2, 3)
    y = reshape(Float64.(7:12), 2, 3)
    dx, dy = zeros(size(x)), zeros(size(y))
    factor = [10.0, 20.0, 30.0]
    matrix_factor = reshape(Float64.(2:7), 2, 3)
    @lazy product(x, y; factor, matrix_factor) = factor * matrix_factor * x * y
    tangent = MooncakeExt.tlazy_expr(product, (dx, dy))

    @test Mooncake.tangent_type(typeof(product)) == typeof(tangent)
    @test size(tangent) == size(x)
    @test axes(tangent) == axes(x)
    @test first(eachindex(tangent)) == CartesianIndex(1, 1)

    LazyOperators.addto!(tangent, 2.0, 2, 3)
    LazyOperators.subfrom!(tangent, 0.5, CartesianIndex(2, 3))
    @test dx[2, 3] == 1.5 * factor[3] * matrix_factor[2, 3] * y[2, 3]
    @test dy[2, 3] == 1.5 * factor[3] * matrix_factor[2, 3] * x[2, 3]
    @test count(!iszero, dx) == 1
    @test count(!iszero, dy) == 1

    LazyOperators.addto!(tangent, 1.0, 2)
    @test dx[2, 1] == factor[1] * matrix_factor[2, 1] * y[2, 1]
    @test dy[2, 1] == factor[1] * matrix_factor[2, 1] * x[2, 1]

    dz = zeros(size(x))
    @lazy square(x) = x * x
    @lazy double(x) = 2 * x
    nested = LazyExpressions.lazy_expr(double.fun, (square,), ())
    square_tangent = MooncakeExt.tlazy_expr(square, (dz,))
    nested_tangent = MooncakeExt.tlazy_expr(nested, (square_tangent,))
    @test Mooncake.tangent_type(typeof(nested)) == typeof(nested_tangent)
    LazyOperators.addto!(nested_tangent, 1.0, 1, 2)
    @test dz[1, 2] == 4 * x[1, 2]
    @test count(!iszero, dz) == 1
end
