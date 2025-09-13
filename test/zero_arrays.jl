using CFDomains.ZeroArrays: zero_array

@testset "zero_array" begin
    x = randn(3,4)
    z = zero_array(x)
    @test z == zero(x)
    @test (@. z*x) == z
    @test (z+x) == x
end
