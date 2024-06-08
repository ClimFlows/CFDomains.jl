using CFDomains
using SHTnsSpheres: SHTnsSphere
using Test

@testset "CFDomains.jl" begin

    nlat = 16
    sph = SHTnsSphere(nlat)
    @info CFDomains.data_layout(sph)

    # Write your tests here.
end
