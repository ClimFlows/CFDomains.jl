module SHTnsSpheres_Ext

    using SHTnsSpheres: SHTnsSphere
    using CFDomains: CFDomains, HyperDiffusion, HVLayout
    import CFDomains: hyperdiffusion!, hyperdiff_shell!

    CFDomains.data_layout(::SHTnsSphere) = HVLayout(2)

    function hyperdiff_shell!(sph::SHTnsSphere, ::HVLayout{2}, hd::HyperDiffusion{:vector_curl}, storage, coefs, ::Nothing)
        (; niter, nu), (; laplace, lmax) = hd, sph
        @. storage.spheroidal = (1-nu*(-laplace/(lmax*(lmax+1)))^niter)*coefs.spheroidal
        return storage
    end

end # module