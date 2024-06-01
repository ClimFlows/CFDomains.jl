module SHTnsSpheres_Ext

    using SHTnsSpheres: SHTnsSphere
    using CFDomains: CFDomains, HVLayout

    CFDomains.data_layout(::SHTnsSphere) = HVLayout()

end # module