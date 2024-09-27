module CFDomainsAdaptExt

using CFDomains: VoronoiSphere, Shell, HybridCoordinate, HybridMassCoordinate
using Adapt: @adapt_structure 

@adapt_structure VoronoiSphere
@adapt_structure Shell
@adapt_structure HybridCoordinate
@adapt_structure HybridMassCoordinate

end
