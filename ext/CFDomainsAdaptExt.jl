module CFDomainsAdaptExt

using CFDomains: Shell, HybridCoordinate, HybridMassCoordinate
using Adapt: @adapt_structure 

@adapt_structure Shell
@adapt_structure HybridCoordinate
@adapt_structure HybridMassCoordinate

end
