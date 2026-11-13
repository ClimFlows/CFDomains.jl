module CFDomainsAdaptExt

using CFDomains: Shell, HybridCoordinate, HybridMassCoordinate
using CFDomains.LazyExpressions: LazyExpression
import CFDomains.LazyExpressions: adapt_structure
using Adapt: @adapt_structure, adapt
import Adapt

@adapt_structure Shell
@adapt_structure HybridCoordinate
@adapt_structure HybridMassCoordinate

Adapt.adapt_structure(to, lazy::LazyExpression{T,N}) where {T,N} = adapt_structure(adapt, to, lazy)
end
