module Stencils

using CFDomains: HVLayout, VHLayout
using ManagedLoops: @unroll

# Each stencil operation is implemented in three steps:
# 1- Extract only those fields that are relevant for the stencil
#       sph = <stencil>(sph)
#    Returns a named tuple.
# 2- Extract mesh data for a given mesh element ij (cell, edge, dual)
#       op = <stencil>(sph, ij, [::Val{N}]) . 
#    N is the "degree" = number of connected mesh elements, if not known in advance.
#    op is of the form Fix(expr, data...) which is a callable object 
#    such that op(args...) = expr(data..., args...)
# 3- `expr` evaluates the stencil expression, e.g. `sum_weighted`, ...

include("voronoi_stencils_helpers.jl")
include("voronoi_stencils_linear.jl")
include("voronoi_stencils_quadratic.jl")
include("voronoi_stencils_transport.jl")
include("voronoi_stencils_deprecated.jl")

end
