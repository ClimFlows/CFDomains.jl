# CFDomains

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ClimFlows.github.io/CFDomains.jl/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ClimFlows.github.io/CFDomains.jl/dev/)
[![Build Status](https://github.com/ClimFlows/CFDomains.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ClimFlows/CFDomains.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ClimFlows/CFDomains.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ClimFlows/CFDomains.jl)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ClimFlows/CFDomains.jl)

## Change Log

### v0.3

* breaking: `VoronoiSphere` expects additional data from mesh file reader (#11)

* new: 
  * 0.3.6: new Voronoi stencil for dot product of contravariant vectors (#17)
  * 0.3.6: `transpose!` (#17)
  * 0.3.5: single-argument call to Voronoi stencil extracts relevant mesh data ; useful to pass fewer arguments to GPU kernels  (#16)
  * 0.3.3: compute `cen2vertex`, needed for transport scheme on Voronoi meshes  (#14)
  * 0.3.0: Voronoi stencils for `gradient3d` and `perp` operators (#11)

* fixed: 
  * 0.3.2: Voronoi averaging stencils (#13)
  * 0.3.4: dispatch for Trisk operator
