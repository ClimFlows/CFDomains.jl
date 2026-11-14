# CFDomains

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ClimFlows.github.io/CFDomains.jl/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ClimFlows.github.io/CFDomains.jl/dev/)
[![Build Status](https://github.com/ClimFlows/CFDomains.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ClimFlows/CFDomains.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ClimFlows/CFDomains.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ClimFlows/CFDomains.jl)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ClimFlows/CFDomains.jl)

## Change Log

### v0.6.1

* new: Enzyme reverse mode differentiates direct reads from `LazyExpression`
  values by accumulating into input shadows without a full-sized cotangent
  array. Like Mooncake, the Enzyme rule treats `params` as constants; pass
  arrays that need gradients as positional inputs. Nested lazy inputs, mixed
  precision, complex values, active arrays captured by the expression function,
  and lazy expressions used as parameters are unsupported. A generic callable
  that accepts duals but calls a dual-incompatible helper is also unsupported.

### v0.6

* new: Enzyme write-through cotangents for direct array inputs to lazy
  expressions in custom reverse rules; contributions accumulate directly into
  input shadows without a full-sized temporary array.

### v0.5

* 0.5.4: remove inactive tests for Voronoi operators now maintained in
  VoronoiSpheres.jl and trim unused test dependencies.
* 0.5.3: support multidimensional lazy expressions for layered fields.
* 0.5.2: fix reverse-mode dispatch for operators that subtract from an
  existing output (`subfrom!`).
* 0.5.0: move the Voronoi mesh and its operators to
  [VoronoiSpheres.jl](https://github.com/ClimFlows/VoronoiSpheres.jl) (#32).

### v0.4

* breaking: `mass_level` expects an additional argument for horizontal position. This is to allow the argument `metric_cov` of `mass_coordinate` to be a `Vector` describing horizontal variations (#29)

### v0.3

* breaking: `VoronoiSphere` expects additional data from mesh file reader (#11)

* new: 
  * 0.3.10-0.3.14: Mooncake adjoints for Voronoi-mesh operators (#23-#29)
  * 0.3.9: Voronoi divergence yielding a two-form (#22)
  * 0.3.7: `zero_array` (#20)
  * 0.3.6: `transpose!` (#19)
  * 0.3.6: new Voronoi stencil for dot products (#19)
  * 0.3.5: single-argument call to Voronoi stencil extracts relevant mesh data ; useful to pass fewer arguments to GPU kernels  (#16)
  * 0.3.3: compute `cen2vertex`, needed for transport scheme on Voronoi meshes  (#14)
  * 0.3.0: Voronoi stencils for `gradient3d` and `perp` operators (#11)

* fixed: 
  * 0.3.2: Voronoi averaging stencils (#13)
  * 0.3.4: dispatch for Trisk operator
