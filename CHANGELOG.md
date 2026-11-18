# Changelog

## 0.6.2

- Fix Mooncake cotangent accumulation for multidimensional and nested lazy
  expressions, including linear and Cartesian indices.
- Load the Mooncake extension only with ForwardDiff, which it uses to compute
  scalar partial derivatives.
- Add regression tests for one- and two-dimensional cotangents.

Earlier releases are described in the [README](README.md#change-log).
