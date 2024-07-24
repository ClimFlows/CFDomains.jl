abstract type AbstractFilter ; end

"""
    # generic
    filtered = filter!(space, scratch, field, filter::AbstractFilter, dt, mgr::LoopManager)

    # out-of-place, allocates `scratch`, allocates and returns `space`
    filtered = filter!(void, void, field, filter, dt, mgr)

    # mutating, non-allocating
    filter!(filtered, scratch, field, filter, dt, mgr)

    # in-place, non-allocating
    filter!(field, scratch, field, filter, dt, mgr)

Apply `filter` to `field` during time `dt` with loop manager `mgr`. `mgr` can be `nothing`.
`space` is the output field, or `void`. In the latter case, `space` is allocated as `similar(field)` or equivalent.
`scratch` is scratch space allocated with [`scratch_space`](@ref), or `void`. In the latter case, `scratch` is implicitly allocated.
"""
function filter! end

"""
    scratch = scratch_space(filter::AbstractFilter, field, [scratch])

If `scratch` is omitted or `::Void`, allocate and return scratch space for applying `filter` to `field`.
Otherwise just return `scratch`. See also [`filter!`](@ref).
"""
function scratch_space end

# generic fallbacks
scratch_space(filter, field, ::Void) = scratch_space(filter, field)
scratch_space(filter, field, scratch) = scratch

struct HyperDiffusion{fieldtype, D, F, X} <: AbstractFilter
    domain :: D
    niter :: Int
    nu :: F
    extra::X # pre-computed stuff, if any
end
const HD=HyperDiffusion

"""
    filter = HyperDiffusion(fieldtype::Symbol, domain, niter::Int, nu)                  # user-friendly
    filter = HyperDiffusion{fieldtype, D, F, X}(domain::D, niter::Int, nu::F, extra::X) # internal

Return a `filter` that applies Laplacian diffusion iterated `niter` times with hyperdiffusive
coefficient `nu ≥ 0` on fields of type `fieldtype`. Supported field types
are `:scalar`, `:vector`, `:vector_curl`, `:vector_div`.

The filter is to be used as:

    new_data = filter(out_data, in_data, scratch=void)

The above syntax returns `hyperdiffusion!(filter.domain, filter, out_data, in_data, scratch)`. If `scratch::Void`, then

    scratch = scratch_space(filter, in_data)

The user-friendly `HyperDiffusion` may be specialized for specific types of `domain`. The specialized method should call
the internal constructor which takes an extra argument `extra`, stored as `filter.extra` for later use
by [`hyperdiffusion!`](@ref). The default user-friendly constructor sets `extra=nothing`.

Similarly, `scratch_space(filter, in_data)` calls `scratch_hyperdiff(f.domain, Val(fieldtype), in_data)` which by default returns `nothing`.
[`scratch_hyperdiff`](@ref) may be specialized for specific domain types and `fieldtype`.
"""
HyperDiffusion(domain::D, niter, nu::F, fieldtype) where {D, F} =
    HyperDiffusion{fieldtype, D, F}(domain, niter, nu, nothing)

(hd::HyperDiffusion)(out_data, in_data) = hyperdiffusion!(hd.domain, hd, storage, coefs)

"""
    new_data = hyperdiffusion!(filter.domain, filter, out_data, in_data, scratch)

Apply hyperdiffusive `filter`` to `in_data`. The result is written into `out_data` and returned.
If `domain::SpectralDomain`, the filter applies to spectral coefficients. If `out_data::Void`, it is adequately allocated.
See `[`HyperDiffusion`](@ref).
"""
function hyperdiffusion! end

scratch_space(f::HD{fieldtype}, field) where fieldtype = scratch_hyperdiff(f.domain, Val(fieldtype), field)

"""
    scratch = scratch_hyperdiff!(domain, Val(fieldtype), field)

Return scratch space used to apply hyperdiffusion on `domain` for a `field` of a certain `fieldtype`.
See [`HyperDiffusion`](@ref) and [`hyperdiffusion!`](@ref).
"""
scratch_hyperdiff!(domain, ::Val, field) = nothing

filter!(out, field, f::HyperDiffusion, dt, scratch, mgr=nothing) = hyperdiff!(out, field, f, f.domain, dt, scratch, mgr)
