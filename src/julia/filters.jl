"""
    filter = HyperDiffusion(fieldtype::Symbol, domain, niter::Int, nu)                  # user-friendly
    filter = HyperDiffusion{fieldtype, D, F, X}(domain::D, niter::Int, nu::F, extra::X) # internal

Return a `filter` that applies Laplacian diffusion iterated `niter` times with hyperdiffusive
coefficient `nu ≥ 0` on fields of type `fieldtype`. Supported field types
are `:scalar`, `:vector`, `:vector_curl`, `:vector_div`.

The filter is to be used as:

    # out-of-place, allocates `scratch`, allocates and returns `filtered_data`
    filtered_data = filter(void, in_data, void)

    # mutating, non-allocating
    out_data = filter(out_data, in_data, scratch)

    # in-place, non-allocating
    inout_data = filter(inout_data, inout_data, scratch)

`filter(out_data, in_data, scratch)` returns `hyperdiffusion!(filter.domain, filter, out_data, in_data, scratch)`. If `scratch::Void`, then

    scratch = scratch_space(filter, in_data)

The user-friendly `HyperDiffusion` may be specialized for specific types of `domain`. The specialized method should call
the internal constructor which takes an extra argument `extra`, stored as `filter.extra` for later use
by [`hyperdiffusion!`](@ref). The default user-friendly constructor sets `extra=nothing`.

Similarly, `scratch_space(filter, in_data)` calls `scratch_hyperdiff(f.domain, Val(fieldtype), in_data)` which by default returns `nothing`.
[`scratch_hyperdiff`](@ref) may be specialized for specific domain types and `fieldtype`.
"""
HyperDiffusion(domain::D, niter, nu::F, fieldtype) where {D, F} =
    HyperDiffusion{fieldtype, D, F, Nothing}(domain, niter, nu, nothing)

function (hd::HyperDiffusion)(out_data, in_data, scratch=void)
    scratch = scratch_space(hd, in_data, scratch)
    return hyperdiffusion!(hd.domain, hd, out_data, in_data, scratch)
end
"""
    new_data = hyperdiffusion!(domain, filter, out_data, in_data, scratch)

Apply hyperdiffusive `filter`` to `in_data`. The result is written into `out_data` and returned.
If `domain::SpectralDomain`, the filter applies to spectral coefficients. If `out_data::Void`, it is adequately allocated.
See `[`HyperDiffusion`](@ref).
"""
hyperdiffusion!(shell::Shell, filter, out_data, in_data, scratch) =
    hyperdiff_shell!(shell.layer, shell.layout, filter, out_data, in_data, scratch)

"""
    new_data = hyperdiff_shell!(layer, layout, filter, out_data, in_data, scratch)

Apply hyperdiffusive `filter`` to `in_data`. The result is written into `out_data` and returned.
If `domain::SpectralDomain`, the filter applies to spectral coefficients. If `out_data::Void`, it is adequately allocated.
See `[`HyperDiffusion`](@ref).
"""
function hyperdiff_shell! end

"""
    scratch = scratch_space(filter::AbstractFilter, field, [scratch])

If `scratch` is omitted or `::Void`, return scratch space for applying `filter` to `field`,
allocated by [`scratch_hyperdiff`](@ref). Otherwise just return `scratch`. See also [`HyperDiffusion`](@ref).
"""
scratch_space(_, _, scratch) = scratch
scratch_space(filter, field, ::Void) = scratch_space(filter, field)
scratch_space(f::HyperDiffusion{fieldtype}, field) where fieldtype = scratch_hyperdiff(f.domain, Val(fieldtype), field)

"""
    scratch = scratch_hyperdiff(domain, Val(fieldtype), field)

Return scratch space used to apply hyperdiffusion on `domain` for a `field` of a certain `fieldtype`.
See [`HyperDiffusion`](@ref) and [`hyperdiffusion!`](@ref).
"""
scratch_hyperdiff(domain, ::Val, field) = nothing
