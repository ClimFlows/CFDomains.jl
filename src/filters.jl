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

struct HyperDiffusion{fieldtype, D, F} <: AbstractFilter
    domain :: D
    niter :: Int
    nu :: F
end
const HD=HyperDiffusion

"""
    filter = Hyperdiffusion(domain, niter, nu, fieldtype::Symbol)

Return a `filter` that applies Laplacian diffusion iterated `niter` times with hyperdiffusive
coefficient `nu ≥ 0` on fields of type `fieldtype`. Supported field types for `domain::VoronoiSphere`
are `:scalar`, `:vector_curl`, `:vector_div`.
"""
HyperDiffusion(domain::D, niter, nu::F, fieldtype) where {D, F} =
    HyperDiffusion{fieldtype, D, F}(domain, niter, nu)

scratch_space(f::HD{fieldtype}, field) where fieldtype = scratch_hyperdiff(f.domain, Val(fieldtype), field)

filter!(out, field, f::HyperDiffusion, dt, scratch, mgr=nothing) = hyperdiff!(out, field, f, f.domain, dt, scratch, mgr)
