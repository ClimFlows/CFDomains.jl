"""
    abstract type VerticalCoordinate{N} end

Parent type for generalized vertical coordinates ranging from `0` to `N`.
See also [`PressureCoordinate`](@ref).
"""
abstract type VerticalCoordinate{N} end

"""
    abstract type PressureCoordinate{N} <: VerticalCoordinate{N} end

Parent type for a pressure-based vertical coordinate.
Children types should specialize [`pressure_level`](@ref) and [`mass_level`](@ref).
See also [`VerticalCoordinate`](@ref).
"""
abstract type PressureCoordinate{N} <: VerticalCoordinate{N} end

"""
    p = pressure_level(k, ps, vcoord::PressureCoordinate{N})

Returns pressure `p` corresponding to level *`k/2`* as prescribed by vertical coordinate `vcoord`
and surface pressure `ps`.

So-called full levels correspond to odd
values `k=1,2...2N-1` while interfaces between full levels
(so-called half-levels) correspond to even values `k=0,2...2N`
"""
function pressure_level end

"""
    abstract type MassCoordinate{N} <: VerticalCoordinate{N} end

Parent type for a mass-based vertical coordinate.
Children types should specialize [`mass_level`](@ref).
See also [`VerticalCoordinate`](@ref) and [`mass_coordinate`](@ref).
"""
abstract type MassCoordinate{N} <: VerticalCoordinate{N} end

"""
    mcoord = mass_coordinate(pcoord::PressureCoordinate, metric_cov=1)

Return the mass-based coordinate deduced from `pcoord` and
the covariant metric factor `metric_cov`. `metric_cov` can be a scalar or 
a vector. In the latter case `metric_cov[ij]) is the metric factor
at horizontal position `ij`. The object `mcoord` can then
be used with:

    m = mass_level(k, ij, masstot, mcoord)

With `metric_cov==1`, masstot should be in Pa ( kg/m²⋅(m/s²) ). With `metric_cov`
in m², `masstot` should be in kg⋅(m/s²). `m` has the same unit as `mass_tot`.
If `masstot` is covariant (integral over a cell) then `metric_cov` should
include the cell area.
See also [`mass_level`](@ref).
"""
function mass_coordinate end

"""
    m = mass_level(k, ij, masstot, mcoord::MassCoordinate{N})

Return mass `m` in level *`k/2`* and at horizontal position `ij`, as prescribed
by vertical coordinate mcoord and total mass `masstot`.

So-called full levels correspond to odd
values `k=1,2...2N-1` while interfaces between full levels
(so-called half-levels) correspond to even values `k=0,2...2N`

`masstot` may be:
* per unit surface, with unit Pa
* per unit non-dimensional surface (e.g. on the unit sphere), with unit kg.(m/s²))
Which convention is appropriate depends on the `metric_factor` provided when constructing `mcoord`.
See also [`mass_coordinate`](@ref).
"""
function mass_level end

@inline nlayer(::VerticalCoordinate{nz}) where nz = nz

#=============================== Sigma coordinate ===========================#

"""
    sigma = SigmaCoordinate(N, ptop) <: PressureCoordinate{N}
Pressure based sigma-coordinate for `N` levels with top pressure `ptop`.
Pressure levels are linear in vertical coordinate `k` :
    k/N = (ps-p)/(ps-ptop)
where `k` ranges from `0` (ground) to `N` (model top).
"""
struct SigmaCoordinate{N,F} <: PressureCoordinate{N}
    ptop::F # pressure at model top
end

SigmaCoordinate(N::Int, ptop::F) where {F} = SigmaCoordinate{N,F}(ptop)

pressure_level(k, ps, sigma::SigmaCoordinate{N}) where {N} =
    (k * sigma.ptop + (2N - k) * ps) / 2N

struct SigmaMassCoordinate{N,F} <: MassCoordinate{N}
end

"""
    mcoord = mass_coordinate(mcoord::MassCoordinate)
Return `mcoord` itself, unchanged. Interim function for backwards compatibility.
"""
mass_coordinate(mc::MassCoordinate) = mc

# metric_cov is unused
mass_coordinate(::SigmaCoordinate{N,F}, metric_cov) where {N,F} = SigmaMassCoordinate{N,F}()
# position (k,ij) is unused
mass_level(k, ij, masstot, ::SigmaMassCoordinate{N}, metric_cov) where N = masstot/N

#============================== Hybrid coordinate =============================#

struct HybridCoordinate{F, N, VF<:AbstractVector{F}} <: PressureCoordinate{N}
    ptop::F
    a::VF
    b::VF
    v::Val{N} # for Adapt.@adapt_structure
end
function HybridCoordinate(ptop::F, ai::VF, am::VF, bi::VF, bm::VF) where {F, VF<:AbstractVector{F} }
    N = length(eachindex(am,bm))
    @assert length(eachindex(ai,bi)) == N+1
    HybridCoordinate(ptop, interleave(ai,am), interleave(bi,bm), Val(N))
end

function interleave(i,m)
    v = [i[1]] # first interface
    for j in eachindex(m)
        push!(v, m[j]) # full level
        push!(v, i[j+1]) # upper interface
    end
    return v
end

pressure_level(k, ps, vc::HybridCoordinate) = vc.a[k+1] + ps*vc.b[k+1]

struct HybridMassCoordinate{F, N, VF<:AbstractVector{F}, A<:Union{F,AbstractVector{F}}} <: MassCoordinate{N}
    metric_cov::A
    ptop::F
    a::VF
    b::VF
    v::Val{N}
end

mass_coordinate(vc::HybridCoordinate{F,N}, metric_cov) where {F,N} =
    HybridMassCoordinate(metric_cov, vc.ptop, vc.a, vc.b, Val(N))

Base.@propagate_inbounds function mass_level(k, ij, masstot, vc::HybridMassCoordinate)
    # k==1 for first full level, k==3 for second full level, etc.
    (; ptop, a, b) = vc
    metric_cov = get(vc.metric_cov, ij)
    ps_cov = masstot + metric_cov*ptop
    p_down = metric_cov*a[k] + ps_cov*b[k]
    p_up = metric_cov*a[k+2] + ps_cov*b[k+2]
    return p_down-p_up
end

@inline get(metric::Number, _) = metric
Base.@propagate_inbounds get(metric, ij) = metric[ij]
