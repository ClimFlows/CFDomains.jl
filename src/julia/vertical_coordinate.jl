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
the covariant metric factor `metric_cov`. This object `mcoord` can then
be used with:

    m = mass_level(k, masstot, vcoord)

With `metric_cov==1`, masstot should be in Pa ( kg/m²⋅(m/s²) ). With `metric_cov`
the covariant metric factor, `masstot` should be in kg⋅(m/s²).
See also [`mass_level`](@ref).
"""
function mass_coordinate end

"""
    m = mass_level(k, masstot, mcoord::MassCoordinate{N})

Return mass `m` in level *`k/2`* as prescribed
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

# we accept the "metric_cov" argument, although it is not used
mass_coordinate(vc::SigmaCoordinate{N,F}, metric_cov=nothing) where {N,F} = SigmaMassCoordinate{N,F}()

mass_level(k, masstot, ::SigmaMassCoordinate{N}, metric_cov=nothing) where N = masstot/N
