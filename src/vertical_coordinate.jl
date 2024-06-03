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
    m = mass_level(k, masstot, vcoord::PressureCoordinate{N})

Return mass `m`
in level *`k/2`* as prescribed by vertical coordinate vcoord and total mass `masstot`,
i.e. surface pressure minus top pressure.

So-called full levels correspond to odd
values `k=1,2...2N-1` while interfaces between full levels
(so-called half-levels) correspond to even values `k=0,2...2N`
"""
function mass_level end

"""
remap_fluxes_ps!(flux, newmg, mg, layout, vcoord::PressureCoordinate)

Computes the target (pseudo-)mass distribution `newmg`
prescribed by pressure-based coordinate `vcoord` and surface pressure `ps`,
as well as the vertical (pseudo-)mass flux `flux` needed to remap
from current mass distribution `mg` to target `newmg`.
`layout` specifies the data layout, see `VHLayout` and `HVLayout`.
"""
function remap_fluxes_ps! end



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
mass_level(k, masstot, ::SigmaCoordinate{N}) where N = masstot/N
