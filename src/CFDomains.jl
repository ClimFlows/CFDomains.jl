module CFDomains
using MutatingOrNot: void, Void
using ManagedLoops: @loops, @unroll, @with
using Random: MersenneTwister

macro fast(code)
    debug = haskey(ENV, "GF_DEBUG") && (ENV["GF_DEBUG"] != "")
    return debug ? esc(code) : esc(quote
        @inbounds $code
    end)
end

#====================  Abstract Domain types ====================#

"""
Parent type of [`SpectralDomain`](@ref) and [`FDDomain`](@ref)
"""
abstract type AbstractDomain end
"""
    SpectralDomain <: AbstractDomain
parent type of [`SpectralSphere`](@ref)
"""
abstract type SpectralDomain <: AbstractDomain end
"""
    FDDomain <: AbstractDomain`
"""
abstract type FDDomain <: AbstractDomain end

#=============== Types for filters ================#

abstract type AbstractFilter end

struct HyperDiffusion{fieldtype,D,F,X} <: AbstractFilter
    domain::D
    niter::Int
    nu::F
    extra::X # pre-computed stuff, if any
end


#=============== Allocations ================#

"""
    field = allocate_field(kind::Symbol, domain::AbstractDomain, precision::Type)

Allocate a field of the given `kind` and `precision` over the given domain.
Typical values for `precision` are `Float32`, `Float64` or `ForwardDiff.Dual`.
Depending on the domain, valid values for `kind` may include `:scalar`, `:vector`,
`:scalar_spec`, `scalar_spat` (spectral/spatial representation of a scalar field),
`:vector_spec`, `vector_spat` (spectral/spatial representation of a vector field).

Internally, `allocate_field(kind::Symbol, domain, F)` returns
`allocate_field(Val(kind), domain, F)`. To specialize `allocate_field`
for `MyDomain <: AbstractDomain`, one must provide methods for:

    allocate_field(::Val{kind}, domain::MyDomain, F)

where symbol `kind::Symbol` is one of the valid field kinds for that domain.
"""
function allocate_field end

"""
    fields = allocate_fields(kinds::Tuple, domain::AbstractDomain, F::Type)
    fields = allocate_fields(kinds::NamedTuple, domain::AbstractDomain, F::Type)

Allocate a (named) tuple of fields according to the provided `kinds`. For instance:

    fields = allocate_fields((:vector, :scalar), domain, F)
    fields = allocate_fields((a=:vector, b=:scalar), domain, F)

are equivalent to, respectively:

    fields = (allocate_field(:vector, domain, F), allocate_field(:scalar, domain, F))
    fields = (a=allocate_field(:vector, domain, F), b=allocate_field(:scalar, domain, F))
"""
function allocate_fields end

@inline allocate_fields(syms::NamedTuple, domain::AbstractDomain, F::Type) =
    map(sym -> allocate_field(Val(sym), domain, F), syms)
@inline allocate_fields(syms::Tuple, domain::AbstractDomain, F::Type) =
    Tuple(allocate_field(Val(sym), domain, F) for sym in syms)
@inline allocate_fields(syms::Tuple, domain::AbstractDomain, F::Type, mgr) =
    Tuple(allocate_field(Val(sym), domain, F, mgr) for sym in syms)
@inline allocate_field(sym::Symbol, domain::AbstractDomain, F::Type) =
    allocate_field(Val(sym), domain, F)
@inline allocate_field(sym::Symbol, nq::Int, domain::AbstractDomain, F::Type) =
    allocate_field(Val(sym), nq, domain, F)
@inline allocate_field(sym::Symbol, domain::AbstractDomain, F::Type, mgr) =
    allocate_field(Val(sym), domain, F, mgr)
@inline allocate_field(sym::Symbol, nq::Int, domain::AbstractDomain, F::Type, mgr) =
    allocate_field(Val(sym), nq, domain, F, mgr)

# belongs to ManagedLoops
# array(T, ::Union{Nothing, ManagedLoops.HostManager}, size...) = Array{T}(undef, size...)
# array(T, mgr::Loops.DeviceBackend, size...) = Loops.to_device(Array{T}(undef, size...), mgr)
# array(T, mgr::Loops.WrapperBackend, size...) = array(T, mgr.mgr, size...)

#===================== Shell (multi-layer domain)=====================#

#=
struct DimX end # first horizontal dimension
struct DimY end # second horizontal dimension
struct DimXY end # unique dimension indexing horizontal points
abstract type DimZ end # vertical dimension
struct BottomUp <: DimZ end # vertical dimension, model levels increase from bottom to top
struct TopDown <: DimZ end # vertical dimension, model levels increase from top to bottom

struct Layout{Dims<:Tuple}
    dims::Dims
end
=#

"""
    struct HVLayout{rank} end
    layout = HVLayout(rank)
Singleton type describing a multi-layer data layout where horizontal layers are contiguous in memory. `rank`
is the number of horizontal indices (1 or 2).
"""
struct HVLayout{rank} end
HVLayout(rank = 1) = HVLayout{rank}()

"""
    struct VHLayout{rank} end
    layout = VHLayout(rank)

Singleton type describing a multi-layer data layout where vertical columns are contiguous. `rank`
is the number of horizontal indices (1 or 2).
"""
struct VHLayout{rank} end
VHLayout(rank = 1) = VHLayout{rank}()

"""
    multi_layer_domain = Shell(nz::Int, layer::AbstractDomain, layout)

Return a multi-layer domain made of `nz` layers with data layout specified by `layout`.
Unless you know what you are doing, it is recommended to use rather:

    multi_layer_domain = shell(nz::Int, layer::AbstractDomain)

which gets the data layout from `data_layout(layer)`. Otherwise, `multi_layer_domain` may be non-optimal or non-usable.
"""
struct Shell{nz,Domain,Layout}
    v::Val{nz}  # for Adapt.@adapt_structure
    layer::Domain
    layout::Layout
end
Shell(nz::Int, layer, layout) = Shell(Val(nz), layer, layout)

"""
    multi_layer_domain = shell(nz::Int, layer::AbstractDomain)

Return a multi-layer domain made of `nz` layers with data layout specified by `data_layout(layer)`.
"""
shell(nz, layer) = Shell(nz, layer, data_layout(layer))

"""
    layout = data_layout(shell::Shell)

Return `layout` describing the data layout of multi-layer domain `shell`.

    layout = data_layout(domain::Domain)

Return `layout` describing the preferred data layout for a shell made of layers
of type `Domain`.

Typical values for `layout` are the singletons `HVLayout()` (layers are contiguous in memory)
and `VHLayout` (columns are contiguous in memory).
"""
data_layout(shell::Shell) = shell.layout

# shell(nz, layer::SHTnsSphere) = Shell(nz, layer, HVLayout())

@inline nlayer(::Shell{nz}) where {nz} = nz
@inline layers(shell::Shell) = shell
@inline interfaces(shell::Shell{nz,M}) where {nz,M} = Shell(nz, shell.layer, shell.layout)

allocate_field(val::Val, shell::Shell{nz}, F) where {nz} =
    allocate_shell(val, shell.layer, shell.layout, nz, F)
allocate_field(val::Val, nq::Int, shell::Shell{nz}, F) where {nz} =
    allocate_shell(val, shell.layer, nz, nq, F)
allocate_field(val::Val, shell::Shell{nz}, F, mgr) where {nz} =
    allocate_shell(val, shell.layer, nz, F, mgr)
allocate_field(val::Val, nq::Int, shell::Shell{nz}, F, mgr) where {nz} =
    allocate_shell(val, shell.layer, nz, nq, F, mgr)

# Dubious functions
# @inline Base.eltype(shell::Shell) = eltype(shell.layer)
# @inline interior(data, domain::Shell) = data
# @inline ijk( ::Type{Shell{nz, M}}, ij, k) where {nz, M} = (ij-1)*nz + k
# @inline kplus( ::Type{Shell{nz, M}})      where {nz, M} = 1
# @inline primal(domain::Shell{nz}) where nz = Shell(primal(domain.layer), nz)
# @inline interior(x::AbstractDomain) = interior(typeof(x))
# @inline interior(domain::Type) = domain

#======================== Box ========================#

meshgrid(ai, bj) = [a for a in ai, b in bj], [b for a in ai, b in bj]

"""
    periodize!(data, box::AbstractBox, mgr)
Enforce horizontally-periodic boundary conditions on array `data` representing
grid point values in `box`. `data` may also be a collection, in which case
`periodize!` is applied to each element of the collection. Call `periodize!`
on data obtained by computations involving horizontal averaging/differencing.
"""
@inline periodize!(datas::Tuple, box::AbstractDomain, args...) =
    periodize_tuple!(datas, box, args...)
function periodize_tuple!(datas::Tuple, box, args...)
    for data in datas
        periodize!(data, box, args...)
    end
    return datas
end

#============= Spherical harmonics on the unit sphere ==========#

"Parent type for spherical domains using spherical harmonics."
abstract type SpectralSphere <: SpectralDomain end

#===================== Spherical Voronoi mesh =================#

abstract type UnstructuredDomain <: AbstractDomain end

struct SubMesh{sym,Dom<:UnstructuredDomain} <: UnstructuredDomain
    domain::Dom
end
@inline SubMesh(sym::Symbol, dom::D) where {D} = SubMesh{sym,D}(dom)

include("julia/voronoi_stencils.jl")

include("julia/VoronoiSphere.jl")

shell(nz, layer::VoronoiSphere) = Shell(nz, layer, VHLayout())

#=================== Vertical coordinates ====================#

include("julia/vertical_coordinate.jl")

#================= Vertical interpolation ====================#

include("julia/vertical_interpolation.jl")

#=================== Numerical filters ========================#

include("julia/filters.jl")

using PackageExtensionCompat
function __init__()
    @require_extensions
end

end # module
