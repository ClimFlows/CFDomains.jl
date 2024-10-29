macro fields(expr)
    @assert expr.head == :(::)
    typ = expr.args[2]
    lines = [:($field::$typ) for field in expr.args[1].args]
    return esc(Expr(:block, lines...))
end

struct VoronoiSphere{
    F,
    VI<:AbstractVector{Int32},  # Vectors of integers
    MI<:AbstractMatrix{Int32},  # Matrices of integers
    VR<:AbstractVector{F},      # Vectors of reals
    MR<:AbstractMatrix{F},      # Matrices of reals
    AR<:AbstractArray{F,3},     # 3D array of reals
    VP<:AbstractVector{NTuple{3,F}}, # Vectors of 3D points
    MP<:AbstractMatrix{NTuple{3,F}}, # Matrices of 3D points
} <: UnstructuredDomain
    @fields (primal_num, dual_num, edge_num)::Int32
    @fields (primal_deg, dual_deg, trisk_deg)::VI
    @fields (primal_edge, primal_vertex, dual_edge, dual_vertex)::MI
    @fields (edge_left_right, edge_down_up, trisk, edge_kite)::MI
    @fields (primal_neighbour, primal_ne, dual_ne)::MI
    @fields (Ai, lon_i, lat_i, Av, lon_v, lat_v)::VR
    @fields (le, de, le_de, lon_e, lat_e, angle_e)::VR
    @fields (primal_bounds_lon, primal_bounds_lat, dual_bounds_lon, dual_bounds_lat)::MR
    @fields (Riv2, wee, edge_perp)::MR
    primal_perot_cov::AR
    primal_grad3d::MP
    # computed
    @fields (xyz_i, elon_i, elat_i)::VP
    @fields (xyz_e, elon_e, elat_e, normal_e, tangent_e)::VP
    @fields (xyz_v, elon_v, elat_v)::VP
    cen2edge::MP
end
const VSph = VoronoiSphere

Base.show(io::IO, ::Type{<:VoronoiSphere{F}}) where {F} = print(io, "VoronoiSphere{$F}")
Base.show(io::IO, sphere::VoronoiSphere) =
    print(io, "VoronoiSphere($(length(sphere.Ai)) cells, $(length(sphere.Av)) dual cells)")

# converts Floats to Float, leaves other types alone
@inline convert_float(data, T) = data
@inline convert_float(data::Integer, T) = data
@inline convert_float(data::AbstractFloat, T) = T(data)

struct StructDict
    dict::Dict{Symbol, Any}
    StructDict(itr) = new(Dict(itr))
end
Base.getindex(sd::StructDict, sym::Symbol) = getindex(sd.dict, sym)
Base.setindex!(sd::StructDict, val, sym::Symbol) = setindex!(sd.dict, val, sym)
Base.getproperty(sd::StructDict, sym::Symbol) = (sym==:dict) ? getfield(sd, sym) : sd.dict[sym]
Base.setproperty!(sd::StructDict, sym::Symbol, val) = setindex!(sd.dict, val, sym)

function VoronoiSphere(read_data::Function; prec = Float32)
    # These are read from file
    base_names = (
        (:primal_num, :dual_num, :edge_num)...,
        (:primal_deg, :dual_deg, :trisk_deg)...,
        (:primal_edge, :primal_vertex, :dual_edge, :dual_vertex)...,
        (:primal_neighbour, :primal_grad3d, :edge_kite, :edge_perp)...,
        (:edge_left_right, :edge_down_up, :trisk)...,
        (:primal_ne, :dual_ne)...,
        (:Ai, :lon_i, :lat_i, :Av, :lon_v, :lat_v)...,
        (:le, :de, :le_de, :lon_e, :lat_e, :angle_e)...,
        (:primal_bounds_lon, :primal_bounds_lat, :dual_bounds_lon, :dual_bounds_lat)...,
        (:Riv2, :wee, :primal_perot_cov)...,
    )
    data = StructDict(name => convert_float.(read_data(name), prec) for name in base_names)

    # Convert sizes to Int32
    nums = (:primal_num, :dual_num, :edge_num)
    for name in nums
        data[name] = Int32(data[name])
    end
    nums = Tuple(data[name] for name in nums)

    # Converts 3D vectors to tuples
    vec2tup(x) = [ map(dim->x[i,j,dim], (1,2,3)) for i in axes(x, 1), j in axes(x, 2)]
    data.primal_grad3d = vec2tup(data.primal_grad3d)

    # Extra stuff which can be computed without worrying about halos
    data.xyz_i, data.elon_i, data.elat_i = local_bases(data.lon_i, data.lat_i)
    data.xyz_e, data.elon_e, data.elat_e = local_bases(data.lon_e, data.lat_e)
    data.xyz_v, data.elon_v, data.elat_v = local_bases(data.lon_v, data.lat_v)
    data.normal_e, data.tangent_e =
        normal_tangents(data.elon_e, data.elat_e, data.angle_e)
    data.cen2edge = center_to_edge(data.xyz_i, data.xyz_e, data.primal_deg, data.primal_edge)

    # Store everything into a VoronoiSphere object
    names = fieldnames(VoronoiSphere)
    return VoronoiSphere((crop(nums, data[name], name) for name in names)...)
end

local_bases(lons, lats) = @. zipper = local_basis(lons, lats)
function local_basis(lon, lat)
    sinlon, coslon = sincos(lon)
    sinlat, coslat = sincos(lat)
    return (
        (coslon * coslat, sinlon * coslat, sinlat),
        (-sinlon, coslon, zero(sinlat)),
        (-coslon * sinlat, -sinlon * sinlat, coslat),
    )
end

normal_tangents(elons, elats, angles) = @. zipper = normal_tangent(elons, elats, angles)
function normal_tangent(elon, elat, angle)
    sina, cosa = sincos(angle)
    return (@. cosa * elon + sina * elat), (@. cosa * elat - sina * elon)
end

function center_to_edge(xyz_i, xyz_e, primal_degree, primal_edge)
    dxyz = similar(primal_edge, eltype(xyz_i))
    for cell in eachindex(xyz_i)
        center = xyz_i[cell]
        for edge in 1:primal_degree[cell]
            dxyz[edge, cell] = xyz_e[primal_edge[edge, cell]] .- center
        end
    end
    return dxyz
end

@inline Base.eltype(dom::VSph) = eltype(dom.Ai)

@inline primal(dom::VSph) = SubMesh{:scalar,typeof(dom)}(dom)

function crop((primal_num, dual_num, edge_num), data, name::Symbol)
    if name in (
        :primal_deg,
        :primal_edge,
        :primal_vertex,
        :primal_ne,
        :Ai,
        :lon_i,
        :lat_i,
        :primal_bounds_lon,
        :primal_bounds_lat,
    )
        num = primal_num
    elseif name in (
        :dual_deg,
        :dual_edge,
        :dual_vertex,
        :dual_ne,
        :Av,
        :lon_v,
        :lat_v,
        :dual_bounds_lon,
        :dual_bounds_lat,
        :Riv2,
    )
        num = dual_num
    elseif name in (
        :trisk_deg,
        :edge_left_right,
        :edge_down_up,
        :trisk,
        :le,
        :de,
        :le_de,
        :lon_e,
        :lat_e,
        :angle_e,
        :wee,
    )
        num = edge_num
    elseif name == :primal_perot_cov
        return data[:, 1:primal_num, :]
    else
        return data
    end
    if isa(data, AbstractVector)
        return data[1:num]
    else
        return data[:, 1:num]
    end
end

#====================== Allocate ======================#

array(::Nothing, dom::VSph, F, dims...) = similar(dom.Ai, F, dims...)

allocate_field(::Val{:scalar}, dom::VSph, F::Type{<:Real}, backend = nothing) =
    array(backend, dom, F, length(dom.Ai))
allocate_field(::Val{:dual}, dom::VSph, F::Type{<:Real}, backend = nothing) =
    array(backend, dom, F, length(dom.Av))
allocate_field(::Val{:vector}, dom::VSph, F::Type{<:Real}, backend = nothing) =
    array(backend, dom, F, length(dom.le))

allocate_shell(::Val{:scalar}, dom::VSph, nz, F::Type, backend = nothing) =
    array(backend, dom, F, nz, length(dom.Ai))
allocate_shell(::Val{:dual}, dom::VSph, nz, F::Type, backend = nothing) =
    array(backend, dom, F, nz, length(dom.Av))
allocate_shell(::Val{:vector}, dom::VSph, nz, F::Type, backend = nothing) =
    array(backend, dom, F, length(dom.le))
allocate_shell(::Val{:scalar}, dom::VSph, nz, nq, F::Type, backend = nothing) =
    array(backend, dom, F, length(dom.Ai), nq)
allocate_shell(::Val{:dual}, dom::VSph, nz, nq, F::Type, backend = nothing) =
    array(backend, dom, F, length(dom.Av), nq)
allocate_shell(::Val{:vector}, dom::VSph, nz, nq, F::Type, backend = nothing) =
    array(backend, dom, F, length(dom.le), nq)

@inline periodize!(data, ::Shell{Nz,<:VSph}, backend) where {Nz} = data
@inline periodize!(data, ::Shell{Nz,<:VSph}) where {Nz} = data
@inline periodize!(datas::Tuple, ::Shell{Nz,<:VSph}, args...) where {Nz} = datas

#====================== Effective resolution ======================#

normL2(f) = sqrt(sum(x -> x^2, f) / length(f))

"""
Estimates the largest eigenvalue `-lambda=dx^-2` of the scalar Laplace operator and returns `dx`
which is a (non-dimensional) length on the unit sphere characterizing the mesh resolution.
By design, the Courant number for the wave equation with unit wave speed solved with time step `dt` is `dt/dx`.
"""
function laplace_dx(mesh::VoronoiSphere, mgr = nothing)
    rng = MersenneTwister(1234) # for reproducibility
    h, u = similar(mesh.Ai), similar(mesh.le_de)
    copy!(h, randn(rng, eltype(mesh.Ai), length(mesh.Ai)))
    for i = 1:20
        hmax = normL2(h)
        @. h = inv(hmax) * h
        gradient!(mgr, u, h, mesh)
        divergence!(mgr, h, u, mesh)
    end
    return inv(sqrt(normL2(h)))::eltype(mesh.le_de)
end

function gradient!(mgr, gradcov, f, mesh::VoronoiSphere)
    left_right = mesh.edge_left_right
    @with mgr, let ijrange = eachindex(gradcov)
        @fast for ij in ijrange
            gradcov[ij] = f[left_right[2, ij]] - f[left_right[1, ij]]
        end
    end
end

function divergence!(mgr, divu, ucov, mesh::VoronoiSphere)
    degree, edges, signs = mesh.primal_deg, mesh.primal_edge, mesh.primal_ne
    areas, hodges = mesh.Ai, mesh.le_de
    @with mgr,
    let ijrange = eachindex(divu)
        @fast for ij in ijrange
            deg = degree[ij]
            @unroll deg in 5:7 divu[ij] =
                inv(areas[ij]) * sum(
                    (signs[e, ij] * hodges[edges[e, ij]]) * ucov[edges[e, ij]] for e = 1:deg
                )
        end
    end
end

#========================== Interpolation ===========================#

# First-order interpolation weighted by areas of dual cells
primal_from_dual!(fi, fv, mesh::VoronoiSphere) =
    primal_from_dual!(fi, fv, mesh.primal_deg, mesh.Av, mesh.primal_vertex)

function primal_from_dual!(fi::AbstractVector, fv, degrees, areas, vertices)
    @fast for ij in eachindex(degrees)
        deg = degrees[ij]
        Ai = sum(areas[vertices[vertex, ij]] for vertex = 1:deg)
        fi[ij] =
            inv(Ai) *
            sum(areas[vertices[vertex, ij]] * fv[vertices[vertex, ij]] for vertex = 1:deg)
    end
    return fi
end

function primal_from_dual!(fi::AbstractMatrix, fv, degrees, areas, vertices)
    nz = size(fi, 1)
    @fast for ij in eachindex(degrees)
        deg = degrees[ij]
        inv_Ai = inv(sum(areas[vertices[vertex, ij]] for vertex = 1:deg))
        for k = 1:nz
            fi[k, ij] = 0
        end
        for vertex = 1:deg
            vv = vertices[vertex, ij]
            ww = areas[vv] * inv_Ai
            for k = 1:nz
                fi[k, ij] = muladd(ww, fv[k, vv], fi[k, ij])
            end
        end
    end
    return fi
end

primal_from_dual(fv::AbstractVector, degrees, areas, vertices) =
    primal_from_dual!(similar(degrees, eltype(fv)), fv, degrees, areas, vertices)
primal_from_dual(fv::AbstractMatrix, degrees, areas, vertices) = primal_from_dual!(
    Matrix{eltype(fv)}(undef, size(fv, 1), size(degrees, 1)),
    fv,
    degrees,
    areas,
    vertices,
)
primal_from_dual(fv, mesh::VoronoiSphere) =
    primal_from_dual(fv, mesh.primal_deg, mesh.Av, mesh.primal_vertex)

# Perot reconstruction of vector field given covariant components

function primal3D_from_cov!(
    u::T,
    v::T,
    w::T,
    ucov::T,
    degrees,
    edges,
    weights,
) where {T<:AbstractVector}
    for ij in eachindex(u, v, w, ucov, degrees)
        u[ij], v[ij], w[ij] = primal3D_from_cov!(ij, ucov, degree[ij], edges, weights)
    end
    return u, v, w
end

# x,y,z = ( coslat*coslon, coslat*sinlon, sinlat )
# d(x,y,z)/dlon  = ( -coslat*sinlon, coslat*coslon, 0 )
# d(x,y,w)/dlat  = ( -sinlat*coslon, -sinlat*coslon, coslat )
# function 'fun' is applied to (ulon,ulat), see ShallowWaters.diag_ulonlat
@inline function primal_lonlat_from_cov!(
    fun::Fun,
    ulon::T,
    ulat::T,
    ucov::T,
    degrees,
    edges,
    weights,
    coslon,
    sinlon,
    coslat,
    sinlat,
) where {Fun,T<:AbstractVector}
    for ij in eachindex(ulon, ulat, degrees, coslon, sinlon, coslat, sinlat)
        u, v, w = primal3D_from_cov!(ij, ucov, degrees[ij], edges, weights)
        ulon[ij], ulat[ij] = fun(
            ij,
            v * coslon[ij] - u * sinlon[ij],
            w * coslat[ij] - sinlat[ij] * (u * coslon[ij] + v * sinlon[ij]),
        )
    end
    return ulon, ulat
end

@inline function primal_lonlat_from_cov!(
    fun::Fun,
    ulon::T,
    ulat::T,
    ucov::T,
    degrees,
    edges,
    weights,
    coslon,
    sinlon,
    coslat,
    sinlat,
) where {Fun,T<:AbstractMatrix}
    for ij in eachindex(degrees, coslon, sinlon, coslat, sinlat)
        for k in axes(ulon, 1)
            u, v, w = primal3D_from_cov!(ij, view(ucov, k, :), degrees[ij], edges, weights)
            ulon[k, ij], ulat[k, ij] = fun(
                ij,
                v * coslon[ij] - u * sinlon[ij],
                w * coslat[ij] - sinlat[ij] * (u * coslon[ij] + v * sinlon[ij]),
            )
        end
    end
    return ulon, ulat
end

@inline primal3D_from_cov!(ij, ucov, deg, edges, weights) = (
    sum(weights[iedge, ij, 1] * ucov[edges[iedge, ij]] for iedge = 1:deg),
    sum(weights[iedge, ij, 2] * ucov[edges[iedge, ij]] for iedge = 1:deg),
    sum(weights[iedge, ij, 3] * ucov[edges[iedge, ij]] for iedge = 1:deg),
)

#========================== Hyperviscous filter ===========================#

scratch_hyperdiff(sphere::VoronoiSphere, ::Val{:vector_curl}, u) =
    (zv = similar(u, size(sphere.Av)), gradzv = similar(u, size(sphere.lon_e)))

function hyperdiff!(
    ucov_out,
    ucov,
    dissip::HyperDiffusion{:vector_curl},
    sphere::VoronoiSphere,
    dt,
    scratch,
    mgr,
)
    nudt = dissip.nu * dt
    zv =
        voronoi_curl_2D!(mgr, scratch.zv, ucov, sphere.Av, sphere.dual_edge, sphere.dual_ne)
    gradzv = voronoi_gradv_2D!(mgr, scratch.gradzv, zv, sphere.edge_down_up, sphere.le_de)
    for i = 2:dissip.niter
        zv = voronoi_curl_2D!(
            mgr,
            scratch.zv,
            gradzv,
            sphere.Av,
            sphere.dual_edge,
            sphere.dual_ne,
        )
        gradzv =
            voronoi_gradv_2D!(mgr, scratch.gradzv, zv, sphere.edge_down_up, sphere.le_de)
    end
    @. ucov_out = ucov - nudt * gradzv
end

voronoi_curl_2D!(mgr, ::Void, ucov, areas, edges, signs) =
    voronoi_curl_2D!(mgr, similar(areas, eltype(ucov)), ucov, areas, edges, signs)

function voronoi_curl_2D!(mgr, zv, ucov, areas, edges, signs)
    voronoi_curl_2D_(mgr, zv, ucov, areas, edges, signs)
    return zv
end

@loops function voronoi_curl_2D_(_, zv, ucov, areas, edges, signs)
    let ijrange = axes(zv, 1)
        F = eltype(zv)
        @unroll for ij in ijrange
            aa = inv(areas[ij])
            ee = (edges[e, ij] for e = 1:3) # 3 edge indices
            ss = (F(signs[e, ij]) for e = 1:3) # 3 signs
            zv[ij] = aa * sum(ucov[ee[edge]] * ss[edge] for edge = 1:3)
        end
    end
end

voronoi_gradv_2D!(mgr, ::Void, zv, down_up, le_de) =
    voronoi_gradv_2D!(mgr, similar(le_de, eltype(zv)), zv, down_up, le_de)

function voronoi_gradv_2D!(mgr, grad, zv, down_up, le_de)
    voronoi_gradv_2D_(mgr, grad, zv, down_up, le_de)
    return grad
end

@loops function voronoi_gradv_2D_(_, grad, zv, down_up, le_de)
    let ijrange = axes(grad, 1) # velocity points
        for ij in ijrange
            ij_down, ij_up = down_up[1, ij], down_up[2, ij]
            de_le = inv(le_de[ij]) # covariant => contravariant
            grad[ij] = de_le * (zv[ij_up] - zv[ij_down])
        end
    end
end
