module VerticalInterpolation

using MutatingOrNot: void, Void
using CFDomains: HVLayout

"""
    interpolated = interpolate!(mgr::LoopManager, domain::Shell, field, coord, refs, increasing)
Interpolate `field` defined on 3D `domain` to reference values `refs` of `coord`.
If `coord` increases with level number, `increasing==true` and vice-versa.
`refs` must be sorted according to `increasing`. `out` may be `void`. Example:

T_ref = interpolate!(mgr, void, domain, [850., 500.], temperature, pressure, false)

With these arguments, `interpolate!` is *not* meant to be specialized for a specific domain type.
Instead, specialize:
    interpolated = interpolate!(mgr, out, layout::Layout, refs, field, coord, increasing::Val)
"""
interpolate!(mgr, out, shell, field, coord, refs, increasing=true::Bool) =
    interpolate!(mgr, out, shell.layout, field, coord, refs, Val(increasing))

function interpolate!(mgr, out, layout, refs, field, coord, ::Val{true})
    @assert issorted(refs)
    out = interpolate_out(out, layout, field, refs)
    return interpolate_increasing!(mgr, out, layout, field, coord, refs)
end

function interpolate!(mgr, out, layout, field, coord, refs, ::Val{false})
    @assert issorted(refs ; lt = >)
    out = interpolate_out(out, layout, field, refs)
    return interpolate_decreasing!(mgr, out, layout, field, coord, refs)
end

#============== allocate output array ==========#

interpolate_out(out, _, _, _) = out
function interpolate_out(::Void, ::HVLayout{2}, field::AbstractArray{<:Any,3}, refs)
    (x,y,_) = axes(field)
    return similar(field, (x,y, eachindex(refs)))
end

#=============== decreasing ================#

function interpolate_decreasing!(mgr, out, ::HVLayout{2}, field, coord, refs)
    interpolate_decreasing_XYV!(mgr, out, field, coord, refs)
    return out
end

function interpolate_decreasing_XYV!(_, out, field, coord, refs)
    let (irange, jrange) = (axes(out,1), axes(out,2))
        for i in irange, j in jrange
            @views interpolate_decreasing_V!(out[i,j,:], field[i,j,:], coord[i,j,:], refs)
        end
    end
end

@inline function interpolate_decreasing_V!(out, field, coord, refs)
    krange = eachindex(field)
    k = first(krange)
    for l in eachindex(refs)
        ref = refs[l]
        # we want k such that coord[k-1] >= ref >= coord[k]
        bottom = coord[k]
        while k<last(krange) && bottom > ref
            k += 1
            bottom = coord[k]
        end
        if bottom > ref # k == last(krange)
            out[l] = field[k]
        elseif k>first(krange)
            out[l] = linear_interp(ref, coord[k-1], field[k-1], coord[k], field[k])
        else
            out[l] = field[first(krange)]
        end
    end
    return out
end

linear_interp(ref, x, fx, y, fy) = ((ref-x)*fy+(y-ref)*fx)/(y-x)

end # module