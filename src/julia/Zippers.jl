module Zippers

using Base.Broadcast: Broadcast, Broadcasted, combine_eltypes

struct Zipper end
const zipper = Zipper()

@inline function Broadcast.materialize!(::Zipper, bc::Broadcasted)
    T = combine_eltypes(bc.f, bc.args)
    return materialize_zipper(Tuple(T.types), bc)
end

@inline function materialize_zipper(types, bc)
    x = map(E -> similar(bc, E), types)
    @inbounds for i in eachindex(bc)
        foreach(x, bc[i]) do xx, yy
            xx[i] = yy
        end
    end
    return x
end

end
