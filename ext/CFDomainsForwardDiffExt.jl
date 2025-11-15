module CFDomainsForwardDiffExt

import ForwardDiff as FD
import CFDomains.VoronoiOperators as Ops

function Ops.pdv(fun1::T, x) where {T}
    xx = FD.Dual{T}(x, one(x))
    ff = fun1(xx)
    return ff.partials.values[1]
end

function Ops.pdv(fun2::T, x, y) where {T}
    xx = FD.Dual{T}(x, one(x), zero(x))
    yy = FD.Dual{T}(y, zero(y), one(y))
    ff = fun2(xx, yy)
    return ff.partials.values
end

function Ops.pdv(fun3::T, x, y, z) where {T}
    xx = FD.Dual{T}(x, one(x), zero(x), zero(x))
    yy = FD.Dual{T}(y, zero(y), one(y), zero(y))
    zz = FD.Dual{T}(z, zero(z), zero(z), one(z))
    ff = fun3(xx, yy, zz)
    return ff.partials.values
end

end
