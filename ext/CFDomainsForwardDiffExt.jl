module CFDomainsForwardDiffExt

import ForwardDiff as FD
import CFDomains.LazyExpressions.pdv

function pdv(fun1::T, x) where {T}
    xx = FD.Dual{T}(x, one(x))
    ff = fun1(xx)
    value = ff isa FD.Dual ? FD.value(ff) : ff
    partials = ff isa FD.Dual ? ff.partials.values[1] : zero(x)
    return value, partials
end

function pdv(fun2::T, x, y) where {T}
    xx = FD.Dual{T}(x, one(x), zero(x))
    yy = FD.Dual{T}(y, zero(y), one(y))
    ff = fun2(xx, yy)
    value = ff isa FD.Dual ? FD.value(ff) : ff
    partials = ff isa FD.Dual ? ff.partials.values : (zero(x), zero(y))
    return value, partials
end

function pdv(fun3::T, x, y, z) where {T}
    xx = FD.Dual{T}(x, one(x), zero(x), zero(x))
    yy = FD.Dual{T}(y, zero(y), one(y), zero(y))
    zz = FD.Dual{T}(z, zero(z), zero(z), one(z))
    ff = fun3(xx, yy, zz)
    value = ff isa FD.Dual ? FD.value(ff) : ff
    partials = ff isa FD.Dual ? ff.partials.values : (zero(x), zero(y), zero(z))
    return value, partials
end

end
