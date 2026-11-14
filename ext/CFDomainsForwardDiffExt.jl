module CFDomainsForwardDiffExt

import ForwardDiff as FD
import CFDomains.LazyExpressions.pdv

function pdv(fun::T, inputs::Vararg{Any,N}) where {T,N}
    values = inputs
    dual_inputs = ntuple(i -> FD.Dual{T}(
        values[i], ntuple(j -> i == j ? one(values[i]) : zero(values[i]), N),
    ), N)
    result = fun(dual_inputs...)
    value = result isa FD.Dual ? FD.value(result) : result
    partials = result isa FD.Dual ? result.partials.values : ntuple(_ -> zero(result), N)
    return value, partials
end

end
