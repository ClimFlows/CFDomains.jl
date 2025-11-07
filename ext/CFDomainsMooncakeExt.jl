module CFDomainsMooncakeExt

# codual = reverse codual = primal + rdata
# fcodual = forward codual = primal + fdata

import Mooncake
using Mooncake: CoDual, NoTangent, NoFData, NoRData, zero_fcodual, primal, tangent
using CFDomains.VoronoiOperators: apply!, apply_adj!, apply_internal!, VoronoiOperator

const CoVector{F} = CoDual{<:AbstractVector{F}, <:AbstractVector{F}}
const CoNumber{F} = CoDual{F,NoFData}
const CoOperator{A,B} = CoDual{<:VoronoiOperator{A,B}, NoFData}
CoFunction(f) = CoDual{typeof(f), NoFData}

Mooncake.tangent_type(::Type{<:VoronoiOperator}) = NoTangent

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(apply!), Vararg}
Mooncake.rrule!!(::CoFunction(apply!), fx::Vararg) = apply!_rrule!!(fx...)

function apply!_rrule!!(foutput::CoVector{F}, op::CoOperator{1,1}, finput::CoVector{F}) where F
    output, stencil, input = primal(foutput), primal(op), primal(finput)
    output_ = copy(output) # to undo mutation during backward pass    
    dout, din = tangent(foutput), tangent(finput)
    function apply!_pullback!!(::NoRData)
        copy!(output, output_) # undo mutation
        apply_adj!(dout, stencil, din)
        return NoRData(), NoRData(), NoRData(), NoRData() # rdata for (apply!, output, op, input)
    end
    apply_internal!(output, stencil, input)
    return zero_fcodual(nothing), apply!_pullback!!
end

end
