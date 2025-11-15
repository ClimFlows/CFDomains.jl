module CFDomainsSIMDExt

import ForwardDiff as FD
import SIMD
using SIMD: Vec

@inline FD.can_dual(::Type{Vec{N, F}}) where {N, F} = FD.can_dual(F)

@inline FD._mul_partial(partial::Vec, x::Vec) = partial * x
@inline FD._mul_partial(partial::SVec{N,F}, x::F) where {N,F} = partial * x
@inline FD._mul_partial(partial::F, x::SVec{N,F}) where {N,F} = partial * x

@inline Base.:*(x::Vec, partials::FD.Partials) = partials*x

@inline Base.:*(partials::FD.Partials, x::Vec) =
    FD.Partials(FD.scale_tuple(partials.values, x))

@inline function FD.dual_definition_retval(::Val{T}, val::S, deriv::S, partial::FD.Partials{M,S}) where {T,F,N,M, S<:Vec{N,F}}
    return FD.Dual{T}(val, deriv*partial)   
end
@inline function FD.dual_definition_retval(::Val{T}, val::S, deriv1::S, partial1::FD.Partials{M,S}, deriv2::F, partial2::FD.Partials{M,F}) where {T,F,N,M, S<:Vec{N,F}}
    return FD.Dual{T}(val, FD._mul_partials(partial1, partial2, deriv1, deriv2))   
end
@inline function FD.dual_definition_retval(::Val{T}, val::S, deriv1::F, partial1::FD.Partials{M,F}, deriv2::S, partial2::FD.Partials{M,S}) where {T,F,N,M, S<:Vec{N,F}}
    return FD.Dual{T}(val, FD._mul_partials(partial1, partial2, deriv1, deriv2))   
end
@inline function FD.dual_definition_retval(::Val{T}, val::S, deriv1::S, partial1::FD.Partials{M,S}, deriv2::S, partial2::FD.Partials{M,S}) where {T,F,N,M, S<:Vec{N,F}}
    return FD.Dual{T}(val, FD._mul_partials(partial1, partial2, deriv1, deriv2))   
end

end
