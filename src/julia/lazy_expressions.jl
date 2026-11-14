module LazyExpressions

using MacroTools
using Base:@propagate_inbounds as @prop

macro lazy(expr)
    esc(expand_lazy(expr))
end

function expand_lazy(expr)
    # get function name, body, inputs and params
    def = splitdef(expr)
    name = def[:name]
    inputs = def[:args]
    params = def[:kwargs]
    # anonymous function taking inputs and params as regular args
    args = [inputs...; params...]
    fun = combinedef(Dict(:body=>def[:body], :args=>args, :kwargs=>[], :whereparams=>[]))
    # construct lazy expression
    params = Expr(:tuple, params...)
    inputs = Expr(:tuple, inputs...)
    :( $name = $lazy_expr(($fun), ($inputs), ($params)) )
end

const AA{N,T} = AbstractArray{T,N} # an array of rank N
const Arrays{N} = Tuple{Vararg{AA{N}}} # a tuple of arrays of rank N

function lazy_expr(fun::Fun, inputs::Arrays{N}, params) where {Fun, N}
    T = promote_type((eltype(input) for input in inputs)...)
    LazyExpression{T, N, Fun, typeof(inputs), typeof(params)}(fun, inputs, params)
end

struct LazyExpression{T, N, Fun, Inputs<:Arrays{N}, Params<:Tuple} <: AbstractArray{T,N}
    fun::Fun
    inputs::Inputs
    params::Params
end
Base.size(lazy::LazyExpression) = size(lazy.inputs[1])

function adapt_structure(adapt_field, to, lazy::LazyExpression{T,N}) where {T,N}
    inputs = adapt_field(to, lazy.inputs)
    params = adapt_field(to, lazy.params)
    lazy_expr(lazy.fun, inputs, params)
end

@inline function cartesian_indices(lazy::LazyExpression{T,N}, indices::Tuple) where {T,N}
    M = length(indices)
    if M == N && all(index -> index isa Integer, indices)
        return indices
    elseif M == 1 && only(indices) isa CartesianIndex
        return Tuple(only(indices))
    elseif M == 1
        return Tuple(CartesianIndices(size(lazy))[only(indices)])
    elseif M > N && all(isone, indices[(N + 1):M])
        return ntuple(i -> indices[i], N)
    else
        throw(BoundsError(lazy, indices))
    end
end

@inline function cartesian_indices(
    lazy::LazyExpression, indices::Tuple{CartesianIndex{M},Vararg{Integer}},
) where {M}
    return cartesian_indices(lazy, (Tuple(first(indices))..., Base.tail(indices)...))
end

@prop function Base.getindex(lazy::LazyExpression{T,N}, indices::Vararg{Integer,M}) where {T,N,M}
    cartesian = cartesian_indices(lazy, indices)
    @boundscheck foreach(x -> checkbounds(x, cartesian...), lazy.inputs)
    inputs = map(y -> (@inbounds y[cartesian...]), lazy.inputs)
    params = get(lazy.params, cartesian...)
    @inline lazy.fun(inputs..., params...)
end

@inline function Base.getindex(
    lazy::LazyExpression, index::CartesianIndex, indices::Vararg{Integer,M},
) where {M}
    return getindex(lazy, Tuple(index)..., indices...)
end

@inline get(x::Tuple, indices...) = map(y -> get(y, indices...), x)
@inline get(x::AbstractArray{<:Any,M}, indices::Vararg{Integer,N}) where {M,N} =
    @inbounds getindex(x, parameter_indices(Val(N), Val(M), indices)...)
@inline get(x::Number, indices...) = x

@inline parameter_indices(::Val{1}, ::Val{M}, indices) where {M} = indices
@inline parameter_indices(::Val{N}, ::Val{M}, indices) where {N,M} =
    indices[(N-M+1):N]

#===================== automatic partial derivatives =================#

"""
    fa = pdv(fun, a)
    fa, fb = pdv(fun, a, b)
    fa, fb, fc = pdv(fun, a, b, c)

Return the partial derivatives of scalar function `fun` evaluated at input `a, ...`.
*This function is implemented only when the package ForwardDiff is loaded*
either directly from the main program or via some dependency.
"""
function pdv end

"""
    lazy_cotangent(expr, shadow)

Return a write-through cotangent for a `LazyExpression` during Enzyme reverse
mode. `shadow` is the Enzyme shadow expression; contributions written to the
cotangent are accumulated in its input arrays without allocating a full-sized
cotangent array. Direct scalar reads support homogeneous real floating-point
expressions with array-valued inputs and parameters. Mixed-precision or
complex-valued expressions, scalar active parameters, and differentiable arrays
captured by `fun` are not handled. Parameters that are themselves
`LazyExpression`s are also unsupported. Pass differentiable arrays as explicit
lazy inputs or ordinary array-valued parameters.

This method is available when both Enzyme and ForwardDiff are loaded.
"""
function lazy_cotangent end

end # module

