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

@prop function Base.getindex(lazy::LazyExpression, i)
    @boundscheck foreach(x->checkbounds(x,i), lazy.inputs)
    inputs = map(y-> (@inbounds y[i]), lazy.inputs)
    params = get(lazy.params, i)
    @inline lazy.fun(inputs..., params...)
end

@inline get(x::Tuple, i) = map(y->get(y, i), x)
@inline get(x::AbstractVector, i) = @inbounds x[i]
@inline get(x::Number, _) = x

end
