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

const Arrays{N} = Tuple{Vararg{AbstractArray{<:Any, N}}} # a tuple of arrays of rank N

function lazy_expr(fun::Fun, inputs::Arrays{N}, params) where {Fun, N}
    LazyExpression{Fun, typeof(inputs), typeof(params)}(fun, inputs, params) 
end

struct LazyExpression{Fun, Inputs, Params}
    fun::Fun
    inputs::Inputs
    params::Params
end

@prop function Base.getindex(lazy::LazyExpression{T}, i) where T
    @boundscheck foreach(x->checkbounds(x,i), lazy.inputs)
    inputs = map(y-> (@inbounds y[i]), lazy.inputs)
    params = get(lazy.params, i)
    @inline lazy.fun(inputs..., params...)
end

@inline get(x::Tuple, i) = map(y->get(y, i), x)
@inline get(x::AbstractVector, i) = @inbounds x[i]
@inline get(x::Number, _) = x

end
