function norm_op(f, tmp, op, app!)
    app!(tmp, op, f)
    return LinAlg.norm(tmp)
#    return(sum(tmp))
end

function dnorm_op(t, f, df, tmp, op, app!) # directional derivative
    ff = @. f+t*df # t can be a DualNumber
    tmp = similar(ff, size(tmp))
    return norm_op(ff, tmp, op, app!)
end

function test_op(q, tmp, op)
    backend = DI.AutoMooncake()
    prep = DI.prepare_gradient(norm_op, backend, q, Const(tmp), Const(op), Const(Ops.apply!));
    grad = DI.gradient(norm_op, prep, backend, q, Const(tmp), Const(op), Const(Ops.apply!));

    backendFD, t = DI.AutoForwardDiff(), zero(eltype(q))    
    gradFD = DI.derivative(dnorm_op, backendFD, t, Const(q), Const(grad), Const(tmp), Const(op), Const(Ops.apply!));
    @info "check $(typeof(op))" gradFD LinAlg.dot(grad,grad)
    @test gradFD ≈ LinAlg.dot(grad,grad)

    run() = norm_op(q, tmp,op, Ops.apply!)
    display(@benchmark $run())
end

function norm_div(u, tmp, as_two_form, div_form!)
    tmp2 = as_two_form(tmp) # wraps `tmp` as a writable two-form
    div_form!(tmp2, u)
    return LinAlg.norm(tmp)
end

function dnorm_div(t, u, du, tmp_, as_two_form, div_form) # directional derivative
    uu = @. u+t*du # t can be a DualNumber
    tmp = similar(uu, size(tmp_))
    return norm_div(uu, tmp, as_two_form, div_form)
end

function test_norm_div(ucov, tmp, sphere)
    norm_div(ucov, tmp, Ops.AsDensity(sphere), Ops.Divergence(sphere))

    backend = DI.AutoMooncake()
    as_two_form, div_form = Ops.AsDensity(sphere), Ops.Divergence(sphere)
    prep = DI.prepare_gradient(norm_div, backend, ucov, Const(tmp), Const(as_two_form), Const(div_form));
    grad = DI.gradient(norm_div, prep, backend, ucov, Const(tmp), Const(as_two_form), Const(div_form));

    backendFD, t = DI.AutoForwardDiff(), zero(eltype(ucov))
    grad2 = DI.derivative(dnorm_div, backendFD, t, Const(ucov), Const(grad), Const(tmp), Const(as_two_form), Const(div_form));
    @info "test_norm_div" grad2 LinAlg.dot(grad,grad)
    @test grad2 ≈ LinAlg.dot(grad,grad)

    run() = norm_div(ucov, tmp, Ops.AsDensity(sphere), Ops.Divergence(sphere))
    display(@benchmark $run())
end
