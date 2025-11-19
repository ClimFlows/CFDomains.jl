function MC_gradient(loss, q, args...)
    backend = DI.AutoMooncake()
    prep = DI.prepare_gradient(loss, backend, q, map(Const, args)...);
    return DI.gradient(loss, prep, backend, q, map(Const, args)...);
end
function FD_gradient(loss, q, args...)
    backendFD, t = DI.AutoForwardDiff(), zero(eltype(q))    
    return DI.derivative(loss, backendFD, t, Const(q), map(Const, args)...);
end

# 1 input

function norm_op(f, tmp, op, app!)
    app!(tmp, op, f)
    return LinAlg.norm(tmp)
end

function dnorm_op(t, f, df, tmp, op, app!) # directional derivative
    ff = @. f+t*df # t can be a DualNumber
    tmp = similar(ff, size(tmp))
    return norm_op(ff, tmp, op, app!)
end

function test_op(q, tmp, op)
    grad = MC_gradient(norm_op, q, tmp, op, Ops.apply!); 
    grad2 = FD_gradient(dnorm_op, q, grad, tmp, op, Ops.apply!);
    @info "check $(typeof(op))" grad2 LinAlg.dot(grad, grad)
    @test grad2 ≈ LinAlg.dot(grad,grad)

    run() = norm_op(q, tmp,op, Ops.apply!)
    display(@benchmark $run())
end

# 2 inputs

function norm_op(f, g, tmp, op, app!)
    app!(tmp, op, f, g)
    return LinAlg.norm(tmp)
end

function dnorm_op(t, f, df, g, tmp, op, app!) # directional derivative
    ff = @. f+t*df # t can be a DualNumber
    tmp = similar(ff, size(tmp))
    return norm_op(ff, g, tmp, op, app!)
end

function norm_op_switch(g, f, tmp, op, app!)
    app!(tmp, op, f, g)
    return LinAlg.norm(tmp)
end

function dnorm_op_switch(t, g, dg, f, tmp, op, app!) # directional derivative
    gg = @. g+t*dg # t can be a DualNumber
    tmp = similar(gg, size(tmp))
    return norm_op_switch(gg, f, tmp, op, app!)
end

function test_op(a, b, tmp, op)
    grad = MC_gradient(norm_op, a, b, tmp, op, Ops.apply!);
    grad2 = FD_gradient(dnorm_op, a, grad, b, tmp, op, Ops.apply!);
    @info "check $(typeof(op))" grad2 LinAlg.dot(grad,grad)
    @test grad2 ≈ LinAlg.dot(grad,grad)

    run() = norm_op(a, b, tmp, op, Ops.apply!)
    display(@benchmark $run())

    grad = MC_gradient(norm_op_switch, b, a, tmp, op, Ops.apply!);
    grad2 = FD_gradient(dnorm_op_switch, b, grad, a, tmp, op, Ops.apply!);
    @info "check $(typeof(op))" grad2 LinAlg.dot(grad,grad)
    @test grad2 ≈ LinAlg.dot(grad,grad)
end

# test div with AsDensity output

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

    as_two_form, div_form = Ops.AsDensity(sphere), Ops.Divergence(sphere)
    grad = MC_gradient(norm_div, ucov, tmp, as_two_form, div_form);
    grad2 = FD_gradient(dnorm_div, ucov, grad, tmp, as_two_form, div_form);
    @info "test_norm_div" grad2 LinAlg.dot(grad,grad)
    @test grad2 ≈ LinAlg.dot(grad,grad)

    run() = norm_div(ucov, tmp, Ops.AsDensity(sphere), Ops.Divergence(sphere))
    display(@benchmark $run())
end
