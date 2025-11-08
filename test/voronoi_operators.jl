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
    backend = DI.AutoMooncake()
    prep = DI.prepare_gradient(norm_op, backend, q, Const(tmp), Const(op), Const(Ops.apply!));
    grad = DI.gradient(norm_op, prep, backend, q, Const(tmp), Const(op), Const(Ops.apply!));

    backendFD, t = DI.AutoForwardDiff(), zero(eltype(q))    
    grad2 = DI.derivative(dnorm_op, backendFD, t, Const(q), Const(grad), Const(tmp), Const(op), Const(Ops.apply!));
    @info "check $(typeof(op))" grad2 LinAlg.dot(grad,grad)
    @test grad2 ≈ LinAlg.dot(grad,grad)

    run() = norm_op(q, tmp,op, Ops.apply!)
    display(@benchmark $run())
end
