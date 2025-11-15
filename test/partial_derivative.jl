function loop_FD1(mgr, fun::Fun, fa, a) where Fun
    @with mgr let irange = eachindex(fa)
        @vec for i in irange
            fa[i] = Ops.pdv(fun, a[i])
        end
    end
end

function loop_FD2(mgr, fun2::Fun, fa, fb, a, b) where Fun
    @with mgr let irange = eachindex(fa,fb)
        @vec for i in irange
            @inbounds fa[i], fb[i] = Ops.pdv(fun2, a[i], b[i])
        end
    end
end

function loop_FD3(mgr, fun3::Fun, fa, fb, fc, a, b, c) where Fun
    @with mgr let irange = eachindex(fa,fb,fc)
        @vec for i in irange
            @inbounds fa[i], fb[i], fc[i] = Ops.pdv(fun3, a[i], b[i], c[i])
        end
    end
end

fun2(x,y) = cos(x)*sin(y)
fun3(x,y,z) = cos(x)*sin(y)*exp(z)

Base.sincos(x::SIMD.Vec) = sin(x), cos(x)

@testset "partial_derivative" begin
    let (a, b, c) = (1.0, 2.0, 3.0)
        @test Ops.pdv(sin, a) ≈ cos(a)
        @test all( Ops.pdv(*, a, b) .≈ (b,a))
    end

    mgr = VectorizedCPU(8)
    F, N = Float32, 1024
    a, b, c = (randn(F, N*N) for _ in 1:3)
    fa, fb, fc = (similar(a) for _ in 1:3)

    loop_FD1(mgr, sin, fa, a)
    @test fa ≈ cos.(a)
    loop_FD2(mgr, *, fa, fb, a, b)
    @test fa ≈ b
    @test fb ≈ a
    loop_FD3(mgr, *, fa, fb, fc, a, b, c)
    @test fa ≈ b.*c
    @test fb ≈ a.*c
    @test fc ≈ a.*b
end
