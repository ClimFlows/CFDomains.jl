function trisk_one_step(mgr, qU, sphere, q, U)
    @with mgr,
    let krange = axes(qU,1)
        @inbounds for ij in axes(qU, 2)
            deg = sphere.trisk_deg[ij]
            @unroll deg in 9:10 begin
                trisk = Stencils.TRiSK(sphere, ij, Val(deg))
                @vec for k in krange
                    qU[k, ij] = trisk(U, q, k)
                end
            end
        end
    end
end

function trisk_many_steps(mgr, qU, sphere, q, U)
    @with mgr,
    let krange = axes(qU,1)
        @inbounds for ij in axes(qU, 2)
            deg = sphere.trisk_deg[ij]
            qU[:, ij] .= 0
            for edge in 1:deg
                trisk = Stencils.TRiSK(sphere, ij, edge)
                @vec for k in krange
                    qU[k, ij] = trisk(qU, U, q, k)
                end
            end
        end
    end
end

qU, q, U = (randn(Float32, 64, length(sphere.le_de)) for _=1:3)
vlen(::VectorizedCPU{N}) where N = N
N = vlen(VectorizedCPU()) # native vector size for Float32
(f1, f2) = trisk_one_step, trisk_many_steps

for (fun, vlen, nt) in [(f2, N, 1), (f1, N, 1), (f1, 2N, 1), (f1, N, 2), (f1, N, 4)]
    mgr = VectorizedCPU(vlen)
    mgr = (nt == 1) ? mgr : MultiThread(mgr, nt)
    @info "$fun on $mgr"
    @btime $fun($mgr, $qU, $sphere, $q, $U)
end
