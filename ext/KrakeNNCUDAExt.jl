module KrakeNNCUDAExt

# CUDA launch options for the DSL LBM kernels: a cap on the registers per
# thread (`maxregs`) that KernelAbstractions cannot pass by itself, and the
# default workgroup of the 3D GNF step.
#
# Loaded by `using CUDA`. CUDA is a strong dependency of KrakeNN that the
# package never loads at module load; Julia >= 1.11 accepts a strong dependency
# as extension trigger (Base.loading merges [deps] and [weakdeps] when it looks
# triggers up; Pkg checks "listed in [weakdeps] or [deps]"). Do NOT add CUDA to
# [weakdeps] (see ext/KrakeNNCUDSSExt.jl).
#
# `lbm_cuda_launch_config` follows CUDACore.CUDAKernels.launch_tuple of CUDA.jl
# 6.4 step by step (launch_config, mkcontext, kernel_call, kernel_compile,
# automatic workgroup, rebind) and only adds the compiler option `maxregs`. With
# maxregs = 0 it returns the kernel the plain KernelAbstractions launch uses
# (kernel_compile caches on the compiler options). CUDA.jl versions without the
# kernel_call/kernel_compile API (CUDA.jl 5) fall back to the plain launch,
# without the cap, with one warning.

using KrakeNN
using CUDA

const KA = KrakeNN.KernelAbstractions
const CC = isdefined(CUDA, :CUDACore) ? CUDA.CUDACore : CUDA
const _HAS_CALL_API = all(s -> isdefined(CC, s),
                          (:kernel_call, :kernel_compile, :kernel_launch, :rebind,
                           :launch_configuration, :CUDAKernels)) &&
                      isdefined(CC.CUDAKernels, :threads_to_workgroupsize)

const CUDABackend = CUDA.CUDABackend

function KrakeNN.lbm_cuda_launch_config(rk::KA.Kernel{CUDABackend}, args::Tuple, ndrange,
                                        workgroupsize, maxregs::Integer)
    _HAS_CALL_API || error("lbm_cuda_launch_config needs the CUDA.jl 6 kernel_call API " *
                           "(CUDA.jl $(pkgversion(CUDA)))")
    bk = KA.backend(rk)
    ndr, wg, iterspace, _ = KA.launch_config(rk, ndrange, workgroupsize)
    ctx = KA.mkcontext(rk, ndr, iterspace)
    maxthreads = KA.workgroupsize(rk) <: KA.StaticSize ?
                 prod(KA.get(KA.workgroupsize(rk))) : nothing
    call = CC.kernel_call(rk.f, (ctx, args...))
    kw = maxregs > 0 ? (; maxregs=Int(maxregs)) : (;)
    kernel = CC.kernel_compile(call; always_inline=bk.always_inline, fastmath=bk.fastmath,
                               maxthreads, kw...)
    if KA.workgroupsize(rk) <: KA.DynamicSize && workgroupsize === nothing
        cfg = CC.launch_configuration(kernel.fun; max_threads=prod(ndr))
        threads = if bk.prefer_blocks
            t = min(prod(ndr), cfg.threads)
            cld(prod(ndr), max(cld(prod(ndr), t), cfg.blocks))
        else
            cfg.threads
        end
        wg = CC.CUDAKernels.threads_to_workgroupsize(threads, ndr)
        iterspace, _ = KA.partition(rk, ndr, wg)
        ctx = KA.mkcontext(rk, ndr, iterspace)
        call = CC.rebind(call, ctx, 1)
    end
    return (; kernel, call, workgroupsize=(wg === nothing ? nothing : Tuple(wg)),
            ndrange=ndr, blocks=length(KA.blocks(iterspace)),
            threads=length(KA.workitems(iterspace)))
end

# Default workgroup of the 3D GNF step on CUDA: (32, 4, 2) in Float64 (best with
# maxregs 128 on the H100 sweep, job 28150951); Float32 keeps the automatic choice
# (with maxregs 96, auto beat 32x4x2).
KrakeNN._gnf3d_step_workgroup(::CUDABackend, ::Type{Float64}) = (32, 4, 2)

const _WARNED = Ref(false)

function KrakeNN._lbm_launch_capped!(::CUDABackend, rk::KA.Kernel{CUDABackend}, args::Tuple,
                                     ndrange, workgroupsize, maxregs::Integer)
    if !_HAS_CALL_API
        if !_WARNED[]
            _WARNED[] = true
            @warn "KrakeNN: CUDA.jl $(pkgversion(CUDA)) has no kernel_call API; " *
                  "register cap maxregs=$maxregs ignored (plain launch)"
        end
        rk(args...; ndrange, workgroupsize)
        return nothing
    end
    c = KrakeNN.lbm_cuda_launch_config(rk, args, ndrange, workgroupsize, maxregs)
    c.blocks == 0 && return nothing
    CC.kernel_launch(c.kernel, c.call; threads=c.threads, blocks=c.blocks)
    return nothing
end

end # module
