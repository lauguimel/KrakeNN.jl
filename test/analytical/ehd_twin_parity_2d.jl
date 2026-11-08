using Test
using KernelAbstractions
using KrakeNN

include(joinpath(@__DIR__, "..", "reference", "ehd_twin_reference.jl"))

const STAGE_ORDER = [
    "phi_after_solve",
    "Ex",
    "Ey",
    "macros_pre_charge.rho",
    "macros_pre_charge.ux",
    "macros_pre_charge.uy",
    "q_after_collide.f",
    "q_after_stream_bc.f",
    "q_after_stream_bc.q",
    "Fx",
    "Fy",
    "f_after_ns_collide.f",
    "f_after_ns_stream_bc.f",
    "macros_post.rho",
    "macros_post.ux",
    "macros_post.uy",
    "macros_post.umax",
]

mutable struct KrakenTraceState{T}
    Nx::Int
    Ny::Int
    p::NamedTuple
    phi_f_in::Array{T,3}
    phi_f_out::Array{T,3}
    q_f_in::Array{T,3}
    q_f_out::Array{T,3}
    f_in::Array{T,3}
    f_out::Array{T,3}
    phi::Matrix{T}
    qfield::Matrix{T}
    Ex::Matrix{T}
    Ey::Matrix{T}
    rho::Matrix{T}
    ux::Matrix{T}
    uy::Matrix{T}
    Fx::Matrix{T}
    Fy::Matrix{T}
    Fx_prev::Matrix{T}
    Fy_prev::Matrix{T}
    phi_prev::Matrix{T}
    is_solid::Matrix{Bool}
    diag::Vector{T}
end

function init_kraken_trace_from_twin(tw::EHDTwinState{T}) where {T}
    solid = copy(tw.wall)
    st = KrakenTraceState(tw.Nx, tw.Ny, tw.p,
                          copy(tw.phi_f), similar(tw.phi_f),
                          copy(tw.q_f), similar(tw.q_f),
                          copy(tw.ns_f), similar(tw.ns_f),
                          copy(tw.phi), copy(tw.q), zeros(T, tw.Nx, tw.Ny),
                          copy(tw.Ey), ones(T, tw.Nx, tw.Ny),
                          zeros(T, tw.Nx, tw.Ny), zeros(T, tw.Nx, tw.Ny),
                          zeros(T, tw.Nx, tw.Ny), zeros(T, tw.Nx, tw.Ny),
                          zeros(T, tw.Nx, tw.Ny), zeros(T, tw.Nx, tw.Ny),
                          similar(tw.phi), solid, zeros(T, 2))
    KrakeNN.compute_ehd_scalar_2d!(st.phi, st.phi_f_in)
    KrakeNN.compute_ehd_scalar_2d!(st.qfield, st.q_f_in)
    KrakeNN.compute_electric_field_2d!(st.Ex, st.Ey, st.phi_f_in, st.p.tau_U)
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    return st
end

function kraken_step!(st::KrakenTraceState; phi_tol=1e-4, phi_max_iter=10000,
                      force_projection=:xy, phi_check_every=8, record=false)
    checkpoints = Dict{String,Any}()
    phi_rel = Inf
    phi_iter = 0
    for iter in 1:phi_max_iter
        copyto!(st.phi_prev, st.phi)
        KrakeNN.collide_electric_potential_2d!(st.phi_f_in, st.qfield, st.p.eps, st.p.omega_U, st.p.nu_U)
        KrakeNN.stream_wall_x_wall_y_2d!(st.phi_f_out, st.phi_f_in, st.Nx, st.Ny)
        KrakeNN.compute_ehd_scalar_2d!(st.phi, st.phi_f_out)
        KrakeNN.apply_phi_nee_box_2d!(st.phi_f_out, st.phi, one(eltype(st.phi)), zero(eltype(st.phi)), st.Nx, st.Ny)
        KrakeNN.compute_ehd_scalar_2d!(st.phi, st.phi_f_out)
        st.phi_f_in, st.phi_f_out = st.phi_f_out, st.phi_f_in
        phi_iter = iter
        if iter % phi_check_every == 0 || iter == phi_max_iter
            KrakeNN.ehd_rel_change_2d!(st.diag, st.phi, st.phi_prev, st.Nx, st.Ny)
            KernelAbstractions.synchronize(KernelAbstractions.CPU())
            phi_rel = st.diag[1]
            phi_rel <= phi_tol && break
            iter == phi_max_iter && error("KrakeNN phi solve failed; rel=$(phi_rel)")
        end
    end
    KrakeNN.compute_electric_field_2d!(st.Ex, st.Ey, st.phi_f_in, st.p.tau_U)
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    record && (checkpoints["phi_after_solve"] = copy(st.phi);
               checkpoints["Ex"] = copy(st.Ex);
               checkpoints["Ey"] = copy(st.Ey))

    KrakeNN.compute_macroscopic_guo_field_2d!(st.rho, st.ux, st.uy, st.f_in, st.Fx_prev, st.Fy_prev, st.Nx, st.Ny)
    KrakeNN.enforce_free_side_macros_2d!(st.ux, st.uy, st.Nx, st.Ny)
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    record && (checkpoints["macros_pre_charge.rho"] = copy(st.rho);
               checkpoints["macros_pre_charge.ux"] = copy(st.ux);
               checkpoints["macros_pre_charge.uy"] = copy(st.uy))

    KrakeNN.collide_electric_charge_regularized_2d!(st.q_f_in, st.ux, st.uy, st.Ex, st.Ey, st.p.tau_q, st.p.K)
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    record && (checkpoints["q_after_collide.f"] = copy(st.q_f_in))
    KrakeNN.stream_wall_x_wall_y_2d!(st.q_f_out, st.q_f_in, st.Nx, st.Ny)
    KrakeNN.compute_ehd_scalar_2d!(st.qfield, st.q_f_out)
    KrakeNN.apply_charge_nee_box_2d!(st.q_f_out, st.qfield, st.ux, st.uy, st.Ex, st.Ey,
                                    st.p.q_inj, zero(eltype(st.qfield)), st.p.K, st.Nx, st.Ny)
    KrakeNN.compute_ehd_scalar_2d!(st.qfield, st.q_f_out)
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    st.q_f_in, st.q_f_out = st.q_f_out, st.q_f_in
    record && (checkpoints["q_after_stream_bc.f"] = copy(st.q_f_in);
               checkpoints["q_after_stream_bc.q"] = copy(st.qfield))

    KrakeNN.compute_coulomb_force_2d!(st.Fx, st.Fy, st.qfield, st.Ex, st.Ey, st.Nx, st.Ny)
    KrakeNN._project_coulomb_force_rows!(st.Fx, st.Fy, st.is_solid, force_projection)
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    record && (checkpoints["Fx"] = copy(st.Fx); checkpoints["Fy"] = copy(st.Fy))

    KrakeNN.ehd_collide_mrt_2d!(st.f_in, st.Fx, st.Fy, st.is_solid, st.p.nu)
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    record && (checkpoints["f_after_ns_collide.f"] = copy(st.f_in))
    KrakeNN.stream_wall_x_wall_y_2d!(st.f_out, st.f_in, st.Nx, st.Ny)
    KrakeNN.apply_free_slip_sidewalls_2d!(st.f_out, st.Nx, st.Ny)
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    st.f_in, st.f_out = st.f_out, st.f_in
    record && (checkpoints["f_after_ns_stream_bc.f"] = copy(st.f_in))

    KrakeNN.compute_macroscopic_guo_field_2d!(st.rho, st.ux, st.uy, st.f_in, st.Fx, st.Fy, st.Nx, st.Ny)
    KrakeNN.enforce_free_side_macros_2d!(st.ux, st.uy, st.Nx, st.Ny)
    KrakeNN.ehd_maxspeed_2d!(st.diag, st.ux, st.uy, st.Nx, st.Ny)
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    umax = st.diag[1]
    record && (checkpoints["macros_post.rho"] = copy(st.rho);
               checkpoints["macros_post.ux"] = copy(st.ux);
               checkpoints["macros_post.uy"] = copy(st.uy);
               checkpoints["macros_post.umax"] = [umax])
    copyto!(st.Fx_prev, st.Fx)
    copyto!(st.Fy_prev, st.Fy)
    return (checkpoints=checkpoints, phi_iter=phi_iter, phi_rel=phi_rel, umax=umax)
end

function diff_metrics(a, b)
    av = vec(collect(a))
    bv = vec(collect(b))
    linf = maximum(abs.(av .- bv))
    # These nondimensional checkpoints include intentionally near-zero fields
    # (Ex/Fx/ux). Use a unit floor so rel behaves like an absolute Linf check
    # for sub-unit fields instead of dividing by numerical noise.
    denom = max(maximum(abs.(bv)), one(eltype(float.(bv))))
    return linf, linf / denom
end

function _twin_step_with_phi_check_every!(st::EHDTwinState; phi_tol=1e-4,
                                          phi_max_iter=10000,
                                          phi_check_every=8,
                                          force_projection=:xy,
                                          record=false)
    checkpoints = Dict{String,Any}()
    Up = similar(st.phi)
    phi_rel = Inf
    phi_iter = 0
    for iter in 1:phi_max_iter
        copyto!(Up, st.phi)
        collide_phi!(st.phi_f, st.q, st.p.eps, st.p.omega_U, st.p.nu_U)
        matlab_stream!(st.phi_tmp, st.phi_f)
        st.phi_f, st.phi_tmp = st.phi_tmp, st.phi_f
        scalar!(st.phi, st.phi_f)
        apply_phi_bc!(st.phi_f, st.phi)
        scalar!(st.phi, st.phi_f)
        phi_iter = iter
        if iter % phi_check_every == 0 || iter == phi_max_iter
            phi_rel = maximum(abs.(st.phi .- Up)) / max(maximum(abs.(st.phi)), floatmin(eltype(st.phi)))
            phi_rel <= phi_tol && break
            iter == phi_max_iter && error("Twin phi solve failed; rel=$(phi_rel)")
        end
    end
    calculate_E!(st.Ex, st.Ey, st.phi_f, st.p.tau_U)
    record && (checkpoints["phi_after_solve"] = copy(st.phi);
               checkpoints["Ex"] = copy(st.Ex);
               checkpoints["Ey"] = copy(st.Ey))

    Fx_prev = sum_slots(st.Fx_slots)
    Fy_prev = sum_slots(st.Fy_slots)
    macros_from_force!(st.rho, st.ux, st.uy, st.ns_f, Fx_prev, Fy_prev, st.wall)
    record && (checkpoints["macros_pre_charge.rho"] = copy(st.rho);
               checkpoints["macros_pre_charge.ux"] = copy(st.ux);
               checkpoints["macros_pre_charge.uy"] = copy(st.uy))

    collide_charge_regularized!(st.q_f, st.ux, st.uy, st.Ex, st.Ey, st.p.tau_q, st.p.K)
    record && (checkpoints["q_after_collide.f"] = copy(st.q_f))
    matlab_stream!(st.q_tmp, st.q_f)
    st.q_f, st.q_tmp = st.q_tmp, st.q_f
    scalar!(st.q, st.q_f)
    apply_charge_bc!(st.q_f, st.q, st.ux, st.uy, st.Ex, st.Ey, st.p.q_inj, st.p.K)
    scalar!(st.q, st.q_f)
    record && (checkpoints["q_after_stream_bc.f"] = copy(st.q_f);
               checkpoints["q_after_stream_bc.q"] = copy(st.q))

    apply_ef!(st.Fx_slots, st.Fy_slots, st.Ex, st.Ey, st.q, st.rho, st.wall;
              projection_mode=force_projection, force_ramp=1.0)
    Fx_dens, Fy_dens = force_density_from_slots(st)
    record && (checkpoints["Fx"] = copy(Fx_dens); checkpoints["Fy"] = copy(Fy_dens))
    collide_sp_mrt!(st.ns_f, st.rho, st.ux, st.uy, st.Fx_slots, st.Fy_slots, st.wall,
                    st.mrt_M, st.mrt_N)
    record && (checkpoints["f_after_ns_collide.f"] = copy(st.ns_f))
    matlab_stream!(st.ns_tmp, st.ns_f; bb_links=st.bb_links)
    st.ns_f, st.ns_tmp = st.ns_tmp, st.ns_f
    apply_free_slip_sidewalls!(st.ns_f)
    record && (checkpoints["f_after_ns_stream_bc.f"] = copy(st.ns_f))
    macros_from_force!(st.rho, st.ux, st.uy, st.ns_f, sum_slots(st.Fx_slots),
                       sum_slots(st.Fy_slots), st.wall)
    umax = maximum(sqrt.(st.ux .* st.ux .+ st.uy .* st.uy))
    record && (checkpoints["macros_post.rho"] = copy(st.rho);
               checkpoints["macros_post.ux"] = copy(st.ux);
               checkpoints["macros_post.uy"] = copy(st.uy);
               checkpoints["macros_post.umax"] = [umax])
    return (checkpoints=checkpoints, phi_iter=phi_iter, phi_rel=phi_rel, umax=umax)
end

@testset "EHD twin parity" begin
    cfg = (Nx=31, Ny=48, C=10.0, M=10.0, T=300.0, Ma_E=0.01,
           alpha=1e-4, delta_U=1.0, gamma=0.3, perturb_amplitude=1e-4,
           perturb_mode=1, FT=Float64)
    tol = 1e-5

    tw = init_twin_state(; cfg...)
    kr = init_kraken_trace_from_twin(tw)
    max_rel = 0.0
    max_linf = 0.0
    max_stage = ""
    max_step = 0
    max_twin_phi_iter = 0
    max_kraken_phi_iter = 0

    elapsed = @elapsed begin
        for step in 1:10
            tref = _twin_step_with_phi_check_every!(tw; phi_tol=1e-4,
                                                    force_projection=:xy,
                                                    phi_check_every=8,
                                                    record=true)
            kref = kraken_step!(kr; phi_tol=1e-4, force_projection=:xy,
                                phi_check_every=8, record=true)
            max_twin_phi_iter = max(max_twin_phi_iter, tref.phi_iter)
            max_kraken_phi_iter = max(max_kraken_phi_iter, kref.phi_iter)

            for stage in STAGE_ORDER
                linf, rel = diff_metrics(kref.checkpoints[stage], tref.checkpoints[stage])
                if rel > max_rel
                    max_rel = rel
                    max_linf = linf
                    max_stage = stage
                    max_step = step
                end
                # The post-fix 10-step trace at 59x96 observed macros max|u|
                # relative diff about 3.4e-7 (commit bb066bcc1). The 1e-5
                # tolerance leaves margin for grid-size sensitivity: this
                # regression uses 31x48 to keep the CPU budget small.
                @test rel <= tol
            end
        end
    end

    @info "EHD twin parity" max_rel=max_rel max_linf=max_linf max_stage=max_stage max_step=max_step elapsed_s=elapsed max_twin_phi_iter=max_twin_phi_iter max_kraken_phi_iter=max_kraken_phi_iter
    println("EHD twin parity max relative diff = $(max_rel) at step $(max_step), stage $(max_stage); elapsed_s=$(elapsed)")

    @test elapsed <= 60
    @test all(isfinite, tw.rho)
    @test all(isfinite, tw.ux)
    @test all(isfinite, tw.uy)
    @test all(isfinite, kr.rho)
    @test all(isfinite, kr.ux)
    @test all(isfinite, kr.uy)
end
