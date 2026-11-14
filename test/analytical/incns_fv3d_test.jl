# V0 of the 3D steady FV solver (plan "solveur FV stationnaire 3D", increment
# I1), CPU only, Float64 (~30 s):
#   (a) dimensional reduction: with pins cleared on both sides, Nz = 3 periodic,
#       z-uniform data and w = 0, the 3D residual equals h times the 2D residual
#       cell by cell (face area h^2 and volume h^3 against h and h^2), the w row
#       vanishes and the mean-flow row is unchanged; Newtonian, GNF and law;
#   (pins) placement per closed 6-connected pocket and the pinned row p_P;
#   (b) coloured Jacobian == dense ForwardDiff Jacobian on 4x5x6 (random states,
#       three active components, solids, a pin, the four side types,
#       periodicity, imposed flow rate, Newtonian and law viscosity);
#   (c) manufactured-solution smoke test on 6^3 and 8^3 (orders on GPU: V0');
#   (d) simple shear: the law evaluated from the 9-component invariant I1 equals
#       the law evaluated at the shear rate;
#   CPU inference / zero-allocation guards on the 3D kernel (not GPU validation).
using Test
using LinearAlgebra
using Random
using ForwardDiff
using KrakeNN
# KernelAbstractions through KrakeNN (the run env need not list it); keep an earlier binding
isdefined(@__MODULE__, :KernelAbstractions) || @eval const KernelAbstractions = KrakeNN.KernelAbstractions
using KrakeNN.SparseArrays

# ------------------------------------------------------------------ helpers
_fv3t_concrete(t::Core.Const) = true
_fv3t_concrete(t::Core.PartialStruct) = _fv3t_concrete(t.typ)
_fv3t_concrete(t) = t === Union{} || (t isa Type &&
    (isconcretetype(t) || (t isa Union && all(_fv3t_concrete, Base.uniontypes(t)))))
function _fv3t_has_box(x)
    x === Core.Box && return true
    x isa Core.Const && return _fv3t_has_box(x.val)
    x isa Core.PartialStruct && return _fv3t_has_box(x.typ) || any(_fv3t_has_box, x.fields)
    x isa DataType && return any(_fv3t_has_box, x.parameters)
    x isa Union && return any(_fv3t_has_box, Base.uniontypes(x))
    x isa GlobalRef && return x.mod === Core && x.name === :Box
    x isa Expr && return any(_fv3t_has_box, x.args)
    x isa QuoteNode && return _fv3t_has_box(x.value)
    return false
end
# same checks as incns_fv_adjoint_compiler.jl: boxed captures, abstract SSA
# values, runtime dispatch and foreign calls in the KA inner function
function _fv3t_kernel_issues(args, ndrange)
    kernel = KrakeNN._fv3_residual_kernel!(KernelAbstractions.CPU(), 64)
    ci, rt = only(KernelAbstractions.ka_code_typed(kernel, typeof(args); ndrange))
    issues = String[]
    rt === Nothing || push!(issues, "kernel return is $rt")
    any(_fv3t_has_box, ci.slottypes) && push!(issues, "boxed slot")
    for (i, ex) in enumerate(ci.code)
        t = ci.ssavaluetypes[i]
        (_fv3t_has_box(ex) || _fv3t_has_box(t)) && push!(issues, "boxed SSA $i")
        computes = ex isa Core.PhiNode || ex isa Core.PiNode ||
                   (ex isa Expr && ex.head in (:call, :invoke, :new, :splatnew, :foreigncall))
        computes && !_fv3t_concrete(t) && push!(issues, "abstract SSA $i: $t")
        if ex isa Expr && ex.head === :call
            f = ex.args[1]
            callee = f isa GlobalRef ? getfield(f.mod, f.name) : f isa QuoteNode ? f.value : f
            callee isa Core.Builtin || callee isa Core.IntrinsicFunction ||
                push!(issues, "runtime call at SSA $i: $f")
        elseif ex isa Expr && ex.head === :foreigncall
            push!(issues, "foreign call at SSA $i")
        end
    end
    return unique(issues)
end

_fv3t_dense_jacobian(p, U) =
    ForwardDiff.jacobian((R, X) -> fv_residual!(R, X, p), zeros(length(U)), U)

_fv3t_clear_pins!(tag) = replace!(tag, FV_CELL_PINNED => FV_CELL_FLUID)

function _fv3t_learned(field3)
    law = LearnedViscosity(TinyMLP{2,2,1}(); nu_ref = 0.1, monotone = false)
    theta = law_init(law; rng = MersenneTwister(33))
    theta[3:4] .= (0.3, -0.2)          # non-negligible field dependence
    return FVLawViscosity(law, theta, (field3,); eps = 0.02)
end

# 2D problem and its z-extruded 3D counterpart (Nz = 3, periodic z)
function _fv3t_pair(Nx, Ny; bc, nu, solid = nothing, source = nothing, bvel2,
                    conv = true, rheology = nothing, gnf_eps = 0.0, law = false,
                    u_mean_target = nothing)
    Nz = 3; h = 1 / Ny
    field2 = [0.2 * (i - 0.5) + 0.1 * (j - 0.5) for i in 1:Nx, j in 1:Ny]
    field3 = repeat(field2, 1, 1, Nz)
    v2 = law ? _fv3t_learned(field2) : nothing
    v3 = law ? _fv3t_learned(field3) : nothing
    src3 = source === nothing ? nothing : (x, y, z) -> (s = source(x, y); (s[1], s[2], 0.0, s[3]))
    p2 = fv_steady_problem(Nx, Ny; h, bc, nu, solid, source, conv, rheology, gnf_eps,
                           viscosity = v2, u_mean_target, boundary_velocity = bvel2)
    p3 = fv_steady_problem(Nx, Ny, Nz; h, nu = (x, y, z) -> nu(x, y), conv, rheology, gnf_eps,
                           bc = (; bc..., B = :periodic, T = :periodic),
                           solid = solid === nothing ? nothing : repeat(solid, 1, 1, Nz),
                           source = src3, viscosity = v3, u_mean_target,
                           boundary_velocity = (s, x, y, z) -> (bvel2(s, x, y)..., 0.0))
    _fv3t_clear_pins!(p2.tag); _fv3t_clear_pins!(p3.tag)
    return p2, p3
end

# ------------------------------------------------------------------ manufactured solution
const _FV3T_K = 2pi
const _FV3T_ABC = (1.0, 0.7, -0.4)
function _fv3t_uex(x)
    k = _FV3T_K; a, b, c = _FV3T_ABC
    sx, sy, sz = sin(k * x[1]), sin(k * x[2]), sin(k * x[3])
    cx, cy, cz = cos(k * x[1]), cos(k * x[2]), cos(k * x[3])
    return [sx * (c * cy - b * cz), sy * (a * cz - c * cx), sz * (b * cx - a * cy)]
end
_fv3t_pex(x) = 0.5 * cos(_FV3T_K * x[1]) * sin(2 * _FV3T_K * x[2]) * cos(_FV3T_K * x[3])
_fv3t_nuex(x) = 1 + 0.4 * sin(_FV3T_K * (x[1] + 2 * x[2] + 3 * x[3]))
function _fv3t_source(x, conv)
    G(y) = ForwardDiff.jacobian(_fv3t_uex, y)
    σ(y) = (Gy = G(y); vec(_fv3t_nuex(y) .* (Gy .+ Gy')))
    Dσ = ForwardDiff.jacobian(σ, x)          # row r + 3(c - 1) is σ_rc
    divσ = ntuple(r -> sum(Dσ[r + 3 * (c - 1), c] for c in 1:3), 3)
    gp = ForwardDiff.gradient(_fv3t_pex, x)
    cv = conv ? G(x) * _fv3t_uex(x) : zeros(3)
    return (cv[1] + gp[1] - divσ[1], cv[2] + gp[2] - divσ[2], cv[3] + gp[3] - divσ[3], 0.0)
end
function _fv3t_mms_errors(N, bc, conv)
    p = fv_steady_problem(N, N, N; nu = (x, y, z) -> _fv3t_nuex([x, y, z]), bc, conv,
                          boundary_velocity = (s, x, y, z) -> Tuple(_fv3t_uex([x, y, z])),
                          source = (x, y, z) -> _fv3t_source([x, y, z], conv))
    Ue = fv_state(p, ((x, y, z) -> _fv3t_uex([x, y, z])[c] for c in 1:3)...,
                  (x, y, z) -> _fv3t_pex([x, y, z]))
    s = fv_newton_solver(p)
    U = copy(Ue)
    st = fv_newton!(s, U; atol = 1e-11, rtol = 0.0, maxit = 20)
    f = fv_fields(p, U); fe = fv_fields(p, Ue)
    eu = sqrt(sum(sum(abs2, f[c] - fe[c]) for c in 1:3) / N^3)
    pp = f[4] .- sum(f[4]) / N^3; pe = fe[4] .- sum(fe[4]) / N^3
    return eu, sqrt(sum(abs2, pp - pe) / N^3), st
end
const _FV3T_WALLS = (W = :wall, E = :wall, S = :wall, N = :wall, B = :wall, T = :wall)
const _FV3T_XPER = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)
const _FV3T_XYPER = (W = :periodic, E = :periodic, S = :periodic, N = :periodic, B = :wall, T = :wall)

@testset "IncNS steady FV 3D (I1 V0, CPU)" begin
    rng = MersenneTwister(2026)

    @testset "(a) z-uniform 3D residual = h x 2D residual" begin
        lid(s, x, y) = s === :N ? (1.0, 0.0) : (0.0, 0.0)
        inflow(s, x, y) = s === :W ? (4y * (1 - y) + 0.1, 0.2 * y) : s === :N ? (0.3, 0.0) : (0.0, 0.0)
        solid = falses(7, 5); solid[3:4, 1:2] .= true
        nuxy(x, y) = 0.05 * (1 + x * y) + 0.02 * sin(3x)
        cy = CarreauYasuda(0.1, 0.002, 3.0, 2.0, 0.4)
        cases = (
            ("Newtonian closed box, NS", (; bc = (W = :wall, E = :wall, S = :wall, N = :wall), nu = nuxy, bvel2 = lid)),
            ("Newtonian inlet/outlet + solids, NS",
             (; bc = (W = :inlet, E = :outlet, S = :wall, N = :wall), nu = nuxy, solid, bvel2 = inflow,
              source = (x, y) -> (0.3x, -0.2y, 0.05 * x * y))),
            ("Newtonian periodic x + source, Stokes",
             (; bc = (W = :periodic, E = :periodic, S = :wall, N = :wall), nu = nuxy, conv = false, bvel2 = lid,
              source = (x, y) -> (1.0, 0.1x, 0.0))),
            ("Newtonian imposed flow rate",
             (; bc = (W = :periodic, E = :periodic, S = :wall, N = :wall), nu = nuxy, bvel2 = (s, x, y) -> (0.0, 0.0),
              source = (x, y) -> (0.5, 0.0, 0.0), u_mean_target = 0.2)),
            ("GNF Carreau-Yasuda inlet/outlet + solids",
             (; bc = (W = :inlet, E = :outlet, S = :wall, N = :wall), nu = nuxy, solid, bvel2 = inflow,
              rheology = cy, gnf_eps = 1e-3)),
            ("law (learned, prescribed field) closed box",
             (; bc = (W = :wall, E = :wall, S = :wall, N = :wall), nu = nuxy, bvel2 = lid, law = true)),
            ("law imposed flow rate + solids",
             (; bc = (W = :periodic, E = :periodic, S = :wall, N = :wall), nu = nuxy, solid, law = true,
              bvel2 = (s, x, y) -> (0.0, 0.0), source = (x, y) -> (0.5, 0.0, 0.0), u_mean_target = 0.2)),
        )
        for (name, kw) in cases
            @testset "$name" begin
                p2, p3 = _fv3t_pair(7, 5; kw...)
                g2, g3 = p2.grid, p3.grid
                h = g3.h
                @test !any(==(FV_CELL_PINNED), p3.tag)
                for trial in 1:2
                    u = randn(rng, 7, 5); v = randn(rng, 7, 5); pr = randn(rng, 7, 5)
                    U2 = fv_state(p2, u, v, pr)
                    U3 = fv_state(p3, repeat(u, 1, 1, 3), repeat(v, 1, 1, 3), zeros(7, 5, 3),
                                  repeat(pr, 1, 1, 3))
                    if p2.flowrate !== nothing
                        U2[end] = U3[end] = randn(rng)
                    end
                    R2 = fv_residual(U2, p2); R3 = fv_residual(U3, p3)
                    scale = maximum(abs, R2) * h
                    worst = 0.0; worst_w = 0.0
                    for k in 1:3, j in 1:5, i in 1:7
                        # solid cells carry identity rows on both sides (unscaled)
                        hs = p3.tag[i, j, k] == FV_CELL_SOLID ? 1.0 : h
                        for (a, b) in ((1, 1), (2, 2), (4, 3))
                            worst = max(worst, abs(R3[fv_index(g3, i, j, k, a)] - hs * R2[fv_index(g2, i, j, b)]))
                        end
                        worst_w = max(worst_w, abs(R3[fv_index(g3, i, j, k, 3)]))
                    end
                    @test worst <= 1e-14 * scale
                    @test worst_w <= 1e-14 * scale
                    p2.flowrate === nothing || @test abs(R3[end] - R2[end]) <= 1e-14 * max(1, abs(R2[end]))
                end
            end
        end
    end

    @testset "pressure pins: placement per closed pocket and pinned row" begin
        # two pockets separated by a solid slab at i = 3 in a closed box
        solid = falses(5, 4, 3); solid[3, :, :] .= true
        p = fv_steady_problem(5, 4, 3; solid)
        pins = findall(==(FV_CELL_PINNED), p.tag)
        @test length(pins) == 2
        @test Tuple.(pins) == [(1, 1, 1), (4, 1, 1)]   # lowest linear index of each pocket
        tag = copy(p.tag)
        @test fv_pressure_pins!(tag, p.grid) == 2 && tag == p.tag
        # an outlet opens the pocket; fully periodic has one pocket
        q = fv_steady_problem(5, 4, 3; bc = (W = :inlet, E = :outlet, S = :wall, N = :wall, B = :wall, T = :wall))
        @test !any(==(FV_CELL_PINNED), q.tag)
        per = (W = :periodic, E = :periodic, S = :periodic, N = :periodic, B = :periodic, T = :periodic)
        r = fv_steady_problem(5, 4, 3; bc = per)
        @test findall(==(FV_CELL_PINNED), r.tag) == [CartesianIndex(1, 1, 1)]
        # pocket open through a periodic wrap only: slab + outlet on the far side
        solid2 = falses(6, 4, 3); solid2[3, :, :] .= true
        s = fv_steady_problem(6, 4, 3; solid = solid2,
                              bc = (W = :wall, E = :outlet, S = :wall, N = :wall, B = :periodic, T = :periodic))
        @test findall(==(FV_CELL_PINNED), s.tag) == [CartesianIndex(1, 1, 1)]
        # effect: the pinned continuity row is p_P, every other row unchanged
        U = randn(rng, fv_nunknowns(p))
        R = fv_residual(U, p)
        g = p.grid
        for c in pins
            @test R[fv_index(g, Tuple(c)..., 4)] == U[fv_index(g, Tuple(c)..., 4)]
        end
        tag0 = copy(p.tag); _fv3t_clear_pins!(tag0)
        p0 = FVSteadyProblem3D(g, tag0, p.nu, p.bvel, p.src, p.visc, nothing, nothing, nothing)
        R0 = fv_residual(U, p0)
        pinrows = [fv_index(g, Tuple(c)..., 4) for c in pins]
        keep = setdiff(1:length(R), pinrows)
        @test R[keep] == R0[keep]
        @test all(R0[pinrows] .!= R[pinrows])
    end

    @testset "(b) coloured Jacobian == dense on 4x5x6" begin
        Nx, Ny, Nz = 4, 5, 6
        nu3(x, y, z) = 0.05 * (1 + x * y + 0.5z) + 0.02 * sin(3x + 2z)
        solid = falses(Nx, Ny, Nz); solid[2:3, 1:2, 3] .= true; solid[4, 5, 6] = true
        inflow(s, x, y, z) = s === :W ? (4y * (1 - y) + 0.1, 0.2 * y, 0.1 * z) :
                             s === :N ? (0.3, 0.0, 0.2) : (0.0, 0.0, 0.0)
        lid(s, x, y, z) = s === :T ? (1.0, 0.4, 0.0) : (0.0, 0.0, 0.0)
        field = [0.2 * i + 0.1 * j - 0.05 * k for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        cy = CarreauYasuda(0.1, 0.002, 3.0, 2.0, 0.4)
        io = (W = :inlet, E = :outlet, S = :wall, N = :wall, B = :periodic, T = :periodic)
        xper = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :periodic, T = :periodic)
        cases = (
            ("Newtonian inlet/outlet/walls/periodic z + solids", false,
             fv_steady_problem(Nx, Ny, Nz; h = 0.2, nu = nu3, bc = io, solid, boundary_velocity = inflow)),
            ("Newtonian closed box with lid (pin)", false,
             fv_steady_problem(Nx, Ny, Nz; h = 0.2, nu = nu3, boundary_velocity = lid)),
            ("Newtonian imposed flow rate, Stokes", false,
             fv_steady_problem(Nx, Ny, Nz; h = 0.2, nu = nu3, bc = xper, solid, conv = false,
                               source = (x, y, z) -> (0.5, 0.0, 0.0, 0.0), u_mean_target = 0.2)),
            ("GNF Carreau-Yasuda inlet/outlet + solids", true,
             fv_steady_problem(Nx, Ny, Nz; h = 0.2, nu = nu3, bc = io, solid, boundary_velocity = inflow,
                               rheology = cy, gnf_eps = 1e-3)),
            ("law (learned, field) closed box with lid (pin)", true,
             fv_steady_problem(Nx, Ny, Nz; h = 0.2, boundary_velocity = lid,
                               viscosity = _fv3t_learned(field))),
            ("law imposed flow rate + solids", true,
             fv_steady_problem(Nx, Ny, Nz; h = 0.2, bc = xper, solid, viscosity = _fv3t_learned(field),
                               source = (x, y, z) -> (0.5, 0.0, 0.0, 0.0), u_mean_target = 0.2)),
        )
        for (name, state_visc, p) in cases
            @testset "$name" begin
                S = fv_jacobian_pattern(p)
                colors, ncol = fv_jacobian_coloring(S, p)
                rows = diff(sparse(transpose(S)).colptr)[1:fv_nunknowns(p.grid)]   # not the dense Fx row
                # H7: Newtonian 42 entries per momentum row; face-strain stencil ~70
                @test maximum(rows) <= (state_visc ? 80 : 42)
                n = fv_nunknowns(p)
                s = fv_newton_solver(p)
                for trial in 1:2
                    U = 0.5 .* randn(rng, n)
                    Jd = _fv3t_dense_jacobian(p, U)
                    @test all(S[k] for k in findall(!iszero, Jd))   # pattern is a superset
                    fv_jacobian!(s, U)
                    @test Matrix(s.J) == Jd                         # bit for bit
                    asm = fv_jacobian_assembler(S, colors, ncol; chunk = 8, order = :csr)
                    nz = zeros(nnz(S))
                    fv_assemble_jacobian!(nz, asm, U, p)
                    @test nz == nonzeros(sparse(transpose(s.J)))   # cuDSS (CSR) order
                end
                @info "3D coloured Jacobian" name ncol nnz(S) max_row = maximum(rows)
            end
        end
    end

    @testset "(c) MMS smoke 6^3 -> 8^3" begin
        # measured 2026-11-09 (CPU Float64, 6 -> 8): velocity L2 order 2.27
        # (Stokes, walls), 2.31 (Stokes, periodic x), 2.04 (NS, periodic x,y);
        # pressure 2.05, 1.98, 2.22. Smoke gates only (two coarse grids, NS not
        # yet in its first-order regime); the orders are gated on GPU by V0'.
        for (name, bc, conv) in (("Stokes all walls", _FV3T_WALLS, false),
                                 ("Stokes periodic x", _FV3T_XPER, false),
                                 ("NS periodic x,y", _FV3T_XYPER, true))
            e6 = _fv3t_mms_errors(6, bc, conv)
            e8 = _fv3t_mms_errors(8, bc, conv)
            @test e6[3].converged && e8[3].converged
            ou = log(e6[1] / e8[1]) / log(8 / 6)
            op = log(e6[2] / e8[2]) / log(8 / 6)
            @info "3D MMS smoke" name eu6 = e6[1] eu8 = e8[1] order_u = ou order_p = op
            @test e8[1] < e6[1]
            @test ou >= (conv ? 0.8 : 1.5)
        end
    end

    @testset "(d) simple shear: law(I1) == law(gamma_dot)" begin
        gd = 0.37
        field = fill(0.6, 5, 5, 5)
        cy = CarreauYasuda(0.1, 0.002, 3.0, 2.0, 0.4)
        lv = _fv3t_learned(field)
        ep = 0.02
        lv0 = FVLawViscosity(lv.law, lv.theta, (field,); eps = ep)
        cl = FVLawViscosity(ClassicalLaw(cy), law_init(ClassicalLaw(cy)), (); eps = ep)
        # I1 + eps^2/2 = gd^2/2 + eps^2/2: the law sees gamma = sqrt(gd^2 + eps^2)
        ge = sqrt(gd^2 + ep^2)
        # three shear planes: u = gd y, v = gd z, w = gd x (velocity exact on walls)
        for (comp, axis) in ((1, 2), (2, 3), (3, 1))
            prof(x, y, z) = gd * (x, y, z)[axis]
            vel(x, y, z) = ntuple(c -> c == comp ? prof(x, y, z) : 0.0, 3)
            for visc in (FVGNFViscosity(cy, ep), lv0, cl)
                p = fv_steady_problem(5, 5, 5; h = 0.2, viscosity = visc,
                                      boundary_velocity = (s, x, y, z) -> vel(x, y, z))
                U = fv_state(p, ((x, y, z) -> vel(x, y, z)[c] for c in 1:3)..., (x, y, z) -> 0.0)
                g = p.grid
                ref = visc isa FVGNFViscosity ?
                      0.002 + (0.1 - 0.002) * (1 + (3.0 * ge)^2.0)^((0.4 - 1) / 2.0) :
                      visc === lv0 ? law_value(lv.law, (ge, 0.6), lv.theta) :
                      law_value(cl.law, (ge,), cl.theta)
                worst = 0.0
                for k in 2:4, j in 2:4, i in 2:4, d in 1:6
                    kind, ii, jj, kk = KrakeNN._fv_face_kind(g, p.tag, i, j, k, d)
                    gP = KrakeNN._fv3_cell_grad(U, p.tag, p.bvel, g, i, j, k)
                    gN = KrakeNN._fv3_cell_grad(U, p.tag, p.bvel, g, ii, jj, kk)
                    n = KrakeNN._fv3_axis(d); sx = KrakeNN._fv3_sign(d)
                    uP = KrakeNN._fv3_cell(U, g, i, j, k); uN = KrakeNN._fv3_cell(U, g, ii, jj, kk)
                    G = KrakeNN._fv3_face_grad(KrakeNN._fv3_mean9(gP, gN), n, sx,
                                               ((uN[c] - uP[c]) / g.h for c in 1:3)...)
                    @test G[3 * (comp - 1) + axis] ≈ gd rtol = 1e-14
                    nuf = KrakeNN._fv_face_nu3(visc, 0.0, 0.0, G, i, j, k, ii, jj, kk)
                    worst = max(worst, abs(nuf - ref) / ref)
                end
                @test worst <= 1e-14
            end
        end
    end

    @testset "kernel inference and zero allocation (CPU guard)" begin
        S = ForwardDiff.Dual{KrakeNN.FVJacobianTag,Float64,4}
        D = ForwardDiff.Dual{KrakeNN.FVParamTag,Float64,4}
        field = rand(rng, 4, 3, 2)
        cy = CarreauYasuda(0.1, 0.002, 3.0, 2.0, 0.4)
        base = (; bc = (W = :inlet, E = :outlet, S = :wall, N = :wall, B = :periodic, T = :periodic),
                boundary_velocity = (s, x, y, z) -> (1.0, 0.0, 0.0))
        lv = _fv3t_learned(field)
        lvd = FVLawViscosity(lv.law, D.(lv.theta), (field,); eps = 0.02)
        probs = (("Newtonian", fv_steady_problem(4, 3, 2; base...)),
                 ("GNF", fv_steady_problem(4, 3, 2; base..., rheology = cy, gnf_eps = 1e-3)),
                 ("law", fv_steady_problem(4, 3, 2; base..., viscosity = lv)),
                 ("law dual parameters", fv_steady_problem(4, 3, 2; base..., viscosity = lvd)),
                 ("flow rate", fv_steady_problem(4, 3, 2; bc = (W = :periodic, E = :periodic, S = :wall, N = :wall,
                                                               B = :wall, T = :wall),
                                                 source = (x, y, z) -> (1.0, 0.0, 0.0, 0.0), u_mean_target = 0.1)))
        for (name, p) in probs, U in (fill(0.1, fv_nunknowns(p)), fill(S(0.1), fv_nunknowns(p)))
            # parameter duals and state duals are never mixed (adjoint vs Jacobian)
            name == "law dual parameters" && eltype(U) <: ForwardDiff.Dual && continue
            @testset "$name $(eltype(U) <: ForwardDiff.Dual ? "dual state" : "Float64")" begin
                T = promote_type(eltype(U), eltype(p.nu), eltype(p.bvel), KrakeNN._fv_visc_eltype(p.visc))
                R = zeros(T, length(U))
                args = (R, U, p.tag, p.nu, p.bvel, p.src, p.grid, p.visc)
                @test isempty(_fv3t_kernel_issues(args, (4, 3, 2)))
                @inferred KrakeNN._fv3_cell_residual(U, p.tag, p.nu, p.bvel, p.src, p.grid, 2, 2, 1, p.visc)
                cell!() = KrakeNN._fv3_residual_cell!(R, U, p.tag, p.nu, p.bvel, p.src, p.grid, p.visc, 2, 2, 1)
                cell!()
                @test (@allocated cell!()) == 0
                fv_residual!(R, U, p)
                @test all(isfinite, ForwardDiff.value.(R))
            end
        end
    end
    # Pseudo-time scaling: the PTC shift is V/dtau with V the cell measure (h^3 in 3D,
    # h^2 in 2D), the scale of the residual rows, so dtau0 is the same physical
    # pseudo-step in both. Before the fix the 3D shift was h^2/dtau and a body-force
    # duct needed ~2.5 N Newton iterations (20 at N = 8) with the default dtau0.
    @testset "pseudo-time shift V/dtau: body-force duct, default dtau0" begin
        N = 8; h = 1 / N
        bc = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)
        mk() = fv_steady_problem(N, N, N; h, nu = 0.1, bc,
                                 source = (x, y, z) -> (1.0, 0.0, 0.0, 0.0),
                                 boundary_velocity = (s, x, y, z) -> (0.0, 0.0, 0.0))
        function run(dtau0)
            p = mk(); s = fv_newton_solver(p)
            U = zeros(fv_nunknowns(p))
            st = fv_newton!(s, U; atol = 1e-13, rtol = 1e-15, maxit = 60, dtau0, dtau_max = 1e8)
            return st, U
        end
        st1, U1 = run(1.0)
        @test st1.converged
        # first step: r = r0, so the shift is exactly h^3 / dtau0 (h^2 would be N times larger)
        @test st1.shift[1] ≈ h^3 / 1.0 rtol = 1e-14
        # bound: ~8 expected for this nearly linear case; 15 stays below the 2.5 N = 20 of the old scaling
        @test st1.iterations <= 15
        st2, U2 = run(10.0)
        @test st2.converged
        @test st2.shift[1] ≈ h^3 / 10.0 rtol = 1e-14
        # steady state does not depend on the pseudo-time
        @test norm(U1 - U2) <= 1e-10 * norm(U1)
        @test maximum(abs, U1) > 1e-3          # non-trivial flow
        # 2D unchanged: h^2 / dtau0
        p2 = fv_steady_problem(N, N; h, nu = 0.1, bc = (W = :periodic, E = :periodic, S = :wall, N = :wall),
                               source = (x, y) -> (1.0, 0.0, 0.0), boundary_velocity = (s, x, y) -> (0.0, 0.0))
        s2 = fv_newton_solver(p2)
        st = fv_newton!(s2, zeros(fv_nunknowns(p2)); maxit = 3, dtau0 = 1.0, dtau_max = 1e8)
        @test st.shift[1] ≈ h^2 rtol = 1e-14
    end
end
