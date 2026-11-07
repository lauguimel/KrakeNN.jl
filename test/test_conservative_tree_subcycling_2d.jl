using Test
using KrakeNN

function _test_full_domain_nested_spec_2d(max_level::Integer)
    blocks = ConservativeTreeRefineBlock2D[]
    parent = ""
    for level in 1:Int(max_level)
        nx_level = 16 << (level - 1)
        ny_level = 12 << (level - 1)
        name = "L$(level)"
        push!(blocks, ConservativeTreeRefineBlock2D(
            name, 1:nx_level, 1:ny_level; parent=parent))
        parent = name
    end
    return create_conservative_tree_spec_2d(16, 12, blocks)
end

function _test_center_yband_nested_spec_2d()
    return create_conservative_tree_spec_2d(16, 12, [
        ConservativeTreeRefineBlock2D("C1", 1:16, 3:10),
        ConservativeTreeRefineBlock2D("C2", 1:32, 7:18; parent="C1"),
    ])
end

function _test_wall_refined_ybands_nested_spec_2d()
    return create_conservative_tree_spec_2d(16, 12, [
        ConservativeTreeRefineBlock2D("B1", 1:16, 1:5),
        ConservativeTreeRefineBlock2D("B2", 1:32, 1:8; parent="B1"),
        ConservativeTreeRefineBlock2D("T1", 1:16, 8:12),
        ConservativeTreeRefineBlock2D("T2", 1:32, 17:24; parent="T1"),
    ])
end

function _test_cartesian_poiseuille_profile_2d(max_level::Integer,
                                               steps::Integer;
                                               Fx=1e-7,
                                               omega=1.0,
                                               rho0=1.0)
    scale = 1 << Int(max_level)
    nx = 16 * scale
    ny = 12 * scale
    volume = 1.0 / (scale * scale)
    F = zeros(Float64, nx, ny, 9)
    Ftmp = similar(F)
    fill_equilibrium_integrated_D2Q9!(F, volume, rho0, 0.0, 0.0)

    for _ in 1:(Int(steps) * scale)
        collide_Guo_integrated_D2Q9!(F, volume, omega, Fx, 0.0)
        stream_periodic_x_wall_y_F_2d!(Ftmp, F)
        F, Ftmp = Ftmp, F
    end

    profile = zeros(Float64, ny)
    for j in 1:ny
        ux_sum = 0.0
        for i in 1:nx
            cell = @view F[i, j, :]
            rho = mass_F(cell) / volume
            mx = momentum_F(cell)[1]
            ux_sum += (mx / volume + Fx / 2) / rho
        end
        profile[j] = ux_sum / nx
    end
    return profile
end

@testset "Conservative tree subcycling ledger 2D" begin
    @testset "generic recursive schedule is level agnostic" begin
        schedule = KrakeNN.create_conservative_tree_subcycle_schedule_2d(3)

        @test schedule.max_level == 3
        @test schedule.ratio == 2
        @test schedule.finest_ticks == 8
        @test schedule.level_step_ticks == [8, 4, 2, 1]
        @test KrakeNN.conservative_tree_subcycle_advance_counts_2d(schedule) ==
              [1, 2, 4, 8]

        sync_counts = KrakeNN.conservative_tree_subcycle_sync_counts_2d(schedule)
        for level in 0:2
            expected = 2^level
            @test sync_counts[(:sync_down, level, level + 1)] == expected
            @test sync_counts[(:sync_up, level + 1, level)] == expected
        end

        tick0 = KrakeNN.conservative_tree_subcycle_events_at_tick_2d(schedule, 0)
        @test [(event.phase, event.src_level, event.dst_level)
               for event in tick0] == [(:sync_down, 0, 1),
                                       (:sync_down, 1, 2),
                                       (:sync_down, 2, 3)]

        tick2 = KrakeNN.conservative_tree_subcycle_events_at_tick_2d(schedule, 2)
        @test (:advance, 3, 3) in
              [(event.phase, event.src_level, event.dst_level)
               for event in tick2]
        @test (:sync_up, 3, 2) in
              [(event.phase, event.src_level, event.dst_level)
               for event in tick2]
        @test (:advance, 2, 2) in
              [(event.phase, event.src_level, event.dst_level)
               for event in tick2]
        @test (:sync_down, 2, 3) in
              [(event.phase, event.src_level, event.dst_level)
               for event in tick2]

        tick8 = KrakeNN.conservative_tree_subcycle_events_at_tick_2d(schedule, 8)
        @test [(event.phase, event.src_level, event.dst_level)
               for event in tick8][end-1:end] == [(:sync_up, 1, 0),
                                                  (:advance, 0, 0)]
    end

    @testset "schedule contracts reject invalid inputs" begin
        @test KrakeNN.create_conservative_tree_subcycle_schedule_2d(0).events ==
              [KrakeNN.ConservativeTreeSubcycleEvent2D(1, :advance, 0, 0)]
        @test_throws ArgumentError KrakeNN.create_conservative_tree_subcycle_schedule_2d(-1)
        @test_throws ArgumentError KrakeNN.create_conservative_tree_subcycle_schedule_2d(2; ratio=1)
        schedule = KrakeNN.create_conservative_tree_subcycle_schedule_2d(2)
        @test_throws ArgumentError KrakeNN.conservative_tree_subcycle_events_at_tick_2d(
            schedule, 5)
    end

    @testset "subcycle state buffers keep algorithm roles disjoint" begin
        spec = create_conservative_tree_spec_2d(8, 6, [
            ConservativeTreeRefineBlock2D("outer", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("inner", 7:8, 5:6; parent="outer"),
        ])
        bank = KrakeNN.create_conservative_tree_subcycle_buffer_bank_2d(spec)
        F = allocate_conservative_tree_F_2d(spec)
        for cell_id in spec.active_cells, q in 1:9
            F[cell_id, q] = 10 * spec.cells[cell_id].level + cell_id + q / 10
        end

        KrakeNN.conservative_tree_subcycle_store_active_owned_2d!(bank, F)
        level1_id = first(id for id in spec.active_cells
                          if spec.cells[id].level == 1)
        buffers = bank.levels[2]
        owned_before = buffers.owned[level1_id, 2]
        buffers.ghost_from_coarse[level1_id, 2] = 7.0
        buffers.reflux_to_coarse[level1_id, 2] = 3.0

        @test buffers.owned[level1_id, 2] == owned_before
        KrakeNN.conservative_tree_subcycle_apply_reflux_to_owned_level_2d!(
            bank, 1)
        @test buffers.owned[level1_id, 2] == owned_before + 3.0
        @test buffers.reflux_to_coarse[level1_id, 2] == 0.0
        @test buffers.ghost_from_coarse[level1_id, 2] == 7.0

        Frestored = allocate_conservative_tree_F_2d(spec)
        KrakeNN.conservative_tree_subcycle_restore_owned_level_2d!(
            Frestored, bank, 1)
        @test Frestored[level1_id, 2] == owned_before + 3.0
        level0_id = first(id for id in spec.active_cells
                          if spec.cells[id].level == 0)
        @test Frestored[level0_id, 2] == 0.0
    end

    @testset "subcycle restriction is conservative bottom-up" begin
        spec = create_conservative_tree_spec_2d(8, 6, [
            ConservativeTreeRefineBlock2D("outer", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("inner", 7:8, 5:6; parent="outer"),
        ])
        bank = KrakeNN.create_conservative_tree_subcycle_buffer_bank_2d(spec)
        for cell_id in spec.active_cells, q in 1:9
            level = spec.cells[cell_id].level
            bank.levels[level + 1].owned[cell_id, q] =
                cell_id + q / 10 + 100 * level
        end

        function active_descendant_sum(parent_id, q)
            children = spec.children[parent_id]
            if children == (0, 0, 0, 0)
                cell = spec.cells[parent_id]
                return cell.active ?
                    bank.levels[cell.level + 1].owned[parent_id, q] : 0.0
            end
            return sum(active_descendant_sum(child_id, q)
                       for child_id in children)
        end

        KrakeNN.conservative_tree_subcycle_restrict_all_levels_2d!(bank)
        for (cell_id, cell) in pairs(spec.cells)
            spec.children[cell_id] == (0, 0, 0, 0) && continue
            buffers = bank.levels[cell.level + 1]
            for q in 1:9
                @test buffers.restrict_to_parent[cell_id, q] ==
                      active_descendant_sum(cell_id, q)
            end
        end
    end

    @testset "subcycle coarse ghosts are conservative and non-owned" begin
        spec = create_conservative_tree_spec_2d(6, 5, [
            ConservativeTreeRefineBlock2D("fine", 3:4, 2:3),
        ])
        bank = KrakeNN.create_conservative_tree_subcycle_buffer_bank_2d(spec)
        Fparent = allocate_conservative_tree_F_2d(spec)
        parent_id = conservative_tree_cell_id_2d(spec, 0, 3, 2)
        children = conservative_tree_children_2d(spec, parent_id)
        for q in 1:9
            Fparent[parent_id, q] = 4q
        end

        KrakeNN.conservative_tree_subcycle_prolong_F_to_child_ghost_2d!(
            bank, Fparent, 0)
        child_buffers = bank.levels[2]
        for q in 1:9
            @test sum(child_buffers.ghost_from_coarse[collect(children), q]) ==
                  Fparent[parent_id, q]
            @test all(child_buffers.ghost_from_coarse[child_id, q] == q
                      for child_id in children)
            @test all(child_buffers.owned[child_id, q] == 0
                      for child_id in children)
        end
    end

    @testset "scheduler binds one L/L+1 interface ledger" begin
        schedule = KrakeNN.create_conservative_tree_subcycle_schedule_2d(1)
        bank = KrakeNN.create_conservative_tree_subcycle_ledger_bank_2d(schedule)
        down = only(event for event in schedule.events
                    if event.phase == :sync_down)
        up = only(event for event in schedule.events
                  if event.phase == :sync_up)
        advances = [event for event in schedule.events
                    if event.phase == :advance && event.src_level == 1]

        @test KrakeNN.conservative_tree_subcycle_local_substep_2d(
            schedule, 0, advances[1].tick) == 1
        @test KrakeNN.conservative_tree_subcycle_local_substep_2d(
            schedule, 0, advances[2].tick) == 2

        ledger = KrakeNN.conservative_tree_subcycle_sync_down_face_2d!(
            bank, down, 12.0, 2, :west)
        half1 = zeros(Float64, 2, 2, 9)
        half2 = zeros(Float64, 2, 2, 9)
        half1[2, 1, 2] = 1.25
        half1[2, 2, 2] = 2.75
        half2[2, 1, 2] = 2.0
        half2[2, 2, 2] = 3.0
        KrakeNN.conservative_tree_subcycle_accumulate_advance_face_2d!(
            bank, advances[1], half1, 2, :east)
        KrakeNN.conservative_tree_subcycle_accumulate_advance_face_2d!(
            bank, advances[2], half2, 2, :east)

        @test KrakeNN.conservative_tree_subcycle_sync_up_ledger_2d(bank, up) ===
              ledger
        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
        @test sums.coarse_to_fine[2] == 12.0
        @test sums.fine_to_coarse[2] == 9.0

        KrakeNN.reset_conservative_tree_subcycle_pair_2d!(bank, 0)
        ledger = KrakeNN.conservative_tree_subcycle_pair_ledger_2d(bank, 0)
        KrakeNN.conservative_tree_subcycle_sync_down_corner_2d!(
            bank, down, 7.0, 6, :southwest)
        corner_half = zeros(Float64, 2, 2, 9)
        corner_half[2, 2, 6] = 2.0
        for event in advances
            KrakeNN.conservative_tree_subcycle_accumulate_advance_corner_2d!(
                bank, event, corner_half, 6, :northeast)
        end
        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
        @test sums.coarse_to_fine[6] == 7.0
        @test sums.fine_to_coarse[6] == 4.0
    end

    @testset "scheduler binds all adjacent level-pair ledgers recursively" begin
        schedule = KrakeNN.create_conservative_tree_subcycle_schedule_2d(3)
        bank = KrakeNN.create_conservative_tree_subcycle_ledger_bank_2d(schedule)

        for event in schedule.events
            if event.phase == :sync_down
                Fq = 10.0 * (event.src_level + 1)
                KrakeNN.conservative_tree_subcycle_sync_down_face_2d!(
                    bank, event, Fq, 2, :west)
            elseif event.phase == :advance && event.src_level > 0
                half = zeros(Float64, 2, 2, 9)
                half[2, 1, 2] = 1.0
                half[2, 2, 2] = 1.0
                KrakeNN.conservative_tree_subcycle_accumulate_advance_face_2d!(
                    bank, event, half, 2, :east)
            elseif event.phase == :sync_up
                @test KrakeNN.conservative_tree_subcycle_sync_up_ledger_2d(
                    bank, event) ===
                      KrakeNN.conservative_tree_subcycle_pair_ledger_2d(
                          bank, event.dst_level)
            end
        end

        for parent in 0:2
            ledger = KrakeNN.conservative_tree_subcycle_pair_ledger_2d(bank, parent)
            sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
            sync_down_count = 2^parent
            child_advance_count = 2^(parent + 1)

            @test sums.coarse_to_fine[2] ==
                  sync_down_count * 10.0 * (parent + 1)
            @test sums.fine_to_coarse[2] == child_advance_count * 2.0
            @test ledger.fine_to_coarse[2, 1] == sync_down_count * 2.0
            @test ledger.fine_to_coarse[2, 2] == sync_down_count * 2.0
        end
    end

    @testset "spatial route ledgers apply one L/L+1 interface to F rows" begin
        spec = create_conservative_tree_spec_2d(6, 5, [
            ConservativeTreeRefineBlock2D("fine", 3:4, 2:3),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        schedule = KrakeNN.create_conservative_tree_subcycle_schedule_2d(spec.max_level)
        bank = KrakeNN.create_conservative_tree_subcycle_spatial_ledger_bank_2d(
            spec; schedule=schedule)
        down = only(event for event in schedule.events
                    if event.phase == :sync_down)
        up = only(event for event in schedule.events
                  if event.phase == :sync_up)
        advances = [event for event in schedule.events
                    if event.phase == :advance && event.src_level == 1]

        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        coarse_west = conservative_tree_cell_id_2d(spec, 0, 2, 2)
        refined_parent = conservative_tree_cell_id_2d(spec, 0, 3, 2)
        children = conservative_tree_children_2d(spec, refined_parent)
        Fin[coarse_west, 2] = 12.0
        split_expected = sum(route.weight * Fin[route.src, route.q]
                             for route in table.routes
                             if route.src == coarse_west &&
                                route.q == 2 &&
                                (route.kind == SPLIT_FACE ||
                                 route.kind == SPLIT_CORNER)) * 2

        KrakeNN.conservative_tree_subcycle_sync_down_routes_F_2d!(
            bank, down, Fin, table)
        KrakeNN.conservative_tree_subcycle_apply_child_advance_injection_F_2d!(
            Fout, bank, advances[1])
        @test isapprox(sum(Fout[collect(children), 2]), split_expected / 2;
                       atol=1e-14, rtol=0)
        KrakeNN.conservative_tree_subcycle_apply_child_advance_injection_F_2d!(
            Fout, bank, advances[2])
        @test isapprox(sum(Fout[collect(children), 2]), split_expected;
                       atol=1e-14, rtol=0)

        KrakeNN.reset_conservative_tree_subcycle_spatial_bank_2d!(bank)
        fill!(Fin, 0.0)
        fill!(Fout, 0.0)
        fine_west = conservative_tree_cell_id_2d(spec, 1, 5, 3)
        Fin[fine_west, 4] = 1.25
        KrakeNN.conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
            bank, advances[1], Fin, table)
        Fin[fine_west, 4] = 2.75
        KrakeNN.conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
            bank, advances[2], Fin, table)
        KrakeNN.conservative_tree_subcycle_apply_sync_up_F_2d!(
            Fout, bank, up)
        @test isapprox(Fout[coarse_west, 4], 2.0; atol=1e-14, rtol=0)
    end

    @testset "spatial route ledgers apply eq/neq alpha to interface packets" begin
        spec = create_conservative_tree_spec_2d(6, 5, [
            ConservativeTreeRefineBlock2D("fine", 3:4, 2:3),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        schedule = KrakeNN.create_conservative_tree_subcycle_schedule_2d(spec.max_level)
        bank = KrakeNN.create_conservative_tree_subcycle_spatial_ledger_bank_2d(
            spec; schedule=schedule)
        F = allocate_conservative_tree_F_2d(spec)
        coarse_west = conservative_tree_cell_id_2d(spec, 0, 2, 2)
        fill_equilibrium_integrated_D2Q9!(
            @view(F[coarse_west, :]), 1.0, 1.0, 0.03, 0.0)
        delta = 2e-4 / 4
        F[coarse_west, 6] += delta
        F[coarse_west, 7] -= delta
        F[coarse_west, 8] += delta
        F[coarse_west, 9] -= delta
        route = first(route for route in table.routes
                      if route.src == coarse_west &&
                         route.q == 6 &&
                         route.kind == SPLIT_CORNER)

        KrakeNN.conservative_tree_subcycle_deposit_coarse_to_fine_route_2d!(
            bank, F, route; alpha=0.25)
        parent_id = spec.cells[route.dst].parent
        ledger = KrakeNN.conservative_tree_subcycle_spatial_ledger_2d(
            bank, parent_id)

        expected = ledger.ratio * reconstructed_integrated_D2Q9_packet(
            @view(F[coarse_west, :]), 1.0, route.q, route.weight; alpha=0.25)
        raw = ledger.ratio * route.weight * F[coarse_west, route.q]
        @test isapprox(sum(ledger.coarse_to_fine[:, :, route.q, :]),
                       expected; atol=1e-14, rtol=0)
        @test abs(expected - raw) > 1e-6
    end

    @testset "spatial route ledgers recurse over all adjacent pairs" begin
        spec = create_conservative_tree_spec_2d(8, 6, [
            ConservativeTreeRefineBlock2D("outer", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("inner", 7:8, 5:6; parent="outer"),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        schedule = KrakeNN.create_conservative_tree_subcycle_schedule_2d(spec.max_level)
        bank = KrakeNN.create_conservative_tree_subcycle_spatial_ledger_bank_2d(
            spec; schedule=schedule)
        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        for cell_id in spec.active_cells
            for q in 1:9
                Fin[cell_id, q] = spec.cells[cell_id].metrics.volume
            end
        end

        for event in schedule.events
            if event.phase == :sync_down
                KrakeNN.conservative_tree_subcycle_sync_down_routes_F_2d!(
                    bank, event, Fin, table)
            elseif event.phase == :advance && event.src_level > 0
                KrakeNN.conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
                    bank, event, Fin, table)
                KrakeNN.conservative_tree_subcycle_apply_child_advance_injection_F_2d!(
                    Fout, bank, event)
            elseif event.phase == :sync_up
                KrakeNN.conservative_tree_subcycle_apply_sync_up_F_2d!(
                    Fout, bank, event)
            end
        end

        for parent_level in 0:(spec.max_level - 1)
            pair = KrakeNN.conservative_tree_subcycle_spatial_pair_ledgers_2d(
                bank, parent_level)
            @test !isempty(pair)
            @test sum(sum(ledger.coarse_to_fine) for ledger in values(pair)) > 0
            @test sum(sum(ledger.fine_to_coarse) for ledger in values(pair)) > 0
        end
        @test sum(Fout) > 0
    end

    @testset "subcycled transport matches route scatter without refinement" begin
        spec = create_conservative_tree_spec_2d(
            4, 4, ConservativeTreeRefineBlock2D[])
        table = create_conservative_tree_route_table_2d(spec)
        Fin = allocate_conservative_tree_F_2d(spec)
        Froute = allocate_conservative_tree_F_2d(spec)
        Fsub = allocate_conservative_tree_F_2d(spec)
        for cell_id in spec.active_cells
            for q in 1:9
                Fin[cell_id, q] = cell_id + q / 10
            end
        end

        stream_conservative_tree_routes_F_2d!(
            Froute, Fin, spec, table; boundary=:bounceback)
        KrakeNN.stream_conservative_tree_subcycled_routes_F_2d!(
            Fsub, Fin, spec, table; boundary=:bounceback)
        @test Fsub == Froute
    end

    @testset "buffered subcycled transport matches route scatter without refinement" begin
        spec = create_conservative_tree_spec_2d(
            4, 4, ConservativeTreeRefineBlock2D[])
        table = create_conservative_tree_route_table_2d(spec)
        Fin = allocate_conservative_tree_F_2d(spec)
        Froute = allocate_conservative_tree_F_2d(spec)
        Fsub = allocate_conservative_tree_F_2d(spec)
        for cell_id in spec.active_cells
            for q in 1:9
                Fin[cell_id, q] = 2cell_id + q / 7
            end
        end

        stream_conservative_tree_routes_F_2d!(
            Froute, Fin, spec, table; boundary=:bounceback)
        KrakeNN.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Fsub, Fin, spec, table; boundary=:bounceback)
        @test Fsub == Froute
    end

    @testset "single-level subcycled transport preserves closed rest mass" begin
        spec = create_conservative_tree_spec_2d(6, 5, [
            ConservativeTreeRefineBlock2D("fine", 3:4, 2:3),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        w = weights(D2Q9())
        for cell_id in spec.active_cells
            volume = spec.cells[cell_id].metrics.volume
            for q in 1:9
                Fin[cell_id, q] = w[q] * volume
            end
        end

        KrakeNN.stream_conservative_tree_subcycled_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:bounceback)

        @test isapprox(sum(active_population_sums_F_2d(Fout, spec)),
                       sum(active_population_sums_F_2d(Fin, spec));
                       atol=1e-12, rtol=0)
        @test maximum(abs.(Fout[spec.active_cells, :] .-
                           Fin[spec.active_cells, :])) <= 1e-14
        diag = KrakeNN.diagnose_conservative_tree_subcycled_rest_2d(spec, table)
        @test abs(diag.active_drift) <= 1e-12
        @test diag.max_active_abs <= 1e-14
        @test maximum(abs.(diag.level_drift)) <= 1e-12
        @test maximum(abs.(diag.orientation_drift)) <= 1e-12
    end

    @testset "single-level buffered subcycled transport preserves rest" begin
        spec = create_conservative_tree_spec_2d(6, 5, [
            ConservativeTreeRefineBlock2D("fine", 3:4, 2:3),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        w = weights(D2Q9())
        for cell_id in spec.active_cells
            volume = spec.cells[cell_id].metrics.volume
            for q in 1:9
                Fin[cell_id, q] = w[q] * volume
            end
        end

        KrakeNN.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:bounceback)

        @test isapprox(sum(active_population_sums_F_2d(Fout, spec)),
                       sum(active_population_sums_F_2d(Fin, spec));
                       atol=1e-12, rtol=0)
        @test maximum(abs.(Fout[spec.active_cells, :] .-
                           Fin[spec.active_cells, :])) <= 1e-14
    end

    @testset "nested subcycled transport rest state is the next closure gate" begin
        spec = create_conservative_tree_spec_2d(8, 6, [
            ConservativeTreeRefineBlock2D("outer", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("inner", 7:8, 5:6; parent="outer"),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        w = weights(D2Q9())
        for cell_id in spec.active_cells
            volume = spec.cells[cell_id].metrics.volume
            for q in 1:9
                Fin[cell_id, q] = w[q] * volume
            end
        end

        KrakeNN.stream_conservative_tree_subcycled_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:bounceback)
        diag = KrakeNN.diagnose_conservative_tree_subcycled_rest_2d(spec, table)

        @test_broken isapprox(sum(active_population_sums_F_2d(Fout, spec)),
                              sum(active_population_sums_F_2d(Fin, spec));
                              atol=1e-12, rtol=0)
        @test_broken maximum(abs.(Fout[spec.active_cells, :] .-
                                  Fin[spec.active_cells, :])) <= 1e-14
        @test_broken abs(diag.active_drift) <= 1e-12
        @test_broken diag.max_active_abs <= 1e-14
    end

    @testset "nested buffered subcycled transport preserves rest state" begin
        spec = create_conservative_tree_spec_2d(8, 6, [
            ConservativeTreeRefineBlock2D("outer", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("inner", 7:8, 5:6; parent="outer"),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        w = weights(D2Q9())
        for cell_id in spec.active_cells
            volume = spec.cells[cell_id].metrics.volume
            for q in 1:9
                Fin[cell_id, q] = w[q] * volume
            end
        end

        KrakeNN.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:bounceback)

        @test isapprox(sum(active_population_sums_F_2d(Fout, spec)),
                       sum(active_population_sums_F_2d(Fin, spec));
                       atol=1e-12, rtol=0)
        @test maximum(abs.(Fout[spec.active_cells, :] .-
                           Fin[spec.active_cells, :])) <= 1e-14
    end

    @testset "four-level buffered subcycled transport preserves rest state" begin
        spec = create_conservative_tree_spec_2d(16, 12, [
            ConservativeTreeRefineBlock2D("L1", 5:12, 3:10),
            ConservativeTreeRefineBlock2D("L2", 13:20, 7:14; parent="L1"),
            ConservativeTreeRefineBlock2D("L3", 29:36, 17:24; parent="L2"),
            ConservativeTreeRefineBlock2D("L4", 61:68, 37:44; parent="L3"),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        w = weights(D2Q9())
        for cell_id in spec.active_cells
            volume = spec.cells[cell_id].metrics.volume
            for q in 1:9
                Fin[cell_id, q] = w[q] * volume
            end
        end

        KrakeNN.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:bounceback)

        @test spec.max_level == 4
        @test isapprox(sum(active_population_sums_F_2d(Fout, spec)),
                       sum(active_population_sums_F_2d(Fin, spec));
                       atol=1e-12, rtol=0)
        @test maximum(abs.(Fout[spec.active_cells, :] .-
                           Fin[spec.active_cells, :])) <= 1e-14
    end

    @testset "four-level buffered periodic-x wall-y transport preserves rest" begin
        spec = create_conservative_tree_spec_2d(16, 12, [
            ConservativeTreeRefineBlock2D("L1", 5:12, 3:10),
            ConservativeTreeRefineBlock2D("L2", 13:20, 7:14; parent="L1"),
            ConservativeTreeRefineBlock2D("L3", 29:36, 17:24; parent="L2"),
            ConservativeTreeRefineBlock2D("L4", 61:68, 37:44; parent="L3"),
        ])
        table = create_conservative_tree_route_table_2d(spec; periodic_x=true)
        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        w = weights(D2Q9())
        for cell_id in spec.active_cells
            volume = spec.cells[cell_id].metrics.volume
            for q in 1:9
                Fin[cell_id, q] = w[q] * volume
            end
        end

        KrakeNN.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:periodic_x_wall_y)

        @test spec.max_level == 4
        @test isapprox(sum(active_population_sums_F_2d(Fout, spec)),
                       sum(active_population_sums_F_2d(Fin, spec));
                       atol=1e-12, rtol=0)
        @test maximum(abs.(Fout[spec.active_cells, :] .-
                           Fin[spec.active_cells, :])) <= 1e-14
    end

    @testset "full-domain nested Poiseuille matches Cartesian at same physical time" begin
        spec = _test_full_domain_nested_spec_2d(2)
        amr = run_conservative_tree_poiseuille_subcycled_2d(
            max_level=2, spec=spec, steps=12, Fx=1e-7, omega=1.0)
        cart_profile = _test_cartesian_poiseuille_profile_2d(
            2, 12; Fx=1e-7, omega=1.0)

        @test spec.max_level == 2
        @test length(spec.active_cells) == 16 * 12 * 4^2
        @test maximum(abs.(amr.ux_profile .- cart_profile)) < 1e-14
        @test amr.relative_mass_drift < 1e-13
    end

    @testset "short-time wall-normal Poiseuille improves with wall coverage" begin
        steps = 96
        center = run_conservative_tree_poiseuille_subcycled_2d(
            max_level=2, spec=_test_center_yband_nested_spec_2d(),
            steps=steps, Fx=1e-7, omega=1.0)
        walls = run_conservative_tree_poiseuille_subcycled_2d(
            max_level=2, spec=_test_wall_refined_ybands_nested_spec_2d(),
            steps=steps, Fx=1e-7, omega=1.0)
        cart = _test_cartesian_poiseuille_profile_2d(
            2, steps; Fx=1e-7, omega=1.0)

        center_diff = center.ux_profile .- cart
        walls_diff = walls.ux_profile .- cart
        center_l2 = sqrt(sum(center_diff .^ 2) / length(center_diff))
        walls_l2 = sqrt(sum(walls_diff .^ 2) / length(walls_diff))

        @test maximum(abs.(walls_diff)) < maximum(abs.(center_diff))
        @test walls_l2 < 0.2 * center_l2
        @test abs(maximum(walls.ux_profile) - maximum(cart)) <
              abs(maximum(center.ux_profile) - maximum(cart))
    end

    @testset "subcycled Poiseuille macroflow runs from level 1 to 4" begin
        for max_level in 1:4
            result = run_conservative_tree_poiseuille_subcycled_2d(
                max_level=max_level, steps=8, Fx=1e-7)
            cart = run_cartesian_channel_mass_reference_2d(
                flow=:poiseuille, max_level=max_level, steps=8, Fx=1e-7)
            guard = conservative_tree_mass_roundoff_rtol_2d(
                Float64, result.steps, max_level)
            @test result.flow == :poiseuille_subcycled
            @test result.max_level == max_level
            @test result.steps == 8
            @test all(isfinite, result.ux_profile)
            @test all(isfinite, result.analytic_profile)
            @test isfinite(result.l2_error)
            @test isfinite(result.linf_error)
            @test result.max_raw_relative_mass_drift <= guard
            @test result.relative_mass_drift <=
                  max(cart.relative_mass_drift, 10eps(Float64))
            @test maximum(result.ux_profile) > 0
            @test result.active_cell_count < result.leaf_equivalent_cell_count
        end
    end

    @testset "mass roundoff guard scales with active leaf count" begin
        small = conservative_tree_mass_roundoff_rtol_2d(
            Float64, 1, 4; active_cell_count=1)
        band = conservative_tree_mass_roundoff_rtol_2d(
            Float64, 1, 4; active_cell_count=20_000)
        @test band > small
        @test band < 20 * small
    end

    @testset "leaf-equivalent physics scales by AMR-D level" begin
        spec = KrakeNN.create_conservative_tree_nested_channel_spec_2d(4)
        omega_fine = 1.0
        force_fine = 1e-7

        @test KrakeNN.conservative_tree_leaf_equivalent_level_scale_2d(
            spec, 4) == 1
        @test KrakeNN.conservative_tree_leaf_equivalent_level_scale_2d(
            spec, 0) == 16
        @test KrakeNN.conservative_tree_leaf_equivalent_force_2d(
            force_fine, spec, 0) == 16force_fine
        @test KrakeNN.conservative_tree_leaf_equivalent_force_2d(
            force_fine, spec, 4) == force_fine

        tau_fine = inv(omega_fine)
        tau_coarse = inv(KrakeNN.conservative_tree_leaf_equivalent_omega_2d(
            omega_fine, spec, 0))
        @test tau_coarse - 0.5 ≈ (tau_fine - 0.5) / 16
        @test KrakeNN.conservative_tree_leaf_equivalent_omega_2d(
            omega_fine, spec, 4) == omega_fine
        @test_throws ArgumentError KrakeNN.conservative_tree_leaf_equivalent_omega_2d(
            2.0, spec, 0)
    end

    @testset "subcycled Couette macroflow runs from level 1 to 4" begin
        for max_level in 1:4
            result = run_conservative_tree_couette_subcycled_2d(
                max_level=max_level, steps=8, U=1e-4)
            cart = run_cartesian_channel_mass_reference_2d(
                flow=:couette, max_level=max_level, steps=8, U=1e-4)
            guard = conservative_tree_mass_roundoff_rtol_2d(
                Float64, result.steps, max_level)
            @test result.flow == :couette_subcycled
            @test result.max_level == max_level
            @test all(isfinite, result.ux_profile)
            @test isfinite(result.l2_error)
            @test isfinite(result.linf_error)
            @test result.max_raw_relative_mass_drift <= guard
            @test result.relative_mass_drift <=
                  max(cart.relative_mass_drift, 10eps(Float64))
            @test result.ux_profile[end] > result.ux_profile[1]
        end
    end

    @testset "subcycled macroflow compiles with Float32 storage" begin
        result = run_conservative_tree_poiseuille_subcycled_2d(
            max_level=1, steps=1, Fx=Float32(1e-7), T=Float32)
        @test eltype(result.F) == Float32
        @test eltype(result.ux_profile) == Float32
        @test all(isfinite, result.ux_profile)
        @test result.max_raw_relative_mass_drift <=
              conservative_tree_mass_roundoff_rtol_2d(Float32, 1, 1)
    end

    @testset "scheduled ledger binding rejects wrong events" begin
        schedule = KrakeNN.create_conservative_tree_subcycle_schedule_2d(2)
        bank = KrakeNN.create_conservative_tree_subcycle_ledger_bank_2d(schedule)
        down = first(event for event in schedule.events
                     if event.phase == :sync_down)
        advance0 = first(event for event in schedule.events
                         if event.phase == :advance && event.src_level == 0)
        advance1 = first(event for event in schedule.events
                         if event.phase == :advance && event.src_level == 1)

        half = zeros(Float64, 2, 2, 9)
        @test_throws ArgumentError KrakeNN.conservative_tree_subcycle_sync_down_face_2d!(
            bank, advance1, 1.0, 2, :west)
        @test_throws ArgumentError KrakeNN.conservative_tree_subcycle_accumulate_advance_face_2d!(
            bank, down, half, 2, :east)
        @test_throws ArgumentError KrakeNN.conservative_tree_subcycle_accumulate_advance_face_2d!(
            bank, advance0, half, 2, :east)
        @test_throws ArgumentError KrakeNN.conservative_tree_subcycle_sync_up_ledger_2d(
            bank, down)
        @test_throws ArgumentError KrakeNN.conservative_tree_subcycle_pair_ledger_2d(
            bank, 2)
        @test_throws ArgumentError KrakeNN.conservative_tree_subcycle_local_substep_2d(
            schedule, 0, 1)
    end

    @testset "coarse-to-fine face packet is consumed once over two half steps" begin
        ledger = create_conservative_tree_subcycle_ledger_2d()
        conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(
            ledger, 12.0, 2, :west)
        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
        totals = conservative_tree_subcycle_total_sums_2d(ledger)

        @test ledger.ratio == 2
        @test conservative_tree_subcycle_weights_2d(ledger) == [0.5, 0.5]
        @test sums.coarse_to_fine[2] == 12.0
        @test sum(sums.coarse_to_fine) == 12.0
        @test all(iszero, sums.coarse_to_fine[[1, 3, 4, 5, 6, 7, 8, 9]])
        @test totals.coarse_to_fine == 12.0
        @test totals.fine_to_coarse == 0.0

        for substep in 1:2
            @test ledger.coarse_to_fine[1, 1, 2, substep] == 3.0
            @test ledger.coarse_to_fine[1, 2, 2, substep] == 3.0
            @test sum(ledger.coarse_to_fine[:, :, 2, substep]) == 6.0
        end
    end

    @testset "coarse-to-fine corner packet is split only in time" begin
        ledger = create_conservative_tree_subcycle_ledger_2d()
        conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(
            ledger, 7.0, 6, :southwest)
        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)

        @test sums.coarse_to_fine[6] == 7.0
        @test sum(sums.coarse_to_fine) == 7.0
        for substep in 1:2
            @test ledger.coarse_to_fine[1, 1, 6, substep] == 3.5
            @test sum(ledger.coarse_to_fine[:, :, 6, substep]) == 3.5
        end
    end

    @testset "fine-to-coarse face packets accumulate by half step" begin
        ledger = create_conservative_tree_subcycle_ledger_2d()
        half1 = zeros(Float64, 2, 2, 9)
        half2 = zeros(Float64, 2, 2, 9)
        half1[2, 1, 2] = 1.25
        half1[2, 2, 2] = 2.75
        half2[2, 1, 2] = 2.0
        half2[2, 2, 2] = 3.0

        conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
            ledger, half1, 2, :east, 1)
        conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
            ledger, half2, 2, :east, 2)
        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)

        @test ledger.fine_to_coarse[2, 1] == 4.0
        @test ledger.fine_to_coarse[2, 2] == 5.0
        @test sums.fine_to_coarse[2] == 9.0
        @test sum(sums.fine_to_coarse) == 9.0
        @test sum(sums.coarse_to_fine) == 0.0
    end

    @testset "full cycle ledger preserves expected orientation totals" begin
        ledger = create_conservative_tree_subcycle_ledger_2d()
        conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(
            ledger, 6.0, 2, :west)
        conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(
            ledger, 4.0, 6, :southwest)

        face_half = zeros(Float64, 2, 2, 9)
        face_half[2, 1, 2] = 1.5
        face_half[2, 2, 2] = 1.5
        corner_half = zeros(Float64, 2, 2, 9)
        corner_half[2, 2, 6] = 2.0
        for substep in 1:2
            conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
                ledger, face_half, 2, :east, substep)
            conservative_tree_subcycle_accumulate_fine_to_coarse_corner_2d!(
                ledger, corner_half, 6, :northeast, substep)
        end

        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
        totals = conservative_tree_subcycle_total_sums_2d(ledger)
        @test sums.coarse_to_fine[2] == 6.0
        @test sums.coarse_to_fine[6] == 4.0
        @test sums.fine_to_coarse[2] == 6.0
        @test sums.fine_to_coarse[6] == 4.0
        @test totals.coarse_to_fine == 10.0
        @test totals.fine_to_coarse == 10.0

        reset_conservative_tree_subcycle_ledger_2d!(ledger)
        empty = conservative_tree_subcycle_total_sums_2d(ledger)
        @test empty.coarse_to_fine == 0.0
        @test empty.fine_to_coarse == 0.0
    end

    @testset "rest equilibrium: axis face deposit and accumulate balance per cycle" begin
        # Setup: a coarse cell at rest equilibrium (rho=1, V_c=1) sends its q-th
        # population through the entry face into the fine patch. The fine patch
        # children are at fine equilibrium (rho=1, V_f=0.25) and contribute back
        # via the opposite direction through the same physical face over 2 sub-steps.
        # The ledger must reflect a balanced cycle for axis directions q in 2:5
        # because a face has 2 children on each side and a cycle has 2 sub-steps:
        #   coarse->fine[q]   = w_q * V_c
        #   fine->coarse[opp] = 2 children * w_opp * V_f * 2 sub-steps = w_opp * V_c
        # With w_q == w_opp for axis directions, both totals are equal.
        Vc = 1.0
        Vf = 0.25
        wq_axis = 1 / 9  # D2Q9 weight for axis directions
        face_in_for = Dict(2 => :west, 3 => :south, 4 => :east, 5 => :north)
        face_out_for = Dict(2 => :east, 3 => :north, 4 => :west, 5 => :south)

        for q in (2, 3, 4, 5)
            opp = d2q9_opposite(q)
            ledger = create_conservative_tree_subcycle_ledger_2d()

            Fq_eq_coarse = wq_axis * Vc
            conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(
                ledger, Fq_eq_coarse, q, face_in_for[q])

            fine_block = zeros(Float64, 2, 2, 9)
            for ic in 1:2, jc in 1:2
                fine_block[ic, jc, opp] = wq_axis * Vf
            end
            for substep in 1:2
                conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
                    ledger, fine_block, opp, face_out_for[opp], substep)
            end

            sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
            @test isapprox(sums.coarse_to_fine[q], wq_axis * Vc;
                           atol=1e-15, rtol=0)
            @test isapprox(sums.fine_to_coarse[opp], wq_axis * Vc;
                           atol=1e-15, rtol=0)
            @test isapprox(sums.coarse_to_fine[q], sums.fine_to_coarse[opp];
                           atol=1e-15, rtol=0)
            # All other orientations must be zero.
            for qz in 1:9
                qz == q && continue
                qz == opp && continue
                @test sums.coarse_to_fine[qz] == 0.0
                @test sums.fine_to_coarse[qz] == 0.0
            end
        end
    end

    @testset "rest equilibrium: corner deposit per substep matches fine equilibrium" begin
        # Setup: coarse cell sends its diagonal q (in 6:9) through one corner.
        # The ledger splits the packet evenly across 2 sub-steps; each sub-step's
        # corner deposit lands in 1 fine child cell. The deposit value per
        # sub-step (= w_q * V_c / 2) must equal the fine equilibrium of that q
        # (= w_q * V_f * 2) up to the well-known 2x corner geometric factor.
        # Concretely, ledger.coarse_to_fine[ix, iy, q, substep] = w_q * V_c / 2
        # = w_q * V_f * 2; the 2x prefactor reflects that 1 coarse step covers 2
        # fine half-steps so the per-sub-step deposit is twice the per-fine-cell
        # equilibrium. This is documented as a known imbalance that the time
        # integrator must compensate via Filippova-Hanel rescaling and reflux.
        Vc = 1.0
        Vf = 0.25
        wq_corner = 1 / 36
        corner_in_for = Dict(6 => :southwest, 7 => :southeast,
                             8 => :northeast, 9 => :northwest)
        corner_out_for = Dict(6 => :northeast, 7 => :northwest,
                              8 => :southwest, 9 => :southeast)

        for q in (6, 7, 8, 9)
            opp = d2q9_opposite(q)
            ledger = create_conservative_tree_subcycle_ledger_2d()

            Fq_eq_coarse = wq_corner * Vc
            conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(
                ledger, Fq_eq_coarse, q, corner_in_for[q])

            fine_block = zeros(Float64, 2, 2, 9)
            for ic in 1:2, jc in 1:2
                fine_block[ic, jc, opp] = wq_corner * Vf
            end
            for substep in 1:2
                conservative_tree_subcycle_accumulate_fine_to_coarse_corner_2d!(
                    ledger, fine_block, opp, corner_out_for[opp], substep)
            end

            sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
            # coarse->fine[q] = wq_corner * Vc (full coarse packet over 1 cycle)
            @test isapprox(sums.coarse_to_fine[q], wq_corner * Vc;
                           atol=1e-15, rtol=0)
            # fine->coarse[opp] = 1 child * wq_corner * Vf * 2 substeps
            #                   = wq_corner * Vc / 2
            @test isapprox(sums.fine_to_coarse[opp], wq_corner * Vc / 2;
                           atol=1e-15, rtol=0)
            # The 2x corner imbalance is INTRINSIC to the subcycling geometry,
            # not a ledger bug. The future time integrator must rescale
            # (Filippova-Hanel) or accept this as residual on coarse.
            @test sums.fine_to_coarse[opp] == sums.coarse_to_fine[q] / 2
        end
    end

    @testset "ledger rejects unsupported contracts" begin
        ledger = create_conservative_tree_subcycle_ledger_2d()
        bad_block = zeros(Float64, 3, 2, 9)

        @test_throws ArgumentError create_conservative_tree_subcycle_ledger_2d(ratio=3)
        @test_throws ArgumentError conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(
            ledger, 1.0, 2, :east)
        @test_throws ArgumentError conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(
            ledger, 1.0, 2, :southwest)
        @test_throws ArgumentError conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
            ledger, bad_block, 2, :east, 1)
        @test_throws ArgumentError conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
            ledger, zeros(Float64, 2, 2, 9), 2, :east, 3)
    end
end
