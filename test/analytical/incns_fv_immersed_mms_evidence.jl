# Evidence helpers shared by the registered test and the standalone receipt.
# A pair touching the roundoff budget is not a measured convergence order.
function _c5m_metric_budget(r, metric)
    t = _c5m_roundoff(r)
    startswith(string(metric), "trunc_") && return t*r.N^2
    (occursin("gradient",string(metric)) || occursin("traction",string(metric))) && return 4t*r.N
    return t
end
function _c5m_pair_order(a,b,metric)
    ea,eb = getproperty(a,metric),getproperty(b,metric)
    if ea <= _c5m_metric_budget(a,metric) || eb <= _c5m_metric_budget(b,metric)
        return "roundoff_limited"
    end
    return log(ea/eb)/log(b.N/a.N)
end
include(joinpath(@__DIR__,"incns_fv_immersed_mms_gate.jl"))
function _c5m_write_evidence(out,results,timings; a0_reference=_c5m_a0_reference())
    mkpath(out)
    orders, selections, final_rows = NamedTuple[], NamedTuple[], NamedTuple[]
    metrics = (:velocity,:pressure,:ghost,:face,:nominal_pressure,
        :cell_gradient,:surface_gradient,:surface_pressure,
        :traction_rms,:traction_linf,:pressure_traction_rms,
        :pressure_traction_linf,:viscous_traction_rms,:viscous_traction_linf,
        :trunc_cut_momentum,:trunc_cut_mass)
    for closure in (:linear,:quadratic), pressure in (:zero_gradient,:linear)
        group = filter(r->r.closure==closure && r.residual_pressure==pressure,results)
        current = NamedTuple[]
        for angle in _C5M_ANGLES, degree in (2,3), state in (:injected,:solved), metric in metrics
            rows = sort(filter(r->r.angle==angle && r.degree==degree && r.state==state,group);by=r->r.N)
            err = [getproperty(r,metric) for r in rows]
            pair = [_c5m_pair_order(rows[k],rows[k+1],metric) for k in 1:3]
            push!(current,(;closure,mass=:M1,residual_pressure=pressure,angle,degree,state,metric,
                error_16=err[1],error_32=err[2],error_64=err[3],error_128=err[4],
                order_16_32=pair[1],order_32_64=pair[2],order_64_128=pair[3]))
        end
        append!(orders,current)
        quad = filter(r->r.degree==2 && r.state==:solved && r.metric==:traction_rms,current)
        probe = filter(r->r.degree==3 && r.state==:solved && r.metric==:traction_rms,current)
        numeric_min(rows,key) = all(r->getproperty(r,key) isa Number,rows) ?
            minimum(getproperty(r,key) for r in rows) : "roundoff_limited"
        affine_rows = filter(r->r.degree==1 && r.state==:injected,group)
        solved_quad = filter(r->r.degree==2 && r.state==:solved,group)
        quadratic_exact = all(r->r.traction_rms<=_c5m_metric_budget(r,:traction_rms),solved_quad)
        gate_rows = quadratic_exact ? probe : quad
        gate_orders = [getproperty(r,k) for r in gate_rows for k in
            (:order_16_32,:order_32_64,:order_64_128)]
        original_selected = _c5m_affine_exact(group) && all(x->x isa Number,gate_orders) &&
            minimum(gate_orders)>=_C5M_ORDER_GATE
        revised = closure==:quadratic && pressure==:linear
        previous_selected = revised ? _c5m_revised_gate(group,a0_reference).checks.selected : original_selected
        final = _c5m_final_gate(group,a0_reference)
        append!(final_rows,final.rows)
        push!(selections,(;closure,mass=:M1,residual_pressure=pressure,
            affine_pass=_c5m_affine_exact(group),
            max_affine_residual=maximum(r.equation_residual for r in affine_rows),
            quadratic_roundoff=quadratic_exact,
            quadratic_min_order_16_32=numeric_min(quad,:order_16_32),
            quadratic_min_order_32_64=numeric_min(quad,:order_32_64),
            quadratic_min_order_64_128=numeric_min(quad,:order_64_128),
            cubic_min_order_16_32=numeric_min(probe,:order_16_32),
            cubic_min_order_32_64=numeric_min(probe,:order_32_64),
            cubic_min_order_64_128=numeric_min(probe,:order_64_128),
            original_traction_gate_field=quadratic_exact ? "cubic_probe" : "quadratic",
            original_selected,previous_selected,traction_gate_field="cubic_probe",
            criterion="final_A0_referenced_all_pressure_traction_components",
            selected=final.checks.selected))
    end
    group = filter(r->r.closure==:quadratic && r.mass==:M1 && r.residual_pressure==:linear,results)
    revised = _c5m_revised_gate(group,a0_reference)
    for (name,rows) in (("local_mms_final_gate.csv",final_rows),
                       ("local_mms_revised_gate.csv",revised.rows),
                       ("local_mms_viscous_gate.csv",revised.viscous))
        open(joinpath(out,name),"w") do io
            println(io,join(string.(keys(first(rows))),","))
            foreach(r->println(io,join(string.(values(r)),",")),rows)
        end
    end
    for (name,rows) in (("local_mms_metrics.csv",results),("local_mms_timings.csv",timings),
                       ("local_mms_orders.csv",orders),("local_mms_selection.csv",selections))
        open(joinpath(out,name),"w") do io
            println(io,join(string.(keys(first(rows))),","))
            foreach(r->println(io,join(string.(values(r)),",")),rows)
        end
    end
    # Every physical-square cut, once per geometry: no corner/q/angle filter.
    open(joinpath(out,"local_mms_links.csv"),"w") do io
        println(io,"N,angle,i,j,d,q,wall_x,wall_y,normal_x,normal_y")
        for N in _C5M_NS, angle in _C5M_ANGLES
            f = _c5m_patch(N,angle,:quadratic,:M1,:linear,1)
            for (i,j,d) in f.links
                x,y = f.xy(i,j)
                q,h = f.p.immersed.qcut[d,i,j],f.p.grid.h
                xw,yw = x+q*h*((d==1)-(d==2)),y+q*h*((d==3)-(d==4))
                println(io,join((N,angle,i,j,d,q,xw,yw,f.normal...),","))
            end
        end
    end
end
