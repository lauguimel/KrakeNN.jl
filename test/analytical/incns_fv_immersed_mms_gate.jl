# Historical revised and final local selection contracts, authorized after A0.
# The stored A0 baseline avoids a duplicate solve in the ordinary MMS suite.
const _C5M_REVISED_ORDER = 1.8
const _C5M_A0_MARGIN = 0.2
const _C5M_A0_MAX_PRESSURE_ORDER = 1.3
const _C5M_REVISED_PAIRS = ((16,32),(32,64),(64,128))

function _c5m_a0_reference()
    path = joinpath(@__DIR__, "..", "..", "benchmarks", "fv_c5", "aligned_c1_orders.csv")
    lines = readlines(path)
    columns = split(first(lines),',')
    required = ("velocity_degree","pressure_degree","state","metric","coarse","fine",
        "error_coarse","error_fine","order")
    all(k->count(==(k),columns)==1,required) || error("A0 baseline has an invalid schema")
    refs = Dict{Tuple{Int,Symbol,Int,Int},Float64}()
    for line in lines[2:end]
        values = split(line,',')
        length(values)==length(columns) || error("A0 baseline has an invalid row")
        row = Dict(zip(columns,values))
        row["velocity_degree"]=="3" && row["pressure_degree"] in ("1","2") &&
            row["state"]=="solved" || continue
        pd = parse(Int,row["pressure_degree"])
        metric = Symbol(row["metric"])
        metric in (:pressure,:traction_rms,:pressure_traction_rms,:viscous_traction_rms) || continue
        pd==1 && metric!=:pressure && continue
        pair = (parse(Int,row["coarse"]),parse(Int,row["fine"]))
        pair in _C5M_REVISED_PAIRS || error("A0 baseline has an unexpected refinement pair")
        key = (pd,metric,pair...)
        haskey(refs,key) && error("A0 baseline has a duplicate probe/pair")
        ea,eb,order = parse.(Float64,(row["error_coarse"],row["error_fine"],row["order"]))
        all(isfinite,(ea,eb,order)) && ea>0 && eb>0 || error("A0 baseline has a nonnumeric measured pair")
        isapprox(order,log(ea/eb)/log(pair[2]/pair[1]);rtol=8eps(),atol=8eps()) ||
            error("A0 baseline order does not match its errors")
        refs[key] = order
    end
    length(refs)==15 || error("A0 baseline requires both pressure probes and all quadratic-pressure traction components on all pairs")
    all(refs[(pd,:pressure,pair...)] <= _C5M_A0_MAX_PRESSURE_ORDER
        for pd in (1,2) for pair in _C5M_REVISED_PAIRS) ||
        error("A0 pressure order exceeds 1.3 for at least one probe/pair: revised selection must STOP")
    return (; source="benchmarks/fv_c5/aligned_c1_orders.csv",velocity_degree=3,pressure_degree=2,
        pressure=[refs[(2,:pressure,pair...)] for pair in _C5M_REVISED_PAIRS],
        linear_pressure=[refs[(1,:pressure,pair...)] for pair in _C5M_REVISED_PAIRS],
        pressure_traction=[refs[(2,:pressure_traction_rms,pair...)] for pair in _C5M_REVISED_PAIRS],
        viscous=[refs[(2,:viscous_traction_rms,pair...)] for pair in _C5M_REVISED_PAIRS],
        traction=[refs[(2,:traction_rms,pair...)] for pair in _C5M_REVISED_PAIRS])
end

# Final contract: every pair must meet every component's matching aligned C1
# order minus 0.2. The historical 1.8-on-two-viscous-pairs gate remains
# reproducible; it is not relabelled as passing under its original criterion.
function _c5m_final_criteria(affine,velocity,pressure,pressure_traction,viscous,traction,reference)
    measured = (;pressure,pressure_traction,viscous,traction)
    all(length(x)==3 for x in (velocity,values(measured)...)) &&
        all(length(getproperty(reference,k))==3 for k in keys(measured)) ||
        error("final gate requires all three refinement pairs for every metric")
    velocity_pass = [_c5m_meets(x,_C5M_REVISED_ORDER) for x in velocity]
    matched = map(keys(measured)) do metric
        [_c5m_meets(getproperty(measured,metric)[k],
            getproperty(reference,metric)[k]-_C5M_A0_MARGIN) for k in 1:3]
    end
    pressure_pass,pressure_traction_pass,viscous_pass,traction_pass = matched
    selected = affine && all(velocity_pass) && all(all,matched)
    return (;affine,velocity_pass,pressure_pass,pressure_traction_pass,viscous_pass,traction_pass,selected)
end

function _c5m_final_gate(group,reference)
    isempty(group) && error("final gate requires measured evidence")
    firstrow = first(group)
    all(r->(r.closure,r.mass,r.residual_pressure)==
        (firstrow.closure,:M1,firstrow.residual_pressure),group) ||
        error("final gate requires one M1 closure/pressure configuration")
    velocity = _c5m_gate_orders(group,:velocity)
    pressure = _c5m_gate_orders(group,:pressure)
    pressure_traction = _c5m_gate_orders(group,:pressure_traction_rms)
    viscous = _c5m_gate_orders(group,:viscous_traction_rms)
    traction = _c5m_gate_orders(group,:traction_rms)
    checks = _c5m_final_criteria(_c5m_affine_exact(group),velocity.minima,pressure.minima,
        pressure_traction.minima,viscous.minima,traction.minima,reference)
    rows = NamedTuple[]
    for (metric,orders,baseline,passes) in (
            (:velocity,velocity,nothing,checks.velocity_pass),
            (:pressure,pressure,reference.pressure,checks.pressure_pass),
            (:pressure_traction_rms,pressure_traction,reference.pressure_traction,checks.pressure_traction_pass),
            (:viscous_traction_rms,viscous,reference.viscous,checks.viscous_pass),
            (:traction_rms,traction,reference.traction,checks.traction_pass))
        for (k,(coarse,fine)) in enumerate(_C5M_REVISED_PAIRS)
            push!(rows,(;closure=firstrow.closure,mass=:M1,residual_pressure=firstrow.residual_pressure,
                coarse,fine,reference=reference.source,velocity_degree=3,pressure_degree=2,
                affine_pass=checks.affine,metric,min_order=orders.minima[k],
                aligned_c1_order=isnothing(baseline) ? "not_used" : baseline[k],
                margin=isnothing(baseline) ? 0.0 : _C5M_A0_MARGIN,
                threshold=isnothing(baseline) ? _C5M_REVISED_ORDER : baseline[k]-_C5M_A0_MARGIN,
                pass=passes[k],selected=checks.selected))
        end
    end
    return (;checks,rows)
end

_c5m_meets(x,t) = x isa Real && isfinite(x) && x>=t
function _c5m_revised_criteria(affine,velocity,viscous,pressure,traction,reference)
    all(length(x)==3 for x in (velocity,viscous,pressure,traction)) ||
        error("revised gate requires all three refinement pairs")
    velocity_pass = [_c5m_meets(x,_C5M_REVISED_ORDER) for x in velocity]
    viscous_pass = [_c5m_meets(x,_C5M_REVISED_ORDER) for x in viscous]
    pressure_pass = [_c5m_meets(pressure[k],reference.pressure[k]-_C5M_A0_MARGIN) for k in 1:3]
    traction_pass = [_c5m_meets(traction[k],reference.traction[k]-_C5M_A0_MARGIN) for k in 1:3]
    viscous_pairs = count(viscous_pass)
    selected = affine && all(velocity_pass) && viscous_pairs>=2 && all(pressure_pass) && all(traction_pass)
    return (;affine,velocity_pass,viscous_pass,pressure_pass,traction_pass,viscous_pairs,selected)
end

function _c5m_gate_orders(group,metric)
    pairs = NamedTuple[]
    for angle in _C5M_ANGLES
        rows = sort(filter(r->r.degree==3 && r.state==:solved && r.angle==angle,group);by=r->r.N)
        [r.N for r in rows] == collect(_C5M_NS) || error("gate requires every grid once at every angle")
        for k in 1:3
            a,b = rows[k],rows[k+1]
            ea,eb = getproperty(a,metric),getproperty(b,metric)
            ba,bb = _c5m_metric_budget(a,metric),_c5m_metric_budget(b,metric)
            push!(pairs,(;angle,degree=3,state=:solved,metric,coarse=a.N,fine=b.N,
                error_coarse=ea,error_fine=eb,budget_coarse=ba,budget_fine=bb,
                error_to_budget_coarse=ea/ba,error_to_budget_fine=eb/bb,
                order=_c5m_pair_order(a,b,metric)))
        end
    end
    minima = map(_C5M_REVISED_PAIRS) do (coarse,fine)
        orders = [r.order for r in pairs if r.coarse==coarse && r.fine==fine]
        all(x->x isa Real && isfinite(x),orders) ? minimum(orders) : "roundoff_limited"
    end
    return (;minima,pairs)
end

function _c5m_revised_gate(group,reference)
    all(r->r.closure==:quadratic && r.mass==:M1 && r.residual_pressure==:linear,group) ||
        error("revised gate is defined only for Q/M1/linear")
    velocity = _c5m_gate_orders(group,:velocity)
    viscous = _c5m_gate_orders(group,:viscous_traction_rms)
    pressure = _c5m_gate_orders(group,:pressure)
    traction = _c5m_gate_orders(group,:traction_rms)
    checks = _c5m_revised_criteria(_c5m_affine_exact(group),velocity.minima,viscous.minima,
        pressure.minima,traction.minima,reference)
    rows = [(;coarse,fine,reference=reference.source,velocity_degree=3,pressure_degree=2,
        affine_pass=checks.affine,velocity_min_order=velocity.minima[k],
        velocity_threshold=_C5M_REVISED_ORDER,velocity_pass=checks.velocity_pass[k],
        viscous_min_order=viscous.minima[k],viscous_threshold=_C5M_REVISED_ORDER,
        viscous_pass=checks.viscous_pass[k],viscous_passing_pairs=checks.viscous_pairs,
        viscous_required_pairs=2,pressure_min_order=pressure.minima[k],
        pressure_a0_order=reference.pressure[k],pressure_margin=_C5M_A0_MARGIN,
        pressure_pass=checks.pressure_pass[k],traction_min_order=traction.minima[k],
        traction_a0_order=reference.traction[k],traction_margin=_C5M_A0_MARGIN,
        traction_pass=checks.traction_pass[k],selected=checks.selected)
        for (k,(coarse,fine)) in enumerate(_C5M_REVISED_PAIRS)]
    return (;checks,rows,viscous=viscous.pairs)
end
