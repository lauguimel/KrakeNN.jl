function _fmt(x)
    x isa AbstractFloat && return string(x)
    return string(x)
end

function _markdown_report(plan::SimulationPlan)
    io = IOBuffer()
    println(io, "# KrakeNN.Units plan")
    println(io)
    println(io, "- source: `", plan.audit_source, "`")
    println(io, "- physics: `", typeof(plan.physics_spec), "`")
    println(io, "- geometry: `", plan.geometry.type, "` R_LU=", plan.units.R_LU)
    println(io, "- scaling: `", plan.units.scaling, "`")
    println(io, "- tau_hydro: ", _fmt(plan.units.tau_hydro))
    println(io, "- nu_total_LU: ", _fmt(plan.units.nu_total_LU))
    println(io, "- u_LU: ", _fmt(plan.units.u_LU))
    println(io, "- Ma: ", _fmt(plan.units.Ma))
    println(io, "- max_steps: ", plan.units.max_steps)
    if isfinite(plan.units.lambda_LU)
        println(io, "- lambda_LU: ", _fmt(plan.units.lambda_LU))
    end
    if isempty(plan.warnings)
        println(io, "- issues: none")
    else
        println(io, "- issues:")
        for issue in plan.warnings
            println(io, "  - `", issue.severity, "` `", issue.code, "`: ", issue.message)
        end
    end
    return String(take!(io))
end

function _json_escape(s::AbstractString)
    return replace(replace(replace(s, "\\" => "\\\\"), "\"" => "\\\""), "\n" => "\\n")
end

function _jsonl_report(plan::SimulationPlan)
    rows = String[]
    push!(rows, "{\"kind\":\"plan\",\"source\":\"$(plan.audit_source)\",\"scaling\":\"$(plan.units.scaling)\",\"tau_hydro\":$(plan.units.tau_hydro),\"nu_total_LU\":$(plan.units.nu_total_LU),\"u_LU\":$(plan.units.u_LU),\"R_LU\":$(plan.units.R_LU)}")
    for issue in plan.warnings
        push!(rows, "{\"kind\":\"issue\",\"severity\":\"$(issue.severity)\",\"code\":\"$(issue.code)\",\"message\":\"$(_json_escape(issue.message))\"}")
    end
    return join(rows, "\n")
end

function report(plan::SimulationPlan; io=stdout, format::Symbol=:markdown)
    text = format === :markdown ? _markdown_report(plan) :
           format === :jsonl ? _jsonl_report(plan) :
           throw(ArgumentError("report format must be :markdown or :jsonl"))
    io === nothing && return text
    print(io, text)
    return nothing
end
