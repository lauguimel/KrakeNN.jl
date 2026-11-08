"""
    source_extract.jl

JuliaSyntax-based source code extraction helpers for the KrakeNN
living documentation. Used by Phases 4.2-4.6 to embed real source code
from `src/` into Literate.jl pages, keeping theory/code synchronized.
"""

using JuliaSyntax

# ---------- internal tree helpers ----------

"""Return the Symbol name of a `function` or `struct` SyntaxNode, or `nothing`."""
function _node_name(n)
    k = kind(n)
    ch = JuliaSyntax.children(n)
    ch === nothing && return nothing
    if k == K"function" || k == K"="
        isempty(ch) && return nothing
        sig = ch[1]
        while sig !== nothing && kind(sig) in (K"where", K"::")
            sc = JuliaSyntax.children(sig)
            (sc === nothing || isempty(sc)) && return nothing
            sig = sc[1]
        end
        sig === nothing && return nothing
        if kind(sig) == K"call"
            sc = JuliaSyntax.children(sig)
            (sc === nothing || isempty(sc)) && return nothing
            name = sc[1]
            while kind(name) == K"."
                nc = JuliaSyntax.children(name)
                (nc === nothing || isempty(nc)) && break
                name = nc[end]
            end
            return Symbol(strip(string(name)))
        elseif kind(sig) == K"Identifier"
            return Symbol(strip(string(sig)))
        end
    elseif k == K"struct"
        isempty(ch) && return nothing
        nm = ch[1]
        while nm !== nothing && kind(nm) in (K"<:", K"curly")
            nc = JuliaSyntax.children(nm)
            (nc === nothing || isempty(nc)) && return nothing
            nm = nc[1]
        end
        nm === nothing && return nothing
        return Symbol(strip(string(nm)))
    end
    return nothing
end

"""
Walk top-level declarations. For each function/struct (possibly wrapped
in a `doc` node), call `f(name, node, docstring_or_nothing)`.
"""
function _each_toplevel(f, tree)
    ch = JuliaSyntax.children(tree)
    ch === nothing && return
    for c in ch
        k = kind(c)
        if k == K"doc"
            dc = JuliaSyntax.children(c)
            (dc === nothing || length(dc) < 2) && continue
            docstr = dc[1]
            inner = dc[2]
            name = _node_name(inner)
            name === nothing && continue
            f(name, inner, docstr)
        elseif k == K"function" || k == K"struct"
            name = _node_name(c)
            name === nothing && continue
            f(name, c, nothing)
        end
    end
end

function _byte_slice(src::String, n)
    r = JuliaSyntax.byte_range(n)
    # Slice by raw UTF-8 bytes: JuliaSyntax byte ranges may end on the last
    # byte of a multi-byte char (e.g. `λ`), which is not a valid String index.
    return String(codeunits(src)[first(r):last(r)])
end

function _line_of(src::String, byte::Int)
    # 1-indexed line number for a byte offset
    return count(==('\n'), @view src[1:min(byte, lastindex(src))]) + 1
end

# ---------- public API ----------

"""
    extract_function(filepath, name; with_docstring=true, language="julia")

Extract the source code of a top-level function (or struct) named `name`
from a Julia source file.

# Arguments
- `filepath::String`: path to the `.jl` file.
- `name::Symbol`: function or struct name to locate.
- `with_docstring::Bool=true`: include the `\"\"\"...\"\"\"` docstring if present.
- `language::String="julia"`: language tag for the returned fenced code block.

# Returns
A `NamedTuple` with fields:
- `signature::String`  — first line of the definition (best-effort).
- `docstring::String`  — the docstring text (empty if none or disabled).
- `body::String`       — the full `function ... end` / `struct ... end` source.
- `full_text::String`  — markdown-ready fenced code block (docstring + body).
- `line_start::Int`, `line_end::Int` — line range in the source file.

# Examples
```julia
r = extract_function("src/drivers/basic.jl", :run_cavity_2d)
println(r.full_text)
```
"""
function extract_function(filepath::String, name::Symbol;
                          with_docstring::Bool=true, language::String="julia")
    src = read(filepath, String)
    tree = parseall(SyntaxNode, src; filename=basename(filepath))
    found_node = nothing
    found_doc = nothing
    _each_toplevel(tree) do nm, node, docstr
        if found_node === nothing && nm === name
            found_node = node
            found_doc = docstr
        end
    end
    if found_node === nothing
        error("extract_function: symbol $(name) not found in $(filepath)")
    end
    body = _byte_slice(src, found_node)
    docstring = ""
    if with_docstring && found_doc !== nothing
        docstring = _byte_slice(src, found_doc)
    end
    signature = first(split(body, '\n'))
    r = JuliaSyntax.byte_range(found_node)
    line_start = _line_of(src, first(r))
    line_end = _line_of(src, last(r))
    combined = isempty(docstring) ? body : (docstring * "\n" * body)
    full_text = "```" * language * "\n" * combined * "\n```"
    return (; signature, docstring, body, full_text, line_start, line_end)
end

"""
    extract_struct(filepath, name; kwargs...)

Extract the source of a `struct` / `mutable struct` / `Base.@kwdef` struct
by name. Same return shape as [`extract_function`](@ref).
"""
function extract_struct(filepath::String, name::Symbol; kwargs...)
    return extract_function(filepath, name; kwargs...)
end

"""
    include_function(filepath, name; kwargs...) -> String

Convenience wrapper that returns only the markdown-ready `full_text` string
from [`extract_function`](@ref). Designed for direct interpolation inside
a Literate.jl page.
"""
function include_function(filepath::String, name::Symbol; kwargs...)
    return extract_function(filepath, name; kwargs...).full_text
end
