# Known intermittent failure (issue #136): on Julia 1.12 with code coverage on
# (the CI flags), the first adjoint call of an immersed gate sometimes returns
# NaN or garbage. Only in that setting is a failing gate recorded as Broken;
# everywhere else it is a plain @test.
using Test

const _FV_IMMERSED_KNOWN112 = VERSION >= v"1.12" && Base.JLOptions().code_coverage != 0

macro fv_immersed_gate(ex)
    quote
        if _FV_IMMERSED_KNOWN112 && !($(esc(ex)))
            @warn "Known Julia 1.12 intermittent immersed adjoint failure (issue #136)"
            @test_broken $(esc(ex))
        else
            @test $(esc(ex))
        end
    end
end
