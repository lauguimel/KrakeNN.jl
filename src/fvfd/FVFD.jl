using KernelAbstractions

include("specs.jl")
include("lowering_2d.jl")
include("operators_2d.jl")
include("operators_3d.jl")
include("operators_3d_openbc.jl")
include("operators_3d_velocity_gradient.jl")
include("halfway_wall_gradient_correction_2d.jl")
include("muscl_boundary.jl")
# IncNS velocity-operator trio (gdl_* divergence/gradient/laplacian + embedded
# variants). Its FVFD_BC_* fallback guard no-ops here: specs.jl above already
# defines the same UInt8 codes with identical values.
include("operators_2d_grad_div_laplacian.jl")
