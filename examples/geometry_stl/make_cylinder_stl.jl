const DEFAULT_RADIUS = 0.2
const DEFAULT_CX = 1.0
const DEFAULT_CY = 0.5
const DEFAULT_Z0 = 0.0
const DEFAULT_Z1 = 1.0
const DEFAULT_SEGMENTS = 64

function _write_triangle(io, normal, v1, v2, v3)
    for value in normal
        write(io, Float32(value))
    end
    for vertex in (v1, v2, v3), value in vertex
        write(io, Float32(value))
    end
    write(io, UInt16(0))
    return nothing
end

function write_cylinder_stl(filename::AbstractString;
                            radius::Real=DEFAULT_RADIUS,
                            cx::Real=DEFAULT_CX,
                            cy::Real=DEFAULT_CY,
                            z0::Real=DEFAULT_Z0,
                            z1::Real=DEFAULT_Z1,
                            segments::Integer=DEFAULT_SEGMENTS)
    segments >= 3 || throw(ArgumentError("segments must be >= 3"))
    z0 < z1 || throw(ArgumentError("z0 must be strictly less than z1"))

    mkpath(dirname(abspath(filename)))
    open(filename, "w") do io
        header = zeros(UInt8, 80)
        label = codeunits("KrakeNN M-GEO-1 closed cylinder prism")
        header[1:length(label)] .= label
        write(io, header)
        write(io, UInt32(4 * segments))

        cb = (Float64(cx), Float64(cy), Float64(z0))
        ct = (Float64(cx), Float64(cy), Float64(z1))
        r = Float64(radius)

        for k in 1:segments
            theta1 = 2.0 * pi * (k - 1) / segments
            theta2 = 2.0 * pi * k / segments
            x1 = Float64(cx) + r * cos(theta1)
            y1 = Float64(cy) + r * sin(theta1)
            x2 = Float64(cx) + r * cos(theta2)
            y2 = Float64(cy) + r * sin(theta2)

            p1b = (x1, y1, Float64(z0))
            p2b = (x2, y2, Float64(z0))
            p1t = (x1, y1, Float64(z1))
            p2t = (x2, y2, Float64(z1))
            mid = 0.5 * (theta1 + theta2)
            side_normal = (cos(mid), sin(mid), 0.0)

            _write_triangle(io, side_normal, p1b, p2b, p2t)
            _write_triangle(io, side_normal, p1b, p2t, p1t)
            _write_triangle(io, (0.0, 0.0, -1.0), cb, p2b, p1b)
            _write_triangle(io, (0.0, 0.0, 1.0), ct, p1t, p2t)
        end
    end
    return abspath(filename)
end

if abspath(PROGRAM_FILE) == @__FILE__
    output = joinpath(@__DIR__, "cylinder.stl")
    path = write_cylinder_stl(output)
    println(path)
end
