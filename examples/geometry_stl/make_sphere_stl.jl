const DEFAULT_RADIUS = 0.2
const DEFAULT_CX = 1.0
const DEFAULT_CY = 0.5
const DEFAULT_CZ = 0.5
const DEFAULT_LATITUDES = 32
const DEFAULT_LONGITUDES = 64

function _sub(a, b)
    return (a[1] - b[1], a[2] - b[2], a[3] - b[3])
end

function _cross(a, b)
    return (a[2] * b[3] - a[3] * b[2],
            a[3] * b[1] - a[1] * b[3],
            a[1] * b[2] - a[2] * b[1])
end

function _unit(v)
    n = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
    n == 0.0 && return (0.0, 0.0, 0.0)
    return (v[1] / n, v[2] / n, v[3] / n)
end

function _write_triangle(io, v1, v2, v3)
    normal = _unit(_cross(_sub(v2, v1), _sub(v3, v1)))
    for value in normal
        write(io, Float32(value))
    end
    for vertex in (v1, v2, v3), value in vertex
        write(io, Float32(value))
    end
    write(io, UInt16(0))
    return nothing
end

function _sphere_point(cx, cy, cz, radius, theta, phi)
    s = sin(theta)
    return (Float64(cx) + Float64(radius) * s * cos(phi),
            Float64(cy) + Float64(radius) * s * sin(phi),
            Float64(cz) + Float64(radius) * cos(theta))
end

function write_sphere_stl(filename::AbstractString;
                          radius::Real=DEFAULT_RADIUS,
                          cx::Real=DEFAULT_CX,
                          cy::Real=DEFAULT_CY,
                          cz::Real=DEFAULT_CZ,
                          latitudes::Integer=DEFAULT_LATITUDES,
                          longitudes::Integer=DEFAULT_LONGITUDES)
    latitudes >= 2 || throw(ArgumentError("latitudes must be >= 2"))
    longitudes >= 3 || throw(ArgumentError("longitudes must be >= 3"))

    mkpath(dirname(abspath(filename)))
    open(filename, "w") do io
        header = zeros(UInt8, 80)
        label = codeunits("KrakeNN M-GEO-5 closed sphere")
        header[1:length(label)] .= label
        write(io, header)
        write(io, UInt32(2 * longitudes * (latitudes - 1)))

        for j in 1:latitudes
            theta0 = pi * (j - 1) / latitudes
            theta1 = pi * j / latitudes
            for k in 1:longitudes
                phi0 = 2.0 * pi * (k - 1) / longitudes
                phi1 = 2.0 * pi * k / longitudes

                p00 = _sphere_point(cx, cy, cz, radius, theta0, phi0)
                p01 = _sphere_point(cx, cy, cz, radius, theta0, phi1)
                p10 = _sphere_point(cx, cy, cz, radius, theta1, phi0)
                p11 = _sphere_point(cx, cy, cz, radius, theta1, phi1)

                if j == 1
                    _write_triangle(io, p00, p10, p11)
                elseif j == latitudes
                    _write_triangle(io, p00, p10, p01)
                else
                    _write_triangle(io, p00, p10, p11)
                    _write_triangle(io, p00, p11, p01)
                end
            end
        end
    end
    return abspath(filename)
end

if abspath(PROGRAM_FILE) == @__FILE__
    output = joinpath(@__DIR__, "sphere.stl")
    path = write_sphere_stl(output)
    println(path)
end
