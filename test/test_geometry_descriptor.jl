using Test
using KrakeNN

@testset "GeometryDescriptor" begin
    is_solid = Bool[true false; false true]
    q_wall = fill(0.5, 2, 2, 9)
    stl_hash = UInt64(0x1234)

    desc = build_geometry_descriptor(:stl, is_solid;
                                     q_wall_dist=q_wall,
                                     stl_hash=stl_hash)

    @test desc isa GeometryDescriptor
    @test desc.type === :stl
    @test desc.blockage == 0.5
    @test desc.q_wall_dist === q_wall
    @test desc.stl_hash == stl_hash
    @test desc.is_solid === is_solid

    override = build_geometry_descriptor(; type=:analytic,
                                         is_solid=is_solid,
                                         blockage=0.25)

    @test override.type === :analytic
    @test override.blockage == 0.25
    @test override.q_wall_dist === nothing
    @test override.stl_hash == UInt64(0)
    @test override.is_solid === is_solid
end
