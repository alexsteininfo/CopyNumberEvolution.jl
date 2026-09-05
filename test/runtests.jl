using CopyNumberEvolution
using Test
using Random
using Distributions   # tests construct length distributions directly

include("fixtures.jl")

@testset "CopyNumberEvolution.jl" begin
    @testset "smoke" begin
        @test isdefined(CopyNumberEvolution, :CopyNumberEvolution)
    end

    include("test_assembly.jl")
    include("test_profile.jl")
    include("test_cna.jl")
    include("test_tree.jl")
    include("test_newick.jl")
    include("test_rates.jl")
    include("test_proposals.jl")
    include("test_wgd.jl")
    include("test_viability.jl")
    include("test_evolve.jl")
    include("test_bingrid.jl")
    include("test_io.jl")
    include("test_ext.jl")
    include("test_science.jl")
    include("test_docs.jl")
end
