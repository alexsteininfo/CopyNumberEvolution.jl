@testset "package extension" begin
    @testset "the extension is wired up in Project.toml" begin
        proj = read(joinpath(@__DIR__, "..", "Project.toml"), String)
        @test occursin("[weakdeps]", proj)
        @test occursin("MutationLoadDynamics = \"7b855ee6-6887-412f-a571-26d20a5a92d7\"", proj)
        @test occursin("CopyNumberEvolutionMutationLoadDynamicsExt = \"MutationLoadDynamics\"", proj)
        # and never a hard dependency: the simulator must not be reachable from an
        # inference-only install
        deps = match(r"\[deps\](.*?)\n\["s, proj)
        @test deps !== nothing
        @test !occursin("MutationLoadDynamics", deps.captures[1])
        @test isfile(joinpath(@__DIR__, "..", "ext",
                              "CopyNumberEvolutionMutationLoadDynamicsExt.jl"))
    end

    mld_loaded = try
        @eval using MutationLoadDynamics
        true
    catch
        false
    end

    if !mld_loaded
        @info """MutationLoadDynamics.jl is not available, so the conversion tests are skipped.
                 Enable them with: julia --project=. -e 'using Pkg; Pkg.develop(path = "../MutationLoadDynamics.jl")'"""
    else
        @testset "converting a lineage tree" begin
            # Build a small BinaryNode tree by hand: founder -> two daughters,
            # the left one dividing again.
            root = MutationLoadDynamics.BinaryNode(
                MutationLoadDynamics.NonMarkovCell(1, 0.0, 3, 1.0))
            MutationLoadDynamics.leftchild!(root,
                MutationLoadDynamics.NonMarkovCell(2, 1.5, 4, 1.0))
            MutationLoadDynamics.rightchild!(root,
                MutationLoadDynamics.NonMarkovCell(3, 1.5, 1, 1.0))
            MutationLoadDynamics.leftchild!(root.left,
                MutationLoadDynamics.NonMarkovCell(4, 2.25, 2, 1.0))
            MutationLoadDynamics.rightchild!(root.left,
                MutationLoadDynamics.NonMarkovCell(5, 2.75, 0, 1.0))

            t = PhyloTree(root)
            @test nnodes(t) == 5
            @test isroot(t, treeroot(t))
            @test node(t, treeroot(t)).source_id == 1
            @test node(t, treeroot(t)).birthtime == 0.0
            @test node(t, treeroot(t)).edge_divisions === nothing
            @test node(t, treeroot(t)).edge_mutations === nothing

            # every non-root edge is exactly one division
            @test all(node(t, i).edge_divisions == 1 for i in 1:nnodes(t) if !isroot(t, i))

            i2 = node_by_source_id(t, 2)
            @test node(t, i2).edge_mutations == 4
            @test edge_time(t, i2) ≈ 1.5
            @test node(t, node_by_source_id(t, 5)).edge_mutations == 0
            @test edge_time(t, node_by_source_id(t, 5)) ≈ 1.25

            @test Set(node(t, l).source_id for l in leaves(t)) == Set([3, 4, 5])
            @test founder_mutations(root) == 3
        end

        @testset "all three rate rules work on a converted tree" begin
            root = MutationLoadDynamics.BinaryNode(
                MutationLoadDynamics.NonMarkovCell(1, 0.0, 2, 1.0))
            MutationLoadDynamics.leftchild!(root,
                MutationLoadDynamics.NonMarkovCell(2, 1.0, 5, 1.0))
            MutationLoadDynamics.rightchild!(root,
                MutationLoadDynamics.NonMarkovCell(3, 2.0, 7, 1.0))
            t = PhyloTree(root)
            rng = Random.Xoshiro(1)
            i2 = node_by_source_id(t, 2)
            @test n_cnas(PerDivision(1.0), t, i2, rng) isa Int
            @test n_cnas(PerTime(1.0), t, i2, rng) isa Int
            @test n_cnas(FromEdgeMutations(), t, i2, rng) == 5
        end

        @testset "a unary chain, as pruning leaves behind, converts unchanged" begin
            root = MutationLoadDynamics.BinaryNode(
                MutationLoadDynamics.NonMarkovCell(1, 0.0, 0, 1.0))
            MutationLoadDynamics.leftchild!(root,
                MutationLoadDynamics.NonMarkovCell(2, 1.0, 1, 1.0))
            MutationLoadDynamics.leftchild!(root.left,
                MutationLoadDynamics.NonMarkovCell(3, 2.0, 1, 1.0))
            t = PhyloTree(root)
            @test nnodes(t) == 3
            @test leaves(t) == [node_by_source_id(t, 3)]
            @test depth(t, node_by_source_id(t, 3)) == 2
        end

        @testset "an end-to-end run on a converted tree" begin
            root = MutationLoadDynamics.BinaryNode(
                MutationLoadDynamics.NonMarkovCell(1, 0.0, 0, 1.0))
            MutationLoadDynamics.leftchild!(root,
                MutationLoadDynamics.NonMarkovCell(2, 1.0, 6, 1.0))
            MutationLoadDynamics.rightchild!(root,
                MutationLoadDynamics.NonMarkovCell(3, 1.0, 6, 1.0))
            t = PhyloTree(root)
            a = toy_assembly(nchrom = 2, len = 1000)
            res = simulate_cnas(t, a, CNAModel(rate = FromEdgeMutations(0.5)); seed = 61)
            for i in 1:nnodes(t)
                @test check_invariants(profile(res, i))
            end
            @test replay(res) == [profile(res, i) for i in 1:nnodes(t)]
        end
    end
end
