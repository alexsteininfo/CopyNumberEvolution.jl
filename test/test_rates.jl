@testset "rates" begin
    # Two-edge tree: node 2 has 3 divisions / Δt = 2.0 / 8 mutations,
    #                node 3 has 1 division  / Δt = 0.5 / 0 mutations.
    tree() = phylotree([nothing, 1, 1];
                       birthtimes = [0.0, 2.0, 0.5],
                       edge_divisions = [nothing, 3, 1],
                       edge_mutations = [nothing, 8, 0])

    mean_draws(rule, t, i; n = 40_000, seed = 20260904) = begin
        rng = Random.Xoshiro(seed)
        sum(n_cnas(rule, t, i, rng) for _ in 1:n) / n
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError PerDivision(-1.0)
        @test_throws ArgumentError PerTime(-0.5)
        @test_throws ArgumentError FromEdgeMutations(1.5)
        @test_throws ArgumentError FromEdgeMutations(-0.1)
        @test FromEdgeMutations().p == 1.0
    end

    @testset "PerDivision scales with edge_divisions" begin
        t = tree()
        @test mean_draws(PerDivision(0.5), t, 2) ≈ 1.5 rtol = 0.05
        @test mean_draws(PerDivision(0.5), t, 3) ≈ 0.5 rtol = 0.05
        @test n_cnas(PerDivision(0.0), t, 2, Random.Xoshiro(1)) == 0
    end

    @testset "PerTime scales with elapsed real time" begin
        t = tree()
        @test mean_draws(PerTime(1.5), t, 2) ≈ 3.0 rtol = 0.05
        @test mean_draws(PerTime(1.5), t, 3) ≈ 0.75 rtol = 0.05
    end

    @testset "FromEdgeMutations is exactly identity by default" begin
        t = tree()
        rng = Random.Xoshiro(7)
        @test all(n_cnas(FromEdgeMutations(), t, 2, rng) == 8 for _ in 1:50)
        @test all(n_cnas(FromEdgeMutations(), t, 3, rng) == 0 for _ in 1:50)
    end

    @testset "FromEdgeMutations thins binomially when p < 1" begin
        t = tree()
        @test mean_draws(FromEdgeMutations(0.25), t, 2) ≈ 2.0 rtol = 0.05
        @test n_cnas(FromEdgeMutations(0.25), t, 3, Random.Xoshiro(1)) == 0
    end

    @testset "each rule names the field it is missing" begin
        divs_only = phylotree([nothing, 1]; edge_divisions = [nothing, 2])
        rng = Random.Xoshiro(1)
        @test n_cnas(PerDivision(1.0), divs_only, 2, rng) isa Int
        @test_throws ArgumentError n_cnas(PerTime(1.0), divs_only, 2, rng)
        @test_throws ArgumentError n_cnas(FromEdgeMutations(), divs_only, 2, rng)

        time_only = phylotree([nothing, 1]; birthtimes = [0.0, 1.0])
        @test_throws ArgumentError n_cnas(PerDivision(1.0), time_only, 2, rng)
        @test n_cnas(PerTime(1.0), time_only, 2, rng) isa Int
    end

    @testset "the error message says how to fix it" begin
        t = phylotree([nothing, 1]; edge_divisions = [nothing, 2])
        err = try
            n_cnas(PerTime(1.0), t, 2, Random.Xoshiro(1))
        catch e
            e
        end
        @test occursin("branchlength", err.msg)
    end

    @testset "CustomRate wraps a function" begin
        t = tree()
        r = CustomRate((tr, i, rng) -> 2 * something(node(tr, i).edge_divisions, 0))
        @test n_cnas(r, t, 2, Random.Xoshiro(1)) == 6
        @test n_cnas(r, t, 3, Random.Xoshiro(1)) == 2
    end

    @testset "a negative edge time is a corrupt tree, not a rate of zero" begin
        bad = phylotree([nothing, 1]; birthtimes = [1.0, 0.0])
        @test_throws ArgumentError n_cnas(PerTime(1.0), bad, 2, Random.Xoshiro(1))
    end
end
