@testset "wgd" begin
    tree() = phylotree([nothing, 1, 1, 2, 2];
                       birthtimes = [0.0, 1.0, 1.0, 2.0, 2.0],
                       edge_divisions = [nothing, 1, 1, 1, 1],
                       source_ids = [100, 200, 300, 400, 500])

    @testset "modes are validated and reported" begin
        @test wgd_mode(NoWGD()) === :multiply
        @test wgd_mode(ScheduledWGD(Dict(2 => 1))) === :multiply
        @test wgd_mode(ScheduledWGD(Dict(2 => 1); mode = :increment)) === :increment
        @test_throws ArgumentError ScheduledWGD(Dict(2 => 1); mode = :bogus)
        @test_throws ArgumentError ExactlyNWGD(1; mode = :bogus)
        @test_throws ArgumentError RateWGD(0.1; mode = :bogus)
    end

    @testset "NoWGD schedules nothing" begin
        @test isempty(prepare_wgd(NoWGD(), tree(), Random.Xoshiro(1)))
    end

    @testset "ScheduledWGD returns exactly what was asked for" begin
        t = tree()
        s = prepare_wgd(ScheduledWGD(Dict(2 => 1, 5 => 2)), t, Random.Xoshiro(1))
        @test s == Dict(2 => 1, 5 => 2)
        # the pairs constructor is equivalent
        @test prepare_wgd(ScheduledWGD(2 => 1), t, Random.Xoshiro(1)) == Dict(2 => 1)
        # naming the edge via helpers, which is the intended workflow
        node_of_interest = node_by_source_id(t, 400)
        @test prepare_wgd(ScheduledWGD(node_of_interest => 1), t, Random.Xoshiro(1)) ==
              Dict(4 => 1)
        @test prepare_wgd(ScheduledWGD(mrca(t, [4, 5]) => 1), t, Random.Xoshiro(1)) ==
              Dict(2 => 1)
    end

    @testset "ScheduledWGD rejects impossible placements" begin
        t = tree()
        @test_throws ArgumentError prepare_wgd(ScheduledWGD(Dict(99 => 1)), t, Random.Xoshiro(1))
        @test_throws ArgumentError prepare_wgd(ScheduledWGD(Dict(1 => 1)), t, Random.Xoshiro(1))
        @test_throws ArgumentError ScheduledWGD(Dict(2 => 0))
        @test_throws ArgumentError ScheduledWGD(Dict(2 => -1))
    end

    @testset "ExactlyNWGD places exactly n doublings on distinct non-root edges" begin
        t = tree()
        for n in 1:4
            s = prepare_wgd(ExactlyNWGD(n), t, Random.Xoshiro(42))
            @test sum(values(s)) == n
            @test length(s) == n
            @test all(v == 1 for v in values(s))
            @test all(k != treeroot(t) for k in keys(s))
        end
        @test_throws ArgumentError prepare_wgd(ExactlyNWGD(5), t, Random.Xoshiro(1))
        @test_throws ArgumentError ExactlyNWGD(0)
    end

    @testset "ExactlyNWGD is deterministic given a seed" begin
        t = tree()
        a = prepare_wgd(ExactlyNWGD(2), t, Random.Xoshiro(7))
        b = prepare_wgd(ExactlyNWGD(2), t, Random.Xoshiro(7))
        c = prepare_wgd(ExactlyNWGD(2), t, Random.Xoshiro(8))
        @test a == b
        # `c` may legitimately coincide with `a` on a 4-edge tree, so assert only the
        # property that must hold: it is still a valid 2-doubling schedule.
        @test sum(values(c)) == 2
    end

    @testset "RateWGD reuses the CNARate machinery" begin
        t = tree()
        # 4 non-root edges, one division each
        total(seed) = sum(values(prepare_wgd(RateWGD(PerDivision(0.5)), t, Random.Xoshiro(seed))); init = 0)
        m = sum(total(s) for s in 1:2000) / 2000
        @test m ≈ 2.0 rtol = 0.08          # 4 edges × Poisson(0.5)
        @test isempty(prepare_wgd(RateWGD(PerDivision(0.0)), t, Random.Xoshiro(1)))
        # the numeric shorthand is PerDivision
        @test RateWGD(0.25).rate isa PerDivision
        # per-time works too
        tt = sum(values(prepare_wgd(RateWGD(PerTime(1.0)), t, Random.Xoshiro(3))); init = 0)
        @test tt isa Int
    end

    @testset "RateWGD never schedules on the root" begin
        t = tree()
        for seed in 1:50
            s = prepare_wgd(RateWGD(PerDivision(3.0)), t, Random.Xoshiro(seed))
            @test treeroot(t) ∉ keys(s)
        end
    end
end
