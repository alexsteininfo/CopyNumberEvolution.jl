@testset "evolve" begin
    # Balanced 4-leaf tree, one division per edge:
    #        1
    #      /   \
    #     2     3
    #    / \   / \
    #   4   5 6   7
    bal() = phylotree([nothing, 1, 1, 2, 2, 3, 3];
                      birthtimes = [0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                      edge_divisions = [nothing, 1, 1, 1, 1, 1, 1],
                      edge_mutations = [nothing, 2, 2, 1, 1, 1, 1],
                      source_ids = collect(101:107))

    A() = toy_assembly(nchrom = 2, len = 1000)

    @testset "CNAModel defaults" begin
        m = CNAModel()
        @test m.rate isa PerDivision
        @test m.target isa UniformChromosome
        @test m.extent isa ExtentMixture
        @test m.kind isa GainLoss
        @test m.wgd isa NoWGD
        @test m.viability isa RejectAndRedraw
        @test m.initial isa Diploid
    end

    @testset "zero alterations: every tip is exactly the root state" begin
        res = simulate_cnas(bal(), A(), CNAModel(rate = PerDivision(0.0)); seed = 1)
        d = diploid(A())
        for i in 1:nnodes(res.tree)
            @test profile(res, i) == d
            @test check_invariants(profile(res, i))
        end
        @test nevents(res) == 0
        @test rejection_count(res) == 0
    end

    @testset "profiles are retained for every node by default" begin
        res = simulate_cnas(bal(), A(), CNAModel(rate = PerDivision(1.0)); seed = 2)
        @test res.retain_internal
        @test all(res.profiles[i] !== nothing for i in 1:nnodes(res.tree))
        @test length(leaf_profiles(res)) == 4
        for i in 1:nnodes(res.tree)
            @test check_invariants(profile(res, i))
        end
    end

    @testset "retain_internal = false keeps only the leaves" begin
        res = simulate_cnas(bal(), A(), CNAModel(rate = PerDivision(1.0));
                            seed = 3, retain_internal = false)
        for i in internal_nodes(res.tree)
            @test res.profiles[i] === nothing
            @test_throws ArgumentError profile(res, i)
        end
        @test all(res.profiles[i] !== nothing for i in leaves(res.tree))
        @test length(leaf_profiles(res)) == 4
    end

    @testset "inheritance: a scheduled doubling reaches exactly its descendants" begin
        t = bal()
        res = simulate_cnas(t, A(),
                            CNAModel(rate = PerDivision(0.0),
                                     wgd = ScheduledWGD(2 => 1));
                            seed = 4)
        d = diploid(A())
        # node 2 and everything below it is doubled
        for i in (2, 4, 5)
            @test cn_at(slot_segments(profile(res, i), 1, 1), 1) == 2
        end
        # node 1, 3, 6, 7 are untouched
        for i in (1, 3, 6, 7)
            @test profile(res, i) == d
        end
        @test nevents(res) == 1
        @test only(res.events).node == 2
        @test only(res.events).event isa WholeGenomeDoubling
    end

    @testset "inheritance: a segmental alteration reaches exactly its descendants" begin
        t = bal()
        fixed_target = (p, rng) -> (1, 1)
        fixed_extent = (p, c, h, rng) -> (101, 200, :focal)
        # one alteration, only on the edge into node 3
        onlyon3 = CustomRate((tr, i, rng) -> i == 3 ? 1 : 0)
        res = simulate_cnas(t, A(),
                            CNAModel(rate = onlyon3, target = fixed_target,
                                     extent = fixed_extent, kind = GainLoss(1.0));
                            seed = 5)
        for i in (3, 6, 7)
            @test cn_at(slot_segments(profile(res, i), 1, 1), 150) == 2
        end
        for i in (1, 2, 4, 5)
            @test cn_at(slot_segments(profile(res, i), 1, 1), 150) == 1
        end
        @test nevents(res) == 1
        @test events_on(res, 3) |> length == 1
        @test events_on(res, 6) |> isempty
        @test length(events_below(res, 1)) == 1
        @test isempty(events_below(res, 2))
    end

    @testset "determinism" begin
        t = bal()
        m = CNAModel(rate = PerDivision(2.0), extent = ExtentMixture(p_chromosome = 0.2, p_arm = 0.2))
        a = simulate_cnas(t, A(), m; seed = 99)
        b = simulate_cnas(t, A(), m; seed = 99)
        c = simulate_cnas(t, A(), m; seed = 100)
        @test [profile(a, i) for i in 1:nnodes(t)] == [profile(b, i) for i in 1:nnodes(t)]
        @test a.events == b.events
        @test [profile(a, i) for i in 1:nnodes(t)] != [profile(c, i) for i in 1:nnodes(t)]
        @test a.seed == 99
    end

    @testset "an explicit rng is honoured and records no seed" begin
        t = bal()
        m = CNAModel(rate = PerDivision(1.0))
        a = simulate_cnas(t, A(), m; rng = Random.Xoshiro(5))
        b = simulate_cnas(t, A(), m; rng = Random.Xoshiro(5))
        @test a.events == b.events
        @test a.seed === nothing
    end

    @testset "events are stored in preorder, then by order within a node" begin
        t = bal()
        res = simulate_cnas(t, A(), CNAModel(rate = PerDivision(3.0)); seed = 6)
        rank = Dict(n => k for (k, n) in enumerate(preorder(t)))
        keys_ = [(rank[e.node], e.order) for e in res.events]
        @test issorted(keys_)
        for i in 1:nnodes(t)
            ons = events_on(res, i)
            @test [e.order for e in ons] == collect(1:length(ons))
        end
    end

    @testset "TruncalCNAs sets the root state and logs its events there" begin
        t = bal()
        res = simulate_cnas(t, A(),
                            CNAModel(rate = PerDivision(0.0),
                                     initial = TruncalCNAs(3),
                                     kind = GainLoss(1.0));
                            seed = 7)
        @test nevents(res) == 3
        @test all(e.node == treeroot(t) for e in res.events)
        @test [e.order for e in res.events] == [1, 2, 3]
        @test profile(res, treeroot(t)) != diploid(A())
        # with no further alterations every tip equals the root
        for i in leaves(t)
            @test profile(res, i) == profile(res, treeroot(t))
        end
        @test_throws ArgumentError TruncalCNAs(-1)
    end

    @testset "Given sets the root state and must match the assembly" begin
        t = bal()
        start = diploid(A())
        apply!(start, SegmentalCNA(1, 1, 1, 1000, 1, :chromosome))
        res = simulate_cnas(t, A(), CNAModel(rate = PerDivision(0.0), initial = Given(start)); seed = 8)
        @test profile(res, treeroot(t)) == start
        for i in leaves(t)
            @test profile(res, i) == start
        end
        # the given profile is not mutated by the simulation
        @test cn_at(slot_segments(start, 1, 1), 1) == 2
        @test_throws ArgumentError simulate_cnas(t, toy_assembly(nchrom = 3, len = 1000),
                                                 CNAModel(initial = Given(start)); seed = 9)
    end

    @testset "event-log replay reproduces every profile" begin
        t = bal()
        m = CNAModel(rate = PerDivision(2.0), initial = TruncalCNAs(2),
                     wgd = ScheduledWGD(3 => 1),
                     extent = ExtentMixture(p_chromosome = 0.1, p_arm = 0.2))
        res = simulate_cnas(t, A(), m; seed = 10)
        rp = replay(res)
        for i in 1:nnodes(t)
            @test rp[i] == profile(res, i)
        end
        # replay also works when nothing was retained
        lean = simulate_cnas(t, A(), m; seed = 10, retain_internal = false)
        rp2 = replay(lean)
        for i in 1:nnodes(t)
            @test rp2[i] == rp[i]
        end
    end

    @testset "rejections are tallied by reason" begin
        t = bal()
        # losses only, on a male karyotype: hemizygous chrX and chrY losses get rejected
        m = CNAModel(rate = PerDivision(4.0), kind = GainLoss(0.0),
                     extent = ExtentMixture(p_chromosome = 1.0),
                     viability = RejectAndRedraw(min_total_cn = 1, max_attempts = 10_000))
        res = simulate_cnas(t, toy_sex_assembly(:male), m; seed = 11)
        @test rejection_count(res) > 0
        @test haskey(res.rejections, :min_total_cn)
        @test res.rejections[:min_total_cn] == rejection_count(res)
    end

    @testset "exhausting max_attempts throws with actionable advice" begin
        t = phylotree([nothing, 1]; edge_divisions = [nothing, 1])
        # a single hemizygous chromosome, whole-chromosome losses only: every
        # proposal drives total copy number to 0 and is rejected
        m = CNAModel(rate = PerDivision(5.0), kind = GainLoss(0.0),
                     extent = ExtentMixture(p_chromosome = 1.0),
                     viability = RejectAndRedraw(min_total_cn = 1, max_attempts = 5))
        err = try
            simulate_cnas(t, hemizygous_assembly(), m; seed = 12)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("max_attempts", err.msg)
        @test occursin("min_total_cn", err.msg)
    end

    @testset "rng_mode = :per_node needs a seed and stays reproducible" begin
        t = bal()
        m = CNAModel(rate = PerDivision(2.0))
        @test_throws ArgumentError simulate_cnas(t, A(), m; rng_mode = :per_node)
        @test_throws ArgumentError simulate_cnas(t, A(), m; seed = 1, rng_mode = :bogus)
        a = simulate_cnas(t, A(), m; seed = 13, rng_mode = :per_node)
        b = simulate_cnas(t, A(), m; seed = 13, rng_mode = :per_node)
        @test a.events == b.events
        for i in 1:nnodes(t)
            @test check_invariants(profile(a, i))
        end
    end

    @testset "a unary chain draws once per edge" begin
        t = phylotree([nothing, 1, 2, 3]; edge_divisions = [nothing, 1, 1, 1])
        onlyone = CNAModel(rate = PerDivision(0.0), wgd = ScheduledWGD(2 => 1, 3 => 1, 4 => 1))
        res = simulate_cnas(t, A(), onlyone; seed = 14)
        @test nevents(res) == 3
        @test cn_at(slot_segments(profile(res, 4), 1, 1), 1) == 8   # doubled three times
        @test cn_at(slot_segments(profile(res, 1), 1, 1), 1) == 1
    end

    @testset "deep trees do not overflow the stack" begin
        n = 20_000
        parents = Vector{Union{Int,Nothing}}(undef, n)
        parents[1] = nothing
        for i in 2:n
            parents[i] = i - 1
        end
        t = phylotree(parents; edge_divisions = vcat(nothing, fill(1, n - 1)))
        res = simulate_cnas(t, A(), CNAModel(rate = PerDivision(0.05));
                            seed = 15, retain_internal = false)
        @test check_invariants(profile(res, n))
        @test nevents(res) > 0
    end

    @testset "show methods do not error" begin
        res = simulate_cnas(bal(), A(), CNAModel(rate = PerDivision(1.0)); seed = 16)
        @test occursin("CNAEvolution", sprint(show, res))
        @test occursin("CNAModel", sprint(show, res.model))
    end
end
