# Alterations on the path from the root to `leaf`, excluding truncal (root) events.
function nonroot_events_to(res, leaf::Int)
    path = Set(ancestors(res.tree, leaf))
    delete!(path, treeroot(res.tree))
    return [e for e in res.events if e.node in path]
end

@testset "scientific properties" begin
    A() = toy_assembly(nchrom = 3, len = 2000)

    @testset "sampling commutes exactly under rng_mode = :per_node" begin
        full = binary_lineage(4)                     # 31 nodes, 16 leaves
        keep = [17, 20, 25, 31]
        sub = induced_subtree(full, keep)
        @test nnodes(sub) < nnodes(full)
        # pruning, not collapsing: a kept leaf's path length is unchanged
        for l in keep
            sid = node(full, l).source_id
            @test depth(sub, node_by_source_id(sub, sid)) == depth(full, l)
        end

        m = CNAModel(rate = PerDivision(1.5),
                     extent = ExtentMixture(p_chromosome = 0.2, p_arm = 0.2),
                     initial = TruncalCNAs(2),
                     wgd = NoWGD())
        rf = simulate_cnas(full, A(), m; seed = 777, rng_mode = :per_node)
        rs = simulate_cnas(sub, A(), m; seed = 777, rng_mode = :per_node)

        # the root state is the same
        @test profile(rs, treeroot(sub)) == profile(rf, treeroot(full))
        # and every kept leaf's profile is identical, not merely similar
        for l in keep
            sid = node(full, l).source_id
            @test profile(rs, node_by_source_id(sub, sid)) == profile(rf, l)
        end
        # so are the events on every retained edge
        for i in 1:nnodes(sub)
            sid = node(sub, i).source_id
            j = node_by_source_id(full, sid)
            @test [e.event for e in events_on(rs, i)] == [e.event for e in events_on(rf, j)]
        end
    end

    @testset "sampling commutes distributionally under rng_mode = :global" begin
        # With one shared stream the draws cannot line up edge-for-edge, so the claim
        # is about distributions. The number of alterations on the retained edges is a
        # sum of independent Poissons with a known mean, so we can bound the tolerance
        # analytically instead of guessing one.
        full = binary_lineage(3)
        keep = [9, 12, 15]
        sub = induced_subtree(full, keep)
        λ = 1.25
        m = CNAModel(rate = PerDivision(λ), viability = AllowAll())

        retained_sids = Set(node(sub, i).source_id for i in 1:nnodes(sub) if !isroot(sub, i))
        divisions = sum(node(full, node_by_source_id(full, s)).edge_divisions
                        for s in retained_sids)
        expected = λ * divisions

        N = 400
        full_total = 0
        sub_total = 0
        for s in 1:N
            rf = simulate_cnas(full, A(), m; seed = s, retain_internal = false)
            rs = simulate_cnas(sub, A(), m; seed = s, retain_internal = false)
            full_total += count(e -> node(full, e.node).source_id in retained_sids, rf.events)
            sub_total += length(rs.events)
        end
        se = sqrt(expected / N)              # standard error of the mean
        @test abs(full_total / N - expected) < 4 * se
        @test abs(sub_total / N - expected) < 4 * se
    end

    @testset "an end-to-end run on hg38, invariants throughout" begin
        a = hg38(:female)
        tree = binary_lineage(7)             # 255 nodes, 128 leaves
        model = CNAModel(rate = PerDivision(0.6),
                         target = CNWeighted(1.0),
                         extent = ExtentMixture(p_chromosome = 0.05, p_arm = 0.15),
                         kind = GainLoss(0.55),
                         wgd = ScheduledWGD(mrca(tree, [128, 191]) => 1),
                         viability = RejectAndRedraw(),
                         initial = TruncalCNAs(3))
        res = simulate_cnas(tree, a, model; seed = 20260904)

        @test nevents(res) > 100
        for i in 1:nnodes(tree)
            @test check_invariants(profile(res, i))
        end
        @test replay(res) == [profile(res, i) for i in 1:nnodes(tree)]

        # the scheduled doubling reached exactly its descendants
        wgd_node = mrca(tree, [128, 191])
        below = Set(descendant_leaves(tree, wgd_node))
        wgd_events = [e for e in res.events if e.event isa WholeGenomeDoubling]
        @test length(wgd_events) == 1
        @test only(wgd_events).node == wgd_node
        @test !isempty(below) && length(below) < length(leaves(tree))

        # every leaf under wgd_node carries exactly one doubling on its root-to-leaf
        # path, and every leaf elsewhere carries none
        for l in leaves(tree)
            onpath = Set(ancestors(tree, l))
            ndoublings = count(e -> e.event isa WholeGenomeDoubling && e.node in onpath,
                               res.events)
            @test ndoublings == (l in below ? 1 : 0)
        end

        # every scale class the model enables actually occurs
        scales = Set(e.event.scale for e in res.events if e.event isa SegmentalCNA)
        @test :focal in scales
        @test :arm in scales

        # total copy number equals the sum over haplotypes, everywhere, on real data
        p = profile(res, first(leaves(tree)))
        for c in 1:nchromosomes(a)
            tot = total_cn(p, c)
            for pos in (1, chromlength(a, c) ÷ 3, chromlength(a, c))
                @test cn_at(tot, pos) ==
                      sum(cn_at(p.segments[s], pos) for s in slots_of(a, c); init = 0)
            end
        end

        # and the whole output pipeline runs
        grid = BinGrid(a, 500_000)
        mat = CNMatrix(res, grid)
        @test ncells(mat) == length(leaves(tree))
        @test mat.total == mat.allele[1] .+ mat.allele[2]
        dir = mktempdir()
        write_medicc2(joinpath(dir, "cells.tsv"), mat)
        write_profiles(joinpath(dir, "truth.tsv"), res)
        write_events(joinpath(dir, "events.tsv"), res)
        write_bins(joinpath(dir, "bins.tsv"), grid)
        write_newick(joinpath(dir, "tree.nwk"), tree; branchlength = :divisions)
        for f in ("cells.tsv", "truth.tsv", "events.tsv", "bins.tsv", "tree.nwk")
            @test filesize(joinpath(dir, f)) > 0
        end
        rt = read_newick(joinpath(dir, "tree.nwk"); branchlength = :divisions)
        @test nnodes(rt) == nnodes(tree)
    end

    @testset "per-division and per-time diverge when timing is non-exponential" begin
        # The package's reason to exist: on a tree where division count and elapsed
        # time are decoupled, the two rate rules put alterations in different places.
        n = 31
        parents = Vector{Union{Int,Nothing}}(undef, n)
        bt = Vector{Float64}(undef, n)
        parents[1] = nothing
        bt[1] = 0.0
        for i in 2:n
            p = i ÷ 2
            parents[i] = p
            # left children divide fast, right children slowly: divisions and time
            # carry different information
            bt[i] = bt[p] + (iseven(i) ? 0.1 : 5.0)
        end
        t = phylotree(parents; birthtimes = bt,
                      edge_divisions = vcat(nothing, fill(1, n - 1)),
                      source_ids = collect(1:n))

        perdiv = CNAModel(rate = PerDivision(2.0), viability = AllowAll())
        pertime = CNAModel(rate = PerTime(2.0 / 2.55), viability = AllowAll())

        fast_leaf = 16      # all-left path: few elapsed time units
        slow_leaf = 31      # all-right path: many elapsed time units
        N = 200
        div_fast = div_slow = time_fast = time_slow = 0
        for s in 1:N
            rd = simulate_cnas(t, A(), perdiv; seed = s)
            rt = simulate_cnas(t, A(), pertime; seed = s)
            div_fast += length(nonroot_events_to(rd, fast_leaf))
            div_slow += length(nonroot_events_to(rd, slow_leaf))
            time_fast += length(nonroot_events_to(rt, fast_leaf))
            time_slow += length(nonroot_events_to(rt, slow_leaf))
        end
        # per division: both leaves are 4 divisions deep, so the burdens match
        @test div_fast / N ≈ div_slow / N rtol = 0.15
        # per time: the slow path accumulates far more
        @test time_slow / N > 3 * (time_fast / N)
    end
end
