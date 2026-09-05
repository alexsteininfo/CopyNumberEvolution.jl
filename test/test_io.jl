@testset "io" begin
    setup() = begin
        t = phylotree([nothing, 1, 1]; edge_divisions = [nothing, 1, 1],
                      labels = [nothing, "left", "right"])
        a = toy_sex_assembly(:male, len = 400)
        res = simulate_cnas(t, a, CNAModel(rate = PerDivision(1.0),
                                           extent = ExtentMixture(p_chromosome = 0.5));
                            seed = 41)
        (t, a, res, BinGrid(a, 100))
    end

    readtsv(path) = [split(l, '\t') for l in readlines(path)]

    @testset "write_profiles" begin
        t, a, res, g = setup()
        path = joinpath(mktempdir(), "profiles.tsv")
        write_profiles(path, res)
        rows = readtsv(path)
        @test rows[1] == ["node_id", "name", "chrom", "haplotype", "start", "stop", "cn"]
        @test length(rows) > 1
        body = rows[2:end]
        @test all(length(r) == 7 for r in body)
        # every chromosome name present is a real one, and coordinates are 1-based
        @test all(r[3] in [chromname(a, c) for c in 1:nchromosomes(a)] for r in body)
        @test minimum(parse(Int, r[5]) for r in body) == 1
        # every chromosome written has slots — the set of names is exactly the
        # positive-ploidy ones, no more and no less
        written = Set(r[3] for r in body)
        @test written == Set(chromname(a, c) for c in 1:nchromosomes(a) if ploidy(a, c) > 0)
    end

    @testset "a zero-ploidy chromosome never appears in a profile table" begin
        t = phylotree([nothing, 1]; edge_divisions = [nothing, 1])
        a = toy_sex_assembly(:female, len = 400)      # chrY has ploidy 0 here
        @test ploidy(a, chromindex(a, "chrY")) == 0
        res = simulate_cnas(t, a, CNAModel(rate = PerDivision(0.0)); seed = 45)
        path = joinpath(mktempdir(), "female.tsv")
        write_profiles(path, res)
        chroms = Set(r[3] for r in readtsv(path)[2:end])
        @test !("chrY" in chroms)
        @test "chrX" in chroms
        @test "chr1" in chroms
    end

    @testset "write_profiles honours the cells argument" begin
        t, a, res, g = setup()
        path = joinpath(mktempdir(), "leaves.tsv")
        write_profiles(path, res; cells = leaves(t))
        ids = Set(parse(Int, r[1]) for r in readtsv(path)[2:end])
        @test ids == Set(leaves(t))
    end

    @testset "write_events" begin
        t, a, res, g = setup()
        path = joinpath(mktempdir(), "events.tsv")
        write_events(path, res)
        rows = readtsv(path)
        @test rows[1] == ["node_id", "name", "order", "type", "chrom", "haplotype",
                          "start", "stop", "delta", "scale", "mode"]
        @test length(rows) - 1 == nevents(res)
        for r in rows[2:end]
            @test r[4] in ("segmental", "wgd")
            if r[4] == "segmental"
                @test r[11] == "NA"
                @test parse(Int, r[9]) != 0
            else
                @test r[5] == "NA" && r[11] in ("multiply", "increment")
            end
        end
    end

    @testset "write_events on a run with a doubling records the mode" begin
        t = phylotree([nothing, 1]; edge_divisions = [nothing, 1])
        a = toy_assembly(nchrom = 1, len = 400)
        res = simulate_cnas(t, a, CNAModel(rate = PerDivision(0.0),
                                           wgd = ScheduledWGD(2 => 1; mode = :increment));
                            seed = 42)
        path = joinpath(mktempdir(), "e.tsv")
        write_events(path, res)
        rows = readtsv(path)
        @test length(rows) == 2
        @test rows[2][4] == "wgd"
        @test rows[2][11] == "increment"
    end

    @testset "write_bins" begin
        t, a, res, g = setup()
        path = joinpath(mktempdir(), "bins.tsv")
        write_bins(path, g)
        rows = readtsv(path)
        @test rows[1] == ["bin_index", "chrom", "start", "stop"]
        @test length(rows) - 1 == nbins(g)
        @test rows[2] == ["1", "chr1", "1", "100"]
    end

    @testset "write_medicc2 shape and coordinates" begin
        t, a, res, g = setup()
        m = CNMatrix(res, g)
        path = joinpath(mktempdir(), "medicc.tsv")
        write_medicc2(path, m)
        rows = readtsv(path)
        @test rows[1] == ["sample_id", "chrom", "start", "end", "cn_a", "cn_b"]
        autosomal = sum(length(bins_of(g, c)) for c in autosomes(a))
        @test length(rows) - 1 == (ncells(m) + 1) * autosomal
        # BED convention: 0-based start, exclusive end
        @test rows[2][2:4] == ["chr1", "0", "100"]
        # the reference rows come first and are all 1/1
        diploid_rows = [r for r in rows[2:end] if r[1] == "diploid"]
        @test length(diploid_rows) == autosomal
        @test all(r[5] == "1" && r[6] == "1" for r in diploid_rows)
        # sample ids match cellname, so they agree with an exported newick
        @test Set(r[1] for r in rows[2:end]) == Set(vcat("diploid", m.names))
    end

    @testset "write_medicc2 excludes sex chromosomes by default" begin
        t, a, res, g = setup()
        m = CNMatrix(res, g)
        path = joinpath(mktempdir(), "auto.tsv")
        write_medicc2(path, m)
        chroms = Set(r[2] for r in readtsv(path)[2:end])
        @test !("chrX" in chroms) && !("chrY" in chroms)
        path2 = joinpath(mktempdir(), "withxy.tsv")
        write_medicc2(path2, m; include_xy = true)
        chroms2 = Set(r[2] for r in readtsv(path2)[2:end])
        @test "chrX" in chroms2 && "chrY" in chroms2
    end

    @testset "write_medicc2 warns above MEDICC2's copy-number ceiling" begin
        t = phylotree([nothing, 1]; edge_divisions = [nothing, 1])
        a = toy_assembly(nchrom = 1, len = 400)
        start = diploid(a)
        apply!(start, SegmentalCNA(1, 1, 1, 400, 11, :chromosome))   # cn 12
        res = simulate_cnas(t, a, CNAModel(rate = PerDivision(0.0), initial = Given(start));
                            seed = 43)
        m = CNMatrix(res, BinGrid(a, 100))
        @test max_cn(m) > 8
        path = joinpath(mktempdir(), "high.tsv")
        @test_logs (:warn, r"MEDICC2") write_medicc2(path, m)
    end

    @testset "write_medicc2 needs allele tracks" begin
        t = phylotree([nothing, 1]; edge_divisions = [nothing, 1])
        a = toy_assembly(nchrom = 1, len = 400)
        res = simulate_cnas(t, a, CNAModel(rate = PerDivision(0.0)); seed = 44)
        m = CNMatrix(res, BinGrid(a, 100); allele = false)
        @test_throws ArgumentError write_medicc2(joinpath(mktempdir(), "x.tsv"), m)
    end

    @testset "an exported newick and matrix agree on names" begin
        t, a, res, g = setup()
        m = CNMatrix(res, g)
        dir = mktempdir()
        write_newick(joinpath(dir, "t.nwk"), t; branchlength = :divisions)
        write_medicc2(joinpath(dir, "m.tsv"), m)
        nwk = read(joinpath(dir, "t.nwk"), String)
        for nm in m.names
            @test occursin(nm, nwk)
        end
    end

    @testset "writers accept an IO as well as a path" begin
        t, a, res, g = setup()
        buf = IOBuffer()
        write_bins(buf, g)
        @test occursin("bin_index", String(take!(buf)))
    end
end
