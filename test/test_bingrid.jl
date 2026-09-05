@testset "bingrid" begin
    S = CopyNumberEvolution.Segment

    @testset "grid layout" begin
        a = toy_assembly(nchrom = 2, len = 1000)
        g = BinGrid(a, 250)
        @test nbins(g) == 8
        @test bins_of(g, 1) == 1:4
        @test bins_of(g, 2) == 5:8
        @test g.bins[1] == CopyNumberEvolution.Bin(1, 1, 250)
        @test g.bins[4] == CopyNumberEvolution.Bin(1, 751, 1000)
        @test_throws ArgumentError BinGrid(a, 0)
    end

    @testset "the last bin of a chromosome is short when the length does not divide" begin
        a = toy_assembly(nchrom = 1, len = 1000)
        g = BinGrid(a, 300)
        @test nbins(g) == 4
        @test g.bins[4] == CopyNumberEvolution.Bin(1, 901, 1000)
        @test length(bins_of(g, 1)) == 4
    end

    @testset "hg38 bin counts" begin
        g = BinGrid(hg38(:female), 500_000)
        @test length(bins_of(g, 1)) == cld(248956422, 500_000)   # 498
        y = chromindex(hg38(:female), "chrY")
        @test isempty(bins_of(g, y))                              # no slots, no bins
        gm = BinGrid(hg38(:male), 500_000)
        @test length(bins_of(gm, chromindex(hg38(:male), "chrY"))) == cld(57227415, 500_000)
    end

    @testset "a bin-aligned segmentation projects exactly" begin
        a = toy_assembly(nchrom = 1, len = 1000)
        g = BinGrid(a, 250)
        segs = [S(1, 250, 1), S(251, 500, 3), S(501, 1000, 0)]
        @test project(segs, g, 1, LengthWeightedMajority()) == [1, 3, 0, 0]
        @test project(segs, g, 1, AreaWeightedMean()) == [1, 3, 0, 0]
    end

    @testset "a chromosome-length segment fills every bin" begin
        a = toy_assembly(nchrom = 1, len = 1000)
        g = BinGrid(a, 250)
        @test project([S(1, 1000, 4)], g, 1, LengthWeightedMajority()) == [4, 4, 4, 4]
    end

    @testset "a straddling bin follows the documented rule" begin
        a = toy_assembly(nchrom = 1, len = 400)
        g = BinGrid(a, 100)
        # bin 2 is 101:200; the breakpoint at 161 gives 60 bp at cn 1 and 40 bp at cn 4
        segs = [S(1, 160, 1), S(161, 400, 4)]
        @test project(segs, g, 1, LengthWeightedMajority()) == [1, 1, 4, 4]
        # area-weighted mean over bin 2 is (60*1 + 40*4)/100 = 2.2 -> 2
        @test project(segs, g, 1, AreaWeightedMean()) == [1, 2, 4, 4]
    end

    @testset "majority ties resolve to the lower copy number" begin
        a = toy_assembly(nchrom = 1, len = 200)
        g = BinGrid(a, 100)
        segs = [S(1, 50, 5), S(51, 200, 2)]     # bin 1 is 50 bp of cn 5 and 50 bp of cn 2
        @test project(segs, g, 1, LengthWeightedMajority())[1] == 2
    end

    @testset "area-weighted mean rounds halves upward" begin
        a = toy_assembly(nchrom = 1, len = 100)
        g = BinGrid(a, 100)
        segs = [S(1, 50, 1), S(51, 100, 2)]     # mean exactly 1.5
        @test project(segs, g, 1, AreaWeightedMean()) == [2]
    end

    @testset "profile projection returns total and per-haplotype tracks" begin
        a = toy_assembly(nchrom = 1, len = 400)
        g = BinGrid(a, 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 1, 200, 1, :focal))
        tot, alleles = project(p, g)
        @test length(alleles) == 2
        @test alleles[1] == [2, 2, 1, 1]
        @test alleles[2] == [1, 1, 1, 1]
        @test tot == [3, 3, 2, 2]
        @test tot == alleles[1] .+ alleles[2]
    end

    @testset "hemizygous chromosomes give zeros on the absent haplotype" begin
        a = toy_sex_assembly(:male, len = 400)
        g = BinGrid(a, 100)
        p = diploid(a)
        tot, alleles = project(p, g)
        x = bins_of(g, chromindex(a, "chrX"))
        @test all(alleles[1][i] == 1 for i in x)
        @test all(alleles[2][i] == 0 for i in x)
        @test all(tot[i] == 1 for i in x)
    end

    @testset "CNMatrix over an evolution result" begin
        t = phylotree([nothing, 1, 1]; edge_divisions = [nothing, 1, 1],
                      labels = [nothing, "left", nothing])
        a = toy_assembly(nchrom = 2, len = 400)
        res = simulate_cnas(t, a, CNAModel(rate = PerDivision(1.0)); seed = 31)
        g = BinGrid(a, 100)
        m = CNMatrix(res, g)
        @test ncells(m) == 2
        @test m.cells == leaves(t)
        @test m.names == ["left", "cell_3"]
        @test size(m.total) == (2, nbins(g))
        @test length(m.allele) == 2
        @test m.total == m.allele[1] .+ m.allele[2]
        @test all(>=(0), m.total)
        @test max_cn(m) >= 0
    end

    @testset "CNMatrix can include internal nodes as a truth set" begin
        t = phylotree([nothing, 1, 1]; edge_divisions = [nothing, 1, 1])
        a = toy_assembly(nchrom = 1, len = 400)
        res = simulate_cnas(t, a, CNAModel(rate = PerDivision(1.0)); seed = 32)
        g = BinGrid(a, 100)
        m = CNMatrix(res, g; cells = collect(1:nnodes(t)))
        @test ncells(m) == 3
        @test m.names == ["cell_1", "cell_2", "cell_3"]
    end

    @testset "allele = false omits the per-haplotype tracks" begin
        t = phylotree([nothing, 1]; edge_divisions = [nothing, 1])
        a = toy_assembly(nchrom = 1, len = 400)
        res = simulate_cnas(t, a, CNAModel(rate = PerDivision(1.0)); seed = 33)
        m = CNMatrix(res, BinGrid(a, 100); allele = false)
        @test m.allele === nothing
        @test size(m.total, 1) == 1
    end

    @testset "grid and result must agree on the assembly" begin
        t = phylotree([nothing, 1]; edge_divisions = [nothing, 1])
        a = toy_assembly(nchrom = 1, len = 400)
        res = simulate_cnas(t, a, CNAModel(rate = PerDivision(0.0)); seed = 34)
        other = BinGrid(toy_assembly(nchrom = 2, len = 400), 100)
        @test_throws ArgumentError CNMatrix(res, other)
    end

    @testset "matrix total is the sum of haplotype projections, not a reprojected total" begin
        # Projection is non-linear, so a length-weighted majority of a sum is not the
        # sum of the majorities. This fixture makes the two definitions disagree:
        #   hap1: 1:60 at cn 3, 61:100 at cn 0   -> majority 3 (60bp beats 40bp)
        #   hap2: 1:40 at cn 0, 41:100 at cn 2   -> majority 2 (60bp beats 40bp)
        #   additive total                        = 3 + 2 = 5
        #   summed segmentation: (1:40,3) (41:60,5) (61:100,2), lengths 40/20/40,
        #     so the majority ties at 40bp between cn 3 and cn 2 and resolves low -> 2
        a = toy_assembly(nchrom = 1, len = 100)
        g = BinGrid(a, 100)                      # exactly one bin, 1:100
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 1, 60, 2, :focal))       # hap1 1:60 -> cn 3
        apply!(p, SegmentalCNA(1, 1, 61, 100, -1, :focal))    # hap1 61:100 -> cn 0
        apply!(p, SegmentalCNA(1, 2, 1, 40, -1, :focal))      # hap2 1:40 -> cn 0
        apply!(p, SegmentalCNA(1, 2, 41, 100, 1, :focal))     # hap2 41:100 -> cn 2

        tot, alleles = project(p, g)
        @test alleles[1] == [3]
        @test alleles[2] == [2]
        @test tot == [5]
        @test tot == alleles[1] .+ alleles[2]

        # and the rejected definition really does give a different answer here,
        # so this test would fail if project ever reprojected the summed segmentation
        @test project(total_cn(p, 1), g, 1, LengthWeightedMajority()) == [2]
    end
end
