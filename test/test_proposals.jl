@testset "proposals" begin
    @testset "UniformChromosome covers eligible chromosomes evenly" begin
        p = diploid(toy_sex_assembly(:female))
        a = p.assembly
        rng = Random.Xoshiro(11)
        counts = zeros(Int, nchromosomes(a))
        for _ in 1:60_000
            c, h = draw_target(UniformChromosome(), p, rng)
            counts[c] += 1
            @test 1 <= h <= ploidy(a, c)
        end
        y = chromindex(a, "chrY")
        @test counts[y] == 0                       # zero ploidy is never targeted
        for c in eligible_chromosomes(a)
            @test counts[c] ≈ 60_000 / 3 rtol = 0.05
        end
    end

    @testset "hemizygous chromosomes only ever yield haplotype 1" begin
        p = diploid(toy_sex_assembly(:male))
        a = p.assembly
        rng = Random.Xoshiro(12)
        x = chromindex(a, "chrX")
        seen = Set{Int}()
        for _ in 1:20_000
            c, h = draw_target(UniformChromosome(), p, rng)
            c == x && push!(seen, h)
        end
        @test seen == Set([1])
    end

    @testset "LengthWeighted is proportional to chromosome length" begin
        specs = [CopyNumberEvolution.ChromosomeSpec("chr1", 3000, 1201:1800),
                 CopyNumberEvolution.ChromosomeSpec("chr2", 1000, 401:600)]
        a = CopyNumberEvolution.GenomeAssembly("w", :female, specs, [2, 2])
        p = diploid(a)
        rng = Random.Xoshiro(13)
        n1 = 0
        for _ in 1:40_000
            c, _ = draw_target(LengthWeighted(), p, rng)
            c == 1 && (n1 += 1)
        end
        @test n1 / 40_000 ≈ 0.75 rtol = 0.03
    end

    @testset "CNWeighted conditions on the mother cell's copy-number state" begin
        a = toy_assembly(nchrom = 2, len = 1000)
        p = diploid(a)
        # take chromosome 1 haplotype 1 to copy number 4
        apply!(p, SegmentalCNA(1, 1, 1, 1000, 3, :chromosome))
        rng = Random.Xoshiro(14)
        gained = 0
        target = slot(a, 1, 1)
        for _ in 1:40_000
            c, h = draw_target(CNWeighted(1.0), p, rng)
            slot(a, c, h) == target && (gained += 1)
        end
        # weights are 4 : 1 : 1 : 1, so the gained slot takes 4/7 of the draws
        @test gained / 40_000 ≈ 4 / 7 rtol = 0.04
    end

    @testset "CNWeighted never targets fully deleted material" begin
        a = toy_assembly(nchrom = 2, len = 1000)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 1, 1000, -1, :chromosome))   # slot now entirely 0
        deleted = slot(a, 1, 1)
        rng = Random.Xoshiro(15)
        for _ in 1:20_000
            c, h = draw_target(CNWeighted(1.0), p, rng)
            @test slot(a, c, h) != deleted
        end
        # β = 0 must not resurrect it via 0^0 == 1
        for _ in 1:20_000
            c, h = draw_target(CNWeighted(0.0), p, rng)
            @test slot(a, c, h) != deleted
        end
    end

    @testset "CNWeighted with every slot deleted is an error, not a silent hang" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 1, 100, -1, :chromosome))
        apply!(p, SegmentalCNA(1, 2, 1, 100, -1, :chromosome))
        @test_throws ArgumentError draw_target(CNWeighted(1.0), p, Random.Xoshiro(1))
    end

    @testset "ExtentMixture validation" begin
        @test_throws ArgumentError ExtentMixture(p_chromosome = 0.6, p_arm = 0.6)
        @test_throws ArgumentError ExtentMixture(p_chromosome = -0.1)
        m = ExtentMixture()
        @test m.p_chromosome == 0.0 && m.p_arm == 0.0
    end

    @testset "whole-chromosome events span exactly the chromosome" begin
        a = toy_assembly(nchrom = 2, len = 1000)
        p = diploid(a)
        d = ExtentMixture(p_chromosome = 1.0)
        rng = Random.Xoshiro(16)
        for _ in 1:200
            s, e, sc = draw_extent(d, p, 1, 1, rng)
            @test (s, e, sc) == (1, 1000, :chromosome)
        end
    end

    @testset "whole-arm events respect the centromere" begin
        a = toy_assembly(nchrom = 1, len = 1000)   # centromere 401:600
        p = diploid(a)
        d = ExtentMixture(p_arm = 1.0)
        rng = Random.Xoshiro(17)
        seen = Set{Tuple{Int,Int}}()
        for _ in 1:400
            s, e, sc = draw_extent(d, p, 1, 1, rng)
            @test sc === :arm
            push!(seen, (s, e))
            @test isempty(intersect(s:e, centromere(a, 1)))
        end
        @test seen == Set([(1, 400), (601, 1000)])
    end

    @testset "focal events stay inside the chromosome" begin
        a = toy_assembly(nchrom = 1, len = 1000)
        p = diploid(a)
        d = ExtentMixture(lengthdist = Distributions.LogUniform(10.0, 5000.0))
        rng = Random.Xoshiro(18)
        truncated = 0
        for _ in 1:5_000
            s, e, sc = draw_extent(d, p, 1, 1, rng)
            @test sc === :focal
            @test 1 <= s <= e <= 1000
            e == 1000 && (truncated += 1)
        end
        @test truncated > 0     # truncation at the boundary does happen, per MEDICC2
    end

    @testset "the three extent classes appear at their stated rates" begin
        a = toy_assembly(nchrom = 1, len = 1000)
        p = diploid(a)
        d = ExtentMixture(p_chromosome = 0.2, p_arm = 0.3)
        rng = Random.Xoshiro(19)
        tally = Dict(:chromosome => 0, :arm => 0, :focal => 0)
        for _ in 1:40_000
            _, _, sc = draw_extent(d, p, 1, 1, rng)
            tally[sc] += 1
        end
        @test tally[:chromosome] / 40_000 ≈ 0.2 rtol = 0.05
        @test tally[:arm] / 40_000 ≈ 0.3 rtol = 0.05
        @test tally[:focal] / 40_000 ≈ 0.5 rtol = 0.05
    end

    @testset "GainLoss" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        rng = Random.Xoshiro(20)
        @test all(draw_kind(GainLoss(1.0), p, 1, 1, 1, 10, rng) == 1 for _ in 1:100)
        @test all(draw_kind(GainLoss(0.0), p, 1, 1, 1, 10, rng) == -1 for _ in 1:100)
        @test all(draw_kind(GainLoss(1.0, 3), p, 1, 1, 1, 10, rng) == 3 for _ in 1:100)
        gains = count(_ -> draw_kind(GainLoss(0.7), p, 1, 1, 1, 10, rng) > 0, 1:40_000)
        @test gains / 40_000 ≈ 0.7 rtol = 0.03
        @test_throws ArgumentError GainLoss(1.2)
        @test_throws ArgumentError GainLoss(0.5, 0)
    end

    @testset "plain functions are accepted as draws" begin
        a = toy_assembly(nchrom = 2, len = 100)
        p = diploid(a)
        rng = Random.Xoshiro(21)
        @test draw_target((prof, r) -> (2, 1), p, rng) == (2, 1)
        @test draw_extent((prof, c, h, r) -> (5, 15, :focal), p, 1, 1, rng) == (5, 15, :focal)
        @test draw_kind((prof, c, h, s, e, r) -> -2, p, 1, 1, 5, 15, rng) == -2
    end
end
