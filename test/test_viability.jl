@testset "viability" begin
    @testset "AllowAll permits everything" begin
        p = diploid(toy_assembly(nchrom = 1, len = 100))
        e = SegmentalCNA(1, 1, 1, 100, -5, :chromosome)
        @test isviable(AllowAll(), p, e)
        @test violation(AllowAll(), p, e) === nothing
        @test max_attempts(AllowAll()) == 1
    end

    @testset "gains and doublings are always viable" begin
        p = diploid(toy_assembly(nchrom = 1, len = 100))
        r = RejectAndRedraw()
        @test isviable(r, p, SegmentalCNA(1, 1, 1, 100, 1, :chromosome))
        @test isviable(r, p, WholeGenomeDoubling(:multiply))
        @test isviable(r, p, WholeGenomeDoubling(:increment))
    end

    @testset "one loss on a diploid autosome is fine, the second is not" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        r = RejectAndRedraw()
        first_loss = SegmentalCNA(1, 1, 21, 40, -1, :focal)
        @test isviable(r, p, first_loss)
        apply!(p, first_loss)
        # total is now 1 over 21:40, so losing the other haplotype there is inviable
        @test !isviable(r, p, SegmentalCNA(1, 2, 21, 40, -1, :focal))
        @test violation(r, p, SegmentalCNA(1, 2, 21, 40, -1, :focal)) === :min_total_cn
        # but losing the other haplotype elsewhere is fine
        @test isviable(r, p, SegmentalCNA(1, 2, 61, 80, -1, :focal))
    end

    @testset "partial overlap is caught" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        r = RejectAndRedraw()
        apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))    # total 1 on 21:40, 2 elsewhere
        # a loss on haplotype 2 that only partly overlaps the depleted span is still inviable
        @test !isviable(r, p, SegmentalCNA(1, 2, 31, 70, -1, :focal))
        @test isviable(r, p, SegmentalCNA(1, 2, 41, 70, -1, :focal))
    end

    @testset "losing a hemizygous chromosome is inviable" begin
        p = diploid(toy_sex_assembly(:male))
        a = p.assembly
        r = RejectAndRedraw()
        x = chromindex(a, "chrX")
        y = chromindex(a, "chrY")
        @test !isviable(r, p, SegmentalCNA(x, 1, 1, 1000, -1, :chromosome))
        @test !isviable(r, p, SegmentalCNA(y, 1, 1, 1000, -1, :chromosome))
        @test !isviable(r, p, SegmentalCNA(x, 1, 100, 200, -1, :focal))
    end

    @testset "zero-copy material is already absent, so a further loss changes nothing" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        r = RejectAndRedraw()
        apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))
        # haplotype 1 is at 0 over 21:40; total there is 1 (from haplotype 2).
        # Losing haplotype 1 again cannot reduce the total, so it stays viable.
        @test isviable(r, p, SegmentalCNA(1, 1, 21, 40, -1, :focal))

        # With BOTH haplotypes at zero over 21:40 the total there is already 0, which
        # is below min_total_cn. A rule that shortcuts on "target already absent" would
        # wrongly call this viable; correctly summing across slots reports the breach.
        q = diploid(a)
        apply!(q, SegmentalCNA(1, 1, 21, 40, -1, :focal))
        apply!(q, SegmentalCNA(1, 2, 21, 40, -1, :focal))
        @test violation(r, q, SegmentalCNA(1, 1, 21, 40, -1, :focal)) === :min_total_cn
    end

    @testset "min_total_cn is a parameter" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        @test isviable(RejectAndRedraw(min_total_cn = 1), p, SegmentalCNA(1, 1, 1, 100, -1, :chromosome))
        @test !isviable(RejectAndRedraw(min_total_cn = 2), p, SegmentalCNA(1, 1, 1, 100, -1, :chromosome))
        @test isviable(RejectAndRedraw(min_total_cn = 0), p, SegmentalCNA(1, 1, 1, 100, -2, :chromosome))
    end

    @testset "constructor validation and max_attempts" begin
        @test_throws ArgumentError RejectAndRedraw(min_total_cn = -1)
        @test_throws ArgumentError RejectAndRedraw(max_attempts = 0)
        @test max_attempts(RejectAndRedraw()) == 100
        @test max_attempts(RejectAndRedraw(max_attempts = 7)) == 7
    end

    @testset "AllRules composes and reports the first violation" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        both = AllRules([AllowAll(), RejectAndRedraw(min_total_cn = 2)])
        @test !isviable(both, p, SegmentalCNA(1, 1, 1, 100, -1, :chromosome))
        @test violation(both, p, SegmentalCNA(1, 1, 1, 100, -1, :chromosome)) === :min_total_cn
        @test isviable(AllRules([AllowAll()]), p, SegmentalCNA(1, 1, 1, 100, -9, :chromosome))
        @test max_attempts(both) == 100
        @test_throws ArgumentError AllRules(ViabilityRule[])
    end
end
