@testset "cna" begin
    S = CopyNumberEvolution.Segment

    @testset "constructor validation" begin
        @test_throws ArgumentError SegmentalCNA(1, 1, 10, 5, 1, :focal)
        @test_throws ArgumentError SegmentalCNA(1, 1, 0, 5, 1, :focal)
        @test_throws ArgumentError SegmentalCNA(1, 1, 1, 5, 0, :focal)
        @test_throws ArgumentError SegmentalCNA(1, 0, 1, 5, 1, :focal)
        @test_throws ArgumentError SegmentalCNA(1, 1, 1, 5, 1, :bogus)
        @test_throws ArgumentError WholeGenomeDoubling(:bogus)
        @test event_span(SegmentalCNA(1, 1, 11, 20, 1, :focal)) == 10
    end

    @testset "interior gain splits into three segments" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 21, 40, 1, :focal))
        @test slot_segments(p, 1, 1) == [S(1, 20, 1), S(21, 40, 2), S(41, 100, 1)]
        @test slot_segments(p, 1, 2) == [S(1, 100, 1)]
        @test check_invariants(p)
    end

    @testset "events at the chromosome boundaries" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 1, 30, 1, :focal))
        @test slot_segments(p, 1, 1) == [S(1, 30, 2), S(31, 100, 1)]
        q = diploid(a)
        apply!(q, SegmentalCNA(1, 1, 71, 100, 1, :focal))
        @test slot_segments(q, 1, 1) == [S(1, 70, 1), S(71, 100, 2)]
        r = diploid(a)
        apply!(r, SegmentalCNA(1, 1, 1, 100, 1, :chromosome))
        @test slot_segments(r, 1, 1) == [S(1, 100, 2)]
        @test check_invariants(p) && check_invariants(q) && check_invariants(r)
    end

    @testset "a loss to zero is LOH and needs no separate event type" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))
        @test slot_segments(p, 1, 1) == [S(1, 20, 1), S(21, 40, 0), S(41, 100, 1)]
        @test total_cn(p, 1) == [S(1, 20, 2), S(21, 40, 1), S(41, 100, 2)]
    end

    @testset "zero is absorbing: a gain never resurrects deleted material" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))       # 21:40 -> cn 0
        apply!(p, SegmentalCNA(1, 1, 1, 100, 1, :chromosome))   # gain the whole chromosome
        @test slot_segments(p, 1, 1) == [S(1, 20, 2), S(21, 40, 0), S(41, 100, 2)]
        @test check_invariants(p)
        # and a further loss over a zeroed run leaves it at zero, not negative
        apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))
        @test cn_at(slot_segments(p, 1, 1), 30) == 0
    end

    @testset "losses clamp at zero" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 1, 100, -3, :chromosome))
        @test slot_segments(p, 1, 1) == [S(1, 100, 0)]
        @test check_invariants(p)
    end

    @testset "overlapping events accumulate and re-canonicalise" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 21, 60, 1, :focal))
        apply!(p, SegmentalCNA(1, 1, 41, 80, 1, :focal))
        @test slot_segments(p, 1, 1) ==
              [S(1, 20, 1), S(21, 40, 2), S(41, 60, 3), S(61, 80, 2), S(81, 100, 1)]
        # a loss that exactly cancels the first gain merges neighbours again
        apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))
        @test slot_segments(p, 1, 1) ==
              [S(1, 40, 1), S(41, 60, 3), S(61, 80, 2), S(81, 100, 1)]
        @test check_invariants(p)
    end

    @testset "events on the wrong chromosome or haplotype throw" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        @test_throws ArgumentError apply!(p, SegmentalCNA(1, 3, 1, 10, 1, :focal))
        @test_throws BoundsError apply!(p, SegmentalCNA(9, 1, 1, 10, 1, :focal))
        @test_throws ArgumentError apply!(p, SegmentalCNA(1, 1, 1, 500, 1, :focal))
    end

    @testset "WGD :multiply doubles every slot and preserves zeros" begin
        a = toy_assembly(nchrom = 2, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))    # a zeroed run
        apply!(p, SegmentalCNA(2, 1, 1, 100, 1, :chromosome)) # a slot at cn 2
        apply!(p, WholeGenomeDoubling(:multiply))
        @test slot_segments(p, 1, 1) == [S(1, 20, 2), S(21, 40, 0), S(41, 100, 2)]
        @test slot_segments(p, 1, 2) == [S(1, 100, 2)]
        @test slot_segments(p, 2, 1) == [S(1, 100, 4)]
        @test check_invariants(p)
    end

    @testset "WGD :increment adds one to non-zero segments only" begin
        a = toy_assembly(nchrom = 2, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))
        apply!(p, SegmentalCNA(2, 1, 1, 100, 1, :chromosome))
        apply!(p, WholeGenomeDoubling(:increment))
        @test slot_segments(p, 1, 1) == [S(1, 20, 2), S(21, 40, 0), S(41, 100, 2)]
        @test slot_segments(p, 2, 1) == [S(1, 100, 3)]   # 2 + 1, not 2 * 2
    end

    @testset "the two WGD modes agree below cn 2 and diverge above it" begin
        a = toy_assembly(nchrom = 1, len = 100)
        flat = diploid(a)
        mult = copy(flat); incr = copy(flat)
        apply!(mult, WholeGenomeDoubling(:multiply))
        apply!(incr, WholeGenomeDoubling(:increment))
        @test mult == incr                     # all copy numbers were 0 or 1

        gained = diploid(a)
        apply!(gained, SegmentalCNA(1, 1, 1, 100, 1, :chromosome))   # cn 2
        mult2 = copy(gained); incr2 = copy(gained)
        apply!(mult2, WholeGenomeDoubling(:multiply))
        apply!(incr2, WholeGenomeDoubling(:increment))
        @test mult2 != incr2
        @test cn_at(slot_segments(mult2, 1, 1), 1) == 4
        @test cn_at(slot_segments(incr2, 1, 1), 1) == 3
    end

    @testset "WGD is not stopped by chromosome boundaries" begin
        a = toy_assembly(nchrom = 3, len = 50)
        p = diploid(a)
        apply!(p, WholeGenomeDoubling(:multiply))
        for c in 1:3, h in 1:2
            @test slot_segments(p, c, h) == [S(1, 50, 2)]
        end
    end
end
