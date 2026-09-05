@testset "profile" begin
    S = CopyNumberEvolution.Segment

    @testset "Segment length" begin
        @test length(S(1, 10, 2)) == 10
        @test length(S(5, 5, 0)) == 1
    end

    @testset "diploid profile" begin
        a = toy_assembly(nchrom = 2, len = 1000)
        p = diploid(a)
        @test check_invariants(p)
        @test nslots(a) == 4
        for c in 1:2, h in 1:2
            @test slot_segments(p, c, h) == [S(1, 1000, 1)]
        end
        @test nsegments(p) == 4
    end

    @testset "sex modes give the documented slot contents" begin
        f = diploid(toy_sex_assembly(:female))
        af = f.assembly
        @test ploidy(af, chromindex(af, "chrX")) == 2
        @test isempty(slots_of(af, chromindex(af, "chrY")))
        m = diploid(toy_sex_assembly(:male))
        am = m.assembly
        @test slot_segments(m, chromindex(am, "chrX"), 1) == [S(1, 1000, 1)]
        @test slot_segments(m, chromindex(am, "chrY"), 1) == [S(1, 1000, 1)]
        @test_throws ArgumentError slot_segments(m, chromindex(am, "chrX"), 2)
    end

    @testset "copy is deep" begin
        p = diploid(toy_assembly())
        q = copy(p)
        push!(q.segments[1], S(1, 1, 9))
        @test length(p.segments[1]) == 1
    end

    @testset "equality and hashing use canonical form" begin
        p = diploid(toy_assembly())
        q = diploid(toy_assembly())
        @test p == q
        @test hash(p) == hash(q)
        r = copy(p)
        r.segments[1][1] = S(1, 1000, 2)
        @test p != r
    end

    @testset "equality requires the same genome, not just the same assembly name" begin
        a1 = toy_assembly(nchrom = 1, len = 100)
        a2 = toy_assembly(nchrom = 2, len = 100)   # same name "toy", same sex
        @test !same_assembly(a1, a2)
        p1 = diploid(a1)
        p2 = diploid(a2)
        @test p1 != p2
    end

    @testset "canonicalize! merges adjacent equal copy numbers" begin
        segs = [S(1, 10, 1), S(11, 20, 1), S(21, 30, 2), S(31, 40, 2), S(41, 50, 1)]
        canonicalize!(segs)
        @test segs == [S(1, 20, 1), S(21, 40, 2), S(41, 50, 1)]
        single = [S(1, 5, 3)]
        @test canonicalize!(single) == [S(1, 5, 3)]
        allsame = [S(1, 2, 0), S(3, 4, 0), S(5, 6, 0)]
        @test canonicalize!(allsame) == [S(1, 6, 0)]
    end

    @testset "check_invariants names every violation" begin
        a = toy_assembly(nchrom = 1, len = 100)
        mk(segs) = CopyNumberEvolution.CNProfile(a, [copy(segs), copy(segs)])
        @test check_invariants(mk([S(1, 100, 1)]))
        # does not start at 1
        @test_throws ErrorException check_invariants(mk([S(2, 100, 1)]))
        # does not reach the chromosome end
        @test_throws ErrorException check_invariants(mk([S(1, 99, 1)]))
        # gap
        @test_throws ErrorException check_invariants(mk([S(1, 40, 1), S(42, 100, 2)]))
        # overlap
        @test_throws ErrorException check_invariants(mk([S(1, 40, 1), S(40, 100, 2)]))
        # adjacent equal cn (non-canonical)
        @test_throws ErrorException check_invariants(mk([S(1, 40, 1), S(41, 100, 1)]))
        # negative cn
        @test_throws ErrorException check_invariants(mk([S(1, 40, -1), S(41, 100, 1)]))
        # start > stop
        @test_throws ErrorException check_invariants(mk([S(40, 1, 1)]))
        # empty
        @test_throws ErrorException check_invariants(mk(S[]))
    end

    @testset "segment_index and cn_at" begin
        segs = [S(1, 10, 1), S(11, 20, 3), S(21, 30, 0)]
        @test segment_index(segs, 1) == 1
        @test segment_index(segs, 10) == 1
        @test segment_index(segs, 11) == 2
        @test segment_index(segs, 30) == 3
        @test cn_at(segs, 5) == 1
        @test cn_at(segs, 11) == 3
        @test cn_at(segs, 25) == 0
    end

    @testset "total_cn sums haplotypes and stays canonical" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        @test total_cn(p, 1) == [S(1, 100, 2)]
        # a gain on haplotype 1 over 21:40 only
        p.segments[slot(a, 1, 1)] = [S(1, 20, 1), S(21, 40, 2), S(41, 100, 1)]
        @test total_cn(p, 1) == [S(1, 20, 2), S(21, 40, 3), S(41, 100, 2)]
        # a matching loss on haplotype 2 makes the total flat again — and canonical
        p.segments[slot(a, 1, 2)] = [S(1, 20, 1), S(21, 40, 0), S(41, 100, 1)]
        @test total_cn(p, 1) == [S(1, 100, 2)]
    end

    @testset "total_cn on hemizygous and absent chromosomes" begin
        m = diploid(toy_sex_assembly(:male))
        am = m.assembly
        @test total_cn(m, chromindex(am, "chrX")) == [S(1, 1000, 1)]
        f = diploid(toy_sex_assembly(:female))
        af = f.assembly
        @test total_cn(f, chromindex(af, "chrY")) == [S(1, 1000, 0)]
        @test total_cn(f, chromindex(af, "chrX")) == [S(1, 1000, 2)]
    end

    @testset "total_cn equals the sum over slots at every position" begin
        rng = Random.Xoshiro(1234)
        a = toy_assembly(nchrom = 2, len = 200)
        p = diploid(a)
        # scatter random breakpoints and copy numbers, keeping canonical form
        for s in 1:nslots(a)
            L = chromlength(a, slot_chrom(a, s))
            bps = sort(unique(rand(rng, 2:L, 6)))
            segs = CopyNumberEvolution.Segment[]
            prev = 1
            for b in vcat(bps, L + 1)
                b > prev || continue
                push!(segs, CopyNumberEvolution.Segment(prev, b - 1, rand(rng, 0:4)))
                prev = b
            end
            canonicalize!(segs)
            p.segments[s] = segs
        end
        @test check_invariants(p)
        for c in 1:nchromosomes(a)
            tot = total_cn(p, c)
            for pos in (1, 37, 100, 199, chromlength(a, c))
                expected = sum(cn_at(p.segments[s], pos) for s in slots_of(a, c))
                @test cn_at(tot, pos) == expected
            end
        end
    end

    @testset "mean_cn" begin
        @test CopyNumberEvolution.mean_cn([S(1, 50, 2), S(51, 100, 0)], 100) ≈ 1.0
        @test CopyNumberEvolution.mean_cn([S(1, 100, 3)], 100) ≈ 3.0
    end
end
