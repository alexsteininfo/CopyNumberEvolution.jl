@testset "assembly" begin
    @testset "hg38/hg19 tables match the UCSC reference" begin
        for (ctor, file) in ((hg38, "hg38.reference.tsv"), (hg19, "hg19.reference.tsv"))
            a = ctor(:male)
            rows = readlines(joinpath(@__DIR__, "data", file))[2:end]
            @test nchromosomes(a) == length(rows)
            for (c, row) in enumerate(rows)
                name, len, cs, ce = split(row, '\t')
                @test chromname(a, c) == name
                @test chromlength(a, c) == parse(Int, len)
                @test centromere(a, c) == parse(Int, cs):parse(Int, ce)
            end
        end
    end

    @testset "sex sets the slot layout" begin
        f = hg38(:female)
        m = hg38(:male)
        @test nchromosomes(f) == 24 && nchromosomes(m) == 24
        @test ploidy(f, chromindex(f, "chrX")) == 2
        @test ploidy(f, chromindex(f, "chrY")) == 0
        @test ploidy(m, chromindex(m, "chrX")) == 1
        @test ploidy(m, chromindex(m, "chrY")) == 1
        # 22 autosome pairs + 2 X, and 22 pairs + X + Y: both are 46
        @test nslots(f) == 46
        @test nslots(m) == 46
        @test isempty(slots_of(f, chromindex(f, "chrY")))
    end

    @testset "slot mapping round-trips" begin
        for sex in (:female, :male)
            a = hg38(sex)
            seen = Int[]
            for c in 1:nchromosomes(a), h in 1:ploidy(a, c)
                s = slot(a, c, h)
                push!(seen, s)
                @test slot_chrom(a, s) == c
                @test slot_haplotype(a, s) == h
                @test s in slots_of(a, c)
            end
            @test sort(seen) == collect(1:nslots(a))
        end
    end

    @testset "slot bounds are checked" begin
        a = hg38(:male)
        @test_throws ArgumentError slot(a, chromindex(a, "chrX"), 2)
        @test_throws ArgumentError slot(a, chromindex(a, "chrY"), 2)
        @test_throws ArgumentError slot(a, 1, 0)
    end

    @testset "arms partition the chromosome around the centromere" begin
        a = hg38(:female)
        for c in 1:nchromosomes(a)
            p, q = arms(a, c)
            cen = centromere(a, c)
            @test last(p) == first(cen) - 1
            @test first(q) == last(cen) + 1
            @test last(q) == chromlength(a, c)
            @test isempty(intersect(p, cen)) && isempty(intersect(q, cen))
            @test length(p) + length(cen) + length(q) == chromlength(a, c)
        end
    end

    @testset "eligible chromosomes exclude zero-ploidy" begin
        f = hg38(:female)
        @test chromindex(f, "chrY") ∉ eligible_chromosomes(f)
        @test length(eligible_chromosomes(f)) == 23
        @test length(eligible_chromosomes(hg38(:male))) == 24
        @test length(autosomes(f)) == 22
    end

    @testset "unknown names and sexes are rejected" begin
        a = hg38(:female)
        @test_throws ArgumentError chromindex(a, "chr99")
        @test_throws ArgumentError hg38(:other)
    end

    @testset "same_assembly compares structure, not just the name" begin
        @test same_assembly(hg38(:female), hg38(:female))
        @test !same_assembly(hg38(:female), hg38(:male))
        @test !same_assembly(hg38(:female), hg19(:female))
        # two assemblies sharing a name but differing in structure are NOT the same
        s1 = [CopyNumberEvolution.ChromosomeSpec("chr1", 100, 41:60)]
        s2 = [CopyNumberEvolution.ChromosomeSpec("chr1", 100, 41:60),
              CopyNumberEvolution.ChromosomeSpec("chr2", 100, 41:60)]
        a1 = CopyNumberEvolution.GenomeAssembly("toy", :female, s1, [2])
        a2 = CopyNumberEvolution.GenomeAssembly("toy", :female, s2, [2, 2])
        @test !same_assembly(a1, a2)
        # differing only in a chromosome length is also not the same
        s3 = [CopyNumberEvolution.ChromosomeSpec("chr1", 200, 41:60)]
        a3 = CopyNumberEvolution.GenomeAssembly("toy", :female, s3, [2])
        @test !same_assembly(a1, a3)
    end
end
