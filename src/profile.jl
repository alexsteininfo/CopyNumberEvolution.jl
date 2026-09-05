"""
    Segment(start, stop, cn)

A run of constant copy number on one haplotype of one chromosome.

Coordinates are **1-based inclusive**, so `Segment(1, 10, 2)` covers ten base pairs.
`cn` is the number of copies of *this haplotype* over the interval, so a normal
diploid autosome is two slots each holding a single `Segment(1, L, 1)` — which is
exactly MEDICC2's `cn_a`/`cn_b` convention.
"""
struct Segment
    start::Int
    stop::Int
    cn::Int
end

Base.length(s::Segment) = s.stop - s.start + 1

Base.show(io::IO, s::Segment) = print(io, "Segment(", s.start, ":", s.stop, ", cn=", s.cn, ")")

"""
    CNProfile(assembly, segments)

One cell's allele-specific copy-number state.

`segments[s]` is the segmentation of haplotype slot `s` (see [`GenomeAssembly`](@ref)):
a sorted, gapless, non-overlapping list of [`Segment`](@ref)s tiling
`1:chromlength(assembly, slot_chrom(assembly, s))`, with no two adjacent segments
sharing a copy number. That canonical form makes `==` meaningful — two profiles are
equal exactly when their segmentations are.

Storage is indexed by slot rather than keyed by `(chromosome, haplotype)` because
`Dict` iteration order is unspecified, and any RNG-consuming pass over haplotypes
would then be irreproducible.

Total copy number is a derived view, never a second representation — see
[`total_cn`](@ref).

Build a normal starting state with [`diploid`](@ref); validate one with
[`check_invariants`](@ref).
"""
struct CNProfile
    assembly::GenomeAssembly
    segments::Vector{Vector{Segment}}
end

"""
    diploid(assembly) -> CNProfile

The normal starting karyotype for `assembly`: every haplotype slot is a single
segment at copy number 1, spanning its whole chromosome.

Sex mode is already baked into the assembly's slot layout, so a male assembly yields
one `chrX` and one `chrY` slot and a female assembly yields two `chrX` slots and no
`chrY` slot.

# Examples
```jldoctest
julia> p = diploid(hg38(:female));

julia> only(total_cn(p, 1)).cn
2

julia> nsegments(p)
46
```
"""
function diploid(a::GenomeAssembly)
    segs = Vector{Vector{Segment}}(undef, nslots(a))
    for c in 1:nchromosomes(a), h in 1:ploidy(a, c)
        segs[slot(a, c, h)] = [Segment(1, chromlength(a, c), 1)]
    end
    return CNProfile(a, segs)
end

Base.copy(p::CNProfile) = CNProfile(p.assembly, [copy(v) for v in p.segments])

# Equality compares the full genome (via `same_assembly`), not just the assembly's
# name and sex. Every test fixture is named "toy" with sex :female, so a name-and-sex
# comparison could not distinguish, say, a 2-chromosome toy assembly from a
# 3-chromosome one — and later tasks depend on that distinction.
Base.:(==)(x::CNProfile, y::CNProfile) =
    same_assembly(x.assembly, y.assembly) && x.segments == y.segments

# Hashing on name and sex only is a valid, cheaper hash that stays consistent with the
# stricter equality above: equal profiles still hash equal, which is the only contract
# `hash` must honour.
Base.hash(p::CNProfile, h::UInt) =
    hash(p.segments, hash(p.assembly.sex, hash(p.assembly.name, h)))

Base.show(io::IO, p::CNProfile) =
    print(io, "CNProfile(", p.assembly.name, ", :", p.assembly.sex, ", ",
          nsegments(p), " segments across ", length(p.segments), " slots)")

"""
    nsegments(profile) -> Int

Total number of segments across all slots — a cheap measure of how fragmented a
profile has become.
"""
nsegments(p::CNProfile) = sum(length, p.segments)

"""
    slot_segments(profile, chrom, haplotype) -> Vector{Segment}

The segmentation of one haplotype of one chromosome. Throws if that haplotype does
not exist at the chromosome's ploidy.
"""
slot_segments(p::CNProfile, c::Integer, h::Integer) =
    p.segments[slot(p.assembly, c, h)]

"""
    canonicalize!(segs) -> segs

Merge adjacent segments with equal copy number, in place. Restores the canonical
form required by [`check_invariants`](@ref) after an edit that may have left two
neighbours sharing a copy number.
"""
function canonicalize!(segs::Vector{Segment})
    i = 1
    while i < length(segs)
        if segs[i].cn == segs[i + 1].cn
            segs[i] = Segment(segs[i].start, segs[i + 1].stop, segs[i].cn)
            deleteat!(segs, i + 1)
        else
            i += 1
        end
    end
    return segs
end

"""
    check_invariants(profile) -> true

Verify every slot's segmentation and throw a descriptive `ErrorException` naming the
slot and segment index on the first violation. Checks that each slot is non-empty,
starts at 1, ends at the chromosome length, has no gaps or overlaps, has no adjacent
segments with equal copy number, and has no negative copy number.

A silently non-canonical segmentation is the failure mode that would poison every
downstream number, so call this liberally in tests.
"""
function check_invariants(p::CNProfile)
    a = p.assembly
    for c in 1:nchromosomes(a), h in 1:ploidy(a, c)
        s = slot(a, c, h)
        segs = p.segments[s]
        L = chromlength(a, c)
        where = "slot $s (chromosome $(chromname(a, c)), haplotype $h)"
        isempty(segs) && error("$where: empty segmentation")
        segs[1].start == 1 ||
            error("$where: first segment starts at $(segs[1].start), expected 1")
        segs[end].stop == L ||
            error("$where: last segment stops at $(segs[end].stop), expected $L")
        for (i, sg) in enumerate(segs)
            sg.start <= sg.stop ||
                error("$where segment $i: start $(sg.start) exceeds stop $(sg.stop)")
            sg.cn >= 0 ||
                error("$where segment $i: negative copy number $(sg.cn)")
            if i > 1
                prev = segs[i - 1]
                sg.start == prev.stop + 1 ||
                    error("$where segment $i: starts at $(sg.start), expected $(prev.stop + 1) (gap or overlap)")
                sg.cn != prev.cn ||
                    error("$where segment $i: adjacent segments both at copy number $(sg.cn); not canonical")
            end
        end
    end
    return true
end

"""
    segment_index(segs, pos) -> Int

Index of the segment containing `pos`, by binary search. `segs` must be sorted and
gapless, and `pos` must lie within it.
"""
function segment_index(segs::Vector{Segment}, pos::Integer)
    lo, hi = 1, length(segs)
    while lo < hi
        mid = (lo + hi + 1) >> 1
        if segs[mid].start <= pos
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end

"""
    cn_at(segs, pos) -> Int

Copy number at position `pos` in a single slot's segmentation.
"""
cn_at(segs::Vector{Segment}, pos::Integer) = segs[segment_index(segs, pos)].cn

"""
    total_cn(profile, chrom) -> Vector{Segment}

Total copy number of chromosome `chrom`, summed over its haplotype slots and returned
in canonical form.

This is a *derived view*, not a stored second representation: simulation is always
allele-specific, and the total-copy-number "mode" affects only output and comparison.
A chromosome with zero ploidy (`chrY` in a female assembly) returns a single segment
at copy number 0.
"""
function total_cn(p::CNProfile, c::Integer)
    a = p.assembly
    L = chromlength(a, c)
    sl = slots_of(a, c)
    isempty(sl) && return [Segment(1, L, 0)]

    breaks = Int[1]
    for s in sl, sg in p.segments[s]
        sg.start > 1 && push!(breaks, sg.start)
    end
    sort!(breaks)
    unique!(breaks)

    out = Vector{Segment}(undef, length(breaks))
    for (i, b) in enumerate(breaks)
        stop = i == length(breaks) ? L : breaks[i + 1] - 1
        tot = 0
        for s in sl
            tot += cn_at(p.segments[s], b)
        end
        out[i] = Segment(b, stop, tot)
    end
    return canonicalize!(out)
end

"""
    total_cn(profile) -> Vector{Vector{Segment}}

Total copy number for every chromosome, in chromosome order.
"""
total_cn(p::CNProfile) = [total_cn(p, c) for c in 1:nchromosomes(p.assembly)]

"""
    mean_cn(segs, len) -> Float64

Length-weighted mean copy number of one slot's segmentation over a chromosome of
`len` bp. Used by [`CNWeighted`](@ref) to condition target choice on the mother
cell's copy-number state.
"""
function mean_cn(segs::Vector{Segment}, len::Integer)
    acc = 0
    for sg in segs
        acc += length(sg) * sg.cn
    end
    return acc / len
end
