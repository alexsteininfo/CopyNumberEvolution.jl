"""
    CNAEvent

Abstract supertype of copy-number-altering events. Concrete subtypes are
[`SegmentalCNA`](@ref) and [`WholeGenomeDoubling`](@ref).

All state changes go through the single method `apply!(profile, event)`. Keeping the
interface this narrow is what allows a karyotype backend to be added later without
touching tree traversal or the output layer.
"""
abstract type CNAEvent end

const EVENT_SCALES = (:focal, :arm, :chromosome)
const WGD_MODES = (:multiply, :increment)

"""
    SegmentalCNA(chrom, haplotype, start, stop, delta, scale)

A gain or loss of `delta` copies over `start:stop` on one haplotype of one chromosome.

Coordinates are 1-based inclusive. `delta` is positive for a gain and negative for a
loss; a loss that drives the copy number to 0 *is* loss of heterozygosity and needs no
separate event type. `scale` records which class of draw produced the event —
`:focal`, `:arm` or `:chromosome` — so events can be tallied by class afterwards; it
carries no semantics of its own, since an arm-level and a focal event differ only in
their extent.
"""
struct SegmentalCNA <: CNAEvent
    chrom::Int
    haplotype::Int
    start::Int
    stop::Int
    delta::Int
    scale::Symbol

    function SegmentalCNA(chrom::Integer, haplotype::Integer, start::Integer,
                          stop::Integer, delta::Integer, scale::Symbol)
        chrom >= 1 || throw(ArgumentError("chrom must be ≥ 1, got $chrom"))
        haplotype >= 1 || throw(ArgumentError("haplotype must be ≥ 1, got $haplotype"))
        start >= 1 || throw(ArgumentError("start must be ≥ 1, got $start"))
        start <= stop || throw(ArgumentError("start $start exceeds stop $stop"))
        delta != 0 || throw(ArgumentError("delta must be non-zero; a zero-delta event is not an event"))
        scale in EVENT_SCALES ||
            throw(ArgumentError("scale must be one of $EVENT_SCALES, got :$scale"))
        new(Int(chrom), Int(haplotype), Int(start), Int(stop), Int(delta), scale)
    end
end

"""
    event_span(event) -> Int

Number of base pairs a [`SegmentalCNA`](@ref) covers.
"""
event_span(e::SegmentalCNA) = e.stop - e.start + 1

Base.show(io::IO, e::SegmentalCNA) = print(io,
    e.delta > 0 ? "gain" : "loss", "(chrom=", e.chrom, ", hap=", e.haplotype,
    ", ", e.start, ":", e.stop, ", Δ=", e.delta, ", ", e.scale, ")")

"""
    WholeGenomeDoubling(mode = :multiply)

A doubling of the entire genome, crossing chromosome boundaries.

`mode` decides the arithmetic and matters as soon as any segment is already at copy
number 2 or more:

- `:multiply` — every copy number is doubled (`cn → 2cn`). This is what
  tetraploidization means, and it preserves zeros for free. **The default.**
- `:increment` — every *non-zero* copy number gains one (`cn → cn + 1`). This is
  MEDICC2's own definition of a WGD event, provided so that profiles can be generated
  on MEDICC2's terms.

The two coincide while all copy numbers are 0 or 1.
"""
struct WholeGenomeDoubling <: CNAEvent
    mode::Symbol

    function WholeGenomeDoubling(mode::Symbol = :multiply)
        mode in WGD_MODES ||
            throw(ArgumentError("WGD mode must be one of $WGD_MODES, got :$mode"))
        new(mode)
    end
end

Base.show(io::IO, e::WholeGenomeDoubling) = print(io, "WGD(:", e.mode, ")")

# Split the segment containing `pos` so that a segment starts exactly at `pos`.
# No-op when `pos` is already a boundary or lies outside the segmentation.
function _split_at!(segs::Vector{Segment}, pos::Int)
    pos <= segs[1].start && return segs
    pos > segs[end].stop && return segs
    i = segment_index(segs, pos)
    sg = segs[i]
    sg.start == pos && return segs
    segs[i] = Segment(sg.start, pos - 1, sg.cn)
    insert!(segs, i + 1, Segment(pos, sg.stop, sg.cn))
    return segs
end

# Zero is absorbing: absent DNA can never be re-gained, and a loss never goes below
# zero. See the manual's "Modelling choices" page.
_shift_cn(cn::Int, delta::Int) = cn == 0 ? 0 : max(0, cn + delta)

"""
    apply!(profile, event) -> profile

Apply `event` to `profile` in place, restoring canonical form.

For a [`SegmentalCNA`](@ref): split at the two breakpoints, shift the copy number of
every covered segment by `delta`, then re-canonicalise. **Zero-copy state is
absorbing** — a gain never raises a segment at copy number 0, because that DNA is
physically absent — and losses clamp at 0.

For a [`WholeGenomeDoubling`](@ref): shift every segment of every slot per the event's
`mode`, ignoring chromosome boundaries.
"""
function apply!(p::CNProfile, e::SegmentalCNA)
    a = p.assembly
    L = chromlength(a, e.chrom)
    e.stop <= L || throw(ArgumentError(
        "event spans $(e.start):$(e.stop) but chromosome $(chromname(a, e.chrom)) is only $L bp"))
    segs = p.segments[slot(a, e.chrom, e.haplotype)]
    _split_at!(segs, e.start)
    _split_at!(segs, e.stop + 1)
    for i in eachindex(segs)
        sg = segs[i]
        if sg.start >= e.start && sg.stop <= e.stop
            segs[i] = Segment(sg.start, sg.stop, _shift_cn(sg.cn, e.delta))
        end
    end
    canonicalize!(segs)
    return p
end

function apply!(p::CNProfile, e::WholeGenomeDoubling)
    for segs in p.segments
        for i in eachindex(segs)
            sg = segs[i]
            newcn = e.mode === :multiply ? 2 * sg.cn : _shift_cn(sg.cn, 1)
            segs[i] = Segment(sg.start, sg.stop, newcn)
        end
        canonicalize!(segs)
    end
    return p
end
