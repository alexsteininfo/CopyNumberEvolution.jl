# Concepts

## A genome is a set of haplotype segmentations

A [`GenomeAssembly`](@ref) lists chromosomes — name, length, centromere — and, from a
sex mode, the number of copies of each in the normal karyotype. That determines the
**slot layout**: one slot per haplotype of per chromosome.

```julia
a = hg38(:female)                        # hg19(:male) works the same way
nchromosomes(a)                          # 24 — chrY is in the table with ploidy 0
nslots(a)                                # 46: 22 autosome pairs + 2 X
chromname(a, 1), chromlength(a, 1)       # "chr1", 248956422
centromere(a, 1)                         # 1-based inclusive
ploidy(a, chromindex(a, "chrY"))         # 0 — present, but no slots
slot(a, 1, 2)                            # linear index of chr1 haplotype 2
slots_of(a, chromindex(a, "chrX"))       # both X slots
slot_chrom(a, 3), slot_haplotype(a, 3)   # and back again
arms(a, 1)                               # p and q, flanking the centromere
```

[`hg38`](@ref) and [`hg19`](@ref) ship real chromosome lengths and centromere positions,
transcribed from UCSC and asserted in the test suite against a checked-in reference
file. Real lengths matter: segment sizes then have physical meaning, which is required
both for comparison with real data and for a realistic size distribution. A non-human
or deliberately synthetic genome is a [`GenomeAssembly`](@ref) built from your own
[`ChromosomeSpec`](@ref) list, with the ploidy vector given explicitly if the sex-mode
rules do not apply.

Sex mode **is** the slot layout rather than a separate switch. A male assembly has one
`chrX` and one `chrY` slot; hemizygosity is therefore representable rather than
special-cased. `chrY` stays in a female table with ploidy 0 so chromosome indices
compare across sexes, and [`eligible_chromosomes`](@ref) is what the proposal
distributions actually draw from. [`autosomes`](@ref) names the 22 non-sex chromosomes,
which is what MEDICC2 export uses by default.

Two assemblies are compared with [`same_assembly`](@ref), which checks name, sex, the
ploidy vector, and every chromosome's name, length and centromere. Name and sex alone
cannot distinguish two structurally different assemblies that happen to share both —
two `"toy"`/`:female` fixtures with a different number of chromosomes, say — so
`same_assembly` is the equality [`CNProfile`](@ref) equality is built on, and it also
guards [`Given`](@ref)'s validation, [`project`](@ref) and [`CNMatrix`](@ref), all of
which require their inputs to sit on the same genome.

A [`CNProfile`](@ref) holds one segmentation per slot. Each is a sorted, gapless,
non-overlapping list of [`Segment`](@ref)s tiling the chromosome — a piecewise-constant
step function of position:

```julia
p = diploid(hg38(:female))
slot_segments(p, 1, 1)        # [Segment(1:248956422, cn=1)]
nsegments(p)                  # 46 — one per slot, to start
```

`cn` counts copies of *that haplotype*, so `cn = 3` on one slot means three copies of
that parental chromosome. This is exactly MEDICC2's `cn_a`/`cn_b` convention.

### Why a segmentation, and why slot-indexed

A dictionary keyed by copy number cannot represent two disjoint segments that happen
to share one — an independent gain at 3p and at 3q, both landing at copy number 3.
That is the common case after a handful of alterations, and such a representation
would silently merge or lose one of them. A segmentation represents everything it can
plus everything it cannot, and it is **exactly what real copy-number callers emit**, so
the same type ingests real and simulated data.

Storage is a `Vector` indexed by slot rather than a `Dict` keyed by
`(chromosome, haplotype)` for a specific reason: `Dict` iteration order is unspecified
in Julia, so any pass over haplotypes that consumes the random number generator —
length-weighted or copy-number-conditioned target choice, doubling, viability checks —
would be irreproducible across Julia versions and insertion histories. Reproducibility
under a fixed seed is a tested property here, not an aspiration.

### Canonical form

Every segmentation is kept canonical: sorted, gapless, covering `1:length`, with **no
two adjacent segments sharing a copy number**, and no negative copy number. That makes
`==` meaningful — two profiles are equal exactly when their segmentations are — and it
is what [`check_invariants`](@ref) enforces:

```julia
check_invariants(p)            # true, or throws naming the slot and segment
canonicalize!(segs)            # merge adjacent equal-cn neighbours after an edit
```

The failure mode of this package is not a crash, it is a plausible-looking wrong
number. A silently non-canonical segmentation would poison everything downstream, so
the invariant checker is called liberally throughout the test suite.

Position lookups are `segment_index(segs, pos)` and [`cn_at`](@ref), both binary
searches.

## Allele-specific is the representation; total is a view

Simulation is *always* allele-specific. Total copy number is derived:

```julia
total_cn(p, 1)      # summed over chr1's slots, re-canonicalised
total_cn(p)         # every chromosome
mean_cn(segs, len)  # length-weighted mean of one slot
```

Two simulation paths would be twice the code and twice the test burden, and they could
disagree. So the "total copy number mode" affects output and comparison only, never
the model. This is also the right call scientifically: everything interesting — loss
of heterozygosity, mirrored subclonal allelic imbalance, parallel evolution on distinct
haplotypes — is invisible in total copy number.

## Four things can happen

The event taxonomy follows MEDICC2's evolutionary model, which is the closest published
statement of which copy-number aberrations matter.

| Event | Effect | Extent |
|:---|:---|:---|
| Segmental gain | `+δ` over a run | any contiguous interval within one chromosome, one haplotype |
| Segmental loss | `−δ` over a run | as above |
| Loss of heterozygosity | a loss reaching `cn = 0` | not a separate type — see below |
| Whole-genome doubling | every slot | the whole genome, crossing chromosome boundaries |

[`CNAEvent`](@ref) has two concrete subtypes, [`SegmentalCNA`](@ref) and
[`WholeGenomeDoubling`](@ref), and [`event_span`](@ref) gives a segmental event's
length in base pairs. Everything is applied through one method:

```julia
apply!(profile, event)
```

Keeping that interface narrow is deliberate: a karyotype backend could later be added
behind it without touching tree traversal or the output layer.

Three consequences of the table worth stating plainly:

**Arbitrary length means arm- and chromosome-level events are not a different kind of
thing.** A focal event and a whole-chromosome event are the same event type at
different extents. They still get their own probabilities in
[`ExtentMixture`](@ref), because no continuous length distribution produces them at a
realistic rate — but nothing downstream treats them specially. The `scale` field
records which class of draw produced an event so it can be tallied afterwards.

**Loss of heterozygosity needs no separate event type.** It is simply a loss that
reaches copy number 0.

**Copy number 0 is absorbing.** A segment at 0 is *absent DNA* and can never be
regained. A gain spanning a run that contains zeroed sub-segments raises the non-zero
parts and leaves the zeros at zero:

```julia
p = diploid(toy)
apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))       # 21:40 -> 0
apply!(p, SegmentalCNA(1, 1, 1, 100, +1, :chromosome))  # 21:40 stays 0
```

This is unconditionally true and is **not** part of the viability policy. Viability is
about states that are *unobserved*; absorption is about states that are *impossible*.
So absorption is never switchable off, and a proposal it blocks costs no rejection
attempt.
