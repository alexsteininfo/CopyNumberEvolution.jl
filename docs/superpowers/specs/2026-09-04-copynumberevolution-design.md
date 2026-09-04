# CopyNumberEvolution.jl — design

**Status:** approved design / spec. Written 2026-09-04. Nothing implemented yet.

**Provenance.** Derived from the handoff written the same day in `gITH-nonMarkovian`
(`docs/handoffs/2026-09-04-copynumberevolution-handoff.md`), with all eight of its open
design questions resolved by the author, plus four corrections established during
design review (§3.2). Modelling reference: MEDICC2 (Kaufmann et al., Genome Biology
2022) — detailed notes in `literature/MEDICC2.md`.

---

## 1. Purpose

Turn a cell-lineage tree into per-cell copy-number profiles.

Given a tree — simulated by `MutationLoadDynamics.jl`, or read from a newick file —
draw somatic copy-number alterations (CNAs) along its edges from a diploid or given
root state, and return the copy-number profile of every node together with a complete
log of the events that produced it. Project those profiles onto a fixed bin grid,
which is the form real low-coverage single-cell DNA data arrives in and the form every
inference method consumes.

This package is a **forward observation model**. It does not infer trees, does not
estimate parameters, and does not compute distances between profiles.

## 2. Position in the package family

```
MutationLoadDynamics.jl            CopyNumberEvolution.jl   <-- THIS PACKAGE
  non-Markovian birth-death,         PhyloTree + CN profile types,
  full lineage trees,                CNA simulation along a tree,
  uniform leaf sampling              sex modes, allele-specific CN,
        |                            projection to a bin grid
        |   (weak dep / pkg ext)           |            |
        v                                  v            v
  gITH-nonMarkovian (paper 1: SNV)   EvoTracer.jl (estimators, ABC,
  scDNA-CN study   (paper 2: CNA)      literature methods, tree inference)
```

**Governing rule.** Packages own algorithms and observation-model types; study repos
own parameter grids, file naming and figures. No parameter sweep, no filename
convention and no figure code belongs here.

**Load-bearing constraint.** `EvoTracer.jl` must never depend on a simulator, because
its estimators run on real patient data. `EvoTracer.jl` will depend on this package
for shared types, therefore **this package must not take a hard dependency on
`MutationLoadDynamics.jl`** — the bridge is a package extension (§4.4). Getting this
wrong makes the simulator transitively required to analyse a clinical dataset.

**Shared vocabulary this package exports** (designed carefully, changed rarely):
`PhyloTree`, `GenomeAssembly`, `CNProfile`, `BinGrid`, `CNMatrix`.

## 3. Settled decisions

### 3.1 The handoff's eight open questions, as resolved

1. **Genome assembly** — real lengths, hg38 as the default; assembly is a parameter.
2. **Viability policy** — reject-and-redraw with an attempt cap and a reported
   rejection count. Designed as an extensible predicate so further classes of
   impossible copy-number state can be added later (§9).
3. **CNA rate model** — all three rules implemented; `PerDivision` is the default.
   The author's per-edge `mutations` field in `MutationLoadDynamics.jl` was written
   with CNA translation in mind, so `FromEdgeMutations` is identity by default.
4. **Whole-arm and whole-chromosome events** — in v1, with their own probabilities
   (§7.4). Requires centromere positions in the assembly.
5. **Whole-genome doubling** — in v1, with **exact placement on a named edge** as a
   first-class requirement (§8).
6. **Bin-straddling projection rule** — unresolved. Implement length-weighted
   majority as the default, expose the alternative, and record the question as open
   in the README (§12.2). To be revisited; not a v1 blocker.
7. **Karyotype backend** — out of scope for v1, deliberately not foreclosed (§16).
8. **Package name / scaffold** — `CopyNumberEvolution.jl`, `PkgTemplates`. Output must
   contain the CN profile of every tip, the CN profile of every internal node, and the
   full record of all CNA events (§11.2).

### 3.2 Corrections to the handoff established in review

1. **Newick branch lengths carry one of three meanings, named at read time**
   (`:divisions`, `:mutations`, `:time`), and `PerDivision` is therefore
   `Poisson(λ · edge_divisions)` rather than `Poisson(λ)` per edge (§4.2, §7.2).
2. **Slot-indexed segment storage replaces the handoff's `Dict{Tuple{Int,Int}, …}`**,
   because `Dict` iteration order is unspecified and would break seeded
   reproducibility (§6.1).
3. **Zero-copy state is absorbing as a hard constraint of `apply!`**, kept separate
   from the viability policy (§6.3).
4. **Dense node ids plus a retained `source_id`**, so "the edge into node *i*" stays
   nameable after upstream leaf sampling has made the original ids sparse (§4.1).

### 3.3 Scientific consequences that must be documented, not just implemented

- Rejection sampling for viability makes the CNA process **conditional on viability**,
  so the realised CNA distribution is not the proposal distribution. Paper 2 must
  state this as a modelling assumption.
- CNAs here are **neutral by construction**. The tree is an input that already encodes
  whatever selection produced it; this package must never kill or reweight a cell.
- Whether CNAs accrue per division or per unit real time is the Markov-versus-non-
  Markov question transposed from SNVs to CNAs. Supporting both is the point of the
  package, not a convenience.

## 4. Input: trees

### 4.1 `PhyloTree`

```julia
struct PhyloNode
    id::Int                              # dense, 1:n; equals the index into nodes
    parent::Union{Int, Nothing}          # nothing for the root
    children::Vector{Int}                # non-binary: newick and inferred trees need it
    birthtime::Union{Float64, Nothing}   # real time of birth; nothing when unknown
    edge_divisions::Union{Int, Nothing}  # cell divisions on parent→this edge
    edge_mutations::Union{Int, Nothing}  # mutations on parent→this edge
    label::Union{String, Nothing}        # newick taxon name, if any
    source_id::Union{Int, Nothing}       # original upstream id (e.g. MLD cell id)
end

struct PhyloTree
    nodes::Vector{PhyloNode}             # index == id
    root::Int
    leaves::Vector{Int}
end
```

Rationale for an owned type rather than reusing `BinaryNode{NonMarkovCell}`: it is what
`EvoTracer.jl` needs for tree-similarity metrics and for returning an *inferred* tree
with no simulator behind it; it is what a newick file parses into; it keeps the
simulator out of the inference dependency chain; and flat-vector storage with integer
ids serialises, hashes and compares more easily than a pointer tree.

The three edge quantities are held separately and are honestly `nothing` when unknown.
Every rate rule requires exactly one of them and throws a named error when it is
absent, so `PerTime` on a divisions-encoded tree fails loudly rather than inventing
times.

**Identity helpers** (all resolve to a dense id, so an edge can be named three ways):

```julia
node_by_source_id(tree, i)      # upstream/MLD numbering
node_by_label(tree, "cell42")   # newick taxon name
mrca(tree, leaves)              # most recent common ancestor of a leaf set
```

Dense ids are required because index-equals-id breaks once upstream leaf sampling
leaves a sparse subset of ids; `source_id` is required because "WGD on edge *i*" must
stay expressible in the author's own numbering.

Other required helpers: `children`, `parent`, `isleaf`, `isroot`, `preorder`,
`postorder`, `descendant_leaves`, `edge_time(tree, id)` (= `birthtime(id) −
birthtime(parent(id))`, erroring if either is `nothing`).

### 4.2 Newick input

```julia
read_newick(path_or_io; branchlength::Symbol)      # no default — must be named
write_newick(io, tree; branchlength::Symbol)
```

| `branchlength` | populates | usable rate rule |
|---|---|---|
| `:divisions` | `edge_divisions = round(Int, bl)` | `PerDivision(λ)` |
| `:mutations` | `edge_mutations = round(Int, bl)` | `FromEdgeMutations([p])` |
| `:time` | `birthtime` by cumulative sum from the root; `edge_divisions = 1` | `PerTime(μ)`, and `PerDivision(λ)` treating each edge as one division |

Fields not implied by the chosen semantics are `nothing`. A single branch-length field
cannot carry both real time and division count, and this project needs both, so the
convention is explicit on both read and write and never inferred.

Newick support required: named and unnamed internal nodes, arbitrary arity, quoted
labels, comments, missing branch lengths (→ `nothing` for the chosen field), and a
trailing semicolon. Round-tripping is tested in all three modes.

### 4.3 Sampling is upstream, not here

Leaf sampling belongs to `MutationLoadDynamics.jl` (`sample_leaves` / `sample_trees`,
specified in its `PLAN-leaf-sampling.md`). **It is specified but not yet implemented
there**, so this package must not assume it exists. Convert whatever tree it is handed.

The property that matters: the upstream sampler **prunes but never collapses**, so
every division ancestral to a sampled cell remains a node and a sampled cell's
root-to-leaf path has the same number of edges as in the full tree — one CNA-drawing
opportunity per real division. Do not add a collapsing step and do not write a
sampler here. A consequence to handle: the root of a sampled tree is the original
founder, not the MRCA of the sample (§10).

### 4.4 The `MutationLoadDynamics.jl` bridge

A package extension (`ext/CopyNumberEvolutionMutationLoadDynamicsExt.jl`, with
`MutationLoadDynamics` under `[weakdeps]`), providing:

```julia
PhyloTree(root::BinaryNode{NonMarkovCell})
```

Mapping: `birthtime` from the cell; `edge_divisions = 1` (one MLD edge is exactly one
division); `edge_mutations = cell.mutations`; `source_id = cell.id`. All three rate
rules therefore work on an MLD tree. Loading this package alone gets the CN modeller
with no simulator dependency; loading both gets the converter for free.

## 5. `GenomeAssembly`

```julia
struct ChromosomeSpec
    name::String
    length::Int
    centromere::UnitRange{Int}   # required for arm-level events
    ploidy::Int                  # copies present in the initial karyotype
end

struct GenomeAssembly
    name::String                 # e.g. "hg38"
    sex::Symbol                  # :female | :male
    chromosomes::Vector{ChromosomeSpec}
    # slot layout: (chrom, haplotype) <-> linear slot index
end
```

Constructors `hg38(sex)` and `hg19(sex)` ship real chromosome lengths and centromere
intervals, so segment sizes have physical meaning — required for comparison with real
data and for a realistic size distribution. The assembly stays a parameter so a
non-human genome is possible.

Sex mode **is** the slot layout, not a separate switch:

- `:female` — haplotypes 1 and 2 for chromosomes 1–22 and X.
- `:male` — haplotypes 1 and 2 for chromosomes 1–22, one X, one Y.

Hemizygosity is therefore representable rather than special-cased, and "46
chromosomes" is 23 pairs addressed as `(chromosome, haplotype)` — never a flat
46-element vector.

**Data provenance requirement.** The hg38/hg19 tables must be transcribed from a named
source (UCSC `hg38.chrom.sizes` for lengths; UCSC `cytoBand` `acen` intervals for
centromeres), the source recorded in a comment beside the table, and a test must
assert the table against a small checked-in reference file. A silently wrong
chromosome length is exactly the plausible-looking wrong number this package's test
strategy exists to catch.

Accessors: `nchromosomes`, `chromlength`, `centromere`, `arms(assembly, chrom)`
(→ p and q ranges), `slots(assembly)`, `slot(assembly, chrom, hap)`,
`slots_of(assembly, chrom)`, `nslots`.

## 6. Copy-number representation

### 6.1 Segmentation, slot-indexed

```julia
struct Segment
    start::Int   # 1-based inclusive genomic coordinate
    stop::Int    # inclusive
    cn::Int      # copies of THIS haplotype over this interval, >= 0
end

struct CNProfile
    assembly::GenomeAssembly
    segments::Vector{Vector{Segment}}   # indexed by haplotype slot
end
```

Each haplotype of each chromosome is a sorted list of non-overlapping segments tiling
the chromosome — a piecewise-constant step function of position.

Why a segmentation rather than the handoff's original `Dict` keyed by copy number: a
`Dict` holds one value per key and so cannot represent two disjoint segments that
happen to share a copy number (an independent gain at 3p and at 3q, both at CN 3).
That is the common case after a handful of CNAs, and the representation would silently
merge or lose one. A segmentation represents everything the dictionary can plus
everything it cannot; it is **exactly what real CN callers emit** (DLP+ / HMMcopy),
so the same type ingests real and simulated data; and applying a CNA to it is a clean,
testable operation.

Why slot-indexed storage rather than `Dict{Tuple{Int,Int}, Vector{Segment}}`:
`Dict` iteration order is unspecified in Julia, so any proposal that iterates
haplotypes — length-weighted or CN-conditioned target choice, WGD, viability checks —
would consume the RNG in an order that is not reproducible across Julia versions or
insertion histories. Determinism under a fixed seed is a required test. Slot indexing
also removes hashing from the inner loop.

A `cn` of 3 on one slot means three copies of that haplotype, which is exactly
MEDICC2's `cn_a`/`cn_b` semantics.

### 6.2 Canonical form and the invariant checker

Invariants, per slot:

1. sorted by `start`;
2. non-overlapping;
3. contiguous coverage of `1:chromlength` with no gaps;
4. no two adjacent segments with equal `cn` (canonical form);
5. `cn >= 0`.

Canonical form means two profiles are equal iff their segment vectors are equal, so
`==` and `hash` are meaningful. **Write the invariant checker first** and call it from
tests after every operation everywhere: a silently non-canonical segmentation is the
failure mode that would poison every downstream number.

```julia
check_invariants(profile)            # throws with slot + segment index on violation
canonicalize!(segments, chromlength) # merge adjacent equal-cn, assert coverage
```

### 6.3 Zero is absorbing — a hard constraint

A segment at `cn = 0` is absent DNA and can never be re-gained. A gain spanning a run
that contains `cn = 0` sub-segments raises the non-zero parts and leaves the zeros at
zero. This is MEDICC2's central physical constraint (`literature/MEDICC2.md` §2) and
is unconditionally true.

It is deliberately **not** part of the viability policy: the policy is about states
that are *unobserved* (whole-chromosome nullisomy), while zero-absorption is about
what is *impossible*. Consequences: zero-absorption is never switchable off, and it
never consumes a rejection attempt.

### 6.4 Allele-specific is the representation; total CN is a projection

Always simulate allele-specific. Expose total copy number as a derived view:

```julia
total_cn(profile, chrom) -> Vector{Segment}   # sum over that chromosome's slots
```

Two simulation code paths would be twice the code and twice the test burden and could
disagree. The "mode" therefore affects output and comparison only, never the model.
A diploid female autosome is two slots, each a single segment at `cn = 1`.

### 6.5 Applying an event

```julia
apply!(profile, event) -> profile     # in place, re-canonicalises, honours §6.3
```

Small interface on purpose: a karyotype backend can later be added behind it without
touching tree traversal or the output layer (§16).

Implementation shape for a segmental event: split at the two breakpoints, add `delta`
to the `cn` of covered segments (clamping at 0, never raising a 0), re-canonicalise.

## 7. The CNA process

### 7.1 The model bundle

```julia
struct CNAModel
    rate::CNARate            # how many CNAs on this edge
    target::TargetDraw       # which chromosome and haplotype
    extent::ExtentDraw       # where, and how long
    kind::KindDraw           # gain or loss, and by how much
    wgd::WGDPolicy
    viability::ViabilityRule
    initial::InitialState
end
```

The three draws are separable and individually replaceable, and each also accepts a
bare function so a study can override one without subtyping. The reason is concrete:
`EvoTracer.jl`'s ABC has to *fit* these parameters, so each must be addressable and
cheap to vary.

### 7.2 Rate

```julia
abstract type CNARate end
struct PerDivision       <: CNARate; λ::Float64; end
struct PerTime           <: CNARate; μ::Float64; end
struct FromEdgeMutations <: CNARate; p::Float64; end   # p = 1.0 default
struct CustomRate        <: CNARate; f; end

n_cnas(rule, tree, node, rng) -> Int
```

- `PerDivision(λ)` → `Poisson(λ · edge_divisions)`. Requires `edge_divisions`. Default
  rate rule. Reduces to `Poisson(λ)` on an MLD tree, where every edge is one division.
- `PerTime(μ)` → `Poisson(μ · edge_time)`. Requires `birthtime` on the node and its
  parent. Implemented from day one, because the comparison with `PerDivision` is the
  scientific point.
- `FromEdgeMutations()` → `k = edge_mutations` exactly, one mutation to one CNA.
  `FromEdgeMutations(p)` with `p < 1` → `Binomial(edge_mutations, p)`, so MLD's
  fitness-coupled mutation process can be kept while lowering the CNA rate. Both are
  exact-count; no additional Poisson layer.
- `CustomRate(f)` wraps `f(tree, node, rng) -> Int`.

Each rule throws a named error naming the missing field when its required field is
`nothing`.

### 7.3 Target — which chromosome and haplotype

Signature `(profile, rng) -> (chrom, haplotype)`.

- `UniformChromosome()` — uniform over chromosomes.
- `LengthWeighted()` — probability proportional to chromosome length.
- `CNWeighted(β)` — slot weight proportional to (mean copy number)^β, so already-gained
  material keeps being gained. This is the author's requirement that the distribution
  may depend on the mother cell's CN state, and it is what produces realistic ploidy
  skew.

Haplotype is uniform over that chromosome's slots unless overridden — which is what
makes mirrored allelic imbalance arise on its own rather than being injected.

### 7.4 Extent — position and size

Signature `(profile, chrom, hap, rng) -> (start, stop, scale)` with
`scale ∈ (:focal, :arm, :chromosome)`.

```julia
struct ExtentMixture <: ExtentDraw
    p_chromosome::Float64   # whole chromosome
    p_arm::Float64          # whole arm (p or q, chosen uniformly) — needs the centromere
    lengthdist              # focal length distribution; default LogUniform
end                         # p_focal = 1 - p_chromosome - p_arm
```

Whole-arm and whole-chromosome events dominate real karyotypes and cannot be produced
at realistic frequency by a continuous length distribution, so they get their own
probabilities. Setting `p_chromosome = p_arm = 0` ignores them without changing the
code path, which is how early analyses will run.

Focal events draw a length, then a uniform start, then **truncate at the chromosome
boundary** — truncation, not rejection, matching MEDICC2, where an event terminates at
the boundary. `scale` is recorded in the event log so events can be tallied by class.

### 7.5 Kind — gain or loss

```julia
struct GainLoss <: KindDraw
    p_gain::Float64
    delta::Int      # magnitude, default 1
end
```

Returns `+delta` or `−delta`. A loss that takes a slot to `cn = 0` is loss of
heterozygosity and needs no separate event type, exactly as in MEDICC2.

### 7.6 Event types

```julia
abstract type CNAEvent end

struct SegmentalCNA <: CNAEvent
    chrom::Int
    haplotype::Int
    start::Int
    stop::Int
    delta::Int       # +k gain, -k loss
    scale::Symbol    # :focal | :arm | :chromosome — provenance of the draw
end

struct WholeGenomeDoubling <: CNAEvent
    mode::Symbol     # :multiply | :increment
end
```

## 8. Whole-genome doubling

```julia
abstract type WGDPolicy end
struct NoWGD <: WGDPolicy end

struct ScheduledWGD <: WGDPolicy      # "exactly one WGD, on the edge into node i"
    at::Dict{Int, Int}                # node id => number of doublings on that edge
    mode::Symbol                      # :multiply (default) | :increment
end

struct ExactlyNWGD <: WGDPolicy       # n doublings, edges drawn uniformly at random
    n::Int
    mode::Symbol
end

struct RateWGD <: WGDPolicy           # per division or per unit time
    p::Float64
    mode::Symbol
end
```

Exact placement is a first-class requirement, not an afterthought. Schedules are built
with the §4.1 helpers, e.g.
`ScheduledWGD(Dict(node_by_source_id(tree, 42) => 1))`, or
`mrca(tree, metastatic_leaves)` for a truncal or subclonal doubling.

**Mode.** `:multiply` (×2 on every segment) is the default: it is what tetraploidization
means and it preserves zeros for free. `:increment` (+1 on every non-zero segment) is
MEDICC2's own definition and exists so profiles can be generated on MEDICC2's terms.
The two coincide while all copy numbers are ≤ 1 and diverge as soon as any segment is
≥ 2 — exactly the regime of interest, which is why the choice is explicit.

**Ordering on an edge is defined, not incidental.** Scheduled and rate-drawn WGDs are
applied before that edge's segmental CNAs, and the event log records the realised
order, so "gained then doubled" versus "doubled then gained" is always recoverable
from the output rather than reconstructed by guesswork.

WGD is exempt from chromosome boundaries by definition (it spans the genome).

## 9. Viability

```julia
abstract type ViabilityRule end
isviable(rule, profile, candidate_event) -> Bool     # the whole interface

struct AllowAll <: ViabilityRule end

struct RejectAndRedraw <: ViabilityRule
    min_total_cn::Int    # total CN summed over a chromosome's slots must stay >= this; default 1
    max_attempts::Int    # default 100
end

struct AllRules <: ViabilityRule; rules::Vector{ViabilityRule}; end
```

A CNA can drive a region, a whole chromosome, or the single X in male mode to
`cn = 0`. Real data contains no cells with whole-chromosome nullisomy, so an
unconstrained process generates profiles that could not exist. On rejection the CNA is
redrawn (target, extent and kind together) up to `max_attempts`; **exceeding the cap
throws**, it does not silently skip the event, because a silent skip would bias the
realised rate without any signal.

"Mark the cell dead" is rejected as an option: the tree is an input with its own
birth-death history, and killing a cell would contradict the given tree and silently
change the sampled population size.

The one-method predicate interface is what makes further classes of impossible state
cheap to add later — a new rule is a new struct, with no change to the traversal.

Rejections are counted **by reason** and returned in the result (§11.2).

## 10. Root / truncal state

```julia
abstract type InitialState end
struct Diploid     <: InitialState end                  # sex mode decides X/Y
struct Given       <: InitialState; profile::CNProfile; end
struct TruncalCNAs <: InitialState; n::Int; end         # apply n CNAs from diploid
```

Because the upstream sampler keeps the founder as the root of a sampled tree, the
handoff's "compute a CN state for the MRCA" is expressed as the **initial state of the
root**, not as an MRCA-specific special case. `TruncalCNAs(n)` is the truncal (clonal)
CN state, the copy-number analogue of clonal SNVs. `Given` also covers the real-data
case of starting from a called ancestral or consensus profile.

## 11. Traversal and result

### 11.1 Traversal

```julia
simulate_cnas(tree, model; rng = Random.default_rng(), retain_internal = true)
    -> CNAEvolution
```

Depth-first from the root, carrying one profile down the current path and copying it
per child before applying that child's events. Segment vectors hold tens of entries,
so copying is cheap.

Per edge, in order: scheduled/drawn WGDs first, then `k = n_cnas(rate, …)` segmental
CNAs, each drawn as target → extent → kind, checked against the viability rule,
redrawn on rejection, applied via `apply!`, and appended to the event log.

### 11.2 Result

```julia
struct LoggedEvent
    node::Int          # the edge parent→node on which it fired
    order::Int         # position within that edge's event sequence
    event::CNAEvent
end

struct CNAEvolution
    tree::PhyloTree
    model::CNAModel
    profiles::Vector{Union{CNProfile, Nothing}}   # by node id; all nodes by default
    events::Vector{LoggedEvent}                   # complete, always
    rejections::Dict{Symbol, Int}                 # by reason
    seed::Union{Int, Nothing}
end
```

`events` is stored in generation order — preorder over nodes, and within a node in
`order` — so replaying it from the root state is a straight forward pass with no
sorting step.

Accessors: `profile(res, node)`, `leaf_profiles(res)`, `events_on(res, node)`,
`events_below(res, node)`.

`retain_internal = true` is the default, because the author requires the CN profile of
every tip **and** every internal node **and** the full event record. `false` remains
available for very large trees, at no scientific cost: **the event log is the primitive
and the retained profiles are a cache** — root state plus log replays every internal
profile exactly. That property is a test (§15).

Rough memory scale with retention: ~2·10⁴ nodes × ~100 segments × 24 B ≈ 50 MB for a
10⁴-leaf tree; ~0.5 GB at 10⁵ leaves, which is why the switch exists.

## 12. Output: the bin grid

### 12.1 Types

```julia
struct Bin
    chrom::Int
    start::Int
    stop::Int
end

struct BinGrid
    assembly::GenomeAssembly
    size::Int                            # bin width in bp; default 500_000 (DLP+)
    bins::Vector{Bin}
    chromranges::Vector{UnitRange{Int}}  # index range of each chromosome's bins
end

struct CNMatrix
    grid::BinGrid
    cells::Vector{Int}                              # PhyloTree node ids
    total::Matrix{Int}                              # cells × bins
    allele::Union{Nothing, Vector{Matrix{Int}}}     # one matrix per haplotype index
end
```

`allele` is a vector of matrices rather than an A/B pair so that a male X (one slot) is
representable without forcing a phantom second haplotype. The final bin of each
chromosome is short; this is documented rather than padded.

### 12.2 Projection rule — an open question

```julia
abstract type BinRule end
struct LengthWeightedMajority <: BinRule end   # default
struct AreaWeightedMean       <: BinRule end   # rounded

project(profile, grid; rule = LengthWeightedMajority()) -> per-bin integers
```

A bin straddling a breakpoint has no unambiguously correct copy number, and either
rule introduces a small systematic difference from a real caller's own binning. The
default is documented, the alternative is available, and **the question is recorded in
the README as unresolved** (handoff open question 6) rather than silently settled.

## 13. Interop and serialisation

### 13.1 MEDICC2 export (v1)

```julia
write_medicc2(path, cnmatrix; include_xy = false, normal_name = "diploid")
```

Long TSV with columns `sample_id, chrom, start, end, cn_a, cn_b`. Requirements taken
from `literature/MEDICC2.md` §9:

- **0-based half-open (BED) coordinates** on output, even though internal `Segment`
  coordinates are 1-based inclusive.
- **Identical segmentation across every sample** — satisfied by construction, since
  every cell is projected onto the same `BinGrid`.
- Integer copy numbers only; a diploid reference row set named by `normal_name`.
- **Autosomes only by default**, matching MEDICC2's own bulk analyses; `include_xy`
  opts in.
- **Warn if any `cn > 8`**, which MEDICC2's alphabet cannot represent.

**Tips only.** The internal-node profiles are the ground truth against which MEDICC2's
ancestral reconstruction is *compared*; they are written to a separate truth file and
must never be fed into its input.

Reading MEDICC2's output back (`_final_cn_profiles.tsv`, `_final_tree.new`) is
`EvoTracer.jl`'s job, per §2's dependency rule.

### 13.2 Plain tables

- Profiles: `node_id, chrom, haplotype, start, stop, cn`.
- Events: `node_id, order, type, chrom, haplotype, start, stop, delta, scale, mode`.
- Bin manifest: `bin_index, chrom, start, stop`.

Newick is the interchange format for trees (§4.2), and `EvoTracer.jl` consumes newick
only.

**No dataset directory layout is defined here.** The on-disk layout of a dataset is
being specified in the `EvoTracer.jl` handoff, because that is where the
many-methods-many-datasets requirement lives. Do not invent a second format; keep the
in-memory types clean and implement a writer against that spec once it exists.

## 14. Package layout and dependencies

```
CopyNumberEvolution.jl/
  Project.toml            # deps: Random, Distributions, StatsBase
                          # weakdeps: MutationLoadDynamics
  src/
    CopyNumberEvolution.jl   # module + exports
    assembly.jl              # ChromosomeSpec, GenomeAssembly, hg38/hg19 tables, slots
    tree.jl                  # PhyloNode, PhyloTree, traversal + identity helpers
    newick.jl                # read_newick / write_newick
    profile.jl               # Segment, CNProfile, invariants, canonicalize, total_cn
    cna.jl                   # CNAEvent types, apply!
    rates.jl                 # CNARate rules
    proposals.jl             # TargetDraw, ExtentDraw, KindDraw
    wgd.jl                   # WGDPolicy
    viability.jl             # ViabilityRule
    initial.jl               # InitialState
    evolve.jl                # CNAModel, simulate_cnas, LoggedEvent, CNAEvolution
    bingrid.jl               # Bin, BinGrid, CNMatrix, BinRule, project
    io.jl                    # plain tables + write_medicc2
  ext/
    CopyNumberEvolutionMutationLoadDynamicsExt.jl
  test/
  literature/MEDICC2.md
  docs/superpowers/specs/
```

Scaffold with `PkgTemplates` so the UUID, CI and docs are standard.

Dependencies stay short — `Random`, `Distributions`, `StatsBase` — so that depending on
this package for its types stays cheap. `AbstractTrees` is deliberately **not** a
dependency: the flat tree does not need it. If the dependency list later grows heavy,
the escape hatch is extracting a small `SomaticEvoCore.jl` holding only the shared
types, but do not start there.

`data/` goes into `.gitignore` on day one. **No real patient data in the repo, ever.**
Fixtures are synthetic and small. Real data lives only in study repos, and only where
the author's data agreement permits.

## 15. Test plan

The failure mode of this package is not a crash, it is a plausible-looking wrong
number. Weight the suite accordingly.

**From the handoff:**

1. **Invariants** — checked on every profile after every operation: sorted,
   non-overlapping, full coverage, canonical, `cn >= 0`.
2. **Zero CNAs** → every tip is exactly the root state. The known-answer test.
3. **Inheritance** — a CNA applied at an internal node appears in *all* tips below it
   and in *no* tip elsewhere. The test that catches a traversal or copy-on-write bug.
4. **Determinism** — same tree, parameters and seed → identical profiles; different
   seeds → different profiles.
5. **Projection consistency** — `total_cn` equals the sum over haplotype slots
   everywhere, on random profiles.
6. **Sex modes** — female has two X and no Y; male has one X and one Y; a loss on a
   hemizygous chromosome behaves per the viability rule.
7. **Sampling commutes** — CNAs simulated on a sampled tree match, in distribution,
   CNAs simulated on the full tree and subset to the same tips. Exact under a fixed
   seed with matched draw order on a small tree; distributional otherwise. This
   validates §4.3, which paper 2's sampling analysis rests on. The fixture builds its
   own pruned tree, since the upstream sampler does not exist yet.
8. **Bin projection** — a bin-aligned segment projects exactly; a straddling segment
   follows the documented rule; a chromosome-length segment fills its chromosome's bins.
9. **Newick round-trip** — in all three `branchlength` modes.

**Added by this design:**

10. **Zero-absorption** — a gain spanning a zeroed run raises the non-zero parts and
    never resurrects a zero; no rejection attempt is consumed.
11. **WGD modes** — `:multiply` and `:increment` agree while all copy numbers are ≤ 1
    and differ exactly where a segment is ≥ 2; `:multiply` preserves zeros.
12. **Scheduled WGD** — lands on exactly the named edge, appears in every tip below it
    and in no other tip, and the count in `at` is respected.
13. **Rate rules** — each scales with its own field (`edge_divisions`, `Δt`,
    `edge_mutations`), and each throws a named error on a tree where that field is
    `nothing`. `FromEdgeMutations()` is exactly identity.
14. **Viability accounting** — the rejection counter increments per reason, and
    exceeding `max_attempts` throws rather than skipping.
15. **Event-log replay** — replaying the log from the root state reproduces every
    retained profile exactly, including with `retain_internal = false` for the tips.
16. **Assembly tables** — hg38/hg19 lengths and centromeres match a checked-in
    reference.
17. **MEDICC2 export shape** — segmentation identical across cells, BED coordinates,
    autosome default, `cn > 8` warning, `diploid` rows present.

## 16. Out of scope for v1

Recorded here and in the README so these are not silently reopened.

- **Karyotype backend / general structural variants.** Ordered lists of genomic
  fragments with orientation would express translocations, inversions and derivative
  chromosomes. Strictly more expressive and strictly more work, and low-coverage scDNA
  data cannot resolve most of it — the observable is a bin-level copy-number profile.
  The author wants the idea kept. Preserved by keeping CNA application behind
  `apply!(profile, event)` (§6.5), so a karyotype backend can be added later without
  touching tree traversal or the output layer. MEDICC2 makes the same omission and
  shows it is tolerable for tree inference.
- **Read-depth noise.** Real data passes through read counts, GC bias and an HMM caller
  before becoming an integer matrix, and the inference paper has sequencing depth as an
  explicit axis. Keep a clean boundary —
  `simulate_readcounts(cnmatrix, depth; …) -> Matrix{Int}` in its own file — and defer
  it past v1. Do not entangle it with CNA simulation.
- **Copy-number-neutral events** (inversions, balanced translocations), breakage–fusion–
  bridge cycles, chromothripsis. Same reasoning as the karyotype backend.
- **Tree inference, distances, estimators, ABC** — all `EvoTracer.jl`.
- **Leaf sampling** — `MutationLoadDynamics.jl` (§4.3).
- **Dataset directory layout** — waits on the `EvoTracer.jl` spec (§13.2).

## 17. Open problems

1. **Bin-straddling projection rule** (§12.2, handoff question 6). Unresolved by the
   author; the default is a documented placeholder to be revisited, not a decision.
2. **Copy-number ceiling.** MEDICC2 cannot represent `cn > 8`. This package imposes no
   internal cap, and warns only on MEDICC2 export. Whether a cap belongs in the
   simulation itself is unsettled.
3. **`PerDivision` on a real newick tree.** An internal edge of an inferred phylogeny
   is not one division, so `edge_divisions` from a `:divisions` newick file is an
   estimate rather than a count. The rate rule is correct given the field; the
   interpretation of the field for real data is a study-level question.
