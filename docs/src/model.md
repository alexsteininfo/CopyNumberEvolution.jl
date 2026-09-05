# The alteration model

[`CNAModel`](@ref) bundles seven independently replaceable components. Each of the
three draws also accepts a plain function. That is deliberate: downstream inference has
to *fit* these parameters, so every one must be addressable and cheap to vary.

```julia
model = CNAModel(
    rate      = PerDivision(1.0),
    target    = UniformChromosome(),
    extent    = ExtentMixture(),
    kind      = GainLoss(0.5),
    wgd       = NoWGD(),
    viability = RejectAndRedraw(),
    initial   = Diploid(),
)
```

## Rate: the question the package exists to ask

Whether alterations accrue **per division** or **per unit real time** is the
Markov-versus-non-Markov question transposed from point mutations to copy number.
Under exponential division timing the two are hard to tell apart; under non-exponential
timing they are not, because division count and elapsed time decouple. The input tree
carries both, so both are computable — and the difference between them is the signal,
not a nuisance.

| [`CNARate`](@ref) | draws | needs |
|:---|:---|:---|
| [`PerDivision`](@ref)`(λ)` | `Poisson(λ · edge_divisions)` | `edge_divisions` |
| [`PerTime`](@ref)`(μ)` | `Poisson(μ · Δt)` | birthtimes |
| [`FromEdgeMutations`](@ref)`()` | exactly `edge_mutations` | `edge_mutations` |
| [`FromEdgeMutations`](@ref)`(p)` | `Binomial(edge_mutations, p)` | `edge_mutations` |
| [`CustomRate`](@ref)`(f)` | `f(tree, node, rng)` | whatever `f` uses |

`PerDivision` is the default. Note it *multiplies* by the division count rather than
applying per edge: a simulated lineage edge is one division, so it reduces to
`Poisson(λ)` there, but a newick edge can stand for many.

`FromEdgeMutations()` is exact identity — one recorded mutation becomes one alteration,
with no extra randomness — which is the natural reading when the upstream simulator's
per-edge mutation count was recorded with this translation in mind. `p < 1` thins
binomially, keeping a fitness-coupled mutation process intact while lowering the
realised alteration rate.

[`n_cnas`](@ref) is the single method to implement for a new rule.

## Target: which chromosome and haplotype

[`TargetDraw`](@ref) rules are callables `(profile, rng) -> (chrom, haplotype)`;
[`draw_target`](@ref) dispatches.

- [`UniformChromosome`](@ref) — uniform over eligible chromosomes. Note: over
  *chromosomes*, not base pairs.
- [`LengthWeighted`](@ref) — proportional to chromosome length, i.e. uniform over base
  pairs.
- [`CNWeighted`](@ref)`(β)` — slot weight proportional to mean copy number to the power
  `β`, so already-gained material keeps being gained. This is the rule that conditions
  the proposal on the mother cell's copy-number state, and it is what produces
  realistic ploidy skew.

Chromosomes with zero ploidy are never eligible. Haplotype choice is uniform over a
chromosome's slots unless a rule says otherwise, which is what lets mirrored allelic
imbalance arise on its own rather than being injected. A slot at mean copy number 0
gets weight 0 under `CNWeighted` for *every* `β`, including 0, consistent with copy
number 0 being absorbing.

## Extent: where, and how far

[`ExtentDraw`](@ref) rules are callables
`(profile, chrom, haplotype, rng) -> (start, stop, scale)`; [`draw_extent`](@ref)
dispatches. [`ExtentMixture`](@ref) is the one provided:

```julia
ExtentMixture(p_chromosome = 0.05, p_arm = 0.15)     # p_focal = 0.80
ExtentMixture()                                      # focal only
ExtentMixture(lengthdist = LogUniform(1e6, 5e7))     # narrower focal sizes
```

Whole-arm and whole-chromosome events dominate real karyotypes and cannot be produced
at a realistic rate by any continuous length distribution, so they get their own
probabilities. Setting both to zero ignores large-scale events entirely without
changing the code path — which is how early analyses will typically run.

Focal events draw a length, then a uniform start, then **truncate** at the chromosome
end. Truncation rather than rejection, matching MEDICC2, where an event terminates at
the boundary. Arm events pick the p or q arm with equal probability and need the
assembly's centromere positions.

## Kind: gain or loss

[`KindDraw`](@ref) rules are callables
`(profile, chrom, haplotype, start, stop, rng) -> delta`; [`draw_kind`](@ref)
dispatches. [`GainLoss`](@ref)`(p_gain, delta = 1)` returns `+delta` or `−delta`.

## Whole-genome doubling

A [`WGDPolicy`](@ref) resolves, once, before traversal, into a schedule mapping a node
id to the number of doublings on its incoming edge — that is [`prepare_wgd`](@ref).
The traversal then only *reads* the schedule, so no `Dict` iteration ever consumes the
random number generator.

```julia
NoWGD()
ScheduledWGD(node_by_source_id(tree, 42) => 1)        # exactly there
ScheduledWGD(mrca(tree, metastatic_leaves) => 1)      # a subclonal doubling
ScheduledWGD(Dict(7 => 2))                            # two successive doublings
ExactlyNWGD(1)                                        # one doubling, position random
RateWGD(PerDivision(0.01))                            # a rate; RateWGD(0.01) is the same
RateWGD(PerTime(0.005))
```

[`ScheduledWGD`](@ref) is what makes statements like *"one whole-genome doubling in the
dataset, on the edge into node i"* expressible. The root cannot be scheduled — it has
no incoming edge; set the root's state with `TruncalCNAs` or `Given` instead.
[`ExactlyNWGD`](@ref) fixes the count but not the position;
[`RateWGD`](@ref) reuses the [`CNARate`](@ref) machinery, so the
per-division-versus-per-time question applies to doublings on the same footing as to
segmental events.

### `:multiply` versus `:increment`

[`wgd_mode`](@ref) reports which arithmetic a policy uses.

- `:multiply` — every copy number doubles (`cn → 2cn`). **The default.** This is what
  tetraploidization means, and it preserves zeros for free.
- `:increment` — every *non-zero* copy number gains one (`cn → cn + 1`). This is
  MEDICC2's own definition.

They coincide while all copy numbers are 0 or 1 and diverge as soon as any segment is
2 or more — exactly the interesting regime, which is why the choice is explicit rather
than assumed. Use `:increment` when you want profiles generated on MEDICC2's own terms;
otherwise a `:multiply` event will be scored by MEDICC2 as one `+1` doubling plus extra
gains.

### Ordering on an edge is defined, not incidental

Doublings are applied **before** that edge's segmental alterations, and the realised
order is recorded in the event log. So "gained then doubled" is always recoverable from
the output rather than reconstructed by guesswork.

## Viability

An alteration can drive a region, a whole chromosome, or the single X of a male
karyotype to copy number 0. Real data contains no cells with whole-chromosome
nullisomy, so an unconstrained process generates profiles that could not exist.

```julia
AllowAll()                                        # no constraint
RejectAndRedraw(min_total_cn = 1, max_attempts = 100)
AllRules([RejectAndRedraw(), my_rule])
```

[`RejectAndRedraw`](@ref) rejects any alteration that would push a chromosome's total
copy number, summed over haplotypes, below `min_total_cn` at any position it touches.
Rejected proposals are **redrawn** — not skipped, and not fatal to the cell. Killing a
cell was rejected as an option outright: the tree is an *input* with its own
birth–death history, so killing here would contradict the given tree and silently
change the sampled population size. Exceeding `max_attempts` throws rather than
skipping, because a silent skip would bias the realised rate with no signal.

The whole interface is one method, [`violation`](@ref), returning a reason symbol or
`nothing`; [`isviable`](@ref) is the predicate on top and [`max_attempts`](@ref) tells
the traversal how many redraws to allow. A new class of impossible state is therefore a
new struct and no change to the traversal.

!!! warning "This conditions the model"
    Rejection sampling makes the alteration process **conditional on viability**, so
    the realised distribution is not the proposal distribution. That is a modelling
    assumption, and any analysis built on these simulations has to state it. The
    rejection tally is returned in the result so the size of the effect is visible
    rather than hidden.

!!! note "The default forbids all homozygous deletions"
    `min_total_cn = 1` forbids biallelic loss **of any size**, not merely
    whole-chromosome nullisomy. Real tumours do contain small homozygous deletions, so
    set `min_total_cn = 0` — or use [`AllowAll`](@ref) — if focal biallelic loss should
    be permitted. See [Limitations and open questions](limitations.md).

## Root state

Because the upstream sampler keeps the founder as the root, the root is generally *not*
the most recent common ancestor of the sample. Truncal state is therefore the root's
[`InitialState`](@ref), not an MRCA special case.

- [`Diploid`](@ref) — a normal karyotype, sex from the assembly. The default.
- [`Given`](@ref)`(profile)` — start from a called ancestral or consensus profile. It
  must be on the same assembly and sex, and is copied rather than mutated.
- [`TruncalCNAs`](@ref)`(n)` — apply `n` alterations from diploid, drawn from the same
  model and **logged against the root**, so they appear in the event record like any
  others.

## Alterations here are neutral by construction

The tree is an input that already encodes whatever selection produced it. This package
therefore never kills or reweights a cell — which is also why "mark the cell dead" was
rejected as a viability option. If you want selection on copy number, it belongs in the
process that generates the tree.
