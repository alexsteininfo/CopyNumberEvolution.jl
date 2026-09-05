# Limitations and open questions

## Deliberately out of scope

**General structural variants and a karyotype backend.** Ordered lists of genomic
fragments with orientation would express translocations, inversions and derivative
chromosomes. That is strictly more expressive and strictly more work, and low-coverage
single-cell data cannot resolve most of it anyway — the observable is a bin-level
copy-number profile. The idea is kept, not discarded: all state changes go through
`apply!(profile, event)`, so a karyotype backend can be added behind that interface
without touching tree traversal or the output layer. MEDICC2 makes the same omission.

**Copy-number-neutral events**, breakage–fusion–bridge cycles, chromothripsis. Same
reasoning.

**Read-depth noise.** Real data passes through read counts, GC bias and a hidden Markov
model caller before becoming an integer matrix. That belongs behind a clean boundary —
a `simulate_readcounts(cnmatrix, depth)` in its own file — and is deliberately not
entangled with alteration simulation. Not implemented yet.

**Tree inference, distances, estimators, approximate Bayesian computation.** A
different package. Keeping them apart is what lets the estimators run on real patient
data with no simulator in their dependency chain.

**Leaf sampling.** `MutationLoadDynamics.jl`'s operation. See
[Input: trees](trees.md).

**Selection on copy number.** The tree is an input that already encodes whatever
selection produced it, so alterations here are neutral by construction.

## Open questions

These are genuinely undecided. The defaults are documented placeholders, not settled
positions.

### The straddling-bin rule

A bin containing a breakpoint has no unambiguously correct copy number.
[`LengthWeightedMajority`](@ref) is the default and [`AreaWeightedMean`](@ref) is
available, but which better matches a given caller's behaviour is unresolved, and
either introduces a small systematic difference from a real caller's own binning.

### The copy-number ceiling

MEDICC2 cannot represent copy numbers above 8. This package imposes no internal cap and
warns only on export ([`max_cn`](@ref) tells you where you stand). Whether a cap
belongs in the simulation itself — and if so, what should happen to a gain that would
breach it — is unsettled.

### `min_total_cn` forbids all homozygous deletions

[`RejectAndRedraw`](@ref)'s default `min_total_cn = 1` forbids biallelic loss **of any
size**, not merely whole-chromosome nullisomy — which was the concern that motivated
the rule. Real tumours do contain small homozygous deletions. Until a size- or
region-aware rule exists, the options are `min_total_cn = 0`, [`AllowAll`](@ref), or a
custom [`ViabilityRule`](@ref); the one-method [`violation`](@ref) interface exists
precisely so that further classes of impossible state are cheap to add.

### `PerDivision` on an inferred tree

An internal edge of an inferred phylogeny is not one cell division, so
`edge_divisions` read from a `:divisions` newick file is an estimate rather than a
count. The rate rule is correct given the field; what the field *means* for real data
is a study-level question.

## Things that are true and easy to forget

- **Rejection sampling conditions the model.** The realised alteration distribution is
  not the proposal distribution whenever `rejection_count(res) > 0`. State it.
- **`:multiply` and `:increment` doublings are not the same event** above copy number 1,
  and MEDICC2 assumes the latter.
- **The root of a sampled tree is the founder**, not the sample's most recent common
  ancestor.
- **No real patient data belongs in this repository.** Fixtures are synthetic and
  small; `data/` is in `.gitignore`. Real data lives only in study repositories, and
  only where the relevant data agreement permits.
