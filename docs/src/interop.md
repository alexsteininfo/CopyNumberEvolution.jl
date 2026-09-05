# MEDICC2 interoperability

[MEDICC2](https://bitbucket.org/schwarzlab/medicc2) (Kaufmann et al., *Genome Biology*
2022) is the reference method for inferring phylogenies and ancestral genomes from
allele-specific copy-number profiles. Its evolutionary model is the closest published
statement of which copy-number aberrations matter, and it is what this package's event
taxonomy is calibrated against. Detailed notes live in `literature/MEDICC2.md` in the
repository.

## Exporting

```julia
grid = BinGrid(hg38(:female), 500_000)
mat  = CNMatrix(res, grid)                     # the leaves — the observable cells
write_medicc2("cells.tsv", mat)
write_medicc2("cells_xy.tsv", mat; include_xy = true)
write_medicc2("cells.tsv", mat; normal_name = "normal")
```

[`write_medicc2`](@ref) emits the long TSV MEDICC2 expects —
`sample_id, chrom, start, end, cn_a, cn_b` — and handles four conversions:

- **0-based half-open (BED) coordinates.** Internal segments are 1-based inclusive;
  the conversion happens at this boundary and nowhere else.
- **Identical segmentation across every sample**, which MEDICC2 requires and which
  holds by construction, since every cell is projected onto the same grid.
- **A reference sample** named by `normal_name` (default `"diploid"`) with
  `cn_a = cn_b = 1` in every bin — the root MEDICC2 measures distances from.
- **Autosomes only** by default, matching MEDICC2's own bulk analyses. With
  `include_xy = true`, a hemizygous chromosome exports as `cn_b = 0`.

It warns if any copy number exceeds 8, which MEDICC2's alphabet cannot represent.

`sample_id` values come from [`cellname`](@ref), the same function
[`write_newick`](@ref) uses, so the matrix and an exported tree always agree on
identity.

Then, from the shell:

```bash
medicc2 cells.tsv medicc2_out --events -j 8
```

## Only the tips go in

The internal-node profiles are the **ground truth** you compare MEDICC2's ancestral
reconstruction *against*. They must never be fed in as input. Write them separately:

```julia
write_medicc2("cells.tsv", mat)                          # input: leaves only
write_profiles("truth_profiles.tsv", res)                # truth: every node
write_events("truth_events.tsv", res)                    # truth: every event
```

Reading MEDICC2's output back — `_final_cn_profiles.tsv`, `_final_tree.new` — is not
this package's job. That belongs downstream, with the estimators, which must not depend
on a simulator.

## What the comparison buys you

MEDICC2's branch length is the number of copy-number events, inferred by parsimony.
This package knows the **true** per-edge event count and the true ancestral profile at
every internal node. That makes two things directly measurable that the published work
could only bound:

**How far parsimony falls short of the truth.** MEDICC2's minimum-event distance is
established as a *lower bound* on the true number of events, and the authors note that
a doubling followed by many chromosome losses inflates the count because each loss is
counted separately. With ground truth in hand, the size and shape of that gap is
measurable rather than assumed.

**How the gap depends on the timing model.** Parsimony has no notion of divisions or
elapsed time. Simulating the same tree under [`PerDivision`](@ref) and
[`PerTime`](@ref) and scoring both with MEDICC2 asks whether the inferred branch
lengths can distinguish them at all — which is the same question the whole
non-Markovian programme asks, transposed to copy number.

MEDICC2's own finding that copy-number trunks are short where point-mutation trunks are
long is a statement about the same gap, seen from the data side.

## Matching MEDICC2's assumptions

Two settings make simulated data directly comparable on MEDICC2's own terms:

```julia
# MEDICC2 defines a doubling as +1 on every non-zero segment, not ×2
wgd = ScheduledWGD(i => 1; mode = :increment)

# and its alphabet caps at 8, so keep copy numbers in range
max_cn(mat) <= 8 || @warn "profiles exceed what MEDICC2 can represent"
```

Simulated profiles are also **phased by construction**, which makes them a clean test
set for MEDICC2's evolutionary phasing: the truth is known.

Three things MEDICC2 does not model, and neither does this package: copy-number-neutral
events, breakage–fusion–bridge cycles, and chromothripsis. MEDICC2's contiguity stress
test is the published argument that omitting them is tolerable for tree inference.
