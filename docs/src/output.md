# Output

## Running the simulation

```julia
res = simulate_cnas(tree, assembly, model; seed = 1)
res = simulate_cnas(tree, assembly, model; rng = Xoshiro(1))
res = simulate_cnas(tree, assembly, model; seed = 1, retain_internal = false)
res = simulate_cnas(tree, assembly, model; seed = 1, rng_mode = :per_node)
```

[`simulate_cnas`](@ref) descends depth-first, carrying one profile down the current
path and copying it per child, so memory scales with tree **depth** rather than tree
size. The descent is iterative: a 10⁵-deep tree cannot overflow the stack.

### `rng_mode`

`:global` threads one stream through the whole traversal — the straightforward choice.

`:per_node` gives each edge its own stream, derived from the seed and the node's
`source_id` (falling back to its dense id). An edge then draws *identically* no matter
which other edges exist, which means simulating on a sampled tree yields **exactly** the
same alterations as simulating on the full tree and subsetting — not merely the same
distribution. That property is what the sampling analysis rests on, so being able to
test it exactly rather than statistically is worth the option. It requires a seed. The
doubling schedule is still drawn from the global stream.

## What comes back

[`CNAEvolution`](@ref) holds the tree, the assembly, the model, the profiles, the
complete event log, the rejection tally and the seed.

```julia
profile(res, i)            # one node's profile; throws if it was not retained
leaf_profiles(res)         # the observable cells, in leaves(res.tree) order
res.profiles               # indexed by node id; entries may be nothing
```

Profiles are retained for **every** node by default — tips and internal nodes alike —
because the ancestral states are the ground truth an inference method's reconstruction
gets compared against. `retain_internal = false` keeps only the leaves for very large
trees.

### The event log is the primitive

```julia
res.events                 # Vector{LoggedEvent}, complete, always
nevents(res)
events_on(res, i)          # alterations on the edge into node i
events_below(res, i)       # everything strictly below node i
replay(res)                # reconstruct every profile from the log
```

A [`LoggedEvent`](@ref) records the edge (by its child), the event's `order` within
that edge's sequence, and the [`CNAEvent`](@ref) itself. Truncal alterations are logged
against the root. Events are stored in preorder of node and then by `order`, so
[`replay`](@ref) is a single forward pass.

The log is complete regardless of `retain_internal`, and the retained profiles are a
cache: `replay` reconstructs every profile from the root state plus the log. Nothing is
lost by running lean, and the equality of the two is a tested property.

### Rejections

```julia
res.rejections             # Dict{Symbol,Int}, keyed by the constraint broken
rejection_count(res)       # the total
```

A non-empty tally means the realised alteration distribution is conditioned on
viability. Report it.

## Bin projection

Real low-coverage single-cell data is called in fixed bins, and every inference method
consumes a cells × bins integer matrix.

```julia
grid = BinGrid(assembly, 500_000)        # 500 kb, the DLP+ default
nbins(grid)
bins_of(grid, 1)                         # chr1's column range
```

Bins run consecutively within each chromosome, chromosomes appear in assembly order,
and chromosomes with zero ploidy contribute none — so a female grid has no `chrY`
columns. The final bin of each chromosome is short whenever the bin size does not
divide the chromosome length; that is documented rather than padded.

```julia
tot, alleles = project(profile(res, i), grid)
mat = CNMatrix(res, grid)                            # leaves by default
truth = CNMatrix(res, grid; cells = internal_nodes(res.tree))
mat = CNMatrix(res, grid; rule = AreaWeightedMean())
mat = CNMatrix(res, grid; allele = false)
ncells(mat); max_cn(mat)
```

[`CNMatrix`](@ref) carries `cells` (the node ids of its rows), `names` (from
[`cellname`](@ref), so rows match an exported newick's leaf labels), `total`, and
`allele` — one matrix per haplotype index. A [`Bin`](@ref) on a chromosome whose ploidy
is below a haplotype index gets 0 on that track, which is how a male `chrX` exports
with `cn_b = 0`.

`total` is defined as the **sum of the haplotype tracks**, not as a reprojection of the
summed segmentation. Projection is non-linear, so the two differ at straddling bins;
defining it additively guarantees `total == A + B`, which every consumer of an
allele-specific matrix relies on.

### The straddling-bin rule

A bin containing a breakpoint has no unambiguously correct copy number. Two
[`BinRule`](@ref)s are provided:

- [`LengthWeightedMajority`](@ref) — the copy number covering the most base pairs in
  the bin, ties going to the lower value. The default.
- [`AreaWeightedMean`](@ref) — the length-weighted mean, rounded with halves up.

Either introduces a small systematic difference from a real caller's own binning.
Which one best matches a given caller is **unresolved** — see
[Limitations and open questions](limitations.md).

## Files

```julia
write_profiles("truth.tsv", res)                # node_id, name, chrom, haplotype, start, stop, cn
write_events("events.tsv", res)                 # node_id, name, order, type, chrom, haplotype, start, stop, delta, scale, mode
write_bins("bins.tsv", grid)                    # bin_index, chrom, start, stop
write_medicc2("cells.tsv", mat)                 # see the interoperability page
write_newick("tree.nwk", tree; branchlength = :divisions)
```

All writers accept an `IO` as well as a path. Tables are tab-separated with one header
line and `NA` where a field does not apply — a doubling has no chromosome or span, a
segmental alteration has no mode. [`write_profiles`](@ref) writes every node by
default, so the file contains the ancestral truth as well as the tips;
[`write_events`](@ref) writes the whole log; [`write_bins`](@ref) says what a matrix's
columns mean.

These are deliberately plain tables. The on-disk layout of a whole dataset directory is
specified downstream, where the many-methods-many-datasets requirement lives, and
inventing a second format here would guarantee a mismatch. Newick is the interchange
format for trees.
