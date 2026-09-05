# CopyNumberEvolution.jl

Forward simulation of somatic copy-number alterations along a cell-lineage tree.

Give it a tree — simulated by
[`MutationLoadDynamics.jl`](https://github.com/alexander-stein/MutationLoadDynamics.jl)
or read from a newick file — and it draws copy-number alterations along the edges from
a diploid or given root state, returning the allele-specific profile of **every** node
together with a **complete log** of the events that produced it. Profiles then project
onto a fixed bin grid, which is the form real low-coverage single-cell DNA data arrives
in and the form every inference method consumes.

This package is an **observation model**. It does not infer trees, estimate parameters,
or compute distances between profiles.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/alexander-stein/CopyNumberEvolution.jl")
```

Dependencies are deliberately minimal — `Random`, `Distributions`, `StatsBase` — so
that depending on this package for its types stays cheap.
`MutationLoadDynamics.jl` is a *weak* dependency: load it alongside and the tree
converter appears; leave it out and nothing is missing but the converter.

## Quickstart

```julia
using CopyNumberEvolution

assembly = hg38(:female)
tree = read_newick("lineage.nwk"; branchlength = :divisions)

model = CNAModel(
    rate      = PerDivision(0.5),                              # Poisson(λ · divisions)
    target    = CNWeighted(1.0),                               # gains beget gains
    extent    = ExtentMixture(p_chromosome = 0.05, p_arm = 0.15),
    kind      = GainLoss(0.6),                                 # 60% gains
    wgd       = ScheduledWGD(mrca(tree, [12, 34]) => 1),       # one doubling, exactly there
    viability = RejectAndRedraw(min_total_cn = 1),
    initial   = TruncalCNAs(4),                                # four clonal alterations
)

res = simulate_cnas(tree, assembly, model; seed = 20260904)

# Every node's profile, and every event
profile(res, treeroot(tree))          # the truncal state
leaf_profiles(res)                    # the observable cells
res.events                            # the complete record
res.rejections                        # how often viability bit

# Project and export
grid = BinGrid(assembly, 500_000)
mat  = CNMatrix(res, grid)
write_medicc2("cells.tsv", mat)       # input for the reference method
write_profiles("truth.tsv", res)      # the ground truth to compare against
write_events("events.tsv", res)
```

## Where to go next

- [Concepts](concepts.md) — how a genome is represented, and the four things that can
  happen to it.
- [Input: trees](trees.md) — `PhyloTree`, the three meanings of a newick branch
  length, and the `MutationLoadDynamics.jl` bridge.
- [The alteration model](model.md) — every injectable component, and the scientific
  consequences of the defaults.
- [Output](output.md) — profiles, the event log, replay, and bin projection.
- [MEDICC2 interoperability](interop.md) — exporting for the reference method, and the
  comparison that makes possible.
- [Limitations and open questions](limitations.md) — what is deliberately out of
  scope, and what is still undecided.

Detailed notes on MEDICC2, the reference method this model is calibrated against, live
in `literature/MEDICC2.md` in the repository.
