# Input: trees

## `PhyloTree`

[`PhyloTree`](@ref) is flat-vector storage of [`PhyloNode`](@ref)s where
`nodes[i].id == i`. Dense integer ids are cheap to serialise, hash and compare, and
tree-similarity metrics want integer leaf labels anyway.

The package owns this type rather than reusing a simulator's node type for three
reasons: it is what a newick file parses into; it is what an *inferred* tree — one with
no simulator behind it — needs to be; and it keeps the simulator out of the dependency
chain of anything that only needs the types.

```julia
t = phylotree([nothing, 1, 1, 2, 2];
              birthtimes     = [0.0, 1.0, 1.5, 2.0, 2.5],
              edge_divisions = [nothing, 1, 1, 1, 1],
              edge_mutations = [nothing, 4, 7, 2, 0],
              labels         = [nothing, nothing, nothing, "a", "b"],
              source_ids     = [10, 20, 30, 40, 50])

nnodes(t); treeroot(t); leaves(t); internal_nodes(t)
node(t, 4); parentof(t, 4); childrenof(t, 2)
isleaf(t, 4); isroot(t, 1); depth(t, 4)
preorder(t); postorder(t); ancestors(t, 4); descendant_leaves(t, 2)
```

Arity is arbitrary and unary nodes are allowed, because newick input and inferred trees
are not guaranteed binary and a pruned lineage tree contains unary nodes. Every
traversal is iterative, so a 10⁵-deep caterpillar tree is safe.

### Naming an edge

Three ways, all resolving to a dense id — needed because you have to be able to say
"the doubling happens on *that* edge":

```julia
node_by_source_id(t, 40)     # the upstream numbering, e.g. a simulator's cell id
node_by_label(t, "a")        # a newick taxon name
mrca(t, [4, 5])              # the most recent common ancestor of a leaf set
```

`source_id` matters because leaf sampling upstream leaves a *sparse* subset of the
original ids, at which point index-equals-id would break. Dense ids keep flat storage;
`source_id` keeps "cell 40" meaningful. [`cellname`](@ref) is the stable output name — a
node's label if it has one, else `cell_<id>` — and both newick writing and MEDICC2
export go through it, so a cell's `sample_id` in an exported matrix always matches its
leaf label in the exported tree.

## The three meanings of a branch length

A newick file carries **one** number per edge. This package needs both real time and a
division count, so [`read_newick`](@ref) and [`parse_newick`](@ref) make you say which
it is. There is deliberately no default.

| `branchlength` | field populated | rate rules enabled |
|:---|:---|:---|
| `:divisions` | `edge_divisions` | [`PerDivision`](@ref) |
| `:mutations` | `edge_mutations` | [`FromEdgeMutations`](@ref) |
| `:time` | `birthtime` by cumulative sum, plus `edge_divisions = 1` | [`PerTime`](@ref), and [`PerDivision`](@ref) treating each edge as one division |

```julia
t = read_newick("lineage.nwk"; branchlength = :divisions)
write_newick("out.nwk", t; branchlength = :divisions)     # explicit on write too
newick_string(t; branchlength = :time, labels = :source_id)
```

The three [`PhyloNode`](@ref) edge fields are held separately and are honestly
`nothing` when unknown, rather than defaulting to a sentinel. Each rate rule requires
exactly one of them and throws a named error, naming the fix, when it is absent — so
`PerTime` on a divisions-encoded tree fails loudly instead of inventing times.
[`edge_time`](@ref) behaves the same way, and throws on the root, which has no incoming
edge.

Parsing handles named and unnamed internal nodes, arbitrary arity, single-quoted
labels, `[...]` comments, and missing branch lengths. Under `:divisions` and
`:mutations` a fractional branch length is rounded with a warning, since a fractional
count usually means the file should have been read as `:time`.

## The `MutationLoadDynamics.jl` bridge

Load both packages and a converter appears:

```julia
using CopyNumberEvolution, MutationLoadDynamics

tree = PhyloTree(root)               # root::BinaryNode{NonMarkovCell}
founder_mutations(root)              # the founder's own mutations
```

Mapping: `birthtime` from the cell; `edge_divisions = 1`, because one lineage-tree edge
is exactly one division; `edge_mutations = cell.mutations`; `source_id = cell.id`. All
three rate rules therefore work on a converted tree.

This is a **package extension**. Loading `CopyNumberEvolution` alone gives the
copy-number modeller with no simulator anywhere in the dependency chain — which
matters, because the downstream inference package has to be installable and runnable
against real patient data, and a hard dependency here would make a simulator
transitively required to analyse a clinical dataset.

The founder has no incoming edge, so its own mutations cannot be attributed to one and
the root's `edge_mutations` is `nothing`. If you want them translated, feed
[`founder_mutations`](@ref) into `TruncalCNAs`.

## Sampling happens upstream

Leaf sampling is `MutationLoadDynamics.jl`'s operation, not this package's. The
converter takes whatever tree it is handed, full or sampled.

The property that matters is that the upstream sampler **prunes but never collapses**.
Every division ancestral to a sampled cell remains a node, so a sampled cell's
root-to-leaf path has the same number of edges as in the full tree — one
alteration-drawing opportunity per real division. That is what makes alterations
simulated on a sampled tree identical in distribution to alterations simulated on the
full tree and then subset. Collapsing unary nodes would turn divisional depth into "a
count of bifurcations that survived sampling", a property of the sample rather than of
the cell.

Two consequences. The root of a sampled tree is the original **founder**, not the most
recent common ancestor of the sample — which is why truncal state is expressed as the
root's [`InitialState`](@ref) rather than as an MRCA special case. And with
`rng_mode = :per_node` the commuting property holds *exactly*, not just
distributionally; see [The alteration model](model.md).
