"""
    PhyloNode

One node of a [`PhyloTree`](@ref).

The three edge quantities describe the edge from `parent` to this node and are
`nothing` when genuinely unknown, rather than defaulting to a sentinel. A single
newick branch-length field cannot carry both real time and a division count, and this
package needs both, so each rate rule requires exactly one of them and throws a named
error when it is absent.

# Fields
- `id::Int` — dense identifier, equal to this node's index in the tree.
- `parent::Union{Int,Nothing}` — `nothing` for the root.
- `children::Vector{Int}` — arbitrary arity; newick and inferred trees are not
  guaranteed binary, and a pruned lineage tree contains unary nodes.
- `birthtime::Union{Float64,Nothing}` — real time at which this cell was born.
- `edge_divisions::Union{Int,Nothing}` — cell divisions on the incoming edge; `1` for
  a simulated lineage-tree edge.
- `edge_mutations::Union{Int,Nothing}` — mutations acquired on the incoming edge.
- `label::Union{String,Nothing}` — taxon name, e.g. from a newick file.
- `source_id::Union{Int,Nothing}` — identifier in the upstream numbering, e.g. a
  `MutationLoadDynamics.jl` cell id. Preserved so that "the edge into cell *i*" stays
  expressible after upstream leaf sampling has made those ids sparse.
"""
struct PhyloNode
    id::Int
    parent::Union{Int,Nothing}
    children::Vector{Int}
    birthtime::Union{Float64,Nothing}
    edge_divisions::Union{Int,Nothing}
    edge_mutations::Union{Int,Nothing}
    label::Union{String,Nothing}
    source_id::Union{Int,Nothing}
end

"""
    PhyloTree(nodes)

A cell-lineage or phylogenetic tree in flat-vector storage, where `nodes[i].id == i`.

Dense integer ids make the tree cheap to serialise, hash and compare, and give
tree-similarity metrics the integer leaf labels they want. Upstream identifiers
survive on each node's `source_id` (see [`PhyloNode`](@ref)).

Construction validates that ids are dense, that there is exactly one root, that
`parent` and `children` agree, and that every node is reachable from the root.

Prefer [`phylotree`](@ref) to build one from a parent vector.
"""
struct PhyloTree
    nodes::Vector{PhyloNode}
    root::Int
    leaves::Vector{Int}

    function PhyloTree(nodes::Vector{PhyloNode})
        n = length(nodes)
        n >= 1 || throw(ArgumentError("a tree needs at least one node"))
        for (i, nd) in enumerate(nodes)
            nd.id == i || throw(ArgumentError(
                "node at index $i has id $(nd.id); ids must be dense and equal to the index"))
        end
        roots = findall(nd -> nd.parent === nothing, nodes)
        length(roots) == 1 || throw(ArgumentError(
            "expected exactly one root (a node with no parent), found $(length(roots))"))
        for nd in nodes
            for c in nd.children
                1 <= c <= n || throw(ArgumentError("node $(nd.id) lists child $c, which is out of range"))
                nodes[c].parent == nd.id || throw(ArgumentError(
                    "node $c is listed as a child of $(nd.id) but its parent is $(nodes[c].parent)"))
            end
            if nd.parent !== nothing
                1 <= nd.parent <= n || throw(ArgumentError(
                    "node $(nd.id) has parent $(nd.parent), which is out of range"))
                nd.id in nodes[nd.parent].children || throw(ArgumentError(
                    "node $(nd.id) has parent $(nd.parent) but is not among that node's children"))
            end
        end
        # reachability, iteratively
        seen = falses(n)
        stack = [roots[1]]
        while !isempty(stack)
            i = pop!(stack)
            seen[i] && throw(ArgumentError("node $i is reachable twice; the tree contains a cycle"))
            seen[i] = true
            append!(stack, nodes[i].children)
        end
        all(seen) || throw(ArgumentError(
            "nodes $(findall(!, seen)) are not reachable from the root"))
        lv = [nd.id for nd in nodes if isempty(nd.children)]
        return new(nodes, roots[1], lv)
    end
end

"""
    phylotree(parents; birthtimes, edge_divisions, edge_mutations, labels, source_ids)

Build a [`PhyloTree`](@ref) from a parent vector.

`parents[i]` is the parent id of node `i`, with `nothing` (or `0`) marking the root.
Children are ordered ascending by id. Every keyword defaults to all-`nothing` and,
when given, must have one entry per node.

# Examples
```jldoctest
julia> t = phylotree([nothing, 1, 1]; labels = [nothing, "a", "b"]);

julia> leaves(t) == [2, 3]
true

julia> cellname(t, 2)
"a"
```
"""
function phylotree(parents::AbstractVector;
                   birthtimes = nothing,
                   edge_divisions = nothing,
                   edge_mutations = nothing,
                   labels = nothing,
                   source_ids = nothing)
    n = length(parents)
    _checklen(v, name) = v === nothing || length(v) == n ||
        throw(ArgumentError("$name has $(length(v)) entries but there are $n nodes"))
    _checklen(birthtimes, "birthtimes")
    _checklen(edge_divisions, "edge_divisions")
    _checklen(edge_mutations, "edge_mutations")
    _checklen(labels, "labels")
    _checklen(source_ids, "source_ids")
    get_(v, i) = v === nothing ? nothing : v[i]

    par = Vector{Union{Int,Nothing}}(undef, n)
    for i in 1:n
        p = parents[i]
        par[i] = (p === nothing || p == 0) ? nothing : Int(p)
    end
    kids = [Int[] for _ in 1:n]
    for i in 1:n
        p = par[i]
        p === nothing && continue
        1 <= p <= n || throw(ArgumentError("node $i has parent $p, which is out of range 1:$n"))
        p == i && throw(ArgumentError("node $i is its own parent"))
        push!(kids[p], i)
    end
    nodes = [PhyloNode(i, par[i], kids[i],
                       get_(birthtimes, i) === nothing ? nothing : Float64(birthtimes[i]),
                       get_(edge_divisions, i) === nothing ? nothing : Int(edge_divisions[i]),
                       get_(edge_mutations, i) === nothing ? nothing : Int(edge_mutations[i]),
                       get_(labels, i) === nothing ? nothing : String(labels[i]),
                       get_(source_ids, i) === nothing ? nothing : Int(source_ids[i]))
             for i in 1:n]
    return PhyloTree(nodes)
end

"""
    nnodes(tree) -> Int

Total number of nodes.
"""
nnodes(t::PhyloTree) = length(t.nodes)

"""
    treeroot(tree) -> Int

Id of the root. For a tree converted from a sampled lineage tree this is the original
**founder**, not the most recent common ancestor of the sample — see [`mrca`](@ref).
"""
treeroot(t::PhyloTree) = t.root

"""
    leaves(tree) -> Vector{Int}

Ids of the leaves, ascending.
"""
leaves(t::PhyloTree) = t.leaves

"""
    internal_nodes(tree) -> Vector{Int}

Ids of the non-leaf nodes, ascending.
"""
internal_nodes(t::PhyloTree) = [nd.id for nd in t.nodes if !isempty(nd.children)]

"""
    node(tree, i) -> PhyloNode

The node with id `i`.
"""
node(t::PhyloTree, i::Integer) = t.nodes[i]

"""
    parentof(tree, i) -> Union{Int,Nothing}

Parent id of node `i`, or `nothing` if it is the root.
"""
parentof(t::PhyloTree, i::Integer) = t.nodes[i].parent

"""
    childrenof(tree, i) -> Vector{Int}

Child ids of node `i`.
"""
childrenof(t::PhyloTree, i::Integer) = t.nodes[i].children

"""
    isleaf(tree, i) -> Bool
"""
isleaf(t::PhyloTree, i::Integer) = isempty(t.nodes[i].children)

"""
    isroot(tree, i) -> Bool
"""
isroot(t::PhyloTree, i::Integer) = t.nodes[i].parent === nothing

"""
    depth(tree, i) -> Int

Number of edges from the root to node `i`; the root has depth 0.
"""
function depth(t::PhyloTree, i::Integer)
    d = 0
    j = i
    while (p = t.nodes[j].parent) !== nothing
        d += 1
        j = p
    end
    return d
end

"""
    preorder(tree) -> Vector{Int}

Node ids in depth-first preorder: each node before its children, children in stored
order. Iterative, so arbitrarily deep trees are safe.
"""
function preorder(t::PhyloTree)
    out = Vector{Int}(undef, nnodes(t))
    k = 0
    stack = [t.root]
    while !isempty(stack)
        i = pop!(stack)
        k += 1
        out[k] = i
        kids = t.nodes[i].children
        for j in Iterators.reverse(eachindex(kids))
            push!(stack, kids[j])
        end
    end
    return out
end

"""
    postorder(tree) -> Vector{Int}

Node ids in depth-first postorder: each node after all of its children.
"""
postorder(t::PhyloTree) = reverse!(_reverse_preorder(t))

function _reverse_preorder(t::PhyloTree)
    out = Vector{Int}(undef, nnodes(t))
    k = 0
    stack = [t.root]
    while !isempty(stack)
        i = pop!(stack)
        k += 1
        out[k] = i
        append!(stack, t.nodes[i].children)
    end
    return out
end

"""
    ancestors(tree, i) -> Vector{Int}

The path from the root to node `i` inclusive, root first.
"""
function ancestors(t::PhyloTree, i::Integer)
    path = Int[i]
    j = i
    while (p = t.nodes[j].parent) !== nothing
        push!(path, p)
        j = p
    end
    return reverse!(path)
end

"""
    descendant_leaves(tree, i) -> Vector{Int}

Leaf ids at or below node `i`, ascending.
"""
function descendant_leaves(t::PhyloTree, i::Integer)
    out = Int[]
    stack = [Int(i)]
    while !isempty(stack)
        j = pop!(stack)
        if isempty(t.nodes[j].children)
            push!(out, j)
        else
            append!(stack, t.nodes[j].children)
        end
    end
    return sort!(out)
end

"""
    edge_time(tree, i) -> Float64

Real-time length of the edge into node `i`, i.e. `birthtime(i) - birthtime(parent(i))`.

Throws if `i` is the root (it has no incoming edge) or if either birthtime is
unknown — which is the case for a tree read with `branchlength = :divisions` or
`:mutations`.
"""
function edge_time(t::PhyloTree, i::Integer)
    nd = t.nodes[i]
    nd.parent === nothing && throw(ArgumentError(
        "node $i is the root and has no incoming edge, so edge_time is undefined"))
    bt = nd.birthtime
    pbt = t.nodes[nd.parent].birthtime
    (bt === nothing || pbt === nothing) && throw(ArgumentError(
        "edge_time needs birthtimes on node $i and its parent $(nd.parent), but at least one is missing; read the tree with branchlength = :time"))
    return bt - pbt
end

"""
    mrca(tree, ids) -> Int

Most recent common ancestor of `ids`. A node is its own ancestor, so
`mrca(tree, [i])` is `i` and `mrca(tree, [parent, child])` is `parent`.

Useful for naming an edge without knowing its id, e.g. placing a subclonal
whole-genome doubling at `mrca(tree, metastatic_leaves)`.
"""
function mrca(t::PhyloTree, ids::AbstractVector{<:Integer})
    isempty(ids) && throw(ArgumentError("the most recent common ancestor of an empty set is undefined"))
    path = ancestors(t, first(ids))
    for x in Iterators.drop(ids, 1)
        other = Set(ancestors(t, x))
        k = findlast(in(other), path)
        k === nothing && throw(ArgumentError(
            "nodes $(first(ids)) and $x have no common ancestor; is this really one tree?"))
        path = path[1:k]
    end
    return last(path)
end

"""
    node_by_source_id(tree, sid) -> Int

Dense id of the node whose `source_id` is `sid` — the upstream numbering, e.g. a
`MutationLoadDynamics.jl` cell id.
"""
function node_by_source_id(t::PhyloTree, sid::Integer)
    for nd in t.nodes
        nd.source_id == sid && return nd.id
    end
    throw(ArgumentError("no node with source_id $sid"))
end

"""
    node_by_label(tree, label) -> Int

Dense id of the node whose `label` is `label`.
"""
function node_by_label(t::PhyloTree, label::AbstractString)
    for nd in t.nodes
        nd.label == label && return nd.id
    end
    throw(ArgumentError("no node labelled $label"))
end

"""
    cellname(tree, i; prefix = "cell") -> String

Stable output name for node `i`: its `label` if it has one, otherwise
`"\$(prefix)_\$(i)"`.

Newick writing and MEDICC2 export both go through this, so a cell's `sample_id` in an
exported matrix always matches its leaf label in the exported tree.
"""
cellname(t::PhyloTree, i::Integer; prefix::AbstractString = "cell") =
    something(t.nodes[i].label, "$(prefix)_$(i)")

Base.show(io::IO, t::PhyloTree) = print(io, "PhyloTree(", nnodes(t), " nodes, ",
                                        length(t.leaves), " leaves, root ", t.root, ")")

"""
    founder_mutations(root) -> Int

Number of mutations the founder cell of a `MutationLoadDynamics.jl` lineage tree
acquired at its own birth.

The founder has no incoming edge, so those mutations cannot be attributed to one, and
[`PhyloTree`](@ref) leaves the root's `edge_mutations` as `nothing`. If you want them
translated into copy-number alterations, feed this to
`CNAModel(initial = TruncalCNAs(founder_mutations(root)))`.

Requires `MutationLoadDynamics` to be loaded; it is provided by a package extension.
"""
function founder_mutations end
