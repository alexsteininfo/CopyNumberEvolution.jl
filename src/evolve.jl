"""
    InitialState

The copy-number state of the tree's root.

Because the upstream leaf sampler keeps the founder as the root of a sampled tree, the
root is generally *not* the most recent common ancestor of the sample. Truncal
alterations are therefore expressed as the root's initial state rather than as an
MRCA-specific special case.

Concrete states: [`Diploid`](@ref), [`Given`](@ref), [`TruncalCNAs`](@ref).
"""
abstract type InitialState end

"""
    Diploid()

Start from a normal karyotype, with the sex mode taken from the assembly. The default.
"""
struct Diploid <: InitialState end

"""
    Given(profile)

Start from `profile`, which must be on the same assembly and sex as the simulation.
Covers the real-data case of starting from a called ancestral or consensus profile.
The profile is copied, never mutated.
"""
struct Given <: InitialState
    profile::CNProfile
end

"""
    TruncalCNAs(n)

Apply `n` alterations to a diploid genome before traversal begins — the truncal
(clonal) copy-number state, the copy-number analogue of clonal point mutations. The
alterations are drawn from the same model as the rest of the tree and are logged
against the root, so they appear in the event record like any others.
"""
struct TruncalCNAs <: InitialState
    n::Int
    function TruncalCNAs(n::Integer)
        n >= 0 || throw(ArgumentError("TruncalCNAs needs n ≥ 0, got $n"))
        new(Int(n))
    end
end

"""
    CNAModel(; rate, target, extent, kind, wgd, viability, initial)

The complete alteration model: how many alterations per edge, where each lands, how
far it runs, whether it is a gain or a loss, where whole-genome doublings fall, which
proposals are allowed, and what the root looks like.

Each component is injected and independently replaceable, and each of the three draws
also accepts a plain function. That is deliberate: downstream inference has to *fit*
these parameters, so every one of them must be addressable and cheap to vary.

# Keyword defaults
- `rate = PerDivision(1.0)` — see [`CNARate`](@ref).
- `target = UniformChromosome()` — see [`TargetDraw`](@ref).
- `extent = ExtentMixture()` — focal only; set `p_arm`/`p_chromosome` to enable
  large-scale events. See [`ExtentMixture`](@ref).
- `kind = GainLoss(0.5)` — see [`KindDraw`](@ref).
- `wgd = NoWGD()` — see [`WGDPolicy`](@ref).
- `viability = RejectAndRedraw()` — see [`ViabilityRule`](@ref).
- `initial = Diploid()` — see [`InitialState`](@ref).

# Examples
```julia
model = CNAModel(
    rate = PerDivision(0.5),
    target = CNWeighted(1.0),
    extent = ExtentMixture(p_chromosome = 0.05, p_arm = 0.15),
    kind = GainLoss(0.6),
    wgd = ScheduledWGD(mrca(tree, metastatic_leaves) => 1),
    initial = TruncalCNAs(4),
)
```
"""
struct CNAModel{R,T,E,K,W,V,I}
    rate::R
    target::T
    extent::E
    kind::K
    wgd::W
    viability::V
    initial::I
end

function CNAModel(; rate = PerDivision(1.0),
                    target = UniformChromosome(),
                    extent = ExtentMixture(),
                    kind = GainLoss(0.5),
                    wgd = NoWGD(),
                    viability = RejectAndRedraw(),
                    initial = Diploid())
    CNAModel(rate, target, extent, kind, wgd, viability, initial)
end

Base.show(io::IO, m::CNAModel) = print(io, "CNAModel(rate=", m.rate, ", wgd=",
    nameof(typeof(m.wgd)), ", viability=", nameof(typeof(m.viability)),
    ", initial=", nameof(typeof(m.initial)), ")")

"""
    LoggedEvent(node, order, event)

One alteration, recorded against the edge it fell on.

`node` identifies the edge by its child, `order` is the event's position within that
edge's sequence (1-based), and `event` is the [`CNAEvent`](@ref) itself. Root-state
alterations from [`TruncalCNAs`](@ref) are logged against the root.
"""
struct LoggedEvent
    node::Int
    order::Int
    event::CNAEvent
end

"""
    CNAEvolution

The result of [`simulate_cnas`](@ref): profiles, the event log, and the rejection
tally.

# Fields
- `tree::PhyloTree` — the input tree.
- `assembly::GenomeAssembly` — the genome the simulation ran on.
- `model` — the [`CNAModel`](@ref) used.
- `profiles::Vector{Union{CNProfile,Nothing}}` — indexed by node id. Every node by
  default; leaves only when `retain_internal = false`.
- `events::Vector{LoggedEvent}` — **complete**, always, in preorder of node and then
  by `order` within a node.
- `rejections::Dict{Symbol,Int}` — rejected proposals, keyed by the constraint they
  broke. Non-empty means the realised alteration distribution is conditioned on
  viability; see [`ViabilityRule`](@ref).
- `seed::Union{Int,Nothing}` — the seed, when one was given.
- `retain_internal::Bool`.

The event log is the primitive and the profiles are a cache: [`replay`](@ref)
reconstructs every profile from the root state plus the log, so nothing is lost by
running with `retain_internal = false`.
"""
struct CNAEvolution{M}
    tree::PhyloTree
    assembly::GenomeAssembly
    model::M
    profiles::Vector{Union{CNProfile,Nothing}}
    events::Vector{LoggedEvent}
    rejections::Dict{Symbol,Int}
    seed::Union{Int,Nothing}
    retain_internal::Bool
end

Base.show(io::IO, r::CNAEvolution) = print(io, "CNAEvolution(",
    nnodes(r.tree), " nodes, ", length(leaves(r.tree)), " leaves, ",
    length(r.events), " events, ", rejection_count(r), " rejections)")

"""
    simulate_cnas(tree, assembly, model; rng, seed, retain_internal = true, rng_mode = :global)
        -> CNAEvolution

Draw copy-number alterations along `tree` and return every node's profile plus the
complete event log.

Traversal is depth-first, carrying one profile down the current path and copying it
per child, so memory scales with tree *depth* rather than tree size. Per edge, in
order: whole-genome doublings from the policy's schedule first, then
`n_cnas(model.rate, …)` segmental alterations, each drawn as target → extent → kind,
checked against the viability rule and redrawn on rejection. The realised order is
recorded, so "gained then doubled" is always distinguishable from "doubled then
gained".

Alterations here are **neutral by construction**. The tree is an input that already
encodes whatever selection produced it, so this function never kills or reweights a
cell.

# Arguments
- `tree::PhyloTree` — from [`read_newick`](@ref) or converted from a
  `MutationLoadDynamics.jl` lineage tree.
- `assembly::GenomeAssembly` — e.g. `hg38(:female)`.
- `model::CNAModel`.

# Keywords
- `rng` — a random number generator. Ignored if `seed` is given.
- `seed::Union{Int,Nothing}` — convenience for `rng = Xoshiro(seed)`; recorded in the
  result.
- `retain_internal::Bool = true` — keep internal-node profiles. `false` keeps only
  leaves, for very large trees; the event log stays complete either way.
- `rng_mode::Symbol = :global` — `:global` threads one stream through the whole
  traversal. `:per_node` gives each edge its own stream derived from `seed` and the
  node's `source_id` (falling back to its dense id), so an edge draws identically no
  matter which other edges exist. That makes simulating on a sampled tree give
  *exactly* the same alterations as simulating on the full tree and subsetting, rather
  than only the same distribution. Requires `seed`. The doubling schedule is still
  drawn from the global stream.

# Examples
```julia
tree  = read_newick("lineage.nwk"; branchlength = :divisions)
res   = simulate_cnas(tree, hg38(:female), CNAModel(rate = PerDivision(0.4)); seed = 1)
grid  = BinGrid(hg38(:female), 500_000)
mat   = CNMatrix(res, grid)
write_medicc2("cells.tsv", mat)
```
"""
function simulate_cnas(tree::PhyloTree, assembly::GenomeAssembly, model::CNAModel;
                       rng::Random.AbstractRNG = Random.default_rng(),
                       seed::Union{Integer,Nothing} = nothing,
                       retain_internal::Bool = true,
                       rng_mode::Symbol = :global)
    rng_mode in (:global, :per_node) ||
        throw(ArgumentError("rng_mode must be :global or :per_node, got :$rng_mode"))
    if seed !== nothing
        rng = Random.Xoshiro(seed)
    elseif rng_mode === :per_node
        throw(ArgumentError(
            "rng_mode = :per_node derives each edge's stream from the seed, so a seed is required"))
    end
    iseed = seed === nothing ? nothing : Int(seed)

    rejections = Dict{Symbol,Int}()
    events = LoggedEvent[]
    profiles = Vector{Union{CNProfile,Nothing}}(nothing, nnodes(tree))
    schedule = prepare_wgd(model.wgd, tree, rng)

    r = treeroot(tree)
    rootrng = _edge_rng(rng_mode, rng, iseed, tree, r)
    rootp = _initial_profile(model.initial, assembly, model, rootrng, rejections, events, r)
    (retain_internal || isleaf(tree, r)) && (profiles[r] = rootp)

    # Iterative depth-first descent. A frame holds a node, its profile, and the index
    # of the next child to visit, so only the current root-to-node path is in memory
    # and a 10^5-deep tree cannot overflow the stack.
    fnode = Int[r]
    fprof = CNProfile[rootp]
    fnext = Int[1]
    while !isempty(fnode)
        i = fnode[end]
        kids = childrenof(tree, i)
        k = fnext[end]
        if k > length(kids)
            pop!(fnode); pop!(fprof); pop!(fnext)
            continue
        end
        fnext[end] = k + 1
        child = kids[k]
        cp = copy(fprof[end])       # the parent's stored profile is never mutated
        erng = _edge_rng(rng_mode, rng, iseed, tree, child)
        _evolve_edge!(cp, tree, child, model, schedule, erng, rejections, events)
        (retain_internal || isleaf(tree, child)) && (profiles[child] = cp)
        push!(fnode, child); push!(fprof, cp); push!(fnext, 1)
    end

    return CNAEvolution(tree, assembly, model, profiles, events, rejections,
                        iseed, retain_internal)
end

function _edge_rng(mode::Symbol, rng::Random.AbstractRNG, seed::Union{Int,Nothing},
                   t::PhyloTree, i::Integer)
    mode === :global && return rng
    key = something(node(t, i).source_id, i)
    return Random.Xoshiro(hash((seed, key)))
end

_initial_profile(::Diploid, assembly, model, rng, rejections, events, r) = diploid(assembly)

function _initial_profile(init::Given, assembly, model, rng, rejections, events, r)
    same_assembly(init.profile.assembly, assembly) || throw(ArgumentError(
        "Given initial profile is on $(init.profile.assembly.name)/:$(init.profile.assembly.sex) " *
        "but the simulation runs on $(assembly.name)/:$(assembly.sex)"))
    return copy(init.profile)
end

function _initial_profile(init::TruncalCNAs, assembly, model, rng, rejections, events, r)
    p = diploid(assembly)
    for order in 1:init.n
        ev = _draw_cna(model, p, rng, rejections)
        apply!(p, ev)
        push!(events, LoggedEvent(r, order, ev))
    end
    return p
end

function _evolve_edge!(p::CNProfile, t::PhyloTree, i::Int, model::CNAModel,
                       schedule::Dict{Int,Int}, rng::Random.AbstractRNG,
                       rejections::Dict{Symbol,Int}, events::Vector{LoggedEvent})
    order = 0
    ndoublings = get(schedule, i, 0)     # a read, never an iteration
    if ndoublings > 0
        ev = WholeGenomeDoubling(wgd_mode(model.wgd))
        for _ in 1:ndoublings
            apply!(p, ev)
            order += 1
            push!(events, LoggedEvent(i, order, ev))
        end
    end
    for _ in 1:n_cnas(model.rate, t, i, rng)
        ev = _draw_cna(model, p, rng, rejections)
        apply!(p, ev)
        order += 1
        push!(events, LoggedEvent(i, order, ev))
    end
    return p
end

function _draw_cna(model::CNAModel, p::CNProfile, rng::Random.AbstractRNG,
                   rejections::Dict{Symbol,Int})
    attempts = max_attempts(model.viability)
    for _ in 1:attempts
        c, h = draw_target(model.target, p, rng)
        s, e, scale = draw_extent(model.extent, p, c, h, rng)
        δ = draw_kind(model.kind, p, c, h, s, e, rng)
        ev = SegmentalCNA(c, h, s, e, δ, scale)
        why = violation(model.viability, p, ev)
        why === nothing && return ev
        rejections[why] = get(rejections, why, 0) + 1
    end
    error("viability rejection exceeded max_attempts = $attempts: the model is proposing " *
          "almost nothing viable. Loosen the rule (lower min_total_cn, raise max_attempts), " *
          "raise p_gain so losses are less frequent, or reduce event sizes.")
end

"""
    profile(res, i) -> CNProfile

The copy-number profile of node `i`. Throws if it was not retained; use
[`replay`](@ref) to reconstruct all profiles from the event log.
"""
function profile(r::CNAEvolution, i::Integer)
    p = r.profiles[i]
    p === nothing && throw(ArgumentError(
        "no profile retained for node $i (the simulation ran with retain_internal = false); " *
        "call replay(res) to reconstruct every profile from the event log"))
    return p
end

"""
    leaf_profiles(res) -> Vector{CNProfile}

Profiles of the tree's leaves, in `leaves(res.tree)` order.
"""
leaf_profiles(r::CNAEvolution) = [profile(r, i) for i in leaves(r.tree)]

"""
    events_on(res, i) -> Vector{LoggedEvent}

Alterations that fell on the edge into node `i`, in the order they were applied.
"""
events_on(r::CNAEvolution, i::Integer) = [e for e in r.events if e.node == i]

"""
    events_below(res, i) -> Vector{LoggedEvent}

Alterations on every edge strictly below node `i`, i.e. those inherited by some but
not all of the tree.
"""
function events_below(r::CNAEvolution, i::Integer)
    below = Set{Int}()
    stack = collect(childrenof(r.tree, i))
    while !isempty(stack)
        j = pop!(stack)
        push!(below, j)
        append!(stack, childrenof(r.tree, j))
    end
    return [e for e in r.events if e.node in below]
end

"""
    nevents(res) -> Int

Total number of logged alterations.
"""
nevents(r::CNAEvolution) = length(r.events)

"""
    rejection_count(res) -> Int

Total number of proposals rejected by the viability rule, across all reasons.
"""
rejection_count(r::CNAEvolution) = isempty(r.rejections) ? 0 : sum(values(r.rejections))

"""
    replay(res) -> Vector{CNProfile}

Reconstruct every node's profile from the root state plus the event log.

The event log is the primitive and the retained profiles are a cache, so this returns
exactly `res.profiles` wherever those were retained — which is what the replay test
asserts — and fills in the rest. Use it after a `retain_internal = false` run, or to
verify the traversal.
"""
function replay(r::CNAEvolution)
    t = r.tree
    grouped = [LoggedEvent[] for _ in 1:nnodes(t)]
    for le in r.events
        push!(grouped[le.node], le)
    end
    for v in grouped
        sort!(v; by = e -> e.order)
    end
    out = Vector{CNProfile}(undef, nnodes(t))
    root = treeroot(t)
    base = r.model.initial isa Given ? copy(r.model.initial.profile) : diploid(r.assembly)
    for le in grouped[root]
        apply!(base, le.event)
    end
    out[root] = base
    for i in preorder(t)
        i == root && continue
        p = copy(out[parentof(t, i)])
        for le in grouped[i]
            apply!(p, le.event)
        end
        out[i] = p
    end
    return out
end
