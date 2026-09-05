"""
    CNARate

How many copy-number alterations fall on one edge of the tree.

Whether CNAs accrue **per division** or **per unit real time** is the
Markov-versus-non-Markov question transposed from point mutations to copy number.
Under exponential division timing the two are hard to tell apart; under
non-exponential timing they are not, because division count and elapsed time
decouple. The input tree carries both, so both are computable — and the difference
between them is the signal, not a nuisance. Supporting both is the point of this
package.

Implement a new rule by adding a method to
`n_cnas(rule, tree, node, rng) -> Int`.

Concrete rules: [`PerDivision`](@ref), [`PerTime`](@ref),
[`FromEdgeMutations`](@ref), [`CustomRate`](@ref).
"""
abstract type CNARate end

"""
    PerDivision(λ)

`Poisson(λ · edge_divisions)` alterations on each edge. **The default rate rule.**

Requires `edge_divisions`, i.e. a tree converted from `MutationLoadDynamics.jl` or
read with `branchlength = :divisions` or `:time`. On a simulated lineage tree every
edge is exactly one division, so this reduces to `Poisson(λ)` — but a newick edge can
represent many divisions, which is why the rate multiplies rather than being applied
per edge.
"""
struct PerDivision <: CNARate
    λ::Float64
    function PerDivision(λ::Real)
        λ >= 0 || throw(ArgumentError("PerDivision rate λ must be non-negative, got $λ"))
        new(Float64(λ))
    end
end

"""
    PerTime(μ)

`Poisson(μ · Δt)` alterations on each edge, where `Δt` is the edge's elapsed real
time. Requires birthtimes, i.e. a tree from `MutationLoadDynamics.jl` or read with
`branchlength = :time`.
"""
struct PerTime <: CNARate
    μ::Float64
    function PerTime(μ::Real)
        μ >= 0 || throw(ArgumentError("PerTime rate μ must be non-negative, got $μ"))
        new(Float64(μ))
    end
end

"""
    FromEdgeMutations(p = 1.0)

Take the edge's mutation count as the alteration count.

With the default `p = 1.0` this is **exact identity**: one recorded mutation becomes
one CNA, with no extra randomness. With `p < 1` each mutation independently becomes a
CNA, giving `Binomial(edge_mutations, p)` — which keeps an upstream
fitness-coupled mutation process intact while lowering the realised CNA rate.

Requires `edge_mutations`, i.e. a tree from `MutationLoadDynamics.jl` or read with
`branchlength = :mutations`.
"""
struct FromEdgeMutations <: CNARate
    p::Float64
    function FromEdgeMutations(p::Real = 1.0)
        0 <= p <= 1 || throw(ArgumentError("FromEdgeMutations thinning probability p must lie in [0, 1], got $p"))
        new(Float64(p))
    end
end

"""
    CustomRate(f)

Wrap a user function `f(tree, node, rng) -> Int`, for a rate rule this package does
not provide.
"""
struct CustomRate{F} <: CNARate
    f::F
end

"""
    n_cnas(rule, tree, i, rng) -> Int

Number of copy-number alterations to draw on the edge into node `i`.

Throws an `ArgumentError` naming the missing field, and how to obtain it, when the
tree does not carry what `rule` needs.
"""
function n_cnas(r::PerDivision, t::PhyloTree, i::Integer, rng::Random.AbstractRNG)
    d = node(t, i).edge_divisions
    d === nothing && throw(ArgumentError(
        "PerDivision needs edge_divisions, which is missing on node $i. " *
        "Convert a MutationLoadDynamics tree, or read the newick file with branchlength = :divisions or :time."))
    λ = r.λ * d
    λ == 0 && return 0
    return rand(rng, Poisson(λ))
end

function n_cnas(r::PerTime, t::PhyloTree, i::Integer, rng::Random.AbstractRNG)
    Δ = edge_time(t, i)   # throws with a branchlength hint when birthtimes are missing
    Δ >= 0 || throw(ArgumentError(
        "edge into node $i has negative elapsed time $Δ; the tree's birthtimes are inconsistent"))
    μ = r.μ * Δ
    μ == 0 && return 0
    return rand(rng, Poisson(μ))
end

function n_cnas(r::FromEdgeMutations, t::PhyloTree, i::Integer, rng::Random.AbstractRNG)
    m = node(t, i).edge_mutations
    m === nothing && throw(ArgumentError(
        "FromEdgeMutations needs edge_mutations, which is missing on node $i. " *
        "Convert a MutationLoadDynamics tree, or read the newick file with branchlength = :mutations."))
    m == 0 && return 0
    r.p == 1.0 && return m
    return rand(rng, Binomial(m, r.p))
end

n_cnas(r::CustomRate, t::PhyloTree, i::Integer, rng::Random.AbstractRNG) = r.f(t, i, rng)
