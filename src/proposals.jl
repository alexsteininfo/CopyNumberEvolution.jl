# Sample an index from non-negative weights in one pass. Deterministic given `rng`,
# and it never iterates a Dict — see the package's determinism constraint.
function _sample_weighted(rng::Random.AbstractRNG, weights::AbstractVector{Float64})
    total = sum(weights)
    total > 0 || throw(ArgumentError(
        "all proposal weights are zero, so no target can be drawn; " *
        "the profile may have no material left to alter"))
    u = rand(rng) * total
    acc = 0.0
    @inbounds for i in eachindex(weights)
        acc += weights[i]
        u <= acc && return i
    end
    return lastindex(weights)
end

"""
    TargetDraw

Which chromosome and haplotype a copy-number alteration lands on.

A rule is a callable `(profile, rng) -> (chrom, haplotype)`; implement a new one by
adding a method to [`draw_target`](@ref), or just pass a plain function.

Chromosomes with zero ploidy — `chrY` in a female assembly — are never eligible.
Haplotype choice is uniform over a chromosome's slots unless the rule says otherwise,
which is what lets mirrored allelic imbalance arise on its own rather than being
injected.

Concrete rules: [`UniformChromosome`](@ref), [`LengthWeighted`](@ref),
[`CNWeighted`](@ref).
"""
abstract type TargetDraw end

"""
    UniformChromosome()

Uniform over eligible chromosomes, then uniform over that chromosome's haplotypes.
Note this is uniform over *chromosomes*, not over base pairs.
"""
struct UniformChromosome <: TargetDraw end

"""
    LengthWeighted()

Chromosome chosen with probability proportional to its length, then uniform over its
haplotypes — i.e. uniform over base pairs.
"""
struct LengthWeighted <: TargetDraw end

"""
    CNWeighted(β)

Haplotype slot chosen with weight proportional to its mean copy number raised to `β`,
so already-gained material keeps being gained. This is the rule that conditions the
proposal on the mother cell's copy-number state, and it is what produces realistic
ploidy skew.

A slot whose mean copy number is 0 gets weight 0 for every `β`, including `β = 0`, so
fully deleted material is never targeted — consistent with copy number 0 being
absorbing. Drawing from a profile in which every slot is empty throws rather than
looping.
"""
struct CNWeighted <: TargetDraw
    β::Float64
    CNWeighted(β::Real = 1.0) = new(Float64(β))
end

"""
    draw_target(rule, profile, rng) -> (chrom, haplotype)

Choose the chromosome and haplotype for the next alteration.
"""
function draw_target(::UniformChromosome, p::CNProfile, rng::Random.AbstractRNG)
    a = p.assembly
    elig = eligible_chromosomes(a)
    c = elig[rand(rng, 1:length(elig))]
    return (c, rand(rng, 1:ploidy(a, c)))
end

function draw_target(::LengthWeighted, p::CNProfile, rng::Random.AbstractRNG)
    a = p.assembly
    elig = eligible_chromosomes(a)
    w = Float64[chromlength(a, c) for c in elig]
    c = elig[_sample_weighted(rng, w)]
    return (c, rand(rng, 1:ploidy(a, c)))
end

function draw_target(d::CNWeighted, p::CNProfile, rng::Random.AbstractRNG)
    a = p.assembly
    w = Vector{Float64}(undef, nslots(a))
    @inbounds for s in 1:nslots(a)
        m = mean_cn(p.segments[s], chromlength(a, slot_chrom(a, s)))
        w[s] = m == 0 ? 0.0 : m^d.β
    end
    s = _sample_weighted(rng, w)
    return (slot_chrom(a, s), slot_haplotype(a, s))
end

draw_target(f, p::CNProfile, rng::Random.AbstractRNG) = f(p, rng)

"""
    ExtentDraw

Where an alteration starts and how far it runs.

A rule is a callable `(profile, chrom, haplotype, rng) -> (start, stop, scale)`, where
`scale` is `:focal`, `:arm` or `:chromosome` and is recorded in the event log so
events can be tallied by class. Pass a plain function or add a method to
[`draw_extent`](@ref).

The only concrete rule is [`ExtentMixture`](@ref).
"""
abstract type ExtentDraw end

"""
    ExtentMixture(; p_chromosome = 0.0, p_arm = 0.0, lengthdist = LogUniform(1e5, 1e8))

Mix whole-chromosome, whole-arm and focal events.

Whole-arm and whole-chromosome events dominate real karyotypes and cannot be produced
at a realistic rate by any continuous length distribution, so they get their own
probabilities. `p_focal` is whatever the two leave over. Setting both to zero ignores
large-scale events entirely without changing the code path.

Focal events draw a length from `lengthdist`, then a uniform start, then **truncate**
at the chromosome end — truncation rather than rejection, matching MEDICC2, where an
event terminates at the boundary. Arm events pick the p or q arm with equal
probability and need the assembly's centromere positions.

The default `lengthdist` spans 100 kb to 100 Mb log-uniformly, so focal and near-arm
sizes are represented across the range real callers report.
"""
struct ExtentMixture{D} <: ExtentDraw
    p_chromosome::Float64
    p_arm::Float64
    lengthdist::D

    function ExtentMixture(; p_chromosome::Real = 0.0, p_arm::Real = 0.0,
                           lengthdist = LogUniform(1e5, 1e8))
        p_chromosome >= 0 || throw(ArgumentError("p_chromosome must be non-negative, got $p_chromosome"))
        p_arm >= 0 || throw(ArgumentError("p_arm must be non-negative, got $p_arm"))
        p_chromosome + p_arm <= 1 || throw(ArgumentError(
            "p_chromosome + p_arm must not exceed 1, got $(p_chromosome + p_arm)"))
        new{typeof(lengthdist)}(Float64(p_chromosome), Float64(p_arm), lengthdist)
    end
end

"""
    draw_extent(rule, profile, chrom, haplotype, rng) -> (start, stop, scale)

Choose the genomic interval for the next alteration.
"""
function draw_extent(d::ExtentMixture, p::CNProfile, c::Integer, h::Integer,
                     rng::Random.AbstractRNG)
    a = p.assembly
    L = chromlength(a, c)
    u = rand(rng)
    if u < d.p_chromosome
        return (1, L, :chromosome)
    elseif u < d.p_chromosome + d.p_arm
        parm, qarm = arms(a, c)
        chosen = rand(rng, Bool) ? parm : qarm
        return (first(chosen), last(chosen), :arm)
    else
        len = clamp(round(Int, rand(rng, d.lengthdist)), 1, L)
        start = rand(rng, 1:L)
        return (start, min(L, start + len - 1), :focal)
    end
end

draw_extent(f, p::CNProfile, c::Integer, h::Integer, rng::Random.AbstractRNG) =
    f(p, c, h, rng)

"""
    KindDraw

Whether an alteration is a gain or a loss, and by how many copies.

A rule is a callable `(profile, chrom, haplotype, start, stop, rng) -> delta` with
`delta` non-zero. The only concrete rule is [`GainLoss`](@ref).
"""
abstract type KindDraw end

"""
    GainLoss(p_gain, delta = 1)

A gain of `delta` copies with probability `p_gain`, otherwise a loss of `delta`.

A loss that drives a haplotype to copy number 0 *is* loss of heterozygosity and needs
no separate event type; a further loss over already-absent material leaves it at 0.
"""
struct GainLoss <: KindDraw
    p_gain::Float64
    delta::Int

    function GainLoss(p_gain::Real, delta::Integer = 1)
        0 <= p_gain <= 1 || throw(ArgumentError("p_gain must lie in [0, 1], got $p_gain"))
        delta > 0 || throw(ArgumentError("delta must be a positive magnitude, got $delta"))
        new(Float64(p_gain), Int(delta))
    end
end

"""
    draw_kind(rule, profile, chrom, haplotype, start, stop, rng) -> Int

Signed copy-number change for the next alteration: positive for a gain, negative for
a loss.
"""
draw_kind(d::GainLoss, p::CNProfile, c::Integer, h::Integer, s::Integer, e::Integer,
          rng::Random.AbstractRNG) = rand(rng) < d.p_gain ? d.delta : -d.delta

draw_kind(f, p::CNProfile, c::Integer, h::Integer, s::Integer, e::Integer,
          rng::Random.AbstractRNG) = f(p, c, h, s, e, rng)
