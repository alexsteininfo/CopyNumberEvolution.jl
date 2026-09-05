"""
    ViabilityRule

Which proposed alterations are allowed to happen.

An alteration can drive a region, a whole chromosome, or the single X of a male
karyotype to copy number 0. Real data contains no cells with whole-chromosome
nullisomy, so an unconstrained process generates profiles that could not exist.
Rejected proposals are **redrawn**, not skipped and not fatal to the cell: the tree is
an *input* with its own birth–death history, so killing a cell here would contradict
the given tree and silently change the sampled population size.

!!! warning "This conditions the model"
    Rejection sampling makes the alteration process *conditional on viability*, so the
    realised distribution of alterations is not the proposal distribution. That is a
    modelling assumption, and any analysis built on these simulations has to state it.
    The rejection tally is returned in [`CNAEvolution`](@ref) so the size of the effect
    is visible rather than hidden.

Distinct from viability, and never switchable off: copy number 0 is **absorbing**,
because absent DNA cannot be regained (see [`apply!`](@ref)). Viability is about
states that are *unobserved*; absorption is about states that are *impossible*.
A proposal blocked by absorption costs no rejection attempt.

The whole interface is one method — `violation(rule, profile, event)` returning a
reason symbol or `nothing` — so a new class of impossible state is a new struct and no
change to the traversal.

Concrete rules: [`AllowAll`](@ref), [`RejectAndRedraw`](@ref), [`AllRules`](@ref).
"""
abstract type ViabilityRule end

"""
    AllowAll()

Impose no viability constraint. Alterations may drive whole chromosomes to copy number
0, so the output can contain cells that could not exist — filter downstream, or use
[`RejectAndRedraw`](@ref).
"""
struct AllowAll <: ViabilityRule end

"""
    RejectAndRedraw(; min_total_cn = 1, max_attempts = 100)

Reject and redraw any alteration that would push a chromosome's **total** copy number
— summed over its haplotypes — below `min_total_cn` at any position it touches.

With the default `min_total_cn = 1`, no position of any chromosome may lose all its
copies.

!!! note "The default forbids all homozygous deletions"
    `min_total_cn = 1` forbids homozygous deletions **of any size**, not merely
    whole-chromosome nullisomy — which was the concern that motivated the rule. Real
    tumours do contain small homozygous deletions, so set `min_total_cn = 0`, or use
    [`AllowAll`](@ref), if focal biallelic loss should be permitted. This is recorded
    as an open question in the manual.

Exceeding `max_attempts` throws rather than silently skipping the alteration, because
a silent skip would bias the realised rate with no signal that it happened.
"""
struct RejectAndRedraw <: ViabilityRule
    min_total_cn::Int
    max_attempts::Int

    function RejectAndRedraw(; min_total_cn::Integer = 1, max_attempts::Integer = 100)
        min_total_cn >= 0 || throw(ArgumentError("min_total_cn must be non-negative, got $min_total_cn"))
        max_attempts >= 1 || throw(ArgumentError("max_attempts must be ≥ 1, got $max_attempts"))
        new(Int(min_total_cn), Int(max_attempts))
    end
end

"""
    AllRules(rules)

Require every rule in `rules` to be satisfied. Reports the first violation found, in
the order given.
"""
struct AllRules <: ViabilityRule
    rules::Vector{ViabilityRule}

    function AllRules(rules::AbstractVector)
        isempty(rules) && throw(ArgumentError("AllRules needs at least one rule; use AllowAll() for no constraint"))
        new(Vector{ViabilityRule}(rules))
    end
end

"""
    violation(rule, profile, event) -> Union{Symbol,Nothing}

`nothing` if `event` may be applied to `profile`, otherwise a symbol naming the
constraint it breaks. The symbol is what the rejection tally in
[`CNAEvolution`](@ref) is keyed by.
"""
violation(::AllowAll, ::CNProfile, ::CNAEvent) = nothing

# A doubling can only raise copy numbers, so it can never breach a lower bound.
violation(::RejectAndRedraw, ::CNProfile, ::WholeGenomeDoubling) = nothing

function violation(r::RejectAndRedraw, p::CNProfile, e::SegmentalCNA)
    e.delta >= 0 && return nothing          # gains cannot breach a lower bound
    a = p.assembly
    sl = slots_of(a, e.chrom)
    isempty(sl) && return nothing           # nothing to lose on a zero-ploidy chromosome
    target = slot(a, e.chrom, e.haplotype)

    # The copy number is piecewise constant, so checking one position per maximal
    # constant interval inside the event's span is exact.
    for pos in _constant_starts(p, sl, e.start, e.stop)
        total = 0
        for s in sl
            cn = cn_at(p.segments[s], pos)
            s == target && (cn = _shift_cn(cn, e.delta))
            total += cn
        end
        total < r.min_total_cn && return :min_total_cn
    end
    return nothing
end

function violation(r::AllRules, p::CNProfile, e::CNAEvent)
    for rule in r.rules
        why = violation(rule, p, e)
        why === nothing || return why
    end
    return nothing
end

# Positions inside `from:to` at which some slot's copy number changes, plus `from`.
function _constant_starts(p::CNProfile, sl::AbstractUnitRange{Int}, from::Int, to::Int)
    starts = Int[from]
    for s in sl, sg in p.segments[s]
        from < sg.start <= to && push!(starts, sg.start)
    end
    sort!(starts)
    unique!(starts)
    return starts
end

"""
    isviable(rule, profile, event) -> Bool

Whether `event` may be applied to `profile`. Equivalent to
`violation(rule, profile, event) === nothing`.
"""
isviable(r::ViabilityRule, p::CNProfile, e::CNAEvent) = violation(r, p, e) === nothing

"""
    max_attempts(rule) -> Int

How many times a rejected alteration may be redrawn before the simulation gives up
and throws.
"""
max_attempts(::AllowAll) = 1
max_attempts(r::RejectAndRedraw) = r.max_attempts
max_attempts(r::AllRules) = maximum(max_attempts, r.rules)
