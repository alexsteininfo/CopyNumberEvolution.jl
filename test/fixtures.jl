# Shared synthetic fixtures. No real data, ever (see the plan's global constraints).

"""
    toy_assembly(; nchrom = 2, len = 1000, sex = :female)

A tiny assembly for tests: `nchrom` chromosomes of `len` bp with a centromere in the
middle fifth. Chromosome names are `"chr1"`, `"chr2"`, … so sex handling is *not*
triggered; use `toy_sex_assembly` for that.
"""
function toy_assembly(; nchrom::Int = 2, len::Int = 1000, sex::Symbol = :female)
    specs = [CopyNumberEvolution.ChromosomeSpec("chr$(i)", len,
                (2 * len ÷ 5 + 1):(3 * len ÷ 5)) for i in 1:nchrom]
    CopyNumberEvolution.GenomeAssembly("toy", sex, specs, fill(2, nchrom))
end

"""
    toy_sex_assembly(sex; len = 1000)

Two autosomes plus `chrX` and `chrY`, so hemizygosity and zero-ploidy chromosomes are
exercised. `sex` is `:female` or `:male`.
"""
function toy_sex_assembly(sex::Symbol; len::Int = 1000)
    names = ["chr1", "chr2", "chrX", "chrY"]
    specs = [CopyNumberEvolution.ChromosomeSpec(n, len,
                (2 * len ÷ 5 + 1):(3 * len ÷ 5)) for n in names]
    CopyNumberEvolution.GenomeAssembly("toysex", sex, specs)
end

"""
    hemizygous_assembly(; len = 1000)

One chromosome present in a single copy. Used for the pathological-rejection test:
the only whole-chromosome loss available drives total copy number to zero.
"""
function hemizygous_assembly(; len::Int = 1000)
    specs = [CopyNumberEvolution.ChromosomeSpec("chr1", len,
                (2 * len ÷ 5 + 1):(3 * len ÷ 5))]
    CopyNumberEvolution.GenomeAssembly("hemi", :male, specs, [1])
end

"""
    induced_subtree(tree, keep_leaves) -> PhyloTree

Test-only stand-in for `MutationLoadDynamics.sample_leaves`: keep `keep_leaves` plus
every ancestor of a kept leaf, **retaining unary nodes and keeping the founder as the
root**.

This is a fixture, not package API — leaf sampling belongs upstream. It exists so the
"sampling commutes" property can be tested before that sampler is implemented. The
"prune but never collapse" behaviour is the load-bearing part: every division
ancestral to a kept leaf must remain a node, or a kept cell's root-to-leaf path would
lose alteration-drawing opportunities. `source_id`s are preserved, which is what makes
edges identifiable across the two trees.
"""
function induced_subtree(t::PhyloTree, keep_leaves::AbstractVector{<:Integer})
    keepset = Set{Int}()
    for l in keep_leaves, anc in ancestors(t, l)
        push!(keepset, anc)
    end
    old = sort!(collect(keepset))
    newid = Dict(o => i for (i, o) in enumerate(old))
    parents = Vector{Union{Int,Nothing}}(undef, length(old))
    for (i, o) in enumerate(old)
        p = parentof(t, o)
        parents[i] = p === nothing ? nothing : newid[p]
    end
    return phylotree(parents;
        birthtimes     = [node(t, o).birthtime      for o in old],
        edge_divisions = [node(t, o).edge_divisions for o in old],
        edge_mutations = [node(t, o).edge_mutations for o in old],
        labels         = [node(t, o).label          for o in old],
        source_ids     = [something(node(t, o).source_id, o) for o in old])
end

"""
    binary_lineage(depth; divisions = 1, dt = 1.0) -> PhyloTree

A complete binary tree of the given `depth` (so `2^depth` leaves), with `source_id`
equal to the dense id, one division per edge, and birthtimes advancing by `dt` per
level.
"""
function binary_lineage(depth::Int; divisions::Int = 1, dt::Float64 = 1.0)
    n = 2^(depth + 1) - 1
    parents = Vector{Union{Int,Nothing}}(undef, n)
    bt = Vector{Float64}(undef, n)
    parents[1] = nothing
    bt[1] = 0.0
    for i in 2:n
        p = i ÷ 2
        parents[i] = p
        bt[i] = bt[p] + dt
    end
    return phylotree(parents;
        birthtimes = bt,
        edge_divisions = vcat(nothing, fill(divisions, n - 1)),
        edge_mutations = vcat(nothing, fill(2 * divisions, n - 1)),
        source_ids = collect(1:n))
end
