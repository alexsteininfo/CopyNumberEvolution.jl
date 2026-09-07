# Example 1: the full pipeline, from a lineage tree to files a reference method reads.
#
# Mirrors the quickstart in README.md / docs/src/index.md, but actually run so you see
# real numbers instead of a code listing. Run with:
#   julia --project examples/01_quickstart.jl

using CopyNumberEvolution

# A small lineage tree: root -> two divisions -> four leaves. Real trees come from
# `MutationLoadDynamics.jl` or `read_newick`; see example 05 for the newick route.
tree = phylotree([nothing, 1, 1, 2, 2, 3, 3];
                  edge_divisions = [nothing, 1, 1, 1, 1, 1, 1])

assembly = hg38(:female)

model = CNAModel(
    rate    = PerDivision(0.5),
    target  = CNWeighted(1.0),                              # gains beget gains
    extent  = ExtentMixture(p_chromosome = 0.05, p_arm = 0.15),
    kind    = GainLoss(0.6),                                # 60% gains
    initial = TruncalCNAs(4),                                # four clonal alterations
)

res = simulate_cnas(tree, assembly, model; seed = 1)

println("Tree: ", nnodes(tree), " nodes, ", length(leaves(tree)), " leaves")
println("Truncal profile: ", nsegments(profile(res, treeroot(tree))), " segments")
println("Total events logged: ", nevents(res))
println("Rejections: ", res.rejections)

grid = BinGrid(assembly, 5_000_000)   # coarser than the 500kb DLP+ default, for speed
mat  = CNMatrix(res, grid)
println("Bin matrix: ", ncells(mat), " cells x ", nbins(grid),
        " bins, max copy number ", max_cn(mat))

outdir = mktempdir()
write_medicc2(joinpath(outdir, "cells.tsv"), mat)     # input for the reference method
write_profiles(joinpath(outdir, "truth.tsv"), res)    # ground truth: every node
write_events(joinpath(outdir, "events.tsv"), res)     # the complete event log
println("Wrote cells.tsv, truth.tsv, events.tsv to ", outdir)

# Next: example 02 asks the question the package exists to ask — per-division vs
# per-time rates. See docs/src/model.md.
