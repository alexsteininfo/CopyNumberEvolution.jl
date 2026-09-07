# Example 5: reading a tree from newick, and exporting for MEDICC2, the reference
# method this package's event model is calibrated against.
#
# See docs/src/trees.md ("The three meanings of a branch length") and
# docs/src/interop.md.
#
# Run with: julia --project examples/05_newick_and_medicc2.jl

using CopyNumberEvolution

# A newick file carries one number per edge; here it is a division count, so we read
# it with branchlength = :divisions. Written to a temp file to keep this example
# self-contained — in practice this is a file on disk, e.g. from a simulator or an
# inference method.
newick = "((A:2,B:1)AB:1,(C:1,D:3)CD:2)root;"
path = joinpath(mktempdir(), "lineage.nwk")
write(path, newick)

tree = read_newick(path; branchlength = :divisions)
println("Read ", nnodes(tree), "-node tree with leaves ",
        join(sort(cellname.(Ref(tree), leaves(tree))), ", "))

assembly = hg38(:female)
model = CNAModel(
    rate    = PerDivision(0.4),
    extent  = ExtentMixture(p_chromosome = 0.05, p_arm = 0.15),
    initial = TruncalCNAs(3),
)
res = simulate_cnas(tree, assembly, model; seed = 7)

grid = BinGrid(assembly, 5_000_000)   # coarser than the 500kb DLP+ default, for speed
mat  = CNMatrix(res, grid)            # leaves by default — the observable cells

outdir = mktempdir()
write_medicc2(joinpath(outdir, "cells.tsv"), mat)                 # input: leaves only
write_profiles(joinpath(outdir, "truth_profiles.tsv"), res)       # truth: every node
write_events(joinpath(outdir, "truth_events.tsv"), res)           # truth: every event
write_newick(joinpath(outdir, "tree.nwk"), tree; branchlength = :divisions)

println("max copy number in the exported matrix: ", max_cn(mat),
        " (MEDICC2's alphabet caps at 8)")
println("Wrote cells.tsv, truth_profiles.tsv, truth_events.tsv, tree.nwk to ", outdir)
println("\nFrom the shell:  medicc2 ", joinpath(outdir, "cells.tsv"), " ",
        joinpath(outdir, "medicc2_out"), " --events -j 8")
println("\nOnly cells.tsv (the leaves) is MEDICC2 input. truth_profiles.tsv and",
        "\ntruth_events.tsv hold the ancestral ground truth to compare its output",
        "\nagainst — never feed those in.")
