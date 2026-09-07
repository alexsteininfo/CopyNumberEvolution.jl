# Example 2: per-division vs per-time rates — the question the package exists to ask.
#
# Whether copy-number alterations accrue per cell division or per unit of real time is
# the Markov-versus-non-Markov question transposed from point mutations. Under
# exponential division timing the two are hard to tell apart; under non-exponential
# timing they are not, because division count and elapsed time decouple. See
# docs/src/model.md, "Rate: the question the package exists to ask".
#
# Run with: julia --project examples/02_per_division_vs_per_time.jl

using CopyNumberEvolution

# Two lineages from one root, deliberately decoupling divisions from elapsed time:
#   node 2 ("fast"): 10 divisions in 1 unit of real time  — a rapidly cycling clone.
#   node 3 ("slow"):  1 division  in 10 units of real time — a dormant clone.
# A newick file or a MutationLoadDynamics.jl tree would carry both quantities the same
# way; here they are set explicitly to make the contrast obvious.
tree = phylotree([nothing, 1, 1];
                  birthtimes     = [0.0, 1.0, 10.0],
                  edge_divisions = [nothing, 10, 1])

assembly = GenomeAssembly("toy", :female,
    [ChromosomeSpec("chr$(i)", 10_000_000, 4_000_001:6_000_000) for i in 1:5])

model(rate) = CNAModel(rate = rate, extent = ExtentMixture(p_arm = 0.3))

println("Same tree, same model, two rate rules:\n")

res_div = simulate_cnas(tree, assembly, model(PerDivision(1.0)); seed = 1)
println("PerDivision(1.0):  fast lineage (10 divisions) got ", length(events_on(res_div, 2)),
        " events; slow lineage (1 division) got ", length(events_on(res_div, 3)), " events")

res_time = simulate_cnas(tree, assembly, model(PerTime(1.0)); seed = 1)
println("PerTime(1.0):      fast lineage (Δt=1) got ", length(events_on(res_time, 2)),
        " events; slow lineage (Δt=10) got ", length(events_on(res_time, 3)), " events")

println("\nThe fast lineage accrues more alterations under PerDivision (more divisions) but",
        "\nfewer under PerTime (less elapsed time) — and vice versa for the slow lineage.",
        "\nThe two rules make different, testable predictions about the same tree, which is",
        "\nexactly the point: both quantities are computable from one input, so the choice",
        "\nbetween them is a modelling hypothesis, not a formality.")

# Next: example 03 places a whole-genome doubling on a specific clade.
