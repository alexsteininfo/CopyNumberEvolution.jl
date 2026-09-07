# Example 3: placing a whole-genome doubling on a specific clade, and the
# :multiply vs :increment distinction.
#
# `mrca(tree, leaves)` names an edge without knowing its id in advance — the natural
# way to say "one doubling in the dataset, on the branch leading to this clade." See
# docs/src/model.md, "Whole-genome doubling".
#
# Run with: julia --project examples/03_subclonal_wgd.jl

using CopyNumberEvolution

# root (1) -> {2, 3}; 2 -> {4, 5}. Leaves 4 and 5 form a subclade; leaf 3 does not.
tree = phylotree([nothing, 1, 1, 2, 2]; edge_divisions = [nothing, 1, 1, 1, 1])

assembly = GenomeAssembly("toy", :female,
    [ChromosomeSpec("chr$(i)", 1_000_000, 400_001:600_000) for i in 1:3])

# rate = 0 so the only source of copy-number change is the truncal state and the
# scheduled doubling itself — isolating the doubling's effect from unrelated noise.
# p_chromosome = 1.0 keeps every truncal alteration whole-chromosome, so the numbers
# below are easy to read at a glance.
base = (rate = PerDivision(0.0), target = CNWeighted(1.0),
        extent = ExtentMixture(p_chromosome = 1.0), kind = GainLoss(0.9),
        initial = TruncalCNAs(4))

subclade = mrca(tree, [4, 5])   # = node 2: the edge shared by both, and only both

model_multiply  = CNAModel(; base..., wgd = ScheduledWGD(subclade => 1; mode = :multiply))
model_increment = CNAModel(; base..., wgd = ScheduledWGD(subclade => 1; mode = :increment))

res_mult = simulate_cnas(tree, assembly, model_multiply;  seed = 1)
res_incr = simulate_cnas(tree, assembly, model_increment; seed = 1)

println("Truncal (pre-doubling) copy number, chr1 haplotype 1: ",
        only(slot_segments(profile(res_mult, treeroot(tree)), 1, 1)).cn)

println("\nLeaf 3 is outside the doubled clade — unaffected by either policy:")
println("  chr1 hap1 = ", only(slot_segments(profile(res_mult, 3), 1, 1)).cn,
        " (multiply run), ", only(slot_segments(profile(res_incr, 3), 1, 1)).cn, " (increment run)")

println("\nLeaf 4 is inside the doubled clade — :multiply and :increment now disagree",
        "\nwherever the pre-doubling copy number was already >= 2:")
for c in 1:3, h in 1:2
    pre  = only(slot_segments(profile(res_mult, treeroot(tree)), c, h)).cn
    mult = only(slot_segments(profile(res_mult, 4), c, h)).cn
    incr = only(slot_segments(profile(res_incr, 4), c, h)).cn
    tag  = pre >= 2 ? "  (diverge)" : "  (agree: cn was 0 or 1)"
    println("  chr$c hap$h: pre-doubling cn=$pre -> multiply=$mult, increment=$incr", tag)
end

println("\nUse :increment when profiles must be scored on MEDICC2's own terms (its own",
        "\ndoubling definition); otherwise a :multiply event is seen by MEDICC2 as one +1",
        "\ndoubling plus extra gains. See docs/src/interop.md.")

# Next: example 04 looks at what happens when a proposed alteration is not viable.
