"""
    CopyNumberEvolution

Forward simulation of somatic copy-number alterations along a cell-lineage tree.

Takes a lineage tree — simulated by `MutationLoadDynamics.jl` or read from a newick
file — and draws copy-number alterations along its edges from a diploid or given root
state, returning the allele-specific copy-number profile of every node together with a
complete log of the events that produced it. Profiles project onto a fixed bin grid,
the form real low-coverage single-cell DNA data arrives in.

This package is an *observation model*. It does not infer trees, estimate parameters,
or compute distances between profiles.

See the manual for the modelling choices and their consequences, and
`literature/MEDICC2.md` for the reference method this model is calibrated against.
"""
module CopyNumberEvolution

using Random
using Distributions
using StatsBase

export
    # Assembly
    ChromosomeSpec, GenomeAssembly, hg38, hg19,
    nchromosomes, chromname, chromlength, centromere, ploidy,
    nslots, slot, slots_of, slot_chrom, slot_haplotype,
    chromindex, arms, eligible_chromosomes, autosomes, same_assembly

export
    # Profiles
    Segment, CNProfile, diploid, check_invariants, canonicalize!,
    segment_index, cn_at, slot_segments, total_cn, nsegments

export
    # Events
    CNAEvent, SegmentalCNA, WholeGenomeDoubling, apply!, event_span

export
    # Trees
    PhyloNode, PhyloTree, phylotree,
    nnodes, treeroot, leaves, internal_nodes, node, parentof, childrenof,
    isleaf, isroot, depth, preorder, postorder, ancestors, descendant_leaves,
    edge_time, mrca, node_by_source_id, node_by_label, cellname, founder_mutations

export
    # Newick
    parse_newick, read_newick, write_newick, newick_string

export
    # Rates
    CNARate, PerDivision, PerTime, FromEdgeMutations, CustomRate, n_cnas

export
    # Proposals
    TargetDraw, UniformChromosome, LengthWeighted, CNWeighted, draw_target,
    ExtentDraw, ExtentMixture, draw_extent,
    KindDraw, GainLoss, draw_kind

export
    # Whole-genome doubling
    WGDPolicy, NoWGD, ScheduledWGD, ExactlyNWGD, RateWGD, wgd_mode, prepare_wgd

export
    # Viability
    ViabilityRule, AllowAll, RejectAndRedraw, AllRules,
    violation, isviable, max_attempts

export
    # Model, simulation and results
    InitialState, Diploid, Given, TruncalCNAs,
    CNAModel, LoggedEvent, CNAEvolution, simulate_cnas,
    profile, leaf_profiles, events_on, events_below, nevents,
    rejection_count, replay

export
    # Bin grid and matrices
    Bin, BinGrid, nbins, bins_of,
    BinRule, LengthWeightedMajority, AreaWeightedMean, project,
    CNMatrix, ncells, max_cn

export
    # Output
    write_profiles, write_events, write_bins, write_medicc2

include("assembly.jl")
include("profile.jl")
include("cna.jl")
include("tree.jl")
include("newick.jl")
include("rates.jl")
include("proposals.jl")
include("wgd.jl")
include("viability.jl")
include("evolve.jl")
include("bingrid.jl")
include("io.jl")

end # module
