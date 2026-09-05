"""
    Bin(chrom, start, stop)

One fixed-width window of a [`BinGrid`](@ref), 1-based inclusive. The last bin of each
chromosome is shorter than the rest whenever the bin size does not divide the
chromosome length.
"""
struct Bin
    chrom::Int
    start::Int
    stop::Int
end

Base.length(b::Bin) = b.stop - b.start + 1

"""
    BinGrid(assembly, size = 500_000)

A fixed genomic binning — the resolution at which real low-coverage single-cell data
is called, and the form every downstream inference method consumes.

Bins run consecutively within each chromosome and chromosomes appear in assembly
order. Chromosomes with zero ploidy contribute no bins, so a female grid has no `chrY`
columns. The default 500 kb matches DLP+.

Fields: `assembly`, `size`, `bins::Vector{Bin}`, and
`chromranges::Vector{UnitRange{Int}}` giving each chromosome's column range.
"""
struct BinGrid
    assembly::GenomeAssembly
    size::Int
    bins::Vector{Bin}
    chromranges::Vector{UnitRange{Int}}

    function BinGrid(a::GenomeAssembly, size::Integer = 500_000)
        size >= 1 || throw(ArgumentError("bin size must be ≥ 1 bp, got $size"))
        bins = Bin[]
        ranges = Vector{UnitRange{Int}}(undef, nchromosomes(a))
        for c in 1:nchromosomes(a)
            first_i = length(bins) + 1
            if ploidy(a, c) == 0
                ranges[c] = first_i:(first_i - 1)   # empty
                continue
            end
            L = chromlength(a, c)
            s = 1
            while s <= L
                e = min(L, s + size - 1)
                push!(bins, Bin(c, s, e))
                s = e + 1
            end
            ranges[c] = first_i:length(bins)
        end
        new(a, Int(size), bins, ranges)
    end
end

"""
    nbins(grid) -> Int

Total number of bins across all chromosomes.
"""
nbins(g::BinGrid) = length(g.bins)

"""
    bins_of(grid, chrom) -> UnitRange{Int}

Column indices belonging to chromosome `chrom`; empty when it has no slots.
"""
bins_of(g::BinGrid, c::Integer) = g.chromranges[c]

Base.show(io::IO, g::BinGrid) = print(io, "BinGrid(", g.assembly.name, ", ",
                                      g.size, " bp, ", nbins(g), " bins)")

"""
    BinRule

How to assign an integer copy number to a bin that straddles a breakpoint.

There is no unambiguously correct answer, and either rule introduces a small
systematic difference from a real caller's own binning.

!!! note "Open question"
    Which rule best matches a given caller's behaviour is unresolved. The default is a
    documented placeholder, not a settled decision — see the manual's Limitations page.

Concrete rules: [`LengthWeightedMajority`](@ref), [`AreaWeightedMean`](@ref).
"""
abstract type BinRule end

"""
    LengthWeightedMajority()

Give the bin the copy number covering the most base pairs within it. Ties resolve to
the **lower** copy number, so the rule is deterministic. The default.
"""
struct LengthWeightedMajority <: BinRule end

"""
    AreaWeightedMean()

Give the bin the length-weighted mean copy number, rounded to the nearest integer with
halves going **up**.
"""
struct AreaWeightedMean <: BinRule end

"""
    project(segs, grid, chrom, rule) -> Vector{Int}

Project one slot's segmentation onto the bins of chromosome `chrom`, returning one
integer per bin of that chromosome.
"""
function project(segs::Vector{Segment}, g::BinGrid, c::Integer, rule::BinRule)
    cols = bins_of(g, c)
    out = Vector{Int}(undef, length(cols))
    for (k, col) in enumerate(cols)
        b = g.bins[col]
        out[k] = _bin_value(segs, b, rule)
    end
    return out
end

function _bin_value(segs::Vector{Segment}, b::Bin, ::LengthWeightedMajority)
    i = segment_index(segs, b.start)
    bestcn, bestlen = segs[i].cn, 0
    while i <= length(segs) && segs[i].start <= b.stop
        sg = segs[i]
        overlap = min(sg.stop, b.stop) - max(sg.start, b.start) + 1
        if overlap > bestlen || (overlap == bestlen && sg.cn < bestcn)
            bestcn, bestlen = sg.cn, overlap
        end
        i += 1
    end
    return bestcn
end

function _bin_value(segs::Vector{Segment}, b::Bin, ::AreaWeightedMean)
    i = segment_index(segs, b.start)
    acc = 0
    while i <= length(segs) && segs[i].start <= b.stop
        sg = segs[i]
        acc += (min(sg.stop, b.stop) - max(sg.start, b.start) + 1) * sg.cn
        i += 1
    end
    return round(Int, acc / length(b), RoundNearestTiesUp)
end

"""
    project(profile, grid; rule = LengthWeightedMajority()) -> (total, alleles)

Project a whole profile onto `grid`.

Returns `total::Vector{Int}`, one value per bin, and `alleles::Vector{Vector{Int}}`,
one track per haplotype index. A bin on a chromosome whose ploidy is below a haplotype
index gets 0 on that track, which is how a male `chrX` gets `cn_b = 0`.

`total` is the **sum of the haplotype tracks**, not a reprojection of the summed
segmentation. Bin projection is non-linear, so the two differ at straddling bins;
defining it additively guarantees `total == A + B`, which every consumer of an
allele-specific matrix relies on.
"""
function project(p::CNProfile, g::BinGrid; rule::BinRule = LengthWeightedMajority())
    a = p.assembly
    same_assembly(a, g.assembly) || throw(ArgumentError(
        "profile is on $(a.name)/:$(a.sex) but the grid is on $(g.assembly.name)/:$(g.assembly.sex)"))
    maxploidy = maximum(a.ploidy)
    alleles = [zeros(Int, nbins(g)) for _ in 1:maxploidy]
    for c in 1:nchromosomes(a)
        cols = bins_of(g, c)
        isempty(cols) && continue
        for h in 1:ploidy(a, c)
            vals = project(p.segments[slot(a, c, h)], g, c, rule)
            @inbounds for (k, col) in enumerate(cols)
                alleles[h][col] = vals[k]
            end
        end
    end
    total = zeros(Int, nbins(g))
    for track in alleles
        total .+= track
    end
    return (total, alleles)
end

"""
    CNMatrix

Copy-number profiles projected onto a bin grid: the cells × bins integer matrix that
inference methods consume.

# Fields
- `grid::BinGrid`.
- `cells::Vector{Int}` — the tree node ids of the rows, in row order.
- `names::Vector{String}` — output names of the rows, from [`cellname`](@ref), so a
  row's identity matches its leaf label in an exported newick tree.
- `total::Matrix{Int}` — cells × bins total copy number.
- `allele::Union{Nothing,Vector{Matrix{Int}}}` — one cells × bins matrix per haplotype
  index, or `nothing`. `total` is always their sum.
"""
struct CNMatrix
    grid::BinGrid
    cells::Vector{Int}
    names::Vector{String}
    total::Matrix{Int}
    allele::Union{Nothing,Vector{Matrix{Int}}}
end

"""
    CNMatrix(res, grid; cells = leaves(res.tree), rule = LengthWeightedMajority(), allele = true)

Project the profiles of `cells` from a [`CNAEvolution`](@ref) onto `grid`.

`cells` defaults to the tree's leaves — the observable cells. Pass internal node ids
to build the ground-truth ancestral matrix that an inference method's reconstruction
can be compared against; keep that separate from what you hand the method as input.
"""
function CNMatrix(res::CNAEvolution, g::BinGrid;
                  cells::AbstractVector{<:Integer} = leaves(res.tree),
                  rule::BinRule = LengthWeightedMajority(),
                  allele::Bool = true)
    same_assembly(res.assembly, g.assembly) ||
        throw(ArgumentError(
            "result is on $(res.assembly.name)/:$(res.assembly.sex) but the grid is on " *
            "$(g.assembly.name)/:$(g.assembly.sex)"))
    ids = collect(Int, cells)
    n = length(ids)
    maxploidy = maximum(res.assembly.ploidy)
    total = zeros(Int, n, nbins(g))
    tracks = allele ? [zeros(Int, n, nbins(g)) for _ in 1:maxploidy] : nothing
    for (row, id) in enumerate(ids)
        tot, alle = project(profile(res, id), g; rule = rule)
        total[row, :] = tot
        if tracks !== nothing
            for h in 1:maxploidy
                tracks[h][row, :] = alle[h]
            end
        end
    end
    names = [cellname(res.tree, id) for id in ids]
    return CNMatrix(g, ids, names, total, tracks)
end

"""
    ncells(m) -> Int

Number of rows in a [`CNMatrix`](@ref).
"""
ncells(m::CNMatrix) = length(m.cells)

"""
    max_cn(m) -> Int

Largest total copy number anywhere in the matrix. Worth checking before export:
MEDICC2's alphabet cannot represent copy numbers above 8.
"""
max_cn(m::CNMatrix) = isempty(m.total) ? 0 : maximum(m.total)

Base.show(io::IO, m::CNMatrix) = print(io, "CNMatrix(", ncells(m), " cells × ",
    nbins(m.grid), " bins, max cn ", max_cn(m), ")")
