# Every writer takes either an IO or a path. Tab-separated, one header line, `NA` for
# fields that do not apply. These are plain tables on purpose: the on-disk layout of a
# whole dataset directory is specified downstream, and inventing a second format here
# would guarantee a mismatch.
_with_io(f, io::IO) = f(io)
_with_io(f, path::AbstractString) = open(f, path, "w")

"""
    write_profiles(io_or_path, res; cells = 1:nnodes(res.tree))

Write copy-number segmentations as a long table with columns
`node_id, name, chrom, haplotype, start, stop, cn`.

Coordinates are this package's internal 1-based inclusive convention. `cells` selects
which nodes to write and defaults to **every** node, so the file contains the ancestral
truth as well as the observable tips.
"""
function write_profiles(dest, res::CNAEvolution;
                        cells::AbstractVector{<:Integer} = 1:nnodes(res.tree))
    a = res.assembly
    _with_io(dest) do io
        println(io, join(("node_id", "name", "chrom", "haplotype", "start", "stop", "cn"), '\t'))
        for id in cells
            p = profile(res, id)
            nm = cellname(res.tree, id)
            for c in 1:nchromosomes(a), h in 1:ploidy(a, c)
                for sg in p.segments[slot(a, c, h)]
                    println(io, join((id, nm, chromname(a, c), h, sg.start, sg.stop, sg.cn), '\t'))
                end
            end
        end
    end
    return dest
end

"""
    write_events(io_or_path, res)

Write the complete event log as a table with columns
`node_id, name, order, type, chrom, haplotype, start, stop, delta, scale, mode`.

`type` is `segmental` or `wgd`. Columns that do not apply to a row hold `NA`:
a doubling has no chromosome, span, delta or scale, and a segmental alteration has no
mode. `node_id` identifies the edge by its child; the root's rows are truncal
alterations.
"""
function write_events(dest, res::CNAEvolution)
    a = res.assembly
    _with_io(dest) do io
        println(io, join(("node_id", "name", "order", "type", "chrom", "haplotype",
                          "start", "stop", "delta", "scale", "mode"), '\t'))
        for le in res.events
            nm = cellname(res.tree, le.node)
            e = le.event
            if e isa SegmentalCNA
                println(io, join((le.node, nm, le.order, "segmental", chromname(a, e.chrom),
                                  e.haplotype, e.start, e.stop, e.delta, e.scale, "NA"), '\t'))
            else
                println(io, join((le.node, nm, le.order, "wgd", "NA", "NA",
                                  "NA", "NA", "NA", "NA", e.mode), '\t'))
            end
        end
    end
    return dest
end

"""
    write_bins(io_or_path, grid)

Write the bin manifest as `bin_index, chrom, start, stop`, in 1-based inclusive
coordinates. Pairs with a [`CNMatrix`](@ref) to say what its columns mean.
"""
function write_bins(dest, g::BinGrid)
    a = g.assembly
    _with_io(dest) do io
        println(io, join(("bin_index", "chrom", "start", "stop"), '\t'))
        for (i, b) in enumerate(g.bins)
            println(io, join((i, chromname(a, b.chrom), b.start, b.stop), '\t'))
        end
    end
    return dest
end

"""
    write_medicc2(io_or_path, m; include_xy = false, normal_name = "diploid")

Write a [`CNMatrix`](@ref) as MEDICC2 input: a long TSV with columns
`sample_id, chrom, start, end, cn_a, cn_b`.

Conversions and conventions, all of them things MEDICC2 requires:

- **0-based half-open (BED) coordinates**, converted from this package's 1-based
  inclusive segments at this boundary and nowhere else.
- **Identical segmentation across every sample**, which holds by construction because
  every cell is projected onto the same grid.
- A reference sample named `normal_name` with `cn_a = cn_b = 1` in every bin, which is
  the root MEDICC2 measures distances from.
- **Autosomes only** by default, matching MEDICC2's own bulk analyses. Set
  `include_xy = true` to include the sex chromosomes; note a hemizygous chromosome
  exports as `cn_b = 0`.

Warns if any copy number exceeds 8, which MEDICC2's alphabet cannot represent.

Only the observable cells belong in this file. The ancestral profiles are the ground
truth you compare MEDICC2's reconstruction *against* — write those with
[`write_profiles`](@ref) instead, and never feed them in as input.
"""
function write_medicc2(dest, m::CNMatrix; include_xy::Bool = false,
                       normal_name::AbstractString = "diploid")
    m.allele === nothing && throw(ArgumentError(
        "write_medicc2 needs allele-specific tracks; build the matrix with allele = true"))
    length(m.allele) >= 2 || throw(ArgumentError(
        "write_medicc2 expects at least two haplotype tracks, found $(length(m.allele))"))
    g = m.grid
    a = g.assembly
    if max_cn(m) > 8
        @warn "copy numbers above 8 cannot be represented by MEDICC2 (its alphabet caps at 8); the exported values will be out of range" max_cn = max_cn(m)
    end
    keep = include_xy ? collect(1:nchromosomes(a)) : autosomes(a)
    cols = Int[]
    for c in keep
        append!(cols, bins_of(g, c))
    end
    A, B = m.allele[1], m.allele[2]
    _with_io(dest) do io
        println(io, join(("sample_id", "chrom", "start", "end", "cn_a", "cn_b"), '\t'))
        for col in cols
            b = g.bins[col]
            println(io, join((normal_name, chromname(a, b.chrom), b.start - 1, b.stop, 1, 1), '\t'))
        end
        for row in 1:ncells(m)
            nm = m.names[row]
            for col in cols
                b = g.bins[col]
                println(io, join((nm, chromname(a, b.chrom), b.start - 1, b.stop,
                                  A[row, col], B[row, col]), '\t'))
            end
        end
    end
    return dest
end
