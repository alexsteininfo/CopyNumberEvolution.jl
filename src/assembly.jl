"""
    ChromosomeSpec(name, length, centromere)

One chromosome of a [`GenomeAssembly`](@ref).

Coordinates are **1-based inclusive**, so `centromere` is the closed interval the
centromere occupies and `length` is the chromosome's length in base pairs.

# Arguments
- `name::String` — chromosome name, e.g. `"chr1"`. `"chrX"` and `"chrY"` are
  recognised by the sex-mode ploidy rules.
- `length::Int` — chromosome length in bp.
- `centromere::UnitRange{Int}` — the centromeric interval, required for arm-level
  events.
"""
struct ChromosomeSpec
    name::String
    length::Int
    centromere::UnitRange{Int}

    function ChromosomeSpec(name::AbstractString, len::Integer, cen::UnitRange{<:Integer})
        len > 0 || throw(ArgumentError("chromosome $name: length must be positive, got $len"))
        first(cen) >= 2 || throw(ArgumentError("chromosome $name: centromere must start at ≥ 2 so the p arm is non-empty, got $cen"))
        last(cen) <= len - 1 || throw(ArgumentError("chromosome $name: centromere must end at ≤ length-1 so the q arm is non-empty, got $cen with length $len"))
        new(String(name), Int(len), Int(first(cen)):Int(last(cen)))
    end
end

"""
    GenomeAssembly(name, sex, chromosomes[, ploidy])

Chromosome definitions plus the *haplotype slot layout* they imply.

A profile stores one segmentation per **slot**, where a slot is one haplotype of one
chromosome. Sex mode *is* the slot layout rather than a separate switch: a female
assembly gives `chrY` a ploidy of 0 and therefore no slots, while keeping the
chromosome in the table so chromosome indices are comparable across sexes.

When `ploidy` is omitted it is derived from `sex`: 2 for every autosome, 2 X and no Y
for `:female`, one X and one Y for `:male`.

# Arguments
- `name::String` — assembly name, e.g. `"hg38"`.
- `sex::Symbol` — `:female` or `:male`.
- `chromosomes::Vector{ChromosomeSpec}`.
- `ploidy::Vector{Int}` — copies of each chromosome in the initial karyotype.

# Examples
```jldoctest
julia> a = hg38(:male);

julia> nslots(a)
46

julia> ploidy(a, chromindex(a, "chrY"))
1
```
"""
struct GenomeAssembly
    name::String
    sex::Symbol
    chromosomes::Vector{ChromosomeSpec}
    ploidy::Vector{Int}
    slotoffset::Vector{Int}
    slotchrom::Vector{Int}
    nslots::Int
    eligible::Vector{Int}

    function GenomeAssembly(name::AbstractString, sex::Symbol,
                            chromosomes::Vector{ChromosomeSpec}, ploidy::Vector{Int})
        length(ploidy) == length(chromosomes) ||
            throw(ArgumentError("ploidy has $(length(ploidy)) entries but there are $(length(chromosomes)) chromosomes"))
        all(>=(0), ploidy) || throw(ArgumentError("ploidy entries must be non-negative, got $ploidy"))
        any(>(0), ploidy) || throw(ArgumentError("at least one chromosome must have positive ploidy"))
        n = length(chromosomes)
        offset = zeros(Int, n)
        for c in 2:n
            offset[c] = offset[c - 1] + ploidy[c - 1]
        end
        total = sum(ploidy)
        slotchrom = zeros(Int, total)
        for c in 1:n, h in 1:ploidy[c]
            slotchrom[offset[c] + h] = c
        end
        eligible = [c for c in 1:n if ploidy[c] > 0]
        new(String(name), sex, chromosomes, copy(ploidy), offset, slotchrom, total, eligible)
    end
end

function GenomeAssembly(name::AbstractString, sex::Symbol, chromosomes::Vector{ChromosomeSpec})
    GenomeAssembly(name, sex, chromosomes, _sex_ploidy(chromosomes, sex))
end

function _sex_ploidy(chromosomes::Vector{ChromosomeSpec}, sex::Symbol)
    sex in (:female, :male) ||
        throw(ArgumentError("sex must be :female or :male, got :$sex"))
    map(chromosomes) do spec
        if spec.name == "chrX"
            sex === :female ? 2 : 1
        elseif spec.name == "chrY"
            sex === :female ? 0 : 1
        else
            2
        end
    end
end

"""
    nchromosomes(assembly) -> Int

Number of chromosomes in the table, including any with zero ploidy.
"""
nchromosomes(a::GenomeAssembly) = length(a.chromosomes)

"""
    chromname(assembly, chrom) -> String

Name of chromosome index `chrom`.
"""
chromname(a::GenomeAssembly, c::Integer) = a.chromosomes[c].name

"""
    chromlength(assembly, chrom) -> Int

Length of chromosome `chrom` in base pairs.
"""
chromlength(a::GenomeAssembly, c::Integer) = a.chromosomes[c].length

"""
    centromere(assembly, chrom) -> UnitRange{Int}

The centromeric interval of chromosome `chrom`, 1-based inclusive.
"""
centromere(a::GenomeAssembly, c::Integer) = a.chromosomes[c].centromere

"""
    ploidy(assembly, chrom) -> Int

Number of copies of chromosome `chrom` in the initial karyotype; `0` means the
chromosome has no slots (e.g. `chrY` in a female assembly).
"""
ploidy(a::GenomeAssembly, c::Integer) = a.ploidy[c]

"""
    nslots(assembly) -> Int

Total number of haplotype slots. Both `hg38(:female)` and `hg38(:male)` give 46.
"""
nslots(a::GenomeAssembly) = a.nslots

"""
    slot(assembly, chrom, haplotype) -> Int

Linear slot index of haplotype `haplotype` of chromosome `chrom`. Throws if the
haplotype does not exist at that chromosome's ploidy.
"""
function slot(a::GenomeAssembly, c::Integer, h::Integer)
    1 <= h <= a.ploidy[c] ||
        throw(ArgumentError("chromosome $(chromname(a, c)) has ploidy $(a.ploidy[c]); haplotype $h does not exist"))
    return a.slotoffset[c] + h
end

"""
    slots_of(assembly, chrom) -> UnitRange{Int}

Slot indices belonging to chromosome `chrom`; empty when its ploidy is 0.
"""
slots_of(a::GenomeAssembly, c::Integer) =
    (a.slotoffset[c] + 1):(a.slotoffset[c] + a.ploidy[c])

"""
    slot_chrom(assembly, s) -> Int

Chromosome index owning slot `s`.
"""
slot_chrom(a::GenomeAssembly, s::Integer) = a.slotchrom[s]

"""
    slot_haplotype(assembly, s) -> Int

Haplotype index of slot `s` within its chromosome.
"""
slot_haplotype(a::GenomeAssembly, s::Integer) = s - a.slotoffset[a.slotchrom[s]]

"""
    chromindex(assembly, name) -> Int

Index of the chromosome called `name`. Throws if there is no such chromosome.
"""
function chromindex(a::GenomeAssembly, name::AbstractString)
    for c in 1:nchromosomes(a)
        a.chromosomes[c].name == name && return c
    end
    throw(ArgumentError("no chromosome named $name in assembly $(a.name)"))
end

"""
    arms(assembly, chrom) -> (p, q)

The p and q arm intervals of chromosome `chrom`, flanking the centromere. Together
with the centromere they partition `1:chromlength(assembly, chrom)`.
"""
function arms(a::GenomeAssembly, c::Integer)
    cen = centromere(a, c)
    return (1:(first(cen) - 1), (last(cen) + 1):chromlength(a, c))
end

"""
    eligible_chromosomes(assembly) -> Vector{Int}

Chromosomes with positive ploidy — the only ones a CNA can target. Precomputed, so
this is allocation-free to read in the drawing loop.
"""
eligible_chromosomes(a::GenomeAssembly) = a.eligible

"""
    autosomes(assembly) -> Vector{Int}

Indices of chromosomes that are neither `chrX` nor `chrY`.
"""
autosomes(a::GenomeAssembly) =
    [c for c in 1:nchromosomes(a) if !(chromname(a, c) in ("chrX", "chrY"))]

"""
    same_assembly(a, b) -> Bool

Whether two assemblies describe the same genome: same name, same sex, same ploidy
vector, and the same name, length and centromere for every chromosome.

Used wherever two objects must agree on the genome they live on — comparing profiles,
validating a supplied ancestral state, projecting onto a bin grid. Comparing only the
name and sex would let two structurally different assemblies pass as equal, since
nothing stops a caller from building two different genomes under one name.
"""
function same_assembly(a::GenomeAssembly, b::GenomeAssembly)
    a.name == b.name || return false
    a.sex == b.sex || return false
    a.ploidy == b.ploidy || return false
    nchromosomes(a) == nchromosomes(b) || return false
    for c in 1:nchromosomes(a)
        x, y = a.chromosomes[c], b.chromosomes[c]
        (x.name == y.name && x.length == y.length && x.centromere == y.centromere) || return false
    end
    return true
end

Base.show(io::IO, a::GenomeAssembly) =
    print(io, "GenomeAssembly(", a.name, ", :", a.sex, ", ",
          nchromosomes(a), " chromosomes, ", nslots(a), " slots)")

# Chromosome tables
#
# Lengths: UCSC hg38.chrom.sizes / hg19.chrom.sizes.
# Centromeres: merged `acen` intervals of UCSC cytoBand, converted from 0-based
# half-open to 1-based inclusive by adding 1 to the start.
# Retrieved 2026-09-04 from https://hgdownload.soe.ucsc.edu/goldenPath/{hg38,hg19}/.
# `test/data/{hg38,hg19}.reference.tsv` asserts these values; update both together.

const _HG38 = ChromosomeSpec[
    ChromosomeSpec("chr1", 248956422, 121700001:125100000),
    ChromosomeSpec("chr2", 242193529, 91800001:96000000),
    ChromosomeSpec("chr3", 198295559, 87800001:94000000),
    ChromosomeSpec("chr4", 190214555, 48200001:51800000),
    ChromosomeSpec("chr5", 181538259, 46100001:51400000),
    ChromosomeSpec("chr6", 170805979, 58500001:62600000),
    ChromosomeSpec("chr7", 159345973, 58100001:62100000),
    ChromosomeSpec("chr8", 145138636, 43200001:47200000),
    ChromosomeSpec("chr9", 138394717, 42200001:45500000),
    ChromosomeSpec("chr10", 133797422, 38000001:41600000),
    ChromosomeSpec("chr11", 135086622, 51000001:55800000),
    ChromosomeSpec("chr12", 133275309, 33200001:37800000),
    ChromosomeSpec("chr13", 114364328, 16500001:18900000),
    ChromosomeSpec("chr14", 107043718, 16100001:18200000),
    ChromosomeSpec("chr15", 101991189, 17500001:20500000),
    ChromosomeSpec("chr16", 90338345, 35300001:38400000),
    ChromosomeSpec("chr17", 83257441, 22700001:27400000),
    ChromosomeSpec("chr18", 80373285, 15400001:21500000),
    ChromosomeSpec("chr19", 58617616, 24200001:28100000),
    ChromosomeSpec("chr20", 64444167, 25700001:30400000),
    ChromosomeSpec("chr21", 46709983, 10900001:13000000),
    ChromosomeSpec("chr22", 50818468, 13700001:17400000),
    ChromosomeSpec("chrX", 156040895, 58100001:63800000),
    ChromosomeSpec("chrY", 57227415, 10300001:10600000),
]

const _HG19 = ChromosomeSpec[
    ChromosomeSpec("chr1", 249250621, 121500001:128900000),
    ChromosomeSpec("chr2", 243199373, 90500001:96800000),
    ChromosomeSpec("chr3", 198022430, 87900001:93900000),
    ChromosomeSpec("chr4", 191154276, 48200001:52700000),
    ChromosomeSpec("chr5", 180915260, 46100001:50700000),
    ChromosomeSpec("chr6", 171115067, 58700001:63300000),
    ChromosomeSpec("chr7", 159138663, 58000001:61700000),
    ChromosomeSpec("chr8", 146364022, 43100001:48100000),
    ChromosomeSpec("chr9", 141213431, 47300001:50700000),
    ChromosomeSpec("chr10", 135534747, 38000001:42300000),
    ChromosomeSpec("chr11", 135006516, 51600001:55700000),
    ChromosomeSpec("chr12", 133851895, 33300001:38200000),
    ChromosomeSpec("chr13", 115169878, 16300001:19500000),
    ChromosomeSpec("chr14", 107349540, 16100001:19100000),
    ChromosomeSpec("chr15", 102531392, 15800001:20700000),
    ChromosomeSpec("chr16", 90354753, 34600001:38600000),
    ChromosomeSpec("chr17", 81195210, 22200001:25800000),
    ChromosomeSpec("chr18", 78077248, 15400001:19000000),
    ChromosomeSpec("chr19", 59128983, 24400001:28600000),
    ChromosomeSpec("chr20", 63025520, 25600001:29400000),
    ChromosomeSpec("chr21", 48129895, 10900001:14300000),
    ChromosomeSpec("chr22", 51304566, 12200001:17900000),
    ChromosomeSpec("chrX", 155270560, 58100001:63000000),
    ChromosomeSpec("chrY", 59373566, 11600001:13400000),
]

"""
    hg38(sex = :female) -> GenomeAssembly

The GRCh38/hg38 human assembly: 22 autosomes plus X and Y, with real chromosome
lengths and centromere positions. `sex` decides the slot layout (see
[`GenomeAssembly`](@ref)).
"""
hg38(sex::Symbol = :female) = GenomeAssembly("hg38", sex, _HG38)

"""
    hg19(sex = :female) -> GenomeAssembly

The GRCh37/hg19 human assembly. See [`hg38`](@ref).
"""
hg19(sex::Symbol = :female) = GenomeAssembly("hg19", sex, _HG19)
