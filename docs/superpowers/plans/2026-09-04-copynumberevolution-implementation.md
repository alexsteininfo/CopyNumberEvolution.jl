# CopyNumberEvolution.jl Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Julia package that turns a cell-lineage tree (from `MutationLoadDynamics.jl` or a newick file) into allele-specific copy-number profiles for every node, with a complete event log, a bin-grid projection, MEDICC2-compatible export, and comprehensive documentation.

**Architecture:** Pure forward observation model. A flat-vector `PhyloTree` is the input; a slot-indexed segmentation (`CNProfile`) is the state; CNAs are drawn per edge from injectable rate/target/extent/kind rules under a viability policy, applied through a single `apply!` interface, and logged. Profiles project onto a fixed `BinGrid` to produce the cells × bins integer matrix every inference method consumes. The simulator dependency lives behind a package extension so downstream inference packages never inherit it.

**Tech Stack:** Julia ≥ 1.10; `Random`, `Distributions`, `StatsBase`; `Test` stdlib; `Documenter.jl` for docs; `MutationLoadDynamics.jl` as a weak dependency.

**Spec:** `docs/superpowers/specs/2026-09-04-copynumberevolution-design.md` — read it alongside this plan. Every task cites the spec sections it implements.

## Global Constraints

Every task's requirements implicitly include this section.

- **Package identity:** name `CopyNumberEvolution`, UUID `a7fad464-da36-487c-8111-789b46084000`, version `0.1.0`, author `Alexander Stein <alexander.stein@vib.be>`.
- **Julia compat:** `julia = "1.10"` (package extensions need ≥ 1.9). Local toolchain is 1.12.1.
- **Dependencies are capped at three:** `Random` (`9a3f8284-a2c9-5f02-9a11-845980a1fd5c`), `Distributions` (`31c24e10-a181-5473-b8eb-7969acd0382f`), `StatsBase` (`2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91`). **Do not add `AbstractTrees`** — the flat tree does not need it, and cheap type-only dependency is the point (spec §14).
- **Weak dependency only:** `MutationLoadDynamics` (`7b855ee6-6887-412f-a571-26d20a5a92d7`) under `[weakdeps]`, never `[deps]`. A hard dependency would make the simulator transitively required to analyse clinical data (spec §2).
- **Coordinates:** internal `Segment` coordinates are **1-based inclusive**. MEDICC2 export is **0-based half-open (BED)**. Convert only at the writer boundary (spec §13.1).
- **Zero-copy state is absorbing** and is never switchable off: a gain never raises a `cn` of 0 (spec §6.3).
- **Canonical form is mandatory** after every operation: sorted, contiguous, gapless, no adjacent equal `cn`, `cn >= 0` (spec §6.2).
- **RNG determinism:** never iterate a `Dict` in a code path that consumes the RNG. This is why segments are slot-indexed (spec §6.1).
- **No real patient data in the repo, ever.** Fixtures are synthetic and small. `data/` is in `.gitignore` from Task 1 (spec §14).
- **Every exported symbol carries a docstring** with a `# Arguments` or `# Examples` section where it takes parameters. Task 16 enforces this with a test; do not defer docstrings to Task 16.
- **Test commands:** run the full suite with `julia --project=. -e 'using Pkg; Pkg.test()'`. Run a single file during development with `julia --project=. -e 'using Pkg; Pkg.activate("."); include("test/<file>.jl")'` after `using CopyNumberEvolution, Test, Random`.
- **Commit after every task** with a `feat:`/`test:`/`docs:` prefixed message.

## Additions to the spec made by this plan

Flagged explicitly so review can reject them:

1. **`simulate_cnas` takes the assembly as an argument** (`simulate_cnas(tree, assembly, model; ...)`). Spec §11.1 omitted it, but `Diploid()` cannot build a profile without one. Task 11.
2. **`rng_mode = :global | :per_node`** on `simulate_cnas`. `:global` is the default and behaves exactly as the spec describes. `:per_node` derives each edge's RNG from `(seed, source_id)` so that a given edge draws identically regardless of which other edges exist — which turns spec test 7 ("sampling commutes") from a distributional test into an exact one. Tasks 11 and 15.
3. **`violation(rule, profile, event) -> Union{Symbol,Nothing}`** as the primitive behind `isviable`, so the rejection tally can be keyed by reason as spec §11.2 requires. Task 10.
4. **`CNMatrix` carries `names::Vector{String}`** alongside `cells::Vector{Int}`, so the MEDICC2 writer is self-describing and its `sample_id` values agree with newick leaf labels via a shared `cellname`. Tasks 12 and 13.
5. **`CNMatrix.total` is defined as the sum of the per-haplotype projections**, not as a projection of the total segmentation. Bin projection is non-linear, so the two differ at straddling bins; defining it this way guarantees `total == A + B`, which MEDICC2 consumers rely on. Task 12.

---

### Task 1: Package scaffold

Implements spec §14.

**Files:**
- Create: `Project.toml`
- Create: `src/CopyNumberEvolution.jl`
- Create: `test/runtests.jl`
- Create: `test/fixtures.jl`
- Create: `.github/workflows/CI.yml`
- Modify: `.gitignore` (append)

**Interfaces:**
- Consumes: nothing.
- Produces: the `CopyNumberEvolution` module; `test/fixtures.jl` providing `toy_assembly`, used by every later task's tests.

- [ ] **Step 1: Write `Project.toml`**

```toml
name = "CopyNumberEvolution"
uuid = "a7fad464-da36-487c-8111-789b46084000"
authors = ["Alexander Stein <alexander.stein@vib.be>"]
version = "0.1.0"

[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[weakdeps]
MutationLoadDynamics = "7b855ee6-6887-412f-a571-26d20a5a92d7"

[extensions]
CopyNumberEvolutionMutationLoadDynamicsExt = "MutationLoadDynamics"

[compat]
Distributions = "0.25"
StatsBase = "0.33, 0.34"
julia = "1.10"

[extras]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Distributions", "Test"]
```

- [ ] **Step 2: Write the module file with the include order**

The include order is a dependency order; later tasks fill these files in. Create every file as an empty placeholder in this step so the module loads.

```julia
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
```

- [ ] **Step 3: Write `test/fixtures.jl`**

Small synthetic assemblies keep the suite fast and make failures readable. Real hg38
is used only where the test is about hg38.

```julia
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
```

- [ ] **Step 4: Write `test/runtests.jl`**

Add `include` lines as each later task lands. Start with only the smoke test.

```julia
using CopyNumberEvolution
using Test
using Random
using Distributions   # tests construct length distributions directly

include("fixtures.jl")

@testset "CopyNumberEvolution.jl" begin
    @testset "smoke" begin
        @test isdefined(CopyNumberEvolution, :CopyNumberEvolution)
    end
end
```

- [ ] **Step 5: Append to `.gitignore`**

```gitignore

# Julia
Manifest.toml
/docs/build/
/docs/Manifest.toml
*.jl.cov
*.jl.*.cov
*.jl.mem

# Never commit real data — see docs/superpowers/specs/2026-09-04-copynumberevolution-design.md §14
data/
```

- [ ] **Step 6: Write `.github/workflows/CI.yml`**

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        version: ['1.10', '1']
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: ${{ matrix.version }}
      - uses: julia-actions/cache@v2
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
```

- [ ] **Step 7: Run the suite to verify the scaffold loads**

Run: `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`
Expected: PASS, one test.

- [ ] **Step 8: Commit**

```bash
git add Project.toml src test .github .gitignore
git commit -m "feat: scaffold CopyNumberEvolution.jl package"
```

---

### Task 2: `GenomeAssembly` and the hg38/hg19 tables

Implements spec §5. The chromosome tables were transcribed on 2026-09-04 from UCSC
`hg38.chrom.sizes` / `hg19.chrom.sizes` (lengths) and the merged `acen` intervals of
UCSC `cytoBand` (centromeres), converted from 0-based half-open to **1-based
inclusive** by adding 1 to the start.

**Files:**
- Create: `src/assembly.jl`
- Create: `test/data/hg38.reference.tsv`
- Create: `test/data/hg19.reference.tsv`
- Create: `test/test_assembly.jl`
- Modify: `src/CopyNumberEvolution.jl` (exports)
- Modify: `test/runtests.jl` (include)

**Interfaces:**
- Consumes: nothing.
- Produces: `ChromosomeSpec(name::String, length::Int, centromere::UnitRange{Int})`;
  `GenomeAssembly`; `hg38(sex::Symbol=:female)`, `hg19(sex::Symbol=:female)`;
  `nchromosomes(a)::Int`, `chromname(a,c)::String`, `chromlength(a,c)::Int`,
  `centromere(a,c)::UnitRange{Int}`, `ploidy(a,c)::Int`, `nslots(a)::Int`,
  `slot(a,c,h)::Int`, `slots_of(a,c)::UnitRange{Int}`,
  `slot_chrom(a,s)::Int`, `slot_haplotype(a,s)::Int`,
  `chromindex(a,name)::Int`, `arms(a,c)::Tuple{UnitRange{Int},UnitRange{Int}}`,
  `eligible_chromosomes(a)::Vector{Int}`, `autosomes(a)::Vector{Int}`.

- [ ] **Step 1: Write the reference data files**

`test/data/hg38.reference.tsv` (tab-separated, 1-based inclusive centromeres):

```
name	length	cen_start	cen_stop
chr1	248956422	121700001	125100000
chr2	242193529	91800001	96000000
chr3	198295559	87800001	94000000
chr4	190214555	48200001	51800000
chr5	181538259	46100001	51400000
chr6	170805979	58500001	62600000
chr7	159345973	58100001	62100000
chr8	145138636	43200001	47200000
chr9	138394717	42200001	45500000
chr10	133797422	38000001	41600000
chr11	135086622	51000001	55800000
chr12	133275309	33200001	37800000
chr13	114364328	16500001	18900000
chr14	107043718	16100001	18200000
chr15	101991189	17500001	20500000
chr16	90338345	35300001	38400000
chr17	83257441	22700001	27400000
chr18	80373285	15400001	21500000
chr19	58617616	24200001	28100000
chr20	64444167	25700001	30400000
chr21	46709983	10900001	13000000
chr22	50818468	13700001	17400000
chrX	156040895	58100001	63800000
chrY	57227415	10300001	10600000
```

`test/data/hg19.reference.tsv`:

```
name	length	cen_start	cen_stop
chr1	249250621	121500001	128900000
chr2	243199373	90500001	96800000
chr3	198022430	87900001	93900000
chr4	191154276	48200001	52700000
chr5	180915260	46100001	50700000
chr6	171115067	58700001	63300000
chr7	159138663	58000001	61700000
chr8	146364022	43100001	48100000
chr9	141213431	47300001	50700000
chr10	135534747	38000001	42300000
chr11	135006516	51600001	55700000
chr12	133851895	33300001	38200000
chr13	115169878	16300001	19500000
chr14	107349540	16100001	19100000
chr15	102531392	15800001	20700000
chr16	90354753	34600001	38600000
chr17	81195210	22200001	25800000
chr18	78077248	15400001	19000000
chr19	59128983	24400001	28600000
chr20	63025520	25600001	29400000
chr21	48129895	10900001	14300000
chr22	51304566	12200001	17900000
chrX	155270560	58100001	63000000
chrY	59373566	11600001	13400000
```

- [ ] **Step 2: Write the failing tests**

`test/test_assembly.jl`:

```julia
@testset "assembly" begin
    @testset "hg38/hg19 tables match the UCSC reference" begin
        for (ctor, file) in ((hg38, "hg38.reference.tsv"), (hg19, "hg19.reference.tsv"))
            a = ctor(:male)
            rows = readlines(joinpath(@__DIR__, "data", file))[2:end]
            @test nchromosomes(a) == length(rows)
            for (c, row) in enumerate(rows)
                name, len, cs, ce = split(row, '\t')
                @test chromname(a, c) == name
                @test chromlength(a, c) == parse(Int, len)
                @test centromere(a, c) == parse(Int, cs):parse(Int, ce)
            end
        end
    end

    @testset "sex sets the slot layout" begin
        f = hg38(:female)
        m = hg38(:male)
        @test nchromosomes(f) == 24 && nchromosomes(m) == 24
        @test ploidy(f, chromindex(f, "chrX")) == 2
        @test ploidy(f, chromindex(f, "chrY")) == 0
        @test ploidy(m, chromindex(m, "chrX")) == 1
        @test ploidy(m, chromindex(m, "chrY")) == 1
        # 22 autosome pairs + 2 X, and 22 pairs + X + Y: both are 46
        @test nslots(f) == 46
        @test nslots(m) == 46
        @test isempty(slots_of(f, chromindex(f, "chrY")))
    end

    @testset "slot mapping round-trips" begin
        for sex in (:female, :male)
            a = hg38(sex)
            seen = Int[]
            for c in 1:nchromosomes(a), h in 1:ploidy(a, c)
                s = slot(a, c, h)
                push!(seen, s)
                @test slot_chrom(a, s) == c
                @test slot_haplotype(a, s) == h
                @test s in slots_of(a, c)
            end
            @test sort(seen) == collect(1:nslots(a))
        end
    end

    @testset "slot bounds are checked" begin
        a = hg38(:male)
        @test_throws ArgumentError slot(a, chromindex(a, "chrX"), 2)
        @test_throws ArgumentError slot(a, chromindex(a, "chrY"), 2)
        @test_throws ArgumentError slot(a, 1, 0)
    end

    @testset "arms partition the chromosome around the centromere" begin
        a = hg38(:female)
        for c in 1:nchromosomes(a)
            p, q = arms(a, c)
            cen = centromere(a, c)
            @test last(p) == first(cen) - 1
            @test first(q) == last(cen) + 1
            @test last(q) == chromlength(a, c)
            @test isempty(intersect(p, cen)) && isempty(intersect(q, cen))
            @test length(p) + length(cen) + length(q) == chromlength(a, c)
        end
    end

    @testset "eligible chromosomes exclude zero-ploidy" begin
        f = hg38(:female)
        @test chromindex(f, "chrY") ∉ eligible_chromosomes(f)
        @test length(eligible_chromosomes(f)) == 23
        @test length(eligible_chromosomes(hg38(:male))) == 24
        @test length(autosomes(f)) == 22
    end

    @testset "unknown names and sexes are rejected" begin
        a = hg38(:female)
        @test_throws ArgumentError chromindex(a, "chr99")
        @test_throws ArgumentError hg38(:other)
    end
end
```

- [ ] **Step 3: Run the tests to verify they fail**

Add `include("test_assembly.jl")` inside the top-level `@testset` in `test/runtests.jl`, then run:
`julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL with `UndefVarError: hg38 not defined`.

- [ ] **Step 4: Write `src/assembly.jl`**

```julia
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
```

- [ ] **Step 5: Add the exports**

In `src/CopyNumberEvolution.jl`, after the `using` lines and before the `include`s:

```julia
export
    # Assembly
    ChromosomeSpec, GenomeAssembly, hg38, hg19,
    nchromosomes, chromname, chromlength, centromere, ploidy,
    nslots, slot, slots_of, slot_chrom, slot_haplotype,
    chromindex, arms, eligible_chromosomes, autosomes
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/assembly.jl src/CopyNumberEvolution.jl test/data test/test_assembly.jl test/runtests.jl
git commit -m "feat: add GenomeAssembly with hg38/hg19 tables and slot layout"
```

---

### Task 3: `Segment`, `CNProfile`, invariants, total copy number

Implements spec §6.1, §6.2, §6.4.

**Files:**
- Create: `src/profile.jl`
- Create: `test/test_profile.jl`
- Modify: `src/CopyNumberEvolution.jl` (exports)
- Modify: `test/runtests.jl` (include)

**Interfaces:**
- Consumes: `GenomeAssembly` and its accessors from Task 2.
- Produces: `Segment(start::Int, stop::Int, cn::Int)` with `Base.length`;
  `CNProfile(assembly, segments::Vector{Vector{Segment}})`; `diploid(assembly)::CNProfile`;
  `Base.copy`, `==`, `hash` for `CNProfile`;
  `check_invariants(p::CNProfile)::Bool`;
  `canonicalize!(segs::Vector{Segment})::Vector{Segment}`;
  `segment_index(segs, pos)::Int`; `cn_at(segs, pos)::Int`;
  `slot_segments(p, chrom, hap)::Vector{Segment}`;
  `total_cn(p, chrom)::Vector{Segment}`; `mean_cn(segs, len)::Float64`;
  `nsegments(p)::Int`.

- [ ] **Step 1: Write the failing tests**

`test/test_profile.jl`:

```julia
@testset "profile" begin
    S = CopyNumberEvolution.Segment

    @testset "Segment length" begin
        @test length(S(1, 10, 2)) == 10
        @test length(S(5, 5, 0)) == 1
    end

    @testset "diploid profile" begin
        a = toy_assembly(nchrom = 2, len = 1000)
        p = diploid(a)
        @test check_invariants(p)
        @test nslots(a) == 4
        for c in 1:2, h in 1:2
            @test slot_segments(p, c, h) == [S(1, 1000, 1)]
        end
        @test nsegments(p) == 4
    end

    @testset "sex modes give the documented slot contents" begin
        f = diploid(toy_sex_assembly(:female))
        af = f.assembly
        @test ploidy(af, chromindex(af, "chrX")) == 2
        @test isempty(slots_of(af, chromindex(af, "chrY")))
        m = diploid(toy_sex_assembly(:male))
        am = m.assembly
        @test slot_segments(m, chromindex(am, "chrX"), 1) == [S(1, 1000, 1)]
        @test slot_segments(m, chromindex(am, "chrY"), 1) == [S(1, 1000, 1)]
        @test_throws ArgumentError slot_segments(m, chromindex(am, "chrX"), 2)
    end

    @testset "copy is deep" begin
        p = diploid(toy_assembly())
        q = copy(p)
        push!(q.segments[1], S(1, 1, 9))
        @test length(p.segments[1]) == 1
    end

    @testset "equality and hashing use canonical form" begin
        p = diploid(toy_assembly())
        q = diploid(toy_assembly())
        @test p == q
        @test hash(p) == hash(q)
        r = copy(p)
        r.segments[1][1] = S(1, 1000, 2)
        @test p != r
    end

    @testset "canonicalize! merges adjacent equal copy numbers" begin
        segs = [S(1, 10, 1), S(11, 20, 1), S(21, 30, 2), S(31, 40, 2), S(41, 50, 1)]
        canonicalize!(segs)
        @test segs == [S(1, 20, 1), S(21, 40, 2), S(41, 50, 1)]
        single = [S(1, 5, 3)]
        @test canonicalize!(single) == [S(1, 5, 3)]
        allsame = [S(1, 2, 0), S(3, 4, 0), S(5, 6, 0)]
        @test canonicalize!(allsame) == [S(1, 6, 0)]
    end

    @testset "check_invariants names every violation" begin
        a = toy_assembly(nchrom = 1, len = 100)
        mk(segs) = CopyNumberEvolution.CNProfile(a, [copy(segs), copy(segs)])
        @test check_invariants(mk([S(1, 100, 1)]))
        # does not start at 1
        @test_throws ErrorException check_invariants(mk([S(2, 100, 1)]))
        # does not reach the chromosome end
        @test_throws ErrorException check_invariants(mk([S(1, 99, 1)]))
        # gap
        @test_throws ErrorException check_invariants(mk([S(1, 40, 1), S(42, 100, 2)]))
        # overlap
        @test_throws ErrorException check_invariants(mk([S(1, 40, 1), S(40, 100, 2)]))
        # adjacent equal cn (non-canonical)
        @test_throws ErrorException check_invariants(mk([S(1, 40, 1), S(41, 100, 1)]))
        # negative cn
        @test_throws ErrorException check_invariants(mk([S(1, 40, -1), S(41, 100, 1)]))
        # start > stop
        @test_throws ErrorException check_invariants(mk([S(40, 1, 1)]))
        # empty
        @test_throws ErrorException check_invariants(mk(S[]))
    end

    @testset "segment_index and cn_at" begin
        segs = [S(1, 10, 1), S(11, 20, 3), S(21, 30, 0)]
        @test segment_index(segs, 1) == 1
        @test segment_index(segs, 10) == 1
        @test segment_index(segs, 11) == 2
        @test segment_index(segs, 30) == 3
        @test cn_at(segs, 5) == 1
        @test cn_at(segs, 11) == 3
        @test cn_at(segs, 25) == 0
    end

    @testset "total_cn sums haplotypes and stays canonical" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        @test total_cn(p, 1) == [S(1, 100, 2)]
        # a gain on haplotype 1 over 21:40 only
        p.segments[slot(a, 1, 1)] = [S(1, 20, 1), S(21, 40, 2), S(41, 100, 1)]
        @test total_cn(p, 1) == [S(1, 20, 2), S(21, 40, 3), S(41, 100, 2)]
        # a matching loss on haplotype 2 makes the total flat again — and canonical
        p.segments[slot(a, 1, 2)] = [S(1, 20, 1), S(21, 40, 0), S(41, 100, 1)]
        @test total_cn(p, 1) == [S(1, 100, 2)]
    end

    @testset "total_cn on hemizygous and absent chromosomes" begin
        m = diploid(toy_sex_assembly(:male))
        am = m.assembly
        @test total_cn(m, chromindex(am, "chrX")) == [S(1, 1000, 1)]
        f = diploid(toy_sex_assembly(:female))
        af = f.assembly
        @test total_cn(f, chromindex(af, "chrY")) == [S(1, 1000, 0)]
        @test total_cn(f, chromindex(af, "chrX")) == [S(1, 1000, 2)]
    end

    @testset "total_cn equals the sum over slots at every position" begin
        rng = Random.Xoshiro(1234)
        a = toy_assembly(nchrom = 2, len = 200)
        p = diploid(a)
        # scatter random breakpoints and copy numbers, keeping canonical form
        for s in 1:nslots(a)
            L = chromlength(a, slot_chrom(a, s))
            bps = sort(unique(rand(rng, 2:L, 6)))
            segs = CopyNumberEvolution.Segment[]
            prev = 1
            for b in vcat(bps, L + 1)
                b > prev || continue
                push!(segs, CopyNumberEvolution.Segment(prev, b - 1, rand(rng, 0:4)))
                prev = b
            end
            canonicalize!(segs)
            p.segments[s] = segs
        end
        @test check_invariants(p)
        for c in 1:nchromosomes(a)
            tot = total_cn(p, c)
            for pos in (1, 37, 100, 199, chromlength(a, c))
                expected = sum(cn_at(p.segments[s], pos) for s in slots_of(a, c))
                @test cn_at(tot, pos) == expected
            end
        end
    end

    @testset "mean_cn" begin
        @test CopyNumberEvolution.mean_cn([S(1, 50, 2), S(51, 100, 0)], 100) ≈ 1.0
        @test CopyNumberEvolution.mean_cn([S(1, 100, 3)], 100) ≈ 3.0
    end
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Add `include("test_profile.jl")` to `test/runtests.jl`, then run:
`julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL with `UndefVarError: diploid not defined`.

- [ ] **Step 3: Write `src/profile.jl`**

```julia
"""
    Segment(start, stop, cn)

A run of constant copy number on one haplotype of one chromosome.

Coordinates are **1-based inclusive**, so `Segment(1, 10, 2)` covers ten base pairs.
`cn` is the number of copies of *this haplotype* over the interval, so a normal
diploid autosome is two slots each holding a single `Segment(1, L, 1)` — which is
exactly MEDICC2's `cn_a`/`cn_b` convention.
"""
struct Segment
    start::Int
    stop::Int
    cn::Int
end

Base.length(s::Segment) = s.stop - s.start + 1

Base.show(io::IO, s::Segment) = print(io, "Segment(", s.start, ":", s.stop, ", cn=", s.cn, ")")

"""
    CNProfile(assembly, segments)

One cell's allele-specific copy-number state.

`segments[s]` is the segmentation of haplotype slot `s` (see [`GenomeAssembly`](@ref)):
a sorted, gapless, non-overlapping list of [`Segment`](@ref)s tiling
`1:chromlength(assembly, slot_chrom(assembly, s))`, with no two adjacent segments
sharing a copy number. That canonical form makes `==` meaningful — two profiles are
equal exactly when their segmentations are.

Storage is indexed by slot rather than keyed by `(chromosome, haplotype)` because
`Dict` iteration order is unspecified, and any RNG-consuming pass over haplotypes
would then be irreproducible.

Total copy number is a derived view, never a second representation — see
[`total_cn`](@ref).

Build a normal starting state with [`diploid`](@ref); validate one with
[`check_invariants`](@ref).
"""
struct CNProfile
    assembly::GenomeAssembly
    segments::Vector{Vector{Segment}}
end

"""
    diploid(assembly) -> CNProfile

The normal starting karyotype for `assembly`: every haplotype slot is a single
segment at copy number 1, spanning its whole chromosome.

Sex mode is already baked into the assembly's slot layout, so a male assembly yields
one `chrX` and one `chrY` slot and a female assembly yields two `chrX` slots and no
`chrY` slot.

# Examples
```jldoctest
julia> p = diploid(hg38(:female));

julia> only(total_cn(p, 1)).cn
2

julia> nsegments(p)
46
```
"""
function diploid(a::GenomeAssembly)
    segs = Vector{Vector{Segment}}(undef, nslots(a))
    for c in 1:nchromosomes(a), h in 1:ploidy(a, c)
        segs[slot(a, c, h)] = [Segment(1, chromlength(a, c), 1)]
    end
    return CNProfile(a, segs)
end

Base.copy(p::CNProfile) = CNProfile(p.assembly, [copy(v) for v in p.segments])

function Base.:(==)(x::CNProfile, y::CNProfile)
    x.assembly.name == y.assembly.name || return false
    x.assembly.sex == y.assembly.sex || return false
    return x.segments == y.segments
end

Base.hash(p::CNProfile, h::UInt) =
    hash(p.segments, hash(p.assembly.sex, hash(p.assembly.name, h)))

Base.show(io::IO, p::CNProfile) =
    print(io, "CNProfile(", p.assembly.name, ", :", p.assembly.sex, ", ",
          nsegments(p), " segments across ", length(p.segments), " slots)")

"""
    nsegments(profile) -> Int

Total number of segments across all slots — a cheap measure of how fragmented a
profile has become.
"""
nsegments(p::CNProfile) = sum(length, p.segments)

"""
    slot_segments(profile, chrom, haplotype) -> Vector{Segment}

The segmentation of one haplotype of one chromosome. Throws if that haplotype does
not exist at the chromosome's ploidy.
"""
slot_segments(p::CNProfile, c::Integer, h::Integer) =
    p.segments[slot(p.assembly, c, h)]

"""
    canonicalize!(segs) -> segs

Merge adjacent segments with equal copy number, in place. Restores the canonical
form required by [`check_invariants`](@ref) after an edit that may have left two
neighbours sharing a copy number.
"""
function canonicalize!(segs::Vector{Segment})
    i = 1
    while i < length(segs)
        if segs[i].cn == segs[i + 1].cn
            segs[i] = Segment(segs[i].start, segs[i + 1].stop, segs[i].cn)
            deleteat!(segs, i + 1)
        else
            i += 1
        end
    end
    return segs
end

"""
    check_invariants(profile) -> true

Verify every slot's segmentation and throw a descriptive `ErrorException` naming the
slot and segment index on the first violation. Checks that each slot is non-empty,
starts at 1, ends at the chromosome length, has no gaps or overlaps, has no adjacent
segments with equal copy number, and has no negative copy number.

A silently non-canonical segmentation is the failure mode that would poison every
downstream number, so call this liberally in tests.
"""
function check_invariants(p::CNProfile)
    a = p.assembly
    for c in 1:nchromosomes(a), h in 1:ploidy(a, c)
        s = slot(a, c, h)
        segs = p.segments[s]
        L = chromlength(a, c)
        where = "slot $s (chromosome $(chromname(a, c)), haplotype $h)"
        isempty(segs) && error("$where: empty segmentation")
        segs[1].start == 1 ||
            error("$where: first segment starts at $(segs[1].start), expected 1")
        segs[end].stop == L ||
            error("$where: last segment stops at $(segs[end].stop), expected $L")
        for (i, sg) in enumerate(segs)
            sg.start <= sg.stop ||
                error("$where segment $i: start $(sg.start) exceeds stop $(sg.stop)")
            sg.cn >= 0 ||
                error("$where segment $i: negative copy number $(sg.cn)")
            if i > 1
                prev = segs[i - 1]
                sg.start == prev.stop + 1 ||
                    error("$where segment $i: starts at $(sg.start), expected $(prev.stop + 1) (gap or overlap)")
                sg.cn != prev.cn ||
                    error("$where segment $i: adjacent segments both at copy number $(sg.cn); not canonical")
            end
        end
    end
    return true
end

"""
    segment_index(segs, pos) -> Int

Index of the segment containing `pos`, by binary search. `segs` must be sorted and
gapless, and `pos` must lie within it.
"""
function segment_index(segs::Vector{Segment}, pos::Integer)
    lo, hi = 1, length(segs)
    while lo < hi
        mid = (lo + hi + 1) >> 1
        if segs[mid].start <= pos
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end

"""
    cn_at(segs, pos) -> Int

Copy number at position `pos` in a single slot's segmentation.
"""
cn_at(segs::Vector{Segment}, pos::Integer) = segs[segment_index(segs, pos)].cn

"""
    total_cn(profile, chrom) -> Vector{Segment}

Total copy number of chromosome `chrom`, summed over its haplotype slots and returned
in canonical form.

This is a *derived view*, not a stored second representation: simulation is always
allele-specific, and the total-copy-number "mode" affects only output and comparison.
A chromosome with zero ploidy (`chrY` in a female assembly) returns a single segment
at copy number 0.
"""
function total_cn(p::CNProfile, c::Integer)
    a = p.assembly
    L = chromlength(a, c)
    sl = slots_of(a, c)
    isempty(sl) && return [Segment(1, L, 0)]

    breaks = Int[1]
    for s in sl, sg in p.segments[s]
        sg.start > 1 && push!(breaks, sg.start)
    end
    sort!(breaks)
    unique!(breaks)

    out = Vector{Segment}(undef, length(breaks))
    for (i, b) in enumerate(breaks)
        stop = i == length(breaks) ? L : breaks[i + 1] - 1
        tot = 0
        for s in sl
            tot += cn_at(p.segments[s], b)
        end
        out[i] = Segment(b, stop, tot)
    end
    return canonicalize!(out)
end

"""
    total_cn(profile) -> Vector{Vector{Segment}}

Total copy number for every chromosome, in chromosome order.
"""
total_cn(p::CNProfile) = [total_cn(p, c) for c in 1:nchromosomes(p.assembly)]

"""
    mean_cn(segs, len) -> Float64

Length-weighted mean copy number of one slot's segmentation over a chromosome of
`len` bp. Used by [`CNWeighted`](@ref) to condition target choice on the mother
cell's copy-number state.
"""
function mean_cn(segs::Vector{Segment}, len::Integer)
    acc = 0
    for sg in segs
        acc += length(sg) * sg.cn
    end
    return acc / len
end
```

- [ ] **Step 4: Add the exports**

Append to the `export` block in `src/CopyNumberEvolution.jl`:

```julia
export
    # Profiles
    Segment, CNProfile, diploid, check_invariants, canonicalize!,
    segment_index, cn_at, slot_segments, total_cn, nsegments
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/profile.jl src/CopyNumberEvolution.jl test/test_profile.jl test/runtests.jl
git commit -m "feat: add Segment, CNProfile, invariant checker and total copy number"
```

---

### Task 4: CNA events and `apply!`

Implements spec §6.3, §6.5, §7.6. This is where zero-absorption lives.

**Files:**
- Create: `src/cna.jl`
- Create: `test/test_cna.jl`
- Modify: `src/CopyNumberEvolution.jl` (exports)
- Modify: `test/runtests.jl` (include)

**Interfaces:**
- Consumes: `Segment`, `CNProfile`, `canonicalize!`, `segment_index` (Task 3);
  `slot`, `chromlength` (Task 2).
- Produces: `abstract type CNAEvent`;
  `SegmentalCNA(chrom::Int, haplotype::Int, start::Int, stop::Int, delta::Int, scale::Symbol)`;
  `WholeGenomeDoubling(mode::Symbol)`;
  `apply!(p::CNProfile, e::CNAEvent)::CNProfile`;
  `event_span(e::SegmentalCNA)::Int`.

- [ ] **Step 1: Write the failing tests**

`test/test_cna.jl`:

```julia
@testset "cna" begin
    S = CopyNumberEvolution.Segment

    @testset "constructor validation" begin
        @test_throws ArgumentError SegmentalCNA(1, 1, 10, 5, 1, :focal)
        @test_throws ArgumentError SegmentalCNA(1, 1, 0, 5, 1, :focal)
        @test_throws ArgumentError SegmentalCNA(1, 1, 1, 5, 0, :focal)
        @test_throws ArgumentError SegmentalCNA(1, 0, 1, 5, 1, :focal)
        @test_throws ArgumentError SegmentalCNA(1, 1, 1, 5, 1, :bogus)
        @test_throws ArgumentError WholeGenomeDoubling(:bogus)
        @test event_span(SegmentalCNA(1, 1, 11, 20, 1, :focal)) == 10
    end

    @testset "interior gain splits into three segments" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 21, 40, 1, :focal))
        @test slot_segments(p, 1, 1) == [S(1, 20, 1), S(21, 40, 2), S(41, 100, 1)]
        @test slot_segments(p, 1, 2) == [S(1, 100, 1)]
        @test check_invariants(p)
    end

    @testset "events at the chromosome boundaries" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 1, 30, 1, :focal))
        @test slot_segments(p, 1, 1) == [S(1, 30, 2), S(31, 100, 1)]
        q = diploid(a)
        apply!(q, SegmentalCNA(1, 1, 71, 100, 1, :focal))
        @test slot_segments(q, 1, 1) == [S(1, 70, 1), S(71, 100, 2)]
        r = diploid(a)
        apply!(r, SegmentalCNA(1, 1, 1, 100, 1, :chromosome))
        @test slot_segments(r, 1, 1) == [S(1, 100, 2)]
        @test check_invariants(p) && check_invariants(q) && check_invariants(r)
    end

    @testset "a loss to zero is LOH and needs no separate event type" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))
        @test slot_segments(p, 1, 1) == [S(1, 20, 1), S(21, 40, 0), S(41, 100, 1)]
        @test total_cn(p, 1) == [S(1, 20, 2), S(21, 40, 1), S(41, 100, 2)]
    end

    @testset "zero is absorbing: a gain never resurrects deleted material" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))       # 21:40 -> cn 0
        apply!(p, SegmentalCNA(1, 1, 1, 100, 1, :chromosome))   # gain the whole chromosome
        @test slot_segments(p, 1, 1) == [S(1, 20, 2), S(21, 40, 0), S(41, 100, 2)]
        @test check_invariants(p)
        # and a further loss over a zeroed run leaves it at zero, not negative
        apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))
        @test cn_at(slot_segments(p, 1, 1), 30) == 0
    end

    @testset "losses clamp at zero" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 1, 100, -3, :chromosome))
        @test slot_segments(p, 1, 1) == [S(1, 100, 0)]
        @test check_invariants(p)
    end

    @testset "overlapping events accumulate and re-canonicalise" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 21, 60, 1, :focal))
        apply!(p, SegmentalCNA(1, 1, 41, 80, 1, :focal))
        @test slot_segments(p, 1, 1) ==
              [S(1, 20, 1), S(21, 40, 2), S(41, 60, 3), S(61, 80, 2), S(81, 100, 1)]
        # a loss that exactly cancels the first gain merges neighbours again
        apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))
        @test slot_segments(p, 1, 1) ==
              [S(1, 40, 1), S(41, 60, 3), S(61, 80, 2), S(81, 100, 1)]
        @test check_invariants(p)
    end

    @testset "events on the wrong chromosome or haplotype throw" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        @test_throws ArgumentError apply!(p, SegmentalCNA(1, 3, 1, 10, 1, :focal))
        @test_throws BoundsError apply!(p, SegmentalCNA(9, 1, 1, 10, 1, :focal))
        @test_throws ArgumentError apply!(p, SegmentalCNA(1, 1, 1, 500, 1, :focal))
    end

    @testset "WGD :multiply doubles every slot and preserves zeros" begin
        a = toy_assembly(nchrom = 2, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))    # a zeroed run
        apply!(p, SegmentalCNA(2, 1, 1, 100, 1, :chromosome)) # a slot at cn 2
        apply!(p, WholeGenomeDoubling(:multiply))
        @test slot_segments(p, 1, 1) == [S(1, 20, 2), S(21, 40, 0), S(41, 100, 2)]
        @test slot_segments(p, 1, 2) == [S(1, 100, 2)]
        @test slot_segments(p, 2, 1) == [S(1, 100, 4)]
        @test check_invariants(p)
    end

    @testset "WGD :increment adds one to non-zero segments only" begin
        a = toy_assembly(nchrom = 2, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))
        apply!(p, SegmentalCNA(2, 1, 1, 100, 1, :chromosome))
        apply!(p, WholeGenomeDoubling(:increment))
        @test slot_segments(p, 1, 1) == [S(1, 20, 2), S(21, 40, 0), S(41, 100, 2)]
        @test slot_segments(p, 2, 1) == [S(1, 100, 3)]   # 2 + 1, not 2 * 2
    end

    @testset "the two WGD modes agree below cn 2 and diverge above it" begin
        a = toy_assembly(nchrom = 1, len = 100)
        flat = diploid(a)
        mult = copy(flat); incr = copy(flat)
        apply!(mult, WholeGenomeDoubling(:multiply))
        apply!(incr, WholeGenomeDoubling(:increment))
        @test mult == incr                     # all copy numbers were 0 or 1

        gained = diploid(a)
        apply!(gained, SegmentalCNA(1, 1, 1, 100, 1, :chromosome))   # cn 2
        mult2 = copy(gained); incr2 = copy(gained)
        apply!(mult2, WholeGenomeDoubling(:multiply))
        apply!(incr2, WholeGenomeDoubling(:increment))
        @test mult2 != incr2
        @test cn_at(slot_segments(mult2, 1, 1), 1) == 4
        @test cn_at(slot_segments(incr2, 1, 1), 1) == 3
    end

    @testset "WGD is not stopped by chromosome boundaries" begin
        a = toy_assembly(nchrom = 3, len = 50)
        p = diploid(a)
        apply!(p, WholeGenomeDoubling(:multiply))
        for c in 1:3, h in 1:2
            @test slot_segments(p, c, h) == [S(1, 50, 2)]
        end
    end
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Add `include("test_cna.jl")` to `test/runtests.jl`, then run:
`julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL with `UndefVarError: SegmentalCNA not defined`.

- [ ] **Step 3: Write `src/cna.jl`**

```julia
"""
    CNAEvent

Abstract supertype of copy-number-altering events. Concrete subtypes are
[`SegmentalCNA`](@ref) and [`WholeGenomeDoubling`](@ref).

All state changes go through the single method `apply!(profile, event)`. Keeping the
interface this narrow is what allows a karyotype backend to be added later without
touching tree traversal or the output layer.
"""
abstract type CNAEvent end

const EVENT_SCALES = (:focal, :arm, :chromosome)
const WGD_MODES = (:multiply, :increment)

"""
    SegmentalCNA(chrom, haplotype, start, stop, delta, scale)

A gain or loss of `delta` copies over `start:stop` on one haplotype of one chromosome.

Coordinates are 1-based inclusive. `delta` is positive for a gain and negative for a
loss; a loss that drives the copy number to 0 *is* loss of heterozygosity and needs no
separate event type. `scale` records which class of draw produced the event —
`:focal`, `:arm` or `:chromosome` — so events can be tallied by class afterwards; it
carries no semantics of its own, since an arm-level and a focal event differ only in
their extent.
"""
struct SegmentalCNA <: CNAEvent
    chrom::Int
    haplotype::Int
    start::Int
    stop::Int
    delta::Int
    scale::Symbol

    function SegmentalCNA(chrom::Integer, haplotype::Integer, start::Integer,
                          stop::Integer, delta::Integer, scale::Symbol)
        chrom >= 1 || throw(ArgumentError("chrom must be ≥ 1, got $chrom"))
        haplotype >= 1 || throw(ArgumentError("haplotype must be ≥ 1, got $haplotype"))
        start >= 1 || throw(ArgumentError("start must be ≥ 1, got $start"))
        start <= stop || throw(ArgumentError("start $start exceeds stop $stop"))
        delta != 0 || throw(ArgumentError("delta must be non-zero; a zero-delta event is not an event"))
        scale in EVENT_SCALES ||
            throw(ArgumentError("scale must be one of $EVENT_SCALES, got :$scale"))
        new(Int(chrom), Int(haplotype), Int(start), Int(stop), Int(delta), scale)
    end
end

"""
    event_span(event) -> Int

Number of base pairs a [`SegmentalCNA`](@ref) covers.
"""
event_span(e::SegmentalCNA) = e.stop - e.start + 1

Base.show(io::IO, e::SegmentalCNA) = print(io,
    e.delta > 0 ? "gain" : "loss", "(chrom=", e.chrom, ", hap=", e.haplotype,
    ", ", e.start, ":", e.stop, ", Δ=", e.delta, ", ", e.scale, ")")

"""
    WholeGenomeDoubling(mode = :multiply)

A doubling of the entire genome, crossing chromosome boundaries.

`mode` decides the arithmetic and matters as soon as any segment is already at copy
number 2 or more:

- `:multiply` — every copy number is doubled (`cn → 2cn`). This is what
  tetraploidization means, and it preserves zeros for free. **The default.**
- `:increment` — every *non-zero* copy number gains one (`cn → cn + 1`). This is
  MEDICC2's own definition of a WGD event, provided so that profiles can be generated
  on MEDICC2's terms.

The two coincide while all copy numbers are 0 or 1.
"""
struct WholeGenomeDoubling <: CNAEvent
    mode::Symbol

    function WholeGenomeDoubling(mode::Symbol = :multiply)
        mode in WGD_MODES ||
            throw(ArgumentError("WGD mode must be one of $WGD_MODES, got :$mode"))
        new(mode)
    end
end

Base.show(io::IO, e::WholeGenomeDoubling) = print(io, "WGD(:", e.mode, ")")

# Split the segment containing `pos` so that a segment starts exactly at `pos`.
# No-op when `pos` is already a boundary or lies outside the segmentation.
function _split_at!(segs::Vector{Segment}, pos::Int)
    pos <= segs[1].start && return segs
    pos > segs[end].stop && return segs
    i = segment_index(segs, pos)
    sg = segs[i]
    sg.start == pos && return segs
    segs[i] = Segment(sg.start, pos - 1, sg.cn)
    insert!(segs, i + 1, Segment(pos, sg.stop, sg.cn))
    return segs
end

# Zero is absorbing: absent DNA can never be re-gained, and a loss never goes below
# zero. See the manual's "Modelling choices" page.
_shift_cn(cn::Int, delta::Int) = cn == 0 ? 0 : max(0, cn + delta)

"""
    apply!(profile, event) -> profile

Apply `event` to `profile` in place, restoring canonical form.

For a [`SegmentalCNA`](@ref): split at the two breakpoints, shift the copy number of
every covered segment by `delta`, then re-canonicalise. **Zero-copy state is
absorbing** — a gain never raises a segment at copy number 0, because that DNA is
physically absent — and losses clamp at 0.

For a [`WholeGenomeDoubling`](@ref): shift every segment of every slot per the event's
`mode`, ignoring chromosome boundaries.
"""
function apply!(p::CNProfile, e::SegmentalCNA)
    a = p.assembly
    L = chromlength(a, e.chrom)
    e.stop <= L || throw(ArgumentError(
        "event spans $(e.start):$(e.stop) but chromosome $(chromname(a, e.chrom)) is only $L bp"))
    segs = p.segments[slot(a, e.chrom, e.haplotype)]
    _split_at!(segs, e.start)
    _split_at!(segs, e.stop + 1)
    for i in eachindex(segs)
        sg = segs[i]
        if sg.start >= e.start && sg.stop <= e.stop
            segs[i] = Segment(sg.start, sg.stop, _shift_cn(sg.cn, e.delta))
        end
    end
    canonicalize!(segs)
    return p
end

function apply!(p::CNProfile, e::WholeGenomeDoubling)
    for segs in p.segments
        for i in eachindex(segs)
            sg = segs[i]
            newcn = e.mode === :multiply ? 2 * sg.cn : _shift_cn(sg.cn, 1)
            segs[i] = Segment(sg.start, sg.stop, newcn)
        end
        canonicalize!(segs)
    end
    return p
end
```

- [ ] **Step 4: Add the exports**

```julia
export
    # Events
    CNAEvent, SegmentalCNA, WholeGenomeDoubling, apply!, event_span
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cna.jl src/CopyNumberEvolution.jl test/test_cna.jl test/runtests.jl
git commit -m "feat: add CNA event types and apply! with absorbing zero state"
```

---

### Task 5: `PhyloTree` and tree helpers

Implements spec §4.1.

**Files:**
- Create: `src/tree.jl`
- Create: `test/test_tree.jl`
- Modify: `src/CopyNumberEvolution.jl` (exports)
- Modify: `test/runtests.jl` (include)

**Interfaces:**
- Consumes: nothing.
- Produces: `PhyloNode`; `PhyloTree(nodes::Vector{PhyloNode})`;
  `phylotree(parents; birthtimes, edge_divisions, edge_mutations, labels, source_ids)::PhyloTree`;
  `nnodes(t)::Int`, `treeroot(t)::Int`, `leaves(t)::Vector{Int}`,
  `internal_nodes(t)::Vector{Int}`, `node(t,i)::PhyloNode`,
  `parentof(t,i)::Union{Int,Nothing}`, `childrenof(t,i)::Vector{Int}`,
  `isleaf(t,i)::Bool`, `isroot(t,i)::Bool`, `depth(t,i)::Int`,
  `preorder(t)::Vector{Int}`, `postorder(t)::Vector{Int}`,
  `ancestors(t,i)::Vector{Int}`, `descendant_leaves(t,i)::Vector{Int}`,
  `edge_time(t,i)::Float64`, `mrca(t,ids)::Int`,
  `node_by_source_id(t,sid)::Int`, `node_by_label(t,label)::Int`,
  `cellname(t,i; prefix="cell")::String`.

Note: the accessors are named `treeroot`, `parentof`, `childrenof` rather than
`root`, `parent`, `children` to avoid clashing with `Base.parent` and with common
names in consumer packages.

- [ ] **Step 1: Write the failing tests**

`test/test_tree.jl`:

```julia
@testset "tree" begin
    # Fixture shape (ids in parentheses):
    #            1
    #          /   \
    #         2     3
    #        / \   / \
    #       4   5 6   7
    balanced() = phylotree([nothing, 1, 1, 2, 2, 3, 3];
                           birthtimes = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                           edge_divisions = [nothing, 1, 1, 1, 1, 1, 1],
                           edge_mutations = [nothing, 4, 7, 2, 0, 5, 3],
                           labels = [nothing, nothing, nothing, "a", "b", "c", "d"],
                           source_ids = [10, 20, 30, 40, 50, 60, 70])

    @testset "construction and basic accessors" begin
        t = balanced()
        @test nnodes(t) == 7
        @test treeroot(t) == 1
        @test leaves(t) == [4, 5, 6, 7]
        @test internal_nodes(t) == [1, 2, 3]
        @test childrenof(t, 1) == [2, 3]
        @test parentof(t, 4) == 2
        @test parentof(t, 1) === nothing
        @test isroot(t, 1) && !isroot(t, 2)
        @test isleaf(t, 4) && !isleaf(t, 2)
        @test depth(t, 1) == 0 && depth(t, 4) == 2
    end

    @testset "traversal orders" begin
        t = balanced()
        @test preorder(t) == [1, 2, 4, 5, 3, 6, 7]
        @test postorder(t) == [4, 5, 2, 6, 7, 3, 1]
        @test sort(preorder(t)) == collect(1:7)
    end

    @testset "ancestors and descendant leaves" begin
        t = balanced()
        @test ancestors(t, 4) == [1, 2, 4]
        @test ancestors(t, 1) == [1]
        @test descendant_leaves(t, 2) == [4, 5]
        @test descendant_leaves(t, 1) == [4, 5, 6, 7]
        @test descendant_leaves(t, 4) == [4]
    end

    @testset "mrca" begin
        t = balanced()
        @test mrca(t, [4, 5]) == 2
        @test mrca(t, [4, 6]) == 1
        @test mrca(t, [4]) == 4
        @test mrca(t, [4, 4]) == 4
        @test mrca(t, [4, 5, 6, 7]) == 1
        @test mrca(t, [2, 4]) == 2
        @test_throws ArgumentError mrca(t, Int[])
    end

    @testset "edge_time" begin
        t = balanced()
        @test edge_time(t, 2) ≈ 1.0
        @test edge_time(t, 4) ≈ 1.0
        @test edge_time(t, 5) ≈ 1.5
        @test_throws ArgumentError edge_time(t, 1)          # the root has no incoming edge
        u = phylotree([nothing, 1])                          # no birthtimes at all
        @test_throws ArgumentError edge_time(u, 2)
    end

    @testset "identity helpers" begin
        t = balanced()
        @test node_by_source_id(t, 60) == 6
        @test node_by_label(t, "b") == 5
        @test_throws ArgumentError node_by_source_id(t, 999)
        @test_throws ArgumentError node_by_label(t, "zzz")
        @test cellname(t, 4) == "a"
        @test cellname(t, 2) == "cell_2"
        @test cellname(t, 2; prefix = "node") == "node_2"
    end

    @testset "malformed trees are rejected" begin
        @test_throws ArgumentError phylotree([nothing, nothing])       # two roots
        @test_throws ArgumentError phylotree([1, 1])                   # no root
        @test_throws ArgumentError phylotree([nothing, 5])             # parent out of range
        @test_throws ArgumentError phylotree([nothing, 2, 3, 2])       # cycle: 2->3->2 unreachable
        @test_throws ArgumentError phylotree([nothing, 1]; birthtimes = [0.0])  # length mismatch
    end

    @testset "non-binary and unary trees are supported" begin
        # a multifurcation and a unary chain, both of which newick and sampled
        # lineage trees produce
        t = phylotree([nothing, 1, 1, 1])
        @test childrenof(t, 1) == [2, 3, 4]
        @test leaves(t) == [2, 3, 4]
        u = phylotree([nothing, 1, 2, 3])
        @test leaves(u) == [4]
        @test depth(u, 4) == 3
        @test preorder(u) == [1, 2, 3, 4]
    end

    @testset "deep trees do not overflow the stack" begin
        n = 50_000
        parents = Vector{Union{Int,Nothing}}(undef, n)
        parents[1] = nothing
        for i in 2:n
            parents[i] = i - 1
        end
        t = phylotree(parents)
        @test length(preorder(t)) == n
        @test depth(t, n) == n - 1
        @test descendant_leaves(t, 1) == [n]
    end
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Add `include("test_tree.jl")` to `test/runtests.jl`, then run:
`julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL with `UndefVarError: phylotree not defined`.

- [ ] **Step 3: Write `src/tree.jl`**

Every traversal here is iterative. A lineage tree can be a 10⁵-deep caterpillar, so
recursion would overflow the stack.

```julia
"""
    PhyloNode

One node of a [`PhyloTree`](@ref).

The three edge quantities describe the edge from `parent` to this node and are
`nothing` when genuinely unknown, rather than defaulting to a sentinel. A single
newick branch-length field cannot carry both real time and a division count, and this
package needs both, so each rate rule requires exactly one of them and throws a named
error when it is absent.

# Fields
- `id::Int` — dense identifier, equal to this node's index in the tree.
- `parent::Union{Int,Nothing}` — `nothing` for the root.
- `children::Vector{Int}` — arbitrary arity; newick and inferred trees are not
  guaranteed binary, and a pruned lineage tree contains unary nodes.
- `birthtime::Union{Float64,Nothing}` — real time at which this cell was born.
- `edge_divisions::Union{Int,Nothing}` — cell divisions on the incoming edge; `1` for
  a simulated lineage-tree edge.
- `edge_mutations::Union{Int,Nothing}` — mutations acquired on the incoming edge.
- `label::Union{String,Nothing}` — taxon name, e.g. from a newick file.
- `source_id::Union{Int,Nothing}` — identifier in the upstream numbering, e.g. a
  `MutationLoadDynamics.jl` cell id. Preserved so that "the edge into cell *i*" stays
  expressible after upstream leaf sampling has made those ids sparse.
"""
struct PhyloNode
    id::Int
    parent::Union{Int,Nothing}
    children::Vector{Int}
    birthtime::Union{Float64,Nothing}
    edge_divisions::Union{Int,Nothing}
    edge_mutations::Union{Int,Nothing}
    label::Union{String,Nothing}
    source_id::Union{Int,Nothing}
end

"""
    PhyloTree(nodes)

A cell-lineage or phylogenetic tree in flat-vector storage, where `nodes[i].id == i`.

Dense integer ids make the tree cheap to serialise, hash and compare, and give
tree-similarity metrics the integer leaf labels they want. Upstream identifiers
survive on each node's `source_id` (see [`PhyloNode`](@ref)).

Construction validates that ids are dense, that there is exactly one root, that
`parent` and `children` agree, and that every node is reachable from the root.

Prefer [`phylotree`](@ref) to build one from a parent vector.
"""
struct PhyloTree
    nodes::Vector{PhyloNode}
    root::Int
    leaves::Vector{Int}

    function PhyloTree(nodes::Vector{PhyloNode})
        n = length(nodes)
        n >= 1 || throw(ArgumentError("a tree needs at least one node"))
        for (i, nd) in enumerate(nodes)
            nd.id == i || throw(ArgumentError(
                "node at index $i has id $(nd.id); ids must be dense and equal to the index"))
        end
        roots = findall(nd -> nd.parent === nothing, nodes)
        length(roots) == 1 || throw(ArgumentError(
            "expected exactly one root (a node with no parent), found $(length(roots))"))
        for nd in nodes
            for c in nd.children
                1 <= c <= n || throw(ArgumentError("node $(nd.id) lists child $c, which is out of range"))
                nodes[c].parent == nd.id || throw(ArgumentError(
                    "node $c is listed as a child of $(nd.id) but its parent is $(nodes[c].parent)"))
            end
            if nd.parent !== nothing
                1 <= nd.parent <= n || throw(ArgumentError(
                    "node $(nd.id) has parent $(nd.parent), which is out of range"))
                nd.id in nodes[nd.parent].children || throw(ArgumentError(
                    "node $(nd.id) has parent $(nd.parent) but is not among that node's children"))
            end
        end
        # reachability, iteratively
        seen = falses(n)
        stack = [roots[1]]
        while !isempty(stack)
            i = pop!(stack)
            seen[i] && throw(ArgumentError("node $i is reachable twice; the tree contains a cycle"))
            seen[i] = true
            append!(stack, nodes[i].children)
        end
        all(seen) || throw(ArgumentError(
            "nodes $(findall(!, seen)) are not reachable from the root"))
        lv = [nd.id for nd in nodes if isempty(nd.children)]
        return new(nodes, roots[1], lv)
    end
end

"""
    phylotree(parents; birthtimes, edge_divisions, edge_mutations, labels, source_ids)

Build a [`PhyloTree`](@ref) from a parent vector.

`parents[i]` is the parent id of node `i`, with `nothing` (or `0`) marking the root.
Children are ordered ascending by id. Every keyword defaults to all-`nothing` and,
when given, must have one entry per node.

# Examples
```jldoctest
julia> t = phylotree([nothing, 1, 1]; labels = [nothing, "a", "b"]);

julia> leaves(t) == [2, 3]
true

julia> cellname(t, 2)
"a"
```
"""
function phylotree(parents::AbstractVector;
                   birthtimes = nothing,
                   edge_divisions = nothing,
                   edge_mutations = nothing,
                   labels = nothing,
                   source_ids = nothing)
    n = length(parents)
    _checklen(v, name) = v === nothing || length(v) == n ||
        throw(ArgumentError("$name has $(length(v)) entries but there are $n nodes"))
    _checklen(birthtimes, "birthtimes")
    _checklen(edge_divisions, "edge_divisions")
    _checklen(edge_mutations, "edge_mutations")
    _checklen(labels, "labels")
    _checklen(source_ids, "source_ids")
    get_(v, i) = v === nothing ? nothing : v[i]

    par = Vector{Union{Int,Nothing}}(undef, n)
    for i in 1:n
        p = parents[i]
        par[i] = (p === nothing || p == 0) ? nothing : Int(p)
    end
    kids = [Int[] for _ in 1:n]
    for i in 1:n
        p = par[i]
        p === nothing && continue
        1 <= p <= n || throw(ArgumentError("node $i has parent $p, which is out of range 1:$n"))
        p == i && throw(ArgumentError("node $i is its own parent"))
        push!(kids[p], i)
    end
    nodes = [PhyloNode(i, par[i], kids[i],
                       get_(birthtimes, i) === nothing ? nothing : Float64(birthtimes[i]),
                       get_(edge_divisions, i) === nothing ? nothing : Int(edge_divisions[i]),
                       get_(edge_mutations, i) === nothing ? nothing : Int(edge_mutations[i]),
                       get_(labels, i) === nothing ? nothing : String(labels[i]),
                       get_(source_ids, i) === nothing ? nothing : Int(source_ids[i]))
             for i in 1:n]
    return PhyloTree(nodes)
end

"""
    nnodes(tree) -> Int

Total number of nodes.
"""
nnodes(t::PhyloTree) = length(t.nodes)

"""
    treeroot(tree) -> Int

Id of the root. For a tree converted from a sampled lineage tree this is the original
**founder**, not the most recent common ancestor of the sample — see [`mrca`](@ref).
"""
treeroot(t::PhyloTree) = t.root

"""
    leaves(tree) -> Vector{Int}

Ids of the leaves, ascending.
"""
leaves(t::PhyloTree) = t.leaves

"""
    internal_nodes(tree) -> Vector{Int}

Ids of the non-leaf nodes, ascending.
"""
internal_nodes(t::PhyloTree) = [nd.id for nd in t.nodes if !isempty(nd.children)]

"""
    node(tree, i) -> PhyloNode

The node with id `i`.
"""
node(t::PhyloTree, i::Integer) = t.nodes[i]

"""
    parentof(tree, i) -> Union{Int,Nothing}

Parent id of node `i`, or `nothing` if it is the root.
"""
parentof(t::PhyloTree, i::Integer) = t.nodes[i].parent

"""
    childrenof(tree, i) -> Vector{Int}

Child ids of node `i`.
"""
childrenof(t::PhyloTree, i::Integer) = t.nodes[i].children

"""
    isleaf(tree, i) -> Bool
"""
isleaf(t::PhyloTree, i::Integer) = isempty(t.nodes[i].children)

"""
    isroot(tree, i) -> Bool
"""
isroot(t::PhyloTree, i::Integer) = t.nodes[i].parent === nothing

"""
    depth(tree, i) -> Int

Number of edges from the root to node `i`; the root has depth 0.
"""
function depth(t::PhyloTree, i::Integer)
    d = 0
    j = i
    while (p = t.nodes[j].parent) !== nothing
        d += 1
        j = p
    end
    return d
end

"""
    preorder(tree) -> Vector{Int}

Node ids in depth-first preorder: each node before its children, children in stored
order. Iterative, so arbitrarily deep trees are safe.
"""
function preorder(t::PhyloTree)
    out = Vector{Int}(undef, nnodes(t))
    k = 0
    stack = [t.root]
    while !isempty(stack)
        i = pop!(stack)
        k += 1
        out[k] = i
        kids = t.nodes[i].children
        for j in Iterators.reverse(eachindex(kids))
            push!(stack, kids[j])
        end
    end
    return out
end

"""
    postorder(tree) -> Vector{Int}

Node ids in depth-first postorder: each node after all of its children.
"""
postorder(t::PhyloTree) = reverse!(_reverse_preorder(t))

function _reverse_preorder(t::PhyloTree)
    out = Vector{Int}(undef, nnodes(t))
    k = 0
    stack = [t.root]
    while !isempty(stack)
        i = pop!(stack)
        k += 1
        out[k] = i
        append!(stack, t.nodes[i].children)
    end
    return out
end

"""
    ancestors(tree, i) -> Vector{Int}

The path from the root to node `i` inclusive, root first.
"""
function ancestors(t::PhyloTree, i::Integer)
    path = Int[i]
    j = i
    while (p = t.nodes[j].parent) !== nothing
        push!(path, p)
        j = p
    end
    return reverse!(path)
end

"""
    descendant_leaves(tree, i) -> Vector{Int}

Leaf ids at or below node `i`, ascending.
"""
function descendant_leaves(t::PhyloTree, i::Integer)
    out = Int[]
    stack = [Int(i)]
    while !isempty(stack)
        j = pop!(stack)
        if isempty(t.nodes[j].children)
            push!(out, j)
        else
            append!(stack, t.nodes[j].children)
        end
    end
    return sort!(out)
end

"""
    edge_time(tree, i) -> Float64

Real-time length of the edge into node `i`, i.e. `birthtime(i) - birthtime(parent(i))`.

Throws if `i` is the root (it has no incoming edge) or if either birthtime is
unknown — which is the case for a tree read with `branchlength = :divisions` or
`:mutations`.
"""
function edge_time(t::PhyloTree, i::Integer)
    nd = t.nodes[i]
    nd.parent === nothing && throw(ArgumentError(
        "node $i is the root and has no incoming edge, so edge_time is undefined"))
    bt = nd.birthtime
    pbt = t.nodes[nd.parent].birthtime
    (bt === nothing || pbt === nothing) && throw(ArgumentError(
        "edge_time needs birthtimes on node $i and its parent $(nd.parent), but at least one is missing; read the tree with branchlength = :time"))
    return bt - pbt
end

"""
    mrca(tree, ids) -> Int

Most recent common ancestor of `ids`. A node is its own ancestor, so
`mrca(tree, [i])` is `i` and `mrca(tree, [parent, child])` is `parent`.

Useful for naming an edge without knowing its id, e.g. placing a subclonal
whole-genome doubling at `mrca(tree, metastatic_leaves)`.
"""
function mrca(t::PhyloTree, ids::AbstractVector{<:Integer})
    isempty(ids) && throw(ArgumentError("the most recent common ancestor of an empty set is undefined"))
    path = ancestors(t, first(ids))
    for x in Iterators.drop(ids, 1)
        other = Set(ancestors(t, x))
        k = findlast(in(other), path)
        k === nothing && throw(ArgumentError(
            "nodes $(first(ids)) and $x have no common ancestor; is this really one tree?"))
        path = path[1:k]
    end
    return last(path)
end

"""
    node_by_source_id(tree, sid) -> Int

Dense id of the node whose `source_id` is `sid` — the upstream numbering, e.g. a
`MutationLoadDynamics.jl` cell id.
"""
function node_by_source_id(t::PhyloTree, sid::Integer)
    for nd in t.nodes
        nd.source_id == sid && return nd.id
    end
    throw(ArgumentError("no node with source_id $sid"))
end

"""
    node_by_label(tree, label) -> Int

Dense id of the node whose `label` is `label`.
"""
function node_by_label(t::PhyloTree, label::AbstractString)
    for nd in t.nodes
        nd.label == label && return nd.id
    end
    throw(ArgumentError("no node labelled $label"))
end

"""
    cellname(tree, i; prefix = "cell") -> String

Stable output name for node `i`: its `label` if it has one, otherwise
`"\$(prefix)_\$(i)"`.

Newick writing and MEDICC2 export both go through this, so a cell's `sample_id` in an
exported matrix always matches its leaf label in the exported tree.
"""
cellname(t::PhyloTree, i::Integer; prefix::AbstractString = "cell") =
    something(t.nodes[i].label, "$(prefix)_$(i)")

Base.show(io::IO, t::PhyloTree) = print(io, "PhyloTree(", nnodes(t), " nodes, ",
                                        length(t.leaves), " leaves, root ", t.root, ")")
```

- [ ] **Step 4: Add the exports**

```julia
export
    # Trees
    PhyloNode, PhyloTree, phylotree,
    nnodes, treeroot, leaves, internal_nodes, node, parentof, childrenof,
    isleaf, isroot, depth, preorder, postorder, ancestors, descendant_leaves,
    edge_time, mrca, node_by_source_id, node_by_label, cellname
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/tree.jl src/CopyNumberEvolution.jl test/test_tree.jl test/runtests.jl
git commit -m "feat: add PhyloTree with dense ids and iterative traversals"
```

---

### Task 6: Newick read and write

Implements spec §4.2.

**Files:**
- Create: `src/newick.jl`
- Create: `test/test_newick.jl`
- Modify: `src/CopyNumberEvolution.jl` (exports)
- Modify: `test/runtests.jl` (include)

**Interfaces:**
- Consumes: `PhyloNode`, `PhyloTree`, `preorder`, `childrenof`, `edge_time`,
  `cellname`, `node` (Task 5).
- Produces: `parse_newick(str::AbstractString; branchlength::Symbol)::PhyloTree`;
  `read_newick(path_or_io; branchlength::Symbol)::PhyloTree`;
  `write_newick(io_or_path, tree; branchlength::Symbol, labels::Symbol=:label)`;
  `newick_string(tree; branchlength::Symbol, labels::Symbol=:label)::String`.

- [ ] **Step 1: Write the failing tests**

`test/test_newick.jl`:

```julia
@testset "newick" begin
    @testset "branchlength must be named explicitly" begin
        @test_throws ArgumentError parse_newick("(A:1,B:2);")
        @test_throws ArgumentError parse_newick("(A:1,B:2);"; branchlength = :bogus)
    end

    @testset ":time accumulates birthtimes and sets one division per edge" begin
        t = parse_newick("((A:1.0,B:2.0)X:0.5,C:3.0)R;"; branchlength = :time)
        @test nnodes(t) == 6
        @test node_by_label(t, "R") == treeroot(t)
        @test node(t, treeroot(t)).birthtime == 0.0
        x = node_by_label(t, "X")
        @test node(t, x).birthtime ≈ 0.5
        @test node(t, node_by_label(t, "A")).birthtime ≈ 1.5
        @test node(t, node_by_label(t, "C")).birthtime ≈ 3.0
        @test edge_time(t, node_by_label(t, "B")) ≈ 2.0
        @test all(node(t, i).edge_divisions == 1 for i in 1:nnodes(t) if !isroot(t, i))
        @test node(t, treeroot(t)).edge_divisions === nothing
        @test node(t, node_by_label(t, "A")).edge_mutations === nothing
    end

    @testset ":divisions and :mutations fill their own field only" begin
        d = parse_newick("(A:3,B:5)R;"; branchlength = :divisions)
        @test node(d, node_by_label(d, "A")).edge_divisions == 3
        @test node(d, node_by_label(d, "A")).birthtime === nothing
        @test node(d, node_by_label(d, "A")).edge_mutations === nothing

        m = parse_newick("(A:3,B:5)R;"; branchlength = :mutations)
        @test node(m, node_by_label(m, "B")).edge_mutations == 5
        @test node(m, node_by_label(m, "B")).edge_divisions === nothing
        @test node(m, node_by_label(m, "B")).birthtime === nothing
    end

    @testset "structural variety" begin
        # unnamed internal nodes
        t1 = parse_newick("((A:1,B:1):1,C:2);"; branchlength = :time)
        @test length(leaves(t1)) == 3
        # multifurcation
        t2 = parse_newick("(A:1,B:1,C:1,D:1)R;"; branchlength = :time)
        @test length(childrenof(t2, treeroot(t2))) == 4
        # unary node, as a pruned lineage tree produces
        t3 = parse_newick("((A:1):2)R;"; branchlength = :time)
        @test leaves(t3) == [node_by_label(t3, "A")]
        @test depth(t3, node_by_label(t3, "A")) == 2
        # a single leaf
        t4 = parse_newick("A;"; branchlength = :divisions)
        @test nnodes(t4) == 1 && leaves(t4) == [1]
        # quoted label containing a comma and a colon
        t5 = parse_newick("('cell,1:x':1,B:1)R;"; branchlength = :time)
        @test node_by_label(t5, "cell,1:x") isa Int
        # comments are skipped
        t6 = parse_newick("(A:1[a comment],B:1)R;"; branchlength = :time)
        @test length(leaves(t6)) == 2
        # whitespace and newlines
        t7 = parse_newick("(\n  A:1 ,\n  B:1\n) R ;"; branchlength = :time)
        @test node_by_label(t7, "R") == treeroot(t7)
    end

    @testset "missing branch lengths" begin
        d = parse_newick("(A,B)R;"; branchlength = :divisions)
        @test node(d, node_by_label(d, "A")).edge_divisions === nothing
        # under :time a missing length makes that node's birthtime, and every
        # birthtime below it, unknown rather than silently zero
        t = parse_newick("((A:1,B:1)X,C:1)R;"; branchlength = :time)
        @test node(t, node_by_label(t, "X")).birthtime === nothing
        @test node(t, node_by_label(t, "A")).birthtime === nothing
        @test node(t, node_by_label(t, "C")).birthtime ≈ 1.0
    end

    @testset "malformed input is rejected" begin
        @test_throws ArgumentError parse_newick("(A:1,B:1)"; branchlength = :time)      # no semicolon
        @test_throws ArgumentError parse_newick("(A:1,B:1;"; branchlength = :time)      # unbalanced
        @test_throws ArgumentError parse_newick("(A:1,B:1));"; branchlength = :time)    # trailing paren
        @test_throws ArgumentError parse_newick("();"; branchlength = :time)            # empty branchset
        @test_throws ArgumentError parse_newick("(A:-1,B:1);"; branchlength = :time)    # negative length
        @test_throws ArgumentError parse_newick(";"; branchlength = :time)              # empty tree
    end

    @testset "fractional lengths warn under :divisions and :mutations" begin
        t = @test_logs (:warn, r"fractional") parse_newick("(A:2.4,B:1.0)R;"; branchlength = :divisions)
        @test node(t, node_by_label(t, "A")).edge_divisions == 2
    end

    @testset "round-trip in all three modes" begin
        base = phylotree([nothing, 1, 1, 2, 2];
                         birthtimes = [0.0, 1.25, 2.5, 3.0, 4.75],
                         edge_divisions = [nothing, 2, 3, 1, 4],
                         edge_mutations = [nothing, 7, 0, 5, 11],
                         labels = ["R", "X", "c", "a", "b"])
        for mode in (:time, :divisions, :mutations)
            s = newick_string(base; branchlength = mode)
            rt = parse_newick(s; branchlength = mode)
            @test nnodes(rt) == nnodes(base)
            @test [cellname(rt, i) for i in preorder(rt)] ==
                  [cellname(base, i) for i in preorder(base)]
            for i in preorder(rt)
                isroot(rt, i) && continue
                j = node_by_label(base, cellname(rt, i))
                if mode === :time
                    @test edge_time(rt, i) ≈ edge_time(base, j)
                elseif mode === :divisions
                    @test node(rt, i).edge_divisions == node(base, j).edge_divisions
                else
                    @test node(rt, i).edge_mutations == node(base, j).edge_mutations
                end
            end
        end
    end

    @testset "writing requires the field it emits" begin
        t = phylotree([nothing, 1, 1]; labels = ["R", "a", "b"])
        @test_throws ArgumentError newick_string(t; branchlength = :time)
        @test_throws ArgumentError newick_string(t; branchlength = :divisions)
    end

    @testset "labels keyword selects the emitted name" begin
        t = phylotree([nothing, 1, 1];
                      edge_divisions = [nothing, 1, 1],
                      labels = [nothing, "a", nothing],
                      source_ids = [10, 20, 30])
        @test occursin("a", newick_string(t; branchlength = :divisions, labels = :label))
        @test occursin("cell_3", newick_string(t; branchlength = :divisions, labels = :label))
        @test occursin("cell_20", newick_string(t; branchlength = :divisions, labels = :source_id))
        @test occursin("cell_2", newick_string(t; branchlength = :divisions, labels = :id))
        @test_throws ArgumentError newick_string(t; branchlength = :divisions, labels = :bogus)
    end

    @testset "file round-trip" begin
        t = phylotree([nothing, 1, 1]; edge_divisions = [nothing, 2, 2], labels = ["R", "a", "b"])
        path = joinpath(mktempdir(), "tree.nwk")
        write_newick(path, t; branchlength = :divisions)
        rt = read_newick(path; branchlength = :divisions)
        @test [cellname(rt, i) for i in preorder(rt)] == ["R", "a", "b"]
    end

    @testset "labels needing quotes are quoted on write" begin
        t = phylotree([nothing, 1, 1]; edge_divisions = [nothing, 1, 1],
                      labels = ["R", "has,comma", "plain"])
        s = newick_string(t; branchlength = :divisions)
        @test occursin("'has,comma'", s)
        rt = parse_newick(s; branchlength = :divisions)
        @test node_by_label(rt, "has,comma") isa Int
    end
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Add `include("test_newick.jl")` to `test/runtests.jl`, then run:
`julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL with `UndefVarError: parse_newick not defined`.

- [ ] **Step 3: Write `src/newick.jl`**

```julia
const BRANCHLENGTH_MODES = (:divisions, :mutations, :time)

# Mutable staging node used only during parsing.
mutable struct _RawNode
    parent::Union{Int,Nothing}
    children::Vector{Int}
    label::Union{String,Nothing}
    brlen::Union{Float64,Nothing}
end

"""
    parse_newick(str; branchlength) -> PhyloTree

Parse a newick string.

A newick file carries **one** number per edge, but this package needs both real time
and a division count, so you must say which the numbers are. There is deliberately no
default: an implicit convention is the kind of thing that becomes invisible and wrong.

| `branchlength` | field populated | rate rules it enables |
|:---|:---|:---|
| `:divisions` | `edge_divisions` | [`PerDivision`](@ref) |
| `:mutations` | `edge_mutations` | [`FromEdgeMutations`](@ref) |
| `:time` | `birthtime`, by cumulative sum from the root, plus `edge_divisions = 1` | [`PerTime`](@ref), and [`PerDivision`](@ref) treating each edge as one division |

Fields not implied by the chosen mode stay `nothing`, and each rate rule throws a
named error when the field it needs is absent.

Supported syntax: named and unnamed internal nodes, arbitrary arity, single-quoted
labels (with `''` for a literal quote), `[...]` comments, missing branch lengths, and
arbitrary whitespace. Under `:divisions` and `:mutations` a fractional branch length
is rounded and warned about, since a fractional count is usually a sign the file
should have been read as `:time`.

# Examples
```jldoctest
julia> t = parse_newick("((A:1.0,B:2.0)X:0.5,C:3.0)R;"; branchlength = :time);

julia> node(t, node_by_label(t, "A")).birthtime
1.5
```
"""
function parse_newick(str::AbstractString; branchlength::Symbol = :unset)
    branchlength in BRANCHLENGTH_MODES || throw(ArgumentError(
        "branchlength must be one of $BRANCHLENGTH_MODES; got :$branchlength. " *
        "There is no default on purpose — say whether the numbers are divisions, mutations or time."))
    raw = _parse_raw(str)
    return _raw_to_tree(raw, branchlength)
end

"""
    read_newick(path_or_io; branchlength) -> PhyloTree

Read a newick tree from a file path or an `IO`. See [`parse_newick`](@ref) for the
meaning of `branchlength`.
"""
read_newick(io::IO; branchlength::Symbol = :unset) =
    parse_newick(read(io, String); branchlength = branchlength)
read_newick(path::AbstractString; branchlength::Symbol = :unset) =
    open(io -> read_newick(io; branchlength = branchlength), path)

function _parse_raw(str::AbstractString)
    s = collect(str)
    n = length(s)
    nodes = _RawNode[]
    pos = 1

    function skipspace!()
        while pos <= n
            c = s[pos]
            if isspace(c)
                pos += 1
            elseif c == '['
                depth = 0
                while pos <= n
                    s[pos] == '[' && (depth += 1)
                    s[pos] == ']' && (depth -= 1)
                    pos += 1
                    depth == 0 && break
                end
                depth == 0 || throw(ArgumentError("unterminated comment in newick input"))
            else
                return
            end
        end
    end

    function readlabel!()
        skipspace!()
        pos > n && return nothing
        if s[pos] == '\''
            pos += 1
            buf = Char[]
            while true
                pos <= n || throw(ArgumentError("unterminated quoted label in newick input"))
                if s[pos] == '\''
                    if pos + 1 <= n && s[pos + 1] == '\''
                        push!(buf, '\'')
                        pos += 2
                    else
                        pos += 1
                        break
                    end
                else
                    push!(buf, s[pos])
                    pos += 1
                end
            end
            return String(buf)
        end
        start = pos
        while pos <= n && !(s[pos] in ('(', ')', ',', ':', ';', '[')) && !isspace(s[pos])
            pos += 1
        end
        return pos > start ? String(s[start:pos - 1]) : nothing
    end

    function readlength!()
        skipspace!()
        (pos <= n && s[pos] == ':') || return nothing
        pos += 1
        skipspace!()
        start = pos
        while pos <= n && (isdigit(s[pos]) || s[pos] in ('.', '-', '+', 'e', 'E'))
            pos += 1
        end
        pos > start || throw(ArgumentError("expected a number after ':' in newick input"))
        txt = String(s[start:pos - 1])
        val = tryparse(Float64, txt)
        val === nothing && throw(ArgumentError("could not parse branch length '$txt'"))
        val >= 0 || throw(ArgumentError("negative branch length $val is not allowed"))
        return val
    end

    function parsesubtree!(parent::Union{Int,Nothing})
        skipspace!()
        pos <= n || throw(ArgumentError("unexpected end of newick input"))
        push!(nodes, _RawNode(parent, Int[], nothing, nothing))
        me = length(nodes)
        parent === nothing || push!(nodes[parent].children, me)
        if s[pos] == '('
            pos += 1
            while true
                parsesubtree!(me)
                skipspace!()
                pos <= n || throw(ArgumentError("unbalanced parentheses in newick input"))
                if s[pos] == ','
                    pos += 1
                elseif s[pos] == ')'
                    pos += 1
                    break
                else
                    throw(ArgumentError("expected ',' or ')' in newick input, found '$(s[pos])'"))
                end
            end
            isempty(nodes[me].children) && throw(ArgumentError("empty branch set '()' in newick input"))
        end
        nodes[me].label = readlabel!()
        nodes[me].brlen = readlength!()
        return me
    end

    skipspace!()
    pos <= n || throw(ArgumentError("empty newick input"))
    s[pos] == ';' && throw(ArgumentError("empty newick tree ';' has no root node"))
    parsesubtree!(nothing)
    skipspace!()
    (pos <= n && s[pos] == ';') ||
        throw(ArgumentError("newick input must end with ';'" *
            (pos <= n ? " but continues with '$(s[pos])'" : "")))
    pos += 1
    skipspace!()
    pos > n || throw(ArgumentError("trailing content after ';' in newick input"))
    return nodes
end

function _raw_to_tree(raw::Vector{_RawNode}, mode::Symbol)
    n = length(raw)
    order = _raw_preorder(raw)
    birthtimes = Vector{Union{Float64,Nothing}}(nothing, n)
    divisions = Vector{Union{Int,Nothing}}(nothing, n)
    mutations = Vector{Union{Int,Nothing}}(nothing, n)
    fractional = 0

    for i in order
        nd = raw[i]
        if mode === :time
            if nd.parent === nothing
                birthtimes[i] = 0.0
            else
                pbt = birthtimes[nd.parent]
                birthtimes[i] = (pbt === nothing || nd.brlen === nothing) ? nothing : pbt + nd.brlen
                divisions[i] = 1
            end
        elseif nd.parent !== nothing && nd.brlen !== nothing
            v = nd.brlen
            abs(v - round(v)) > 1e-9 && (fractional += 1)
            iv = round(Int, v)
            mode === :divisions ? (divisions[i] = iv) : (mutations[i] = iv)
        end
    end

    fractional > 0 && @warn "newick branch lengths under :$mode should be whole numbers; $fractional fractional value(s) were rounded. If these are real times, read the file with branchlength = :time."

    nodes = [PhyloNode(i, raw[i].parent, raw[i].children,
                       birthtimes[i], divisions[i], mutations[i], raw[i].label, nothing)
             for i in 1:n]
    return PhyloTree(nodes)
end

function _raw_preorder(raw::Vector{_RawNode})
    r = findfirst(nd -> nd.parent === nothing, raw)
    out = Int[]
    stack = [r]
    while !isempty(stack)
        i = pop!(stack)
        push!(out, i)
        append!(stack, raw[i].children)
    end
    return out
end

const _NEWICK_SPECIAL = ('(', ')', ',', ':', ';', '[', ']', '\'', ' ')

_quote_label(name::AbstractString) =
    any(c -> c in _NEWICK_SPECIAL, name) ? "'" * replace(name, "'" => "''") * "'" : name

function _emit_name(t::PhyloTree, i::Integer, labels::Symbol)
    labels === :label && return cellname(t, i)
    labels === :id && return "cell_$(i)"
    labels === :source_id && begin
        sid = node(t, i).source_id
        sid === nothing && throw(ArgumentError("node $i has no source_id, so labels = :source_id cannot name it"))
        return "cell_$(sid)"
    end
    throw(ArgumentError("labels must be :label, :id or :source_id; got :$labels"))
end

function _emit_length(t::PhyloTree, i::Integer, mode::Symbol)
    if mode === :time
        return edge_time(t, i)
    elseif mode === :divisions
        v = node(t, i).edge_divisions
        v === nothing && throw(ArgumentError(
            "cannot write branchlength = :divisions: node $i has no edge_divisions"))
        return v
    else
        v = node(t, i).edge_mutations
        v === nothing && throw(ArgumentError(
            "cannot write branchlength = :mutations: node $i has no edge_mutations"))
        return v
    end
end

"""
    newick_string(tree; branchlength, labels = :label) -> String

Render `tree` as a newick string.

`branchlength` chooses **which** quantity the single branch-length field carries —
`:time`, `:divisions` or `:mutations` — and throws if any non-root node lacks it. The
choice is explicit on write for the same reason it is on read: newick cannot carry
both real time and a division count, and a silent convention is a bug waiting to
happen.

`labels` picks the emitted names: `:label` uses [`cellname`](@ref) (a node's label, or
`cell_<id>`), `:id` always uses `cell_<dense id>`, and `:source_id` uses
`cell_<upstream id>`.
"""
function newick_string(t::PhyloTree; branchlength::Symbol = :unset, labels::Symbol = :label)
    branchlength in BRANCHLENGTH_MODES || throw(ArgumentError(
        "branchlength must be one of $BRANCHLENGTH_MODES; got :$branchlength"))
    labels in (:label, :id, :source_id) || throw(ArgumentError(
        "labels must be :label, :id or :source_id; got :$labels"))
    io = IOBuffer()
    _write_subtree(io, t, treeroot(t), branchlength, labels)
    print(io, ';')
    return String(take!(io))
end

# Iterative emission: a frame is (node, child index, already-opened flag), so a
# 10^5-deep tree writes without recursion.
function _write_subtree(io::IO, t::PhyloTree, start::Int, mode::Symbol, labels::Symbol)
    stack = Tuple{Int,Int}[(start, 0)]
    while !isempty(stack)
        i, k = pop!(stack)
        kids = childrenof(t, i)
        if k == 0 && !isempty(kids)
            print(io, '(')
            push!(stack, (i, 1))
            push!(stack, (kids[1], 0))
        elseif k > 0 && k < length(kids)
            print(io, ',')
            push!(stack, (i, k + 1))
            push!(stack, (kids[k + 1], 0))
        else
            !isempty(kids) && print(io, ')')
            print(io, _quote_label(_emit_name(t, i, labels)))
            if !isroot(t, i)
                print(io, ':', _emit_length(t, i, mode))
            end
        end
    end
    return io
end

"""
    write_newick(io_or_path, tree; branchlength, labels = :label)

Write `tree` in newick format to an `IO` or a file path. See
[`newick_string`](@ref).
"""
write_newick(io::IO, t::PhyloTree; branchlength::Symbol = :unset, labels::Symbol = :label) =
    print(io, newick_string(t; branchlength = branchlength, labels = labels))

write_newick(path::AbstractString, t::PhyloTree; branchlength::Symbol = :unset,
             labels::Symbol = :label) =
    open(io -> write_newick(io, t; branchlength = branchlength, labels = labels), path, "w")
```

- [ ] **Step 4: Add the exports**

```julia
export
    # Newick
    parse_newick, read_newick, write_newick, newick_string
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/newick.jl src/CopyNumberEvolution.jl test/test_newick.jl test/runtests.jl
git commit -m "feat: add newick reader and writer with explicit branch-length semantics"
```

---

### Task 7: CNA rate rules

Implements spec §7.2. This is where the per-division versus per-time question — the
whole point of the package — becomes a switchable parameter.

**Files:**
- Create: `src/rates.jl`
- Create: `test/test_rates.jl`
- Modify: `src/CopyNumberEvolution.jl` (exports)
- Modify: `test/runtests.jl` (include)

**Interfaces:**
- Consumes: `PhyloTree`, `node`, `edge_time` (Task 5).
- Produces: `abstract type CNARate`; `PerDivision(λ)`, `PerTime(μ)`,
  `FromEdgeMutations([p])`, `CustomRate(f)`;
  `n_cnas(rule::CNARate, tree::PhyloTree, i::Int, rng::AbstractRNG)::Int`.

- [ ] **Step 1: Write the failing tests**

`test/test_rates.jl`:

```julia
@testset "rates" begin
    # Two-edge tree: node 2 has 3 divisions / Δt = 2.0 / 8 mutations,
    #                node 3 has 1 division  / Δt = 0.5 / 0 mutations.
    tree() = phylotree([nothing, 1, 1];
                       birthtimes = [0.0, 2.0, 0.5],
                       edge_divisions = [nothing, 3, 1],
                       edge_mutations = [nothing, 8, 0])

    mean_draws(rule, t, i; n = 40_000, seed = 20260904) = begin
        rng = Random.Xoshiro(seed)
        sum(n_cnas(rule, t, i, rng) for _ in 1:n) / n
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError PerDivision(-1.0)
        @test_throws ArgumentError PerTime(-0.5)
        @test_throws ArgumentError FromEdgeMutations(1.5)
        @test_throws ArgumentError FromEdgeMutations(-0.1)
        @test FromEdgeMutations().p == 1.0
    end

    @testset "PerDivision scales with edge_divisions" begin
        t = tree()
        @test mean_draws(PerDivision(0.5), t, 2) ≈ 1.5 rtol = 0.05
        @test mean_draws(PerDivision(0.5), t, 3) ≈ 0.5 rtol = 0.05
        @test n_cnas(PerDivision(0.0), t, 2, Random.Xoshiro(1)) == 0
    end

    @testset "PerTime scales with elapsed real time" begin
        t = tree()
        @test mean_draws(PerTime(1.5), t, 2) ≈ 3.0 rtol = 0.05
        @test mean_draws(PerTime(1.5), t, 3) ≈ 0.75 rtol = 0.05
    end

    @testset "FromEdgeMutations is exactly identity by default" begin
        t = tree()
        rng = Random.Xoshiro(7)
        @test all(n_cnas(FromEdgeMutations(), t, 2, rng) == 8 for _ in 1:50)
        @test all(n_cnas(FromEdgeMutations(), t, 3, rng) == 0 for _ in 1:50)
    end

    @testset "FromEdgeMutations thins binomially when p < 1" begin
        t = tree()
        @test mean_draws(FromEdgeMutations(0.25), t, 2) ≈ 2.0 rtol = 0.05
        @test n_cnas(FromEdgeMutations(0.25), t, 3, Random.Xoshiro(1)) == 0
    end

    @testset "each rule names the field it is missing" begin
        divs_only = phylotree([nothing, 1]; edge_divisions = [nothing, 2])
        rng = Random.Xoshiro(1)
        @test n_cnas(PerDivision(1.0), divs_only, 2, rng) isa Int
        @test_throws ArgumentError n_cnas(PerTime(1.0), divs_only, 2, rng)
        @test_throws ArgumentError n_cnas(FromEdgeMutations(), divs_only, 2, rng)

        time_only = phylotree([nothing, 1]; birthtimes = [0.0, 1.0])
        @test_throws ArgumentError n_cnas(PerDivision(1.0), time_only, 2, rng)
        @test n_cnas(PerTime(1.0), time_only, 2, rng) isa Int
    end

    @testset "the error message says how to fix it" begin
        t = phylotree([nothing, 1]; edge_divisions = [nothing, 2])
        err = try
            n_cnas(PerTime(1.0), t, 2, Random.Xoshiro(1))
        catch e
            e
        end
        @test occursin("branchlength", err.msg)
    end

    @testset "CustomRate wraps a function" begin
        t = tree()
        r = CustomRate((tr, i, rng) -> 2 * something(node(tr, i).edge_divisions, 0))
        @test n_cnas(r, t, 2, Random.Xoshiro(1)) == 6
        @test n_cnas(r, t, 3, Random.Xoshiro(1)) == 2
    end

    @testset "a negative edge time is a corrupt tree, not a rate of zero" begin
        bad = phylotree([nothing, 1]; birthtimes = [1.0, 0.0])
        @test_throws ArgumentError n_cnas(PerTime(1.0), bad, 2, Random.Xoshiro(1))
    end
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Add `include("test_rates.jl")` to `test/runtests.jl`, then run:
`julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL with `UndefVarError: PerDivision not defined`.

- [ ] **Step 3: Write `src/rates.jl`**

```julia
"""
    CNARate

How many copy-number alterations fall on one edge of the tree.

Whether CNAs accrue **per division** or **per unit real time** is the
Markov-versus-non-Markov question transposed from point mutations to copy number.
Under exponential division timing the two are hard to tell apart; under
non-exponential timing they are not, because division count and elapsed time
decouple. The input tree carries both, so both are computable — and the difference
between them is the signal, not a nuisance. Supporting both is the point of this
package.

Implement a new rule by adding a method to
`n_cnas(rule, tree, node, rng) -> Int`.

Concrete rules: [`PerDivision`](@ref), [`PerTime`](@ref),
[`FromEdgeMutations`](@ref), [`CustomRate`](@ref).
"""
abstract type CNARate end

"""
    PerDivision(λ)

`Poisson(λ · edge_divisions)` alterations on each edge. **The default rate rule.**

Requires `edge_divisions`, i.e. a tree converted from `MutationLoadDynamics.jl` or
read with `branchlength = :divisions` or `:time`. On a simulated lineage tree every
edge is exactly one division, so this reduces to `Poisson(λ)` — but a newick edge can
represent many divisions, which is why the rate multiplies rather than being applied
per edge.
"""
struct PerDivision <: CNARate
    λ::Float64
    function PerDivision(λ::Real)
        λ >= 0 || throw(ArgumentError("PerDivision rate λ must be non-negative, got $λ"))
        new(Float64(λ))
    end
end

"""
    PerTime(μ)

`Poisson(μ · Δt)` alterations on each edge, where `Δt` is the edge's elapsed real
time. Requires birthtimes, i.e. a tree from `MutationLoadDynamics.jl` or read with
`branchlength = :time`.
"""
struct PerTime <: CNARate
    μ::Float64
    function PerTime(μ::Real)
        μ >= 0 || throw(ArgumentError("PerTime rate μ must be non-negative, got $μ"))
        new(Float64(μ))
    end
end

"""
    FromEdgeMutations(p = 1.0)

Take the edge's mutation count as the alteration count.

With the default `p = 1.0` this is **exact identity**: one recorded mutation becomes
one CNA, with no extra randomness. With `p < 1` each mutation independently becomes a
CNA, giving `Binomial(edge_mutations, p)` — which keeps an upstream
fitness-coupled mutation process intact while lowering the realised CNA rate.

Requires `edge_mutations`, i.e. a tree from `MutationLoadDynamics.jl` or read with
`branchlength = :mutations`.
"""
struct FromEdgeMutations <: CNARate
    p::Float64
    function FromEdgeMutations(p::Real = 1.0)
        0 <= p <= 1 || throw(ArgumentError("FromEdgeMutations thinning probability p must lie in [0, 1], got $p"))
        new(Float64(p))
    end
end

"""
    CustomRate(f)

Wrap a user function `f(tree, node, rng) -> Int`, for a rate rule this package does
not provide.
"""
struct CustomRate{F} <: CNARate
    f::F
end

"""
    n_cnas(rule, tree, i, rng) -> Int

Number of copy-number alterations to draw on the edge into node `i`.

Throws an `ArgumentError` naming the missing field, and how to obtain it, when the
tree does not carry what `rule` needs.
"""
function n_cnas(r::PerDivision, t::PhyloTree, i::Integer, rng::Random.AbstractRNG)
    d = node(t, i).edge_divisions
    d === nothing && throw(ArgumentError(
        "PerDivision needs edge_divisions, which is missing on node $i. " *
        "Convert a MutationLoadDynamics tree, or read the newick file with branchlength = :divisions or :time."))
    λ = r.λ * d
    λ == 0 && return 0
    return rand(rng, Poisson(λ))
end

function n_cnas(r::PerTime, t::PhyloTree, i::Integer, rng::Random.AbstractRNG)
    Δ = edge_time(t, i)   # throws with a branchlength hint when birthtimes are missing
    Δ >= 0 || throw(ArgumentError(
        "edge into node $i has negative elapsed time $Δ; the tree's birthtimes are inconsistent"))
    μ = r.μ * Δ
    μ == 0 && return 0
    return rand(rng, Poisson(μ))
end

function n_cnas(r::FromEdgeMutations, t::PhyloTree, i::Integer, rng::Random.AbstractRNG)
    m = node(t, i).edge_mutations
    m === nothing && throw(ArgumentError(
        "FromEdgeMutations needs edge_mutations, which is missing on node $i. " *
        "Convert a MutationLoadDynamics tree, or read the newick file with branchlength = :mutations."))
    m == 0 && return 0
    r.p == 1.0 && return m
    return rand(rng, Binomial(m, r.p))
end

n_cnas(r::CustomRate, t::PhyloTree, i::Integer, rng::Random.AbstractRNG) = r.f(t, i, rng)
```

Note: `edge_time` already raises an `ArgumentError` mentioning `branchlength = :time`,
which is what the "error message says how to fix it" test asserts.

- [ ] **Step 4: Add the exports**

```julia
export
    # Rates
    CNARate, PerDivision, PerTime, FromEdgeMutations, CustomRate, n_cnas
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rates.jl src/CopyNumberEvolution.jl test/test_rates.jl test/runtests.jl
git commit -m "feat: add per-division, per-time and per-mutation CNA rate rules"
```

---

### Task 8: Proposal distributions

Implements spec §7.3, §7.4, §7.5. Whole-arm and whole-chromosome events live here.

**Files:**
- Create: `src/proposals.jl`
- Create: `test/test_proposals.jl`
- Modify: `src/CopyNumberEvolution.jl` (exports)
- Modify: `test/runtests.jl` (include)

**Interfaces:**
- Consumes: `GenomeAssembly` accessors (Task 2); `CNProfile`, `mean_cn` (Task 3).
- Produces: `abstract type TargetDraw`; `UniformChromosome()`, `LengthWeighted()`,
  `CNWeighted(β)`; `draw_target(d, p::CNProfile, rng)::Tuple{Int,Int}`;
  `abstract type ExtentDraw`; `ExtentMixture(; p_chromosome, p_arm, lengthdist)`;
  `draw_extent(d, p, chrom, hap, rng)::Tuple{Int,Int,Symbol}`;
  `abstract type KindDraw`; `GainLoss(p_gain[, delta])`;
  `draw_kind(d, p, chrom, hap, start, stop, rng)::Int`.
  Plain functions are accepted wherever a draw is expected.

- [ ] **Step 1: Write the failing tests**

`test/test_proposals.jl`:

```julia
@testset "proposals" begin
    @testset "UniformChromosome covers eligible chromosomes evenly" begin
        p = diploid(toy_sex_assembly(:female))
        a = p.assembly
        rng = Random.Xoshiro(11)
        counts = zeros(Int, nchromosomes(a))
        for _ in 1:60_000
            c, h = draw_target(UniformChromosome(), p, rng)
            counts[c] += 1
            @test 1 <= h <= ploidy(a, c)
        end
        y = chromindex(a, "chrY")
        @test counts[y] == 0                       # zero ploidy is never targeted
        for c in eligible_chromosomes(a)
            @test counts[c] ≈ 60_000 / 3 rtol = 0.05
        end
    end

    @testset "hemizygous chromosomes only ever yield haplotype 1" begin
        p = diploid(toy_sex_assembly(:male))
        a = p.assembly
        rng = Random.Xoshiro(12)
        x = chromindex(a, "chrX")
        seen = Set{Int}()
        for _ in 1:20_000
            c, h = draw_target(UniformChromosome(), p, rng)
            c == x && push!(seen, h)
        end
        @test seen == Set([1])
    end

    @testset "LengthWeighted is proportional to chromosome length" begin
        specs = [CopyNumberEvolution.ChromosomeSpec("chr1", 3000, 1201:1800),
                 CopyNumberEvolution.ChromosomeSpec("chr2", 1000, 401:600)]
        a = CopyNumberEvolution.GenomeAssembly("w", :female, specs, [2, 2])
        p = diploid(a)
        rng = Random.Xoshiro(13)
        n1 = 0
        for _ in 1:40_000
            c, _ = draw_target(LengthWeighted(), p, rng)
            c == 1 && (n1 += 1)
        end
        @test n1 / 40_000 ≈ 0.75 rtol = 0.03
    end

    @testset "CNWeighted conditions on the mother cell's copy-number state" begin
        a = toy_assembly(nchrom = 2, len = 1000)
        p = diploid(a)
        # take chromosome 1 haplotype 1 to copy number 4
        apply!(p, SegmentalCNA(1, 1, 1, 1000, 3, :chromosome))
        rng = Random.Xoshiro(14)
        gained = 0
        target = slot(a, 1, 1)
        for _ in 1:40_000
            c, h = draw_target(CNWeighted(1.0), p, rng)
            slot(a, c, h) == target && (gained += 1)
        end
        # weights are 4 : 1 : 1 : 1, so the gained slot takes 4/7 of the draws
        @test gained / 40_000 ≈ 4 / 7 rtol = 0.04
    end

    @testset "CNWeighted never targets fully deleted material" begin
        a = toy_assembly(nchrom = 2, len = 1000)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 1, 1000, -1, :chromosome))   # slot now entirely 0
        deleted = slot(a, 1, 1)
        rng = Random.Xoshiro(15)
        for _ in 1:20_000
            c, h = draw_target(CNWeighted(1.0), p, rng)
            @test slot(a, c, h) != deleted
        end
        # β = 0 must not resurrect it via 0^0 == 1
        for _ in 1:20_000
            c, h = draw_target(CNWeighted(0.0), p, rng)
            @test slot(a, c, h) != deleted
        end
    end

    @testset "CNWeighted with every slot deleted is an error, not a silent hang" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 1, 100, -1, :chromosome))
        apply!(p, SegmentalCNA(1, 2, 1, 100, -1, :chromosome))
        @test_throws ArgumentError draw_target(CNWeighted(1.0), p, Random.Xoshiro(1))
    end

    @testset "ExtentMixture validation" begin
        @test_throws ArgumentError ExtentMixture(p_chromosome = 0.6, p_arm = 0.6)
        @test_throws ArgumentError ExtentMixture(p_chromosome = -0.1)
        m = ExtentMixture()
        @test m.p_chromosome == 0.0 && m.p_arm == 0.0
    end

    @testset "whole-chromosome events span exactly the chromosome" begin
        a = toy_assembly(nchrom = 2, len = 1000)
        p = diploid(a)
        d = ExtentMixture(p_chromosome = 1.0)
        rng = Random.Xoshiro(16)
        for _ in 1:200
            s, e, sc = draw_extent(d, p, 1, 1, rng)
            @test (s, e, sc) == (1, 1000, :chromosome)
        end
    end

    @testset "whole-arm events respect the centromere" begin
        a = toy_assembly(nchrom = 1, len = 1000)   # centromere 401:600
        p = diploid(a)
        d = ExtentMixture(p_arm = 1.0)
        rng = Random.Xoshiro(17)
        seen = Set{Tuple{Int,Int}}()
        for _ in 1:400
            s, e, sc = draw_extent(d, p, 1, 1, rng)
            @test sc === :arm
            push!(seen, (s, e))
            @test isempty(intersect(s:e, centromere(a, 1)))
        end
        @test seen == Set([(1, 400), (601, 1000)])
    end

    @testset "focal events stay inside the chromosome" begin
        a = toy_assembly(nchrom = 1, len = 1000)
        p = diploid(a)
        d = ExtentMixture(lengthdist = Distributions.LogUniform(10.0, 5000.0))
        rng = Random.Xoshiro(18)
        truncated = 0
        for _ in 1:5_000
            s, e, sc = draw_extent(d, p, 1, 1, rng)
            @test sc === :focal
            @test 1 <= s <= e <= 1000
            e == 1000 && (truncated += 1)
        end
        @test truncated > 0     # truncation at the boundary does happen, per MEDICC2
    end

    @testset "the three extent classes appear at their stated rates" begin
        a = toy_assembly(nchrom = 1, len = 1000)
        p = diploid(a)
        d = ExtentMixture(p_chromosome = 0.2, p_arm = 0.3)
        rng = Random.Xoshiro(19)
        tally = Dict(:chromosome => 0, :arm => 0, :focal => 0)
        for _ in 1:40_000
            _, _, sc = draw_extent(d, p, 1, 1, rng)
            tally[sc] += 1
        end
        @test tally[:chromosome] / 40_000 ≈ 0.2 rtol = 0.05
        @test tally[:arm] / 40_000 ≈ 0.3 rtol = 0.05
        @test tally[:focal] / 40_000 ≈ 0.5 rtol = 0.05
    end

    @testset "GainLoss" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        rng = Random.Xoshiro(20)
        @test all(draw_kind(GainLoss(1.0), p, 1, 1, 1, 10, rng) == 1 for _ in 1:100)
        @test all(draw_kind(GainLoss(0.0), p, 1, 1, 1, 10, rng) == -1 for _ in 1:100)
        @test all(draw_kind(GainLoss(1.0, 3), p, 1, 1, 1, 10, rng) == 3 for _ in 1:100)
        gains = count(_ -> draw_kind(GainLoss(0.7), p, 1, 1, 1, 10, rng) > 0, 1:40_000)
        @test gains / 40_000 ≈ 0.7 rtol = 0.03
        @test_throws ArgumentError GainLoss(1.2)
        @test_throws ArgumentError GainLoss(0.5, 0)
    end

    @testset "plain functions are accepted as draws" begin
        a = toy_assembly(nchrom = 2, len = 100)
        p = diploid(a)
        rng = Random.Xoshiro(21)
        @test draw_target((prof, r) -> (2, 1), p, rng) == (2, 1)
        @test draw_extent((prof, c, h, r) -> (5, 15, :focal), p, 1, 1, rng) == (5, 15, :focal)
        @test draw_kind((prof, c, h, s, e, r) -> -2, p, 1, 1, 5, 15, rng) == -2
    end
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Add `include("test_proposals.jl")` to `test/runtests.jl`, then run:
`julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL with `UndefVarError: UniformChromosome not defined`.

- [ ] **Step 3: Write `src/proposals.jl`**

```julia
# Sample an index from non-negative weights in one pass. Deterministic given `rng`,
# and it never iterates a Dict — see the package's determinism constraint.
function _sample_weighted(rng::Random.AbstractRNG, weights::AbstractVector{Float64})
    total = sum(weights)
    total > 0 || throw(ArgumentError(
        "all proposal weights are zero, so no target can be drawn; " *
        "the profile may have no material left to alter"))
    u = rand(rng) * total
    acc = 0.0
    @inbounds for i in eachindex(weights)
        acc += weights[i]
        u <= acc && return i
    end
    return lastindex(weights)
end

"""
    TargetDraw

Which chromosome and haplotype a copy-number alteration lands on.

A rule is a callable `(profile, rng) -> (chrom, haplotype)`; implement a new one by
adding a method to [`draw_target`](@ref), or just pass a plain function.

Chromosomes with zero ploidy — `chrY` in a female assembly — are never eligible.
Haplotype choice is uniform over a chromosome's slots unless the rule says otherwise,
which is what lets mirrored allelic imbalance arise on its own rather than being
injected.

Concrete rules: [`UniformChromosome`](@ref), [`LengthWeighted`](@ref),
[`CNWeighted`](@ref).
"""
abstract type TargetDraw end

"""
    UniformChromosome()

Uniform over eligible chromosomes, then uniform over that chromosome's haplotypes.
Note this is uniform over *chromosomes*, not over base pairs.
"""
struct UniformChromosome <: TargetDraw end

"""
    LengthWeighted()

Chromosome chosen with probability proportional to its length, then uniform over its
haplotypes — i.e. uniform over base pairs.
"""
struct LengthWeighted <: TargetDraw end

"""
    CNWeighted(β)

Haplotype slot chosen with weight proportional to its mean copy number raised to `β`,
so already-gained material keeps being gained. This is the rule that conditions the
proposal on the mother cell's copy-number state, and it is what produces realistic
ploidy skew.

A slot whose mean copy number is 0 gets weight 0 for every `β`, including `β = 0`, so
fully deleted material is never targeted — consistent with copy number 0 being
absorbing. Drawing from a profile in which every slot is empty throws rather than
looping.
"""
struct CNWeighted <: TargetDraw
    β::Float64
    CNWeighted(β::Real = 1.0) = new(Float64(β))
end

"""
    draw_target(rule, profile, rng) -> (chrom, haplotype)

Choose the chromosome and haplotype for the next alteration.
"""
function draw_target(::UniformChromosome, p::CNProfile, rng::Random.AbstractRNG)
    a = p.assembly
    elig = eligible_chromosomes(a)
    c = elig[rand(rng, 1:length(elig))]
    return (c, rand(rng, 1:ploidy(a, c)))
end

function draw_target(::LengthWeighted, p::CNProfile, rng::Random.AbstractRNG)
    a = p.assembly
    elig = eligible_chromosomes(a)
    w = Float64[chromlength(a, c) for c in elig]
    c = elig[_sample_weighted(rng, w)]
    return (c, rand(rng, 1:ploidy(a, c)))
end

function draw_target(d::CNWeighted, p::CNProfile, rng::Random.AbstractRNG)
    a = p.assembly
    w = Vector{Float64}(undef, nslots(a))
    @inbounds for s in 1:nslots(a)
        m = mean_cn(p.segments[s], chromlength(a, slot_chrom(a, s)))
        w[s] = m == 0 ? 0.0 : m^d.β
    end
    s = _sample_weighted(rng, w)
    return (slot_chrom(a, s), slot_haplotype(a, s))
end

draw_target(f, p::CNProfile, rng::Random.AbstractRNG) = f(p, rng)

"""
    ExtentDraw

Where an alteration starts and how far it runs.

A rule is a callable `(profile, chrom, haplotype, rng) -> (start, stop, scale)`, where
`scale` is `:focal`, `:arm` or `:chromosome` and is recorded in the event log so
events can be tallied by class. Pass a plain function or add a method to
[`draw_extent`](@ref).

The only concrete rule is [`ExtentMixture`](@ref).
"""
abstract type ExtentDraw end

"""
    ExtentMixture(; p_chromosome = 0.0, p_arm = 0.0, lengthdist = LogUniform(1e5, 1e8))

Mix whole-chromosome, whole-arm and focal events.

Whole-arm and whole-chromosome events dominate real karyotypes and cannot be produced
at a realistic rate by any continuous length distribution, so they get their own
probabilities. `p_focal` is whatever the two leave over. Setting both to zero ignores
large-scale events entirely without changing the code path.

Focal events draw a length from `lengthdist`, then a uniform start, then **truncate**
at the chromosome end — truncation rather than rejection, matching MEDICC2, where an
event terminates at the boundary. Arm events pick the p or q arm with equal
probability and need the assembly's centromere positions.

The default `lengthdist` spans 100 kb to 100 Mb log-uniformly, so focal and near-arm
sizes are represented across the range real callers report.
"""
struct ExtentMixture{D} <: ExtentDraw
    p_chromosome::Float64
    p_arm::Float64
    lengthdist::D

    function ExtentMixture(; p_chromosome::Real = 0.0, p_arm::Real = 0.0,
                           lengthdist = LogUniform(1e5, 1e8))
        p_chromosome >= 0 || throw(ArgumentError("p_chromosome must be non-negative, got $p_chromosome"))
        p_arm >= 0 || throw(ArgumentError("p_arm must be non-negative, got $p_arm"))
        p_chromosome + p_arm <= 1 || throw(ArgumentError(
            "p_chromosome + p_arm must not exceed 1, got $(p_chromosome + p_arm)"))
        new{typeof(lengthdist)}(Float64(p_chromosome), Float64(p_arm), lengthdist)
    end
end

"""
    draw_extent(rule, profile, chrom, haplotype, rng) -> (start, stop, scale)

Choose the genomic interval for the next alteration.
"""
function draw_extent(d::ExtentMixture, p::CNProfile, c::Integer, h::Integer,
                     rng::Random.AbstractRNG)
    a = p.assembly
    L = chromlength(a, c)
    u = rand(rng)
    if u < d.p_chromosome
        return (1, L, :chromosome)
    elseif u < d.p_chromosome + d.p_arm
        parm, qarm = arms(a, c)
        chosen = rand(rng, Bool) ? parm : qarm
        return (first(chosen), last(chosen), :arm)
    else
        len = clamp(round(Int, rand(rng, d.lengthdist)), 1, L)
        start = rand(rng, 1:L)
        return (start, min(L, start + len - 1), :focal)
    end
end

draw_extent(f, p::CNProfile, c::Integer, h::Integer, rng::Random.AbstractRNG) =
    f(p, c, h, rng)

"""
    KindDraw

Whether an alteration is a gain or a loss, and by how many copies.

A rule is a callable `(profile, chrom, haplotype, start, stop, rng) -> delta` with
`delta` non-zero. The only concrete rule is [`GainLoss`](@ref).
"""
abstract type KindDraw end

"""
    GainLoss(p_gain, delta = 1)

A gain of `delta` copies with probability `p_gain`, otherwise a loss of `delta`.

A loss that drives a haplotype to copy number 0 *is* loss of heterozygosity and needs
no separate event type; a further loss over already-absent material leaves it at 0.
"""
struct GainLoss <: KindDraw
    p_gain::Float64
    delta::Int

    function GainLoss(p_gain::Real, delta::Integer = 1)
        0 <= p_gain <= 1 || throw(ArgumentError("p_gain must lie in [0, 1], got $p_gain"))
        delta > 0 || throw(ArgumentError("delta must be a positive magnitude, got $delta"))
        new(Float64(p_gain), Int(delta))
    end
end

"""
    draw_kind(rule, profile, chrom, haplotype, start, stop, rng) -> Int

Signed copy-number change for the next alteration: positive for a gain, negative for
a loss.
"""
draw_kind(d::GainLoss, p::CNProfile, c::Integer, h::Integer, s::Integer, e::Integer,
          rng::Random.AbstractRNG) = rand(rng) < d.p_gain ? d.delta : -d.delta

draw_kind(f, p::CNProfile, c::Integer, h::Integer, s::Integer, e::Integer,
          rng::Random.AbstractRNG) = f(p, c, h, s, e, rng)
```

- [ ] **Step 4: Add the exports**

```julia
export
    # Proposals
    TargetDraw, UniformChromosome, LengthWeighted, CNWeighted, draw_target,
    ExtentDraw, ExtentMixture, draw_extent,
    KindDraw, GainLoss, draw_kind
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/proposals.jl src/CopyNumberEvolution.jl test/test_proposals.jl test/runtests.jl
git commit -m "feat: add target, extent and kind proposal distributions"
```

---

### Task 9: Whole-genome doubling policies

Implements spec §8. Exact placement on a named edge is the requirement driving this.

**Files:**
- Create: `src/wgd.jl`
- Create: `test/test_wgd.jl`
- Modify: `src/CopyNumberEvolution.jl` (exports)
- Modify: `test/runtests.jl` (include)

**Interfaces:**
- Consumes: `PhyloTree`, `nnodes`, `preorder`, `isroot`, `treeroot` (Task 5);
  `CNARate`, `n_cnas`, `PerDivision` (Task 7); `WGD_MODES` (Task 4).
- Produces: `abstract type WGDPolicy`; `NoWGD()`,
  `ScheduledWGD(at::Dict{Int,Int}; mode)`, `ScheduledWGD(pairs...; mode)`,
  `ExactlyNWGD(n; mode)`, `RateWGD(rate_or_number; mode)`;
  `wgd_mode(policy)::Symbol`;
  `prepare_wgd(policy, tree, rng)::Dict{Int,Int}`.

- [ ] **Step 1: Write the failing tests**

`test/test_wgd.jl`:

```julia
@testset "wgd" begin
    tree() = phylotree([nothing, 1, 1, 2, 2];
                       birthtimes = [0.0, 1.0, 1.0, 2.0, 2.0],
                       edge_divisions = [nothing, 1, 1, 1, 1],
                       source_ids = [100, 200, 300, 400, 500])

    @testset "modes are validated and reported" begin
        @test wgd_mode(NoWGD()) === :multiply
        @test wgd_mode(ScheduledWGD(Dict(2 => 1))) === :multiply
        @test wgd_mode(ScheduledWGD(Dict(2 => 1); mode = :increment)) === :increment
        @test_throws ArgumentError ScheduledWGD(Dict(2 => 1); mode = :bogus)
        @test_throws ArgumentError ExactlyNWGD(1; mode = :bogus)
        @test_throws ArgumentError RateWGD(0.1; mode = :bogus)
    end

    @testset "NoWGD schedules nothing" begin
        @test isempty(prepare_wgd(NoWGD(), tree(), Random.Xoshiro(1)))
    end

    @testset "ScheduledWGD returns exactly what was asked for" begin
        t = tree()
        s = prepare_wgd(ScheduledWGD(Dict(2 => 1, 5 => 2)), t, Random.Xoshiro(1))
        @test s == Dict(2 => 1, 5 => 2)
        # the pairs constructor is equivalent
        @test prepare_wgd(ScheduledWGD(2 => 1), t, Random.Xoshiro(1)) == Dict(2 => 1)
        # naming the edge via helpers, which is the intended workflow
        node_of_interest = node_by_source_id(t, 400)
        @test prepare_wgd(ScheduledWGD(node_of_interest => 1), t, Random.Xoshiro(1)) ==
              Dict(4 => 1)
        @test prepare_wgd(ScheduledWGD(mrca(t, [4, 5]) => 1), t, Random.Xoshiro(1)) ==
              Dict(2 => 1)
    end

    @testset "ScheduledWGD rejects impossible placements" begin
        t = tree()
        @test_throws ArgumentError prepare_wgd(ScheduledWGD(Dict(99 => 1)), t, Random.Xoshiro(1))
        @test_throws ArgumentError prepare_wgd(ScheduledWGD(Dict(1 => 1)), t, Random.Xoshiro(1))
        @test_throws ArgumentError ScheduledWGD(Dict(2 => 0))
        @test_throws ArgumentError ScheduledWGD(Dict(2 => -1))
    end

    @testset "ExactlyNWGD places exactly n doublings on distinct non-root edges" begin
        t = tree()
        for n in 1:4
            s = prepare_wgd(ExactlyNWGD(n), t, Random.Xoshiro(42))
            @test sum(values(s)) == n
            @test length(s) == n
            @test all(v == 1 for v in values(s))
            @test all(k != treeroot(t) for k in keys(s))
        end
        @test_throws ArgumentError prepare_wgd(ExactlyNWGD(5), t, Random.Xoshiro(1))
        @test_throws ArgumentError ExactlyNWGD(0)
    end

    @testset "ExactlyNWGD is deterministic given a seed" begin
        t = tree()
        a = prepare_wgd(ExactlyNWGD(2), t, Random.Xoshiro(7))
        b = prepare_wgd(ExactlyNWGD(2), t, Random.Xoshiro(7))
        c = prepare_wgd(ExactlyNWGD(2), t, Random.Xoshiro(8))
        @test a == b
        # `c` may legitimately coincide with `a` on a 4-edge tree, so assert only the
        # property that must hold: it is still a valid 2-doubling schedule.
        @test sum(values(c)) == 2
    end

    @testset "RateWGD reuses the CNARate machinery" begin
        t = tree()
        # 4 non-root edges, one division each
        total(seed) = sum(values(prepare_wgd(RateWGD(PerDivision(0.5)), t, Random.Xoshiro(seed))); init = 0)
        m = sum(total(s) for s in 1:2000) / 2000
        @test m ≈ 2.0 rtol = 0.08          # 4 edges × Poisson(0.5)
        @test isempty(prepare_wgd(RateWGD(PerDivision(0.0)), t, Random.Xoshiro(1)))
        # the numeric shorthand is PerDivision
        @test RateWGD(0.25).rate isa PerDivision
        # per-time works too
        tt = sum(values(prepare_wgd(RateWGD(PerTime(1.0)), t, Random.Xoshiro(3))); init = 0)
        @test tt isa Int
    end

    @testset "RateWGD never schedules on the root" begin
        t = tree()
        for seed in 1:50
            s = prepare_wgd(RateWGD(PerDivision(3.0)), t, Random.Xoshiro(seed))
            @test treeroot(t) ∉ keys(s)
        end
    end
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Add `include("test_wgd.jl")` to `test/runtests.jl`, then run:
`julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL with `UndefVarError: NoWGD not defined`.

- [ ] **Step 3: Write `src/wgd.jl`**

```julia
"""
    WGDPolicy

Where whole-genome doublings fall on the tree.

Every policy resolves, before traversal begins, to a schedule mapping a node id to the
number of doublings on its **incoming** edge. That resolution is
[`prepare_wgd`](@ref); the traversal then only ever *reads* the schedule, so no
`Dict` iteration ever consumes the random number generator.

Concrete policies: [`NoWGD`](@ref), [`ScheduledWGD`](@ref), [`ExactlyNWGD`](@ref),
[`RateWGD`](@ref).

All of them carry a `mode` — `:multiply` or `:increment`, see
[`WholeGenomeDoubling`](@ref) — which defaults to `:multiply`.
"""
abstract type WGDPolicy end

"""
    NoWGD()

No whole-genome doublings. Large-scale events can be switched off entirely this way
without changing the model's shape.
"""
struct NoWGD <: WGDPolicy end

"""
    ScheduledWGD(at; mode = :multiply)
    ScheduledWGD(pairs...; mode = :multiply)

Place doublings on exactly the edges you name: `at` maps a node id to the number of
doublings on the edge into it.

This is the policy for statements like *"one whole-genome doubling in the dataset, on
the edge into node i"*. Name the edge however is convenient — the tree helpers all
return dense ids:

```julia
ScheduledWGD(node_by_source_id(tree, 42) => 1)      # by upstream cell id
ScheduledWGD(mrca(tree, metastatic_leaves) => 1)    # a subclonal doubling
ScheduledWGD(node_by_label(tree, "cellA") => 2)     # two successive doublings
```

The root cannot be scheduled: it has no incoming edge. Use
`initial = TruncalCNAs(...)` or `initial = Given(...)` to set the root state instead.
"""
struct ScheduledWGD <: WGDPolicy
    at::Dict{Int,Int}
    mode::Symbol

    function ScheduledWGD(at::AbstractDict; mode::Symbol = :multiply)
        mode in WGD_MODES || throw(ArgumentError("WGD mode must be one of $WGD_MODES, got :$mode"))
        d = Dict{Int,Int}()
        for (k, v) in at
            v >= 1 || throw(ArgumentError("scheduled doubling count for node $k must be ≥ 1, got $v"))
            d[Int(k)] = Int(v)
        end
        new(d, mode)
    end
end

ScheduledWGD(pairs::Pair...; mode::Symbol = :multiply) =
    ScheduledWGD(Dict(pairs...); mode = mode)

"""
    ExactlyNWGD(n; mode = :multiply)

Exactly `n` doublings, one each on `n` distinct non-root edges drawn uniformly at
random without replacement. Use this when the *number* of doublings matters but their
position should not be fixed. Throws if the tree has fewer than `n` non-root edges.
"""
struct ExactlyNWGD <: WGDPolicy
    n::Int
    mode::Symbol

    function ExactlyNWGD(n::Integer; mode::Symbol = :multiply)
        n >= 1 || throw(ArgumentError("ExactlyNWGD needs n ≥ 1, got $n; use NoWGD() for none"))
        mode in WGD_MODES || throw(ArgumentError("WGD mode must be one of $WGD_MODES, got :$mode"))
        new(Int(n), mode)
    end
end

"""
    RateWGD(rate; mode = :multiply)
    RateWGD(λ::Real; mode = :multiply)

Doublings drawn per edge from a [`CNARate`](@ref) rule, so a doubling rate can be
expressed per division (`RateWGD(PerDivision(λ))`, or just `RateWGD(λ)`), per unit
real time (`RateWGD(PerTime(μ))`), or from the edge's mutation count. Reusing the rate
machinery means the per-division-versus-per-time question applies to doublings on the
same footing as to segmental events.
"""
struct RateWGD{R<:CNARate} <: WGDPolicy
    rate::R
    mode::Symbol

    function RateWGD(rate::R; mode::Symbol = :multiply) where {R<:CNARate}
        mode in WGD_MODES || throw(ArgumentError("WGD mode must be one of $WGD_MODES, got :$mode"))
        new{R}(rate, mode)
    end
end

RateWGD(λ::Real; mode::Symbol = :multiply) = RateWGD(PerDivision(λ); mode = mode)

"""
    wgd_mode(policy) -> Symbol

The doubling arithmetic this policy applies, `:multiply` or `:increment`.
"""
wgd_mode(::NoWGD) = :multiply
wgd_mode(p::ScheduledWGD) = p.mode
wgd_mode(p::ExactlyNWGD) = p.mode
wgd_mode(p::RateWGD) = p.mode

"""
    prepare_wgd(policy, tree, rng) -> Dict{Int,Int}

Resolve `policy` into a schedule: node id ⇒ number of doublings on the edge into that
node. Called once, before traversal.
"""
prepare_wgd(::NoWGD, ::PhyloTree, ::Random.AbstractRNG) = Dict{Int,Int}()

function prepare_wgd(p::ScheduledWGD, t::PhyloTree, ::Random.AbstractRNG)
    for k in sort!(collect(keys(p.at)))
        1 <= k <= nnodes(t) || throw(ArgumentError(
            "scheduled WGD names node $k, which is not in this tree (1:$(nnodes(t)))"))
        isroot(t, k) && throw(ArgumentError(
            "node $k is the root and has no incoming edge; set the root's copy-number state " *
            "with initial = TruncalCNAs(n) or initial = Given(profile) instead"))
    end
    return copy(p.at)
end

function prepare_wgd(p::ExactlyNWGD, t::PhyloTree, rng::Random.AbstractRNG)
    candidates = [i for i in preorder(t) if !isroot(t, i)]
    length(candidates) >= p.n || throw(ArgumentError(
        "ExactlyNWGD($(p.n)) needs $(p.n) non-root edges but the tree has only $(length(candidates))"))
    chosen = StatsBase.sample(rng, candidates, p.n; replace = false)
    return Dict{Int,Int}(i => 1 for i in chosen)
end

function prepare_wgd(p::RateWGD, t::PhyloTree, rng::Random.AbstractRNG)
    out = Dict{Int,Int}()
    for i in preorder(t)          # deterministic order, so the schedule is reproducible
        isroot(t, i) && continue
        k = n_cnas(p.rate, t, i, rng)
        k > 0 && (out[i] = k)
    end
    return out
end
```

- [ ] **Step 4: Add the exports**

```julia
export
    # Whole-genome doubling
    WGDPolicy, NoWGD, ScheduledWGD, ExactlyNWGD, RateWGD, wgd_mode, prepare_wgd
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/wgd.jl src/CopyNumberEvolution.jl test/test_wgd.jl test/runtests.jl
git commit -m "feat: add WGD policies including exact placement on a named edge"
```

---

### Task 10: Viability rules

Implements spec §9.

**Files:**
- Create: `src/viability.jl`
- Create: `test/test_viability.jl`
- Modify: `src/CopyNumberEvolution.jl` (exports)
- Modify: `test/runtests.jl` (include)

**Interfaces:**
- Consumes: `CNProfile`, `cn_at`, `slots_of`, `slot` (Tasks 2–3);
  `CNAEvent`, `SegmentalCNA`, `WholeGenomeDoubling`, `_shift_cn` (Task 4).
- Produces: `abstract type ViabilityRule`; `AllowAll()`,
  `RejectAndRedraw(; min_total_cn = 1, max_attempts = 100)`, `AllRules(rules)`;
  `violation(rule, p, e)::Union{Symbol,Nothing}`;
  `isviable(rule, p, e)::Bool`; `max_attempts(rule)::Int`.

- [ ] **Step 1: Write the failing tests**

`test/test_viability.jl`:

```julia
@testset "viability" begin
    @testset "AllowAll permits everything" begin
        p = diploid(toy_assembly(nchrom = 1, len = 100))
        e = SegmentalCNA(1, 1, 1, 100, -5, :chromosome)
        @test isviable(AllowAll(), p, e)
        @test violation(AllowAll(), p, e) === nothing
        @test max_attempts(AllowAll()) == 1
    end

    @testset "gains and doublings are always viable" begin
        p = diploid(toy_assembly(nchrom = 1, len = 100))
        r = RejectAndRedraw()
        @test isviable(r, p, SegmentalCNA(1, 1, 1, 100, 1, :chromosome))
        @test isviable(r, p, WholeGenomeDoubling(:multiply))
        @test isviable(r, p, WholeGenomeDoubling(:increment))
    end

    @testset "one loss on a diploid autosome is fine, the second is not" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        r = RejectAndRedraw()
        first_loss = SegmentalCNA(1, 1, 21, 40, -1, :focal)
        @test isviable(r, p, first_loss)
        apply!(p, first_loss)
        # total is now 1 over 21:40, so losing the other haplotype there is inviable
        @test !isviable(r, p, SegmentalCNA(1, 2, 21, 40, -1, :focal))
        @test violation(r, p, SegmentalCNA(1, 2, 21, 40, -1, :focal)) === :min_total_cn
        # but losing the other haplotype elsewhere is fine
        @test isviable(r, p, SegmentalCNA(1, 2, 61, 80, -1, :focal))
    end

    @testset "partial overlap is caught" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        r = RejectAndRedraw()
        apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))    # total 1 on 21:40, 2 elsewhere
        # a loss on haplotype 2 that only partly overlaps the depleted span is still inviable
        @test !isviable(r, p, SegmentalCNA(1, 2, 31, 70, -1, :focal))
        @test isviable(r, p, SegmentalCNA(1, 2, 41, 70, -1, :focal))
    end

    @testset "losing a hemizygous chromosome is inviable" begin
        p = diploid(toy_sex_assembly(:male))
        a = p.assembly
        r = RejectAndRedraw()
        x = chromindex(a, "chrX")
        y = chromindex(a, "chrY")
        @test !isviable(r, p, SegmentalCNA(x, 1, 1, 1000, -1, :chromosome))
        @test !isviable(r, p, SegmentalCNA(y, 1, 1, 1000, -1, :chromosome))
        @test !isviable(r, p, SegmentalCNA(x, 1, 100, 200, -1, :focal))
    end

    @testset "zero-copy material is already absent, so a further loss changes nothing" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        r = RejectAndRedraw()
        apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))
        # haplotype 1 is at 0 over 21:40; total there is 1 (from haplotype 2).
        # Losing haplotype 1 again cannot reduce the total, so it stays viable.
        @test isviable(r, p, SegmentalCNA(1, 1, 21, 40, -1, :focal))
    end

    @testset "min_total_cn is a parameter" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        @test isviable(RejectAndRedraw(min_total_cn = 1), p, SegmentalCNA(1, 1, 1, 100, -1, :chromosome))
        @test !isviable(RejectAndRedraw(min_total_cn = 2), p, SegmentalCNA(1, 1, 1, 100, -1, :chromosome))
        @test isviable(RejectAndRedraw(min_total_cn = 0), p, SegmentalCNA(1, 1, 1, 100, -2, :chromosome))
    end

    @testset "constructor validation and max_attempts" begin
        @test_throws ArgumentError RejectAndRedraw(min_total_cn = -1)
        @test_throws ArgumentError RejectAndRedraw(max_attempts = 0)
        @test max_attempts(RejectAndRedraw()) == 100
        @test max_attempts(RejectAndRedraw(max_attempts = 7)) == 7
    end

    @testset "AllRules composes and reports the first violation" begin
        a = toy_assembly(nchrom = 1, len = 100)
        p = diploid(a)
        both = AllRules([AllowAll(), RejectAndRedraw(min_total_cn = 2)])
        @test !isviable(both, p, SegmentalCNA(1, 1, 1, 100, -1, :chromosome))
        @test violation(both, p, SegmentalCNA(1, 1, 1, 100, -1, :chromosome)) === :min_total_cn
        @test isviable(AllRules([AllowAll()]), p, SegmentalCNA(1, 1, 1, 100, -9, :chromosome))
        @test max_attempts(both) == 100
        @test_throws ArgumentError AllRules(ViabilityRule[])
    end
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Add `include("test_viability.jl")` to `test/runtests.jl`, then run:
`julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL with `UndefVarError: AllowAll not defined`.

- [ ] **Step 3: Write `src/viability.jl`**

```julia
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
copies. Note the scope of that default: it forbids homozygous deletions **of any
size**, not merely whole-chromosome nullisomy. Real tumours do contain small
homozygous deletions, so set `min_total_cn = 0` (or use [`AllowAll`](@ref)) if focal
biallelic loss should be permitted. This is recorded as an open question in the
manual.

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
```

- [ ] **Step 4: Add the exports**

```julia
export
    # Viability
    ViabilityRule, AllowAll, RejectAndRedraw, AllRules,
    violation, isviable, max_attempts
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/viability.jl src/CopyNumberEvolution.jl test/test_viability.jl test/runtests.jl
git commit -m "feat: add viability rules with reason-keyed rejection reporting"
```

---

### Task 11: The model, the traversal, and the result

Implements spec §7.1, §10, §11. This is the package's core.

**Files:**
- Create: `src/evolve.jl`
- Create: `test/test_evolve.jl`
- Modify: `src/CopyNumberEvolution.jl` (exports)
- Modify: `test/runtests.jl` (include)

**Interfaces:**
- Consumes: everything from Tasks 2–10.
- Produces: `abstract type InitialState`; `Diploid()`, `Given(profile)`,
  `TruncalCNAs(n)`;
  `CNAModel(; rate, target, extent, kind, wgd, viability, initial)`;
  `LoggedEvent(node::Int, order::Int, event::CNAEvent)`;
  `CNAEvolution` with fields `tree`, `assembly`, `model`, `profiles`, `events`,
  `rejections`, `seed`, `retain_internal`;
  `simulate_cnas(tree, assembly, model; rng, seed, retain_internal, rng_mode)::CNAEvolution`;
  `profile(res, i)::CNProfile`, `leaf_profiles(res)::Vector{CNProfile}`,
  `events_on(res, i)::Vector{LoggedEvent}`, `events_below(res, i)::Vector{LoggedEvent}`,
  `nevents(res)::Int`, `rejection_count(res)::Int`, `replay(res)::Vector{CNProfile}`.

- [ ] **Step 1: Write the failing tests**

`test/test_evolve.jl`:

```julia
@testset "evolve" begin
    # Balanced 4-leaf tree, one division per edge:
    #        1
    #      /   \
    #     2     3
    #    / \   / \
    #   4   5 6   7
    bal() = phylotree([nothing, 1, 1, 2, 2, 3, 3];
                      birthtimes = [0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                      edge_divisions = [nothing, 1, 1, 1, 1, 1, 1],
                      edge_mutations = [nothing, 2, 2, 1, 1, 1, 1],
                      source_ids = collect(101:107))

    A() = toy_assembly(nchrom = 2, len = 1000)

    @testset "CNAModel defaults" begin
        m = CNAModel()
        @test m.rate isa PerDivision
        @test m.target isa UniformChromosome
        @test m.extent isa ExtentMixture
        @test m.kind isa GainLoss
        @test m.wgd isa NoWGD
        @test m.viability isa RejectAndRedraw
        @test m.initial isa Diploid
    end

    @testset "zero alterations: every tip is exactly the root state" begin
        res = simulate_cnas(bal(), A(), CNAModel(rate = PerDivision(0.0)); seed = 1)
        d = diploid(A())
        for i in 1:nnodes(res.tree)
            @test profile(res, i) == d
            @test check_invariants(profile(res, i))
        end
        @test nevents(res) == 0
        @test rejection_count(res) == 0
    end

    @testset "profiles are retained for every node by default" begin
        res = simulate_cnas(bal(), A(), CNAModel(rate = PerDivision(1.0)); seed = 2)
        @test res.retain_internal
        @test all(res.profiles[i] !== nothing for i in 1:nnodes(res.tree))
        @test length(leaf_profiles(res)) == 4
        for i in 1:nnodes(res.tree)
            @test check_invariants(profile(res, i))
        end
    end

    @testset "retain_internal = false keeps only the leaves" begin
        res = simulate_cnas(bal(), A(), CNAModel(rate = PerDivision(1.0));
                            seed = 3, retain_internal = false)
        for i in internal_nodes(res.tree)
            @test res.profiles[i] === nothing
            @test_throws ArgumentError profile(res, i)
        end
        @test all(res.profiles[i] !== nothing for i in leaves(res.tree))
        @test length(leaf_profiles(res)) == 4
    end

    @testset "inheritance: a scheduled doubling reaches exactly its descendants" begin
        t = bal()
        res = simulate_cnas(t, A(),
                            CNAModel(rate = PerDivision(0.0),
                                     wgd = ScheduledWGD(2 => 1));
                            seed = 4)
        d = diploid(A())
        # node 2 and everything below it is doubled
        for i in (2, 4, 5)
            @test cn_at(slot_segments(profile(res, i), 1, 1), 1) == 2
        end
        # node 1, 3, 6, 7 are untouched
        for i in (1, 3, 6, 7)
            @test profile(res, i) == d
        end
        @test nevents(res) == 1
        @test only(res.events).node == 2
        @test only(res.events).event isa WholeGenomeDoubling
    end

    @testset "inheritance: a segmental alteration reaches exactly its descendants" begin
        t = bal()
        fixed_target = (p, rng) -> (1, 1)
        fixed_extent = (p, c, h, rng) -> (101, 200, :focal)
        # one alteration, only on the edge into node 3
        onlyon3 = CustomRate((tr, i, rng) -> i == 3 ? 1 : 0)
        res = simulate_cnas(t, A(),
                            CNAModel(rate = onlyon3, target = fixed_target,
                                     extent = fixed_extent, kind = GainLoss(1.0));
                            seed = 5)
        for i in (3, 6, 7)
            @test cn_at(slot_segments(profile(res, i), 1, 1), 150) == 2
        end
        for i in (1, 2, 4, 5)
            @test cn_at(slot_segments(profile(res, i), 1, 1), 150) == 1
        end
        @test nevents(res) == 1
        @test events_on(res, 3) |> length == 1
        @test events_on(res, 6) |> isempty
        @test length(events_below(res, 1)) == 1
        @test isempty(events_below(res, 2))
    end

    @testset "determinism" begin
        t = bal()
        m = CNAModel(rate = PerDivision(2.0), extent = ExtentMixture(p_chromosome = 0.2, p_arm = 0.2))
        a = simulate_cnas(t, A(), m; seed = 99)
        b = simulate_cnas(t, A(), m; seed = 99)
        c = simulate_cnas(t, A(), m; seed = 100)
        @test [profile(a, i) for i in 1:nnodes(t)] == [profile(b, i) for i in 1:nnodes(t)]
        @test a.events == b.events
        @test [profile(a, i) for i in 1:nnodes(t)] != [profile(c, i) for i in 1:nnodes(t)]
        @test a.seed == 99
    end

    @testset "an explicit rng is honoured and records no seed" begin
        t = bal()
        m = CNAModel(rate = PerDivision(1.0))
        a = simulate_cnas(t, A(), m; rng = Random.Xoshiro(5))
        b = simulate_cnas(t, A(), m; rng = Random.Xoshiro(5))
        @test a.events == b.events
        @test a.seed === nothing
    end

    @testset "events are stored in preorder, then by order within a node" begin
        t = bal()
        res = simulate_cnas(t, A(), CNAModel(rate = PerDivision(3.0)); seed = 6)
        rank = Dict(n => k for (k, n) in enumerate(preorder(t)))
        keys_ = [(rank[e.node], e.order) for e in res.events]
        @test issorted(keys_)
        for i in 1:nnodes(t)
            ons = events_on(res, i)
            @test [e.order for e in ons] == collect(1:length(ons))
        end
    end

    @testset "TruncalCNAs sets the root state and logs its events there" begin
        t = bal()
        res = simulate_cnas(t, A(),
                            CNAModel(rate = PerDivision(0.0),
                                     initial = TruncalCNAs(3),
                                     kind = GainLoss(1.0));
                            seed = 7)
        @test nevents(res) == 3
        @test all(e.node == treeroot(t) for e in res.events)
        @test [e.order for e in res.events] == [1, 2, 3]
        @test profile(res, treeroot(t)) != diploid(A())
        # with no further alterations every tip equals the root
        for i in leaves(t)
            @test profile(res, i) == profile(res, treeroot(t))
        end
        @test_throws ArgumentError TruncalCNAs(-1)
    end

    @testset "Given sets the root state and must match the assembly" begin
        t = bal()
        start = diploid(A())
        apply!(start, SegmentalCNA(1, 1, 1, 1000, 1, :chromosome))
        res = simulate_cnas(t, A(), CNAModel(rate = PerDivision(0.0), initial = Given(start)); seed = 8)
        @test profile(res, treeroot(t)) == start
        for i in leaves(t)
            @test profile(res, i) == start
        end
        # the given profile is not mutated by the simulation
        @test cn_at(slot_segments(start, 1, 1), 1) == 2
        @test_throws ArgumentError simulate_cnas(t, toy_assembly(nchrom = 3, len = 1000),
                                                 CNAModel(initial = Given(start)); seed = 9)
    end

    @testset "event-log replay reproduces every profile" begin
        t = bal()
        m = CNAModel(rate = PerDivision(2.0), initial = TruncalCNAs(2),
                     wgd = ScheduledWGD(3 => 1),
                     extent = ExtentMixture(p_chromosome = 0.1, p_arm = 0.2))
        res = simulate_cnas(t, A(), m; seed = 10)
        rp = replay(res)
        for i in 1:nnodes(t)
            @test rp[i] == profile(res, i)
        end
        # replay also works when nothing was retained
        lean = simulate_cnas(t, A(), m; seed = 10, retain_internal = false)
        rp2 = replay(lean)
        for i in 1:nnodes(t)
            @test rp2[i] == rp[i]
        end
    end

    @testset "rejections are tallied by reason" begin
        t = bal()
        # losses only, on a male karyotype: hemizygous chrX and chrY losses get rejected
        m = CNAModel(rate = PerDivision(4.0), kind = GainLoss(0.0),
                     extent = ExtentMixture(p_chromosome = 1.0),
                     viability = RejectAndRedraw(min_total_cn = 1, max_attempts = 10_000))
        res = simulate_cnas(t, toy_sex_assembly(:male), m; seed = 11)
        @test rejection_count(res) > 0
        @test haskey(res.rejections, :min_total_cn)
        @test res.rejections[:min_total_cn] == rejection_count(res)
    end

    @testset "exhausting max_attempts throws with actionable advice" begin
        t = phylotree([nothing, 1]; edge_divisions = [nothing, 1])
        # a single hemizygous chromosome, whole-chromosome losses only: every
        # proposal drives total copy number to 0 and is rejected
        m = CNAModel(rate = PerDivision(5.0), kind = GainLoss(0.0),
                     extent = ExtentMixture(p_chromosome = 1.0),
                     viability = RejectAndRedraw(min_total_cn = 1, max_attempts = 5))
        err = try
            simulate_cnas(t, hemizygous_assembly(), m; seed = 12)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("max_attempts", err.msg)
        @test occursin("min_total_cn", err.msg)
    end

    @testset "rng_mode = :per_node needs a seed and stays reproducible" begin
        t = bal()
        m = CNAModel(rate = PerDivision(2.0))
        @test_throws ArgumentError simulate_cnas(t, A(), m; rng_mode = :per_node)
        @test_throws ArgumentError simulate_cnas(t, A(), m; seed = 1, rng_mode = :bogus)
        a = simulate_cnas(t, A(), m; seed = 13, rng_mode = :per_node)
        b = simulate_cnas(t, A(), m; seed = 13, rng_mode = :per_node)
        @test a.events == b.events
        for i in 1:nnodes(t)
            @test check_invariants(profile(a, i))
        end
    end

    @testset "a unary chain draws once per edge" begin
        t = phylotree([nothing, 1, 2, 3]; edge_divisions = [nothing, 1, 1, 1])
        onlyone = CNAModel(rate = PerDivision(0.0), wgd = ScheduledWGD(2 => 1, 3 => 1, 4 => 1))
        res = simulate_cnas(t, A(), onlyone; seed = 14)
        @test nevents(res) == 3
        @test cn_at(slot_segments(profile(res, 4), 1, 1), 1) == 8   # doubled three times
        @test cn_at(slot_segments(profile(res, 1), 1, 1), 1) == 1
    end

    @testset "deep trees do not overflow the stack" begin
        n = 20_000
        parents = Vector{Union{Int,Nothing}}(undef, n)
        parents[1] = nothing
        for i in 2:n
            parents[i] = i - 1
        end
        t = phylotree(parents; edge_divisions = vcat(nothing, fill(1, n - 1)))
        res = simulate_cnas(t, A(), CNAModel(rate = PerDivision(0.05));
                            seed = 15, retain_internal = false)
        @test check_invariants(profile(res, n))
        @test nevents(res) > 0
    end

    @testset "show methods do not error" begin
        res = simulate_cnas(bal(), A(), CNAModel(rate = PerDivision(1.0)); seed = 16)
        @test occursin("CNAEvolution", sprint(show, res))
        @test occursin("CNAModel", sprint(show, res.model))
    end
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Add `include("test_evolve.jl")` to `test/runtests.jl`, then run:
`julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL with `UndefVarError: CNAModel not defined`.

- [ ] **Step 3: Write `src/evolve.jl`**

```julia
"""
    InitialState

The copy-number state of the tree's root.

Because the upstream leaf sampler keeps the founder as the root of a sampled tree, the
root is generally *not* the most recent common ancestor of the sample. Truncal
alterations are therefore expressed as the root's initial state rather than as an
MRCA-specific special case.

Concrete states: [`Diploid`](@ref), [`Given`](@ref), [`TruncalCNAs`](@ref).
"""
abstract type InitialState end

"""
    Diploid()

Start from a normal karyotype, with the sex mode taken from the assembly. The default.
"""
struct Diploid <: InitialState end

"""
    Given(profile)

Start from `profile`, which must be on the same assembly and sex as the simulation.
Covers the real-data case of starting from a called ancestral or consensus profile.
The profile is copied, never mutated.
"""
struct Given <: InitialState
    profile::CNProfile
end

"""
    TruncalCNAs(n)

Apply `n` alterations to a diploid genome before traversal begins — the truncal
(clonal) copy-number state, the copy-number analogue of clonal point mutations. The
alterations are drawn from the same model as the rest of the tree and are logged
against the root, so they appear in the event record like any others.
"""
struct TruncalCNAs <: InitialState
    n::Int
    function TruncalCNAs(n::Integer)
        n >= 0 || throw(ArgumentError("TruncalCNAs needs n ≥ 0, got $n"))
        new(Int(n))
    end
end

"""
    CNAModel(; rate, target, extent, kind, wgd, viability, initial)

The complete alteration model: how many alterations per edge, where each lands, how
far it runs, whether it is a gain or a loss, where whole-genome doublings fall, which
proposals are allowed, and what the root looks like.

Each component is injected and independently replaceable, and each of the three draws
also accepts a plain function. That is deliberate: downstream inference has to *fit*
these parameters, so every one of them must be addressable and cheap to vary.

# Keyword defaults
- `rate = PerDivision(1.0)` — see [`CNARate`](@ref).
- `target = UniformChromosome()` — see [`TargetDraw`](@ref).
- `extent = ExtentMixture()` — focal only; set `p_arm`/`p_chromosome` to enable
  large-scale events. See [`ExtentMixture`](@ref).
- `kind = GainLoss(0.5)` — see [`KindDraw`](@ref).
- `wgd = NoWGD()` — see [`WGDPolicy`](@ref).
- `viability = RejectAndRedraw()` — see [`ViabilityRule`](@ref).
- `initial = Diploid()` — see [`InitialState`](@ref).

# Examples
```julia
model = CNAModel(
    rate = PerDivision(0.5),
    target = CNWeighted(1.0),
    extent = ExtentMixture(p_chromosome = 0.05, p_arm = 0.15),
    kind = GainLoss(0.6),
    wgd = ScheduledWGD(mrca(tree, metastatic_leaves) => 1),
    initial = TruncalCNAs(4),
)
```
"""
struct CNAModel{R,T,E,K,W,V,I}
    rate::R
    target::T
    extent::E
    kind::K
    wgd::W
    viability::V
    initial::I
end

function CNAModel(; rate = PerDivision(1.0),
                    target = UniformChromosome(),
                    extent = ExtentMixture(),
                    kind = GainLoss(0.5),
                    wgd = NoWGD(),
                    viability = RejectAndRedraw(),
                    initial = Diploid())
    CNAModel(rate, target, extent, kind, wgd, viability, initial)
end

Base.show(io::IO, m::CNAModel) = print(io, "CNAModel(rate=", m.rate, ", wgd=",
    nameof(typeof(m.wgd)), ", viability=", nameof(typeof(m.viability)),
    ", initial=", nameof(typeof(m.initial)), ")")

"""
    LoggedEvent(node, order, event)

One alteration, recorded against the edge it fell on.

`node` identifies the edge by its child, `order` is the event's position within that
edge's sequence (1-based), and `event` is the [`CNAEvent`](@ref) itself. Root-state
alterations from [`TruncalCNAs`](@ref) are logged against the root.
"""
struct LoggedEvent
    node::Int
    order::Int
    event::CNAEvent
end

"""
    CNAEvolution

The result of [`simulate_cnas`](@ref): profiles, the event log, and the rejection
tally.

# Fields
- `tree::PhyloTree` — the input tree.
- `assembly::GenomeAssembly` — the genome the simulation ran on.
- `model` — the [`CNAModel`](@ref) used.
- `profiles::Vector{Union{CNProfile,Nothing}}` — indexed by node id. Every node by
  default; leaves only when `retain_internal = false`.
- `events::Vector{LoggedEvent}` — **complete**, always, in preorder of node and then
  by `order` within a node.
- `rejections::Dict{Symbol,Int}` — rejected proposals, keyed by the constraint they
  broke. Non-empty means the realised alteration distribution is conditioned on
  viability; see [`ViabilityRule`](@ref).
- `seed::Union{Int,Nothing}` — the seed, when one was given.
- `retain_internal::Bool`.

The event log is the primitive and the profiles are a cache: [`replay`](@ref)
reconstructs every profile from the root state plus the log, so nothing is lost by
running with `retain_internal = false`.
"""
struct CNAEvolution{M}
    tree::PhyloTree
    assembly::GenomeAssembly
    model::M
    profiles::Vector{Union{CNProfile,Nothing}}
    events::Vector{LoggedEvent}
    rejections::Dict{Symbol,Int}
    seed::Union{Int,Nothing}
    retain_internal::Bool
end

Base.show(io::IO, r::CNAEvolution) = print(io, "CNAEvolution(",
    nnodes(r.tree), " nodes, ", length(leaves(r.tree)), " leaves, ",
    length(r.events), " events, ", rejection_count(r), " rejections)")

"""
    simulate_cnas(tree, assembly, model; rng, seed, retain_internal = true, rng_mode = :global)
        -> CNAEvolution

Draw copy-number alterations along `tree` and return every node's profile plus the
complete event log.

Traversal is depth-first, carrying one profile down the current path and copying it
per child, so memory scales with tree *depth* rather than tree size. Per edge, in
order: whole-genome doublings from the policy's schedule first, then
`n_cnas(model.rate, …)` segmental alterations, each drawn as target → extent → kind,
checked against the viability rule and redrawn on rejection. The realised order is
recorded, so "gained then doubled" is always distinguishable from "doubled then
gained".

Alterations here are **neutral by construction**. The tree is an input that already
encodes whatever selection produced it, so this function never kills or reweights a
cell.

# Arguments
- `tree::PhyloTree` — from [`read_newick`](@ref) or converted from a
  `MutationLoadDynamics.jl` lineage tree.
- `assembly::GenomeAssembly` — e.g. `hg38(:female)`.
- `model::CNAModel`.

# Keywords
- `rng` — a random number generator. Ignored if `seed` is given.
- `seed::Union{Int,Nothing}` — convenience for `rng = Xoshiro(seed)`; recorded in the
  result.
- `retain_internal::Bool = true` — keep internal-node profiles. `false` keeps only
  leaves, for very large trees; the event log stays complete either way.
- `rng_mode::Symbol = :global` — `:global` threads one stream through the whole
  traversal. `:per_node` gives each edge its own stream derived from `seed` and the
  node's `source_id` (falling back to its dense id), so an edge draws identically no
  matter which other edges exist. That makes simulating on a sampled tree give
  *exactly* the same alterations as simulating on the full tree and subsetting, rather
  than only the same distribution. Requires `seed`. The doubling schedule is still
  drawn from the global stream.

# Examples
```julia
tree  = read_newick("lineage.nwk"; branchlength = :divisions)
res   = simulate_cnas(tree, hg38(:female), CNAModel(rate = PerDivision(0.4)); seed = 1)
grid  = BinGrid(hg38(:female), 500_000)
mat   = CNMatrix(res, grid)
write_medicc2("cells.tsv", mat)
```
"""
function simulate_cnas(tree::PhyloTree, assembly::GenomeAssembly, model::CNAModel;
                       rng::Random.AbstractRNG = Random.default_rng(),
                       seed::Union{Integer,Nothing} = nothing,
                       retain_internal::Bool = true,
                       rng_mode::Symbol = :global)
    rng_mode in (:global, :per_node) ||
        throw(ArgumentError("rng_mode must be :global or :per_node, got :$rng_mode"))
    if seed !== nothing
        rng = Random.Xoshiro(seed)
    elseif rng_mode === :per_node
        throw(ArgumentError(
            "rng_mode = :per_node derives each edge's stream from the seed, so a seed is required"))
    end
    iseed = seed === nothing ? nothing : Int(seed)

    rejections = Dict{Symbol,Int}()
    events = LoggedEvent[]
    profiles = Vector{Union{CNProfile,Nothing}}(nothing, nnodes(tree))
    schedule = prepare_wgd(model.wgd, tree, rng)

    r = treeroot(tree)
    rootrng = _edge_rng(rng_mode, rng, iseed, tree, r)
    rootp = _initial_profile(model.initial, assembly, model, rootrng, rejections, events, r)
    (retain_internal || isleaf(tree, r)) && (profiles[r] = rootp)

    # Iterative depth-first descent. A frame holds a node, its profile, and the index
    # of the next child to visit, so only the current root-to-node path is in memory
    # and a 10^5-deep tree cannot overflow the stack.
    fnode = Int[r]
    fprof = CNProfile[rootp]
    fnext = Int[1]
    while !isempty(fnode)
        i = fnode[end]
        kids = childrenof(tree, i)
        k = fnext[end]
        if k > length(kids)
            pop!(fnode); pop!(fprof); pop!(fnext)
            continue
        end
        fnext[end] = k + 1
        child = kids[k]
        cp = copy(fprof[end])       # the parent's stored profile is never mutated
        erng = _edge_rng(rng_mode, rng, iseed, tree, child)
        _evolve_edge!(cp, tree, child, model, schedule, erng, rejections, events)
        (retain_internal || isleaf(tree, child)) && (profiles[child] = cp)
        push!(fnode, child); push!(fprof, cp); push!(fnext, 1)
    end

    return CNAEvolution(tree, assembly, model, profiles, events, rejections,
                        iseed, retain_internal)
end

function _edge_rng(mode::Symbol, rng::Random.AbstractRNG, seed::Union{Int,Nothing},
                   t::PhyloTree, i::Integer)
    mode === :global && return rng
    key = something(node(t, i).source_id, i)
    return Random.Xoshiro(hash((seed, key)))
end

_initial_profile(::Diploid, assembly, model, rng, rejections, events, r) = diploid(assembly)

function _initial_profile(init::Given, assembly, model, rng, rejections, events, r)
    (init.profile.assembly.name == assembly.name &&
     init.profile.assembly.sex == assembly.sex) || throw(ArgumentError(
        "Given initial profile is on $(init.profile.assembly.name)/:$(init.profile.assembly.sex) " *
        "but the simulation runs on $(assembly.name)/:$(assembly.sex)"))
    return copy(init.profile)
end

function _initial_profile(init::TruncalCNAs, assembly, model, rng, rejections, events, r)
    p = diploid(assembly)
    for order in 1:init.n
        ev = _draw_cna(model, p, rng, rejections)
        apply!(p, ev)
        push!(events, LoggedEvent(r, order, ev))
    end
    return p
end

function _evolve_edge!(p::CNProfile, t::PhyloTree, i::Int, model::CNAModel,
                       schedule::Dict{Int,Int}, rng::Random.AbstractRNG,
                       rejections::Dict{Symbol,Int}, events::Vector{LoggedEvent})
    order = 0
    ndoublings = get(schedule, i, 0)     # a read, never an iteration
    if ndoublings > 0
        ev = WholeGenomeDoubling(wgd_mode(model.wgd))
        for _ in 1:ndoublings
            apply!(p, ev)
            order += 1
            push!(events, LoggedEvent(i, order, ev))
        end
    end
    for _ in 1:n_cnas(model.rate, t, i, rng)
        ev = _draw_cna(model, p, rng, rejections)
        apply!(p, ev)
        order += 1
        push!(events, LoggedEvent(i, order, ev))
    end
    return p
end

function _draw_cna(model::CNAModel, p::CNProfile, rng::Random.AbstractRNG,
                   rejections::Dict{Symbol,Int})
    attempts = max_attempts(model.viability)
    for _ in 1:attempts
        c, h = draw_target(model.target, p, rng)
        s, e, scale = draw_extent(model.extent, p, c, h, rng)
        δ = draw_kind(model.kind, p, c, h, s, e, rng)
        ev = SegmentalCNA(c, h, s, e, δ, scale)
        why = violation(model.viability, p, ev)
        why === nothing && return ev
        rejections[why] = get(rejections, why, 0) + 1
    end
    error("viability rejection exceeded max_attempts = $attempts: the model is proposing " *
          "almost nothing viable. Loosen the rule (lower min_total_cn, raise max_attempts), " *
          "raise p_gain so losses are less frequent, or reduce event sizes.")
end

"""
    profile(res, i) -> CNProfile

The copy-number profile of node `i`. Throws if it was not retained; use
[`replay`](@ref) to reconstruct all profiles from the event log.
"""
function profile(r::CNAEvolution, i::Integer)
    p = r.profiles[i]
    p === nothing && throw(ArgumentError(
        "no profile retained for node $i (the simulation ran with retain_internal = false); " *
        "call replay(res) to reconstruct every profile from the event log"))
    return p
end

"""
    leaf_profiles(res) -> Vector{CNProfile}

Profiles of the tree's leaves, in `leaves(res.tree)` order.
"""
leaf_profiles(r::CNAEvolution) = [profile(r, i) for i in leaves(r.tree)]

"""
    events_on(res, i) -> Vector{LoggedEvent}

Alterations that fell on the edge into node `i`, in the order they were applied.
"""
events_on(r::CNAEvolution, i::Integer) = [e for e in r.events if e.node == i]

"""
    events_below(res, i) -> Vector{LoggedEvent}

Alterations on every edge strictly below node `i`, i.e. those inherited by some but
not all of the tree.
"""
function events_below(r::CNAEvolution, i::Integer)
    below = Set{Int}()
    stack = collect(childrenof(r.tree, i))
    while !isempty(stack)
        j = pop!(stack)
        push!(below, j)
        append!(stack, childrenof(r.tree, j))
    end
    return [e for e in r.events if e.node in below]
end

"""
    nevents(res) -> Int

Total number of logged alterations.
"""
nevents(r::CNAEvolution) = length(r.events)

"""
    rejection_count(res) -> Int

Total number of proposals rejected by the viability rule, across all reasons.
"""
rejection_count(r::CNAEvolution) = isempty(r.rejections) ? 0 : sum(values(r.rejections))

"""
    replay(res) -> Vector{CNProfile}

Reconstruct every node's profile from the root state plus the event log.

The event log is the primitive and the retained profiles are a cache, so this returns
exactly `res.profiles` wherever those were retained — which is what the replay test
asserts — and fills in the rest. Use it after a `retain_internal = false` run, or to
verify the traversal.
"""
function replay(r::CNAEvolution)
    t = r.tree
    grouped = [LoggedEvent[] for _ in 1:nnodes(t)]
    for le in r.events
        push!(grouped[le.node], le)
    end
    for v in grouped
        sort!(v; by = e -> e.order)
    end
    out = Vector{CNProfile}(undef, nnodes(t))
    root = treeroot(t)
    base = r.model.initial isa Given ? copy(r.model.initial.profile) : diploid(r.assembly)
    for le in grouped[root]
        apply!(base, le.event)
    end
    out[root] = base
    for i in preorder(t)
        i == root && continue
        p = copy(out[parentof(t, i)])
        for le in grouped[i]
            apply!(p, le.event)
        end
        out[i] = p
    end
    return out
end
```

- [ ] **Step 4: Add the exports**

```julia
export
    # Model, simulation and results
    InitialState, Diploid, Given, TruncalCNAs,
    CNAModel, LoggedEvent, CNAEvolution, simulate_cnas,
    profile, leaf_profiles, events_on, events_below, nevents,
    rejection_count, replay
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/evolve.jl src/CopyNumberEvolution.jl test/test_evolve.jl test/runtests.jl
git commit -m "feat: add CNAModel, tree traversal, event log and replay"
```

---

### Task 12: Bin grid and copy-number matrix

Implements spec §12.

**Files:**
- Create: `src/bingrid.jl`
- Create: `test/test_bingrid.jl`
- Modify: `src/CopyNumberEvolution.jl` (exports)
- Modify: `test/runtests.jl` (include)

**Interfaces:**
- Consumes: `GenomeAssembly` accessors (Task 2); `CNProfile`, `Segment`, `cn_at`,
  `segment_index` (Task 3); `CNAEvolution`, `profile`, `leaves`, `cellname` (Tasks 5, 11).
- Produces: `Bin(chrom::Int, start::Int, stop::Int)`;
  `BinGrid(assembly, size = 500_000)` with fields `assembly`, `size`, `bins`,
  `chromranges`; `nbins(grid)::Int`; `bins_of(grid, chrom)::UnitRange{Int}`;
  `abstract type BinRule`; `LengthWeightedMajority()`, `AreaWeightedMean()`;
  `project(segs::Vector{Segment}, grid, chrom, rule)::Vector{Int}`;
  `project(p::CNProfile, grid; rule)::Tuple{Vector{Int},Vector{Vector{Int}}}`;
  `CNMatrix` with fields `grid`, `cells`, `names`, `total`, `allele`;
  `CNMatrix(res::CNAEvolution, grid; cells, rule, allele)`;
  `ncells(m)::Int`, `max_cn(m)::Int`.

- [ ] **Step 1: Write the failing tests**

`test/test_bingrid.jl`:

```julia
@testset "bingrid" begin
    S = CopyNumberEvolution.Segment

    @testset "grid layout" begin
        a = toy_assembly(nchrom = 2, len = 1000)
        g = BinGrid(a, 250)
        @test nbins(g) == 8
        @test bins_of(g, 1) == 1:4
        @test bins_of(g, 2) == 5:8
        @test g.bins[1] == CopyNumberEvolution.Bin(1, 1, 250)
        @test g.bins[4] == CopyNumberEvolution.Bin(1, 751, 1000)
        @test_throws ArgumentError BinGrid(a, 0)
    end

    @testset "the last bin of a chromosome is short when the length does not divide" begin
        a = toy_assembly(nchrom = 1, len = 1000)
        g = BinGrid(a, 300)
        @test nbins(g) == 4
        @test g.bins[4] == CopyNumberEvolution.Bin(1, 901, 1000)
        @test length(bins_of(g, 1)) == 4
    end

    @testset "hg38 bin counts" begin
        g = BinGrid(hg38(:female), 500_000)
        @test length(bins_of(g, 1)) == cld(248956422, 500_000)   # 498
        y = chromindex(hg38(:female), "chrY")
        @test isempty(bins_of(g, y))                              # no slots, no bins
        gm = BinGrid(hg38(:male), 500_000)
        @test length(bins_of(gm, chromindex(hg38(:male), "chrY"))) == cld(57227415, 500_000)
    end

    @testset "a bin-aligned segmentation projects exactly" begin
        a = toy_assembly(nchrom = 1, len = 1000)
        g = BinGrid(a, 250)
        segs = [S(1, 250, 1), S(251, 500, 3), S(501, 1000, 0)]
        @test project(segs, g, 1, LengthWeightedMajority()) == [1, 3, 0, 0]
        @test project(segs, g, 1, AreaWeightedMean()) == [1, 3, 0, 0]
    end

    @testset "a chromosome-length segment fills every bin" begin
        a = toy_assembly(nchrom = 1, len = 1000)
        g = BinGrid(a, 250)
        @test project([S(1, 1000, 4)], g, 1, LengthWeightedMajority()) == [4, 4, 4, 4]
    end

    @testset "a straddling bin follows the documented rule" begin
        a = toy_assembly(nchrom = 1, len = 400)
        g = BinGrid(a, 100)
        # bin 2 is 101:200; the breakpoint at 161 gives 60 bp at cn 1 and 40 bp at cn 4
        segs = [S(1, 160, 1), S(161, 400, 4)]
        @test project(segs, g, 1, LengthWeightedMajority()) == [1, 1, 4, 4]
        # area-weighted mean over bin 2 is (60*1 + 40*4)/100 = 2.2 -> 2
        @test project(segs, g, 1, AreaWeightedMean()) == [1, 2, 4, 4]
    end

    @testset "majority ties resolve to the lower copy number" begin
        a = toy_assembly(nchrom = 1, len = 200)
        g = BinGrid(a, 100)
        segs = [S(1, 50, 5), S(51, 200, 2)]     # bin 1 is 50 bp of cn 5 and 50 bp of cn 2
        @test project(segs, g, 1, LengthWeightedMajority())[1] == 2
    end

    @testset "area-weighted mean rounds halves upward" begin
        a = toy_assembly(nchrom = 1, len = 100)
        g = BinGrid(a, 100)
        segs = [S(1, 50, 1), S(51, 100, 2)]     # mean exactly 1.5
        @test project(segs, g, 1, AreaWeightedMean()) == [2]
    end

    @testset "profile projection returns total and per-haplotype tracks" begin
        a = toy_assembly(nchrom = 1, len = 400)
        g = BinGrid(a, 100)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 1, 200, 1, :focal))
        tot, alleles = project(p, g)
        @test length(alleles) == 2
        @test alleles[1] == [2, 2, 1, 1]
        @test alleles[2] == [1, 1, 1, 1]
        @test tot == [3, 3, 2, 2]
        @test tot == alleles[1] .+ alleles[2]
    end

    @testset "hemizygous chromosomes give zeros on the absent haplotype" begin
        a = toy_sex_assembly(:male, len = 400)
        g = BinGrid(a, 100)
        p = diploid(a)
        tot, alleles = project(p, g)
        x = bins_of(g, chromindex(a, "chrX"))
        @test all(alleles[1][i] == 1 for i in x)
        @test all(alleles[2][i] == 0 for i in x)
        @test all(tot[i] == 1 for i in x)
    end

    @testset "CNMatrix over an evolution result" begin
        t = phylotree([nothing, 1, 1]; edge_divisions = [nothing, 1, 1],
                      labels = [nothing, "left", nothing])
        a = toy_assembly(nchrom = 2, len = 400)
        res = simulate_cnas(t, a, CNAModel(rate = PerDivision(1.0)); seed = 31)
        g = BinGrid(a, 100)
        m = CNMatrix(res, g)
        @test ncells(m) == 2
        @test m.cells == leaves(t)
        @test m.names == ["left", "cell_3"]
        @test size(m.total) == (2, nbins(g))
        @test length(m.allele) == 2
        @test m.total == m.allele[1] .+ m.allele[2]
        @test all(>=(0), m.total)
        @test max_cn(m) >= 0
    end

    @testset "CNMatrix can include internal nodes as a truth set" begin
        t = phylotree([nothing, 1, 1]; edge_divisions = [nothing, 1, 1])
        a = toy_assembly(nchrom = 1, len = 400)
        res = simulate_cnas(t, a, CNAModel(rate = PerDivision(1.0)); seed = 32)
        g = BinGrid(a, 100)
        m = CNMatrix(res, g; cells = collect(1:nnodes(t)))
        @test ncells(m) == 3
        @test m.names == ["cell_1", "cell_2", "cell_3"]
    end

    @testset "allele = false omits the per-haplotype tracks" begin
        t = phylotree([nothing, 1]; edge_divisions = [nothing, 1])
        a = toy_assembly(nchrom = 1, len = 400)
        res = simulate_cnas(t, a, CNAModel(rate = PerDivision(1.0)); seed = 33)
        m = CNMatrix(res, BinGrid(a, 100); allele = false)
        @test m.allele === nothing
        @test size(m.total, 1) == 1
    end

    @testset "grid and result must agree on the assembly" begin
        t = phylotree([nothing, 1]; edge_divisions = [nothing, 1])
        a = toy_assembly(nchrom = 1, len = 400)
        res = simulate_cnas(t, a, CNAModel(rate = PerDivision(0.0)); seed = 34)
        other = BinGrid(toy_assembly(nchrom = 2, len = 400), 100)
        @test_throws ArgumentError CNMatrix(res, other)
    end

    @testset "matrix total is the sum of haplotype projections, not a reprojected total" begin
        # Projection is non-linear, so the two definitions genuinely differ at a
        # straddling bin. We define total as the sum, so that total == A + B always.
        a = toy_assembly(nchrom = 1, len = 200)
        g = BinGrid(a, 200)
        p = diploid(a)
        apply!(p, SegmentalCNA(1, 1, 1, 60, 4, :focal))    # hap 1: 60 bp at 5, 140 at 1
        tot, alleles = project(p, g)
        @test alleles[1] == [1]                             # majority of hap 1 is cn 1
        @test alleles[2] == [1]
        @test tot == [2]
        # reprojecting the summed segmentation gives the same here, but the invariant
        # we rely on is the additive one
        @test tot == alleles[1] .+ alleles[2]
    end
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Add `include("test_bingrid.jl")` to `test/runtests.jl`, then run:
`julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL with `UndefVarError: BinGrid not defined`.

- [ ] **Step 3: Write `src/bingrid.jl`**

```julia
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
    (a.name == g.assembly.name && a.sex == g.assembly.sex) || throw(ArgumentError(
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
    (res.assembly.name == g.assembly.name && res.assembly.sex == g.assembly.sex) ||
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
```

- [ ] **Step 4: Add the exports**

```julia
export
    # Bin grid and matrices
    Bin, BinGrid, nbins, bins_of,
    BinRule, LengthWeightedMajority, AreaWeightedMean, project,
    CNMatrix, ncells, max_cn
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/bingrid.jl src/CopyNumberEvolution.jl test/test_bingrid.jl test/runtests.jl
git commit -m "feat: add bin grid, projection rules and CNMatrix"
```

---

### Task 13: Tabular output and MEDICC2 export

Implements spec §13.

**Files:**
- Create: `src/io.jl`
- Create: `test/test_io.jl`
- Modify: `src/CopyNumberEvolution.jl` (exports)
- Modify: `test/runtests.jl` (include)

**Interfaces:**
- Consumes: `CNAEvolution`, `profile`, `events` (Task 11); `CNMatrix`, `BinGrid`,
  `bins_of`, `max_cn` (Task 12); `cellname` (Task 5); assembly accessors (Task 2).
- Produces: `write_profiles(path_or_io, res; cells)`;
  `write_events(path_or_io, res)`; `write_bins(path_or_io, grid)`;
  `write_medicc2(path_or_io, m::CNMatrix; include_xy = false, normal_name = "diploid")`.

- [ ] **Step 1: Write the failing tests**

`test/test_io.jl`:

```julia
@testset "io" begin
    setup() = begin
        t = phylotree([nothing, 1, 1]; edge_divisions = [nothing, 1, 1],
                      labels = [nothing, "left", "right"])
        a = toy_sex_assembly(:male, len = 400)
        res = simulate_cnas(t, a, CNAModel(rate = PerDivision(1.0),
                                           extent = ExtentMixture(p_chromosome = 0.5));
                            seed = 41)
        (t, a, res, BinGrid(a, 100))
    end

    readtsv(path) = [split(l, '\t') for l in readlines(path)]

    @testset "write_profiles" begin
        t, a, res, g = setup()
        path = joinpath(mktempdir(), "profiles.tsv")
        write_profiles(path, res)
        rows = readtsv(path)
        @test rows[1] == ["node_id", "name", "chrom", "haplotype", "start", "stop", "cn"]
        @test length(rows) > 1
        body = rows[2:end]
        @test all(length(r) == 7 for r in body)
        # every chromosome name present is a real one, and coordinates are 1-based
        @test all(r[3] in [chromname(a, c) for c in 1:nchromosomes(a)] for r in body)
        @test minimum(parse(Int, r[5]) for r in body) == 1
        # a female assembly's absent chromosome never appears
        @test all(parse(Int, r[7]) >= 0 for r in body)
    end

    @testset "write_profiles honours the cells argument" begin
        t, a, res, g = setup()
        path = joinpath(mktempdir(), "leaves.tsv")
        write_profiles(path, res; cells = leaves(t))
        ids = Set(parse(Int, r[1]) for r in readtsv(path)[2:end])
        @test ids == Set(leaves(t))
    end

    @testset "write_events" begin
        t, a, res, g = setup()
        path = joinpath(mktempdir(), "events.tsv")
        write_events(path, res)
        rows = readtsv(path)
        @test rows[1] == ["node_id", "name", "order", "type", "chrom", "haplotype",
                          "start", "stop", "delta", "scale", "mode"]
        @test length(rows) - 1 == nevents(res)
        for r in rows[2:end]
            @test r[4] in ("segmental", "wgd")
            if r[4] == "segmental"
                @test r[11] == "NA"
                @test parse(Int, r[9]) != 0
            else
                @test r[5] == "NA" && r[11] in ("multiply", "increment")
            end
        end
    end

    @testset "write_events on a run with a doubling records the mode" begin
        t = phylotree([nothing, 1]; edge_divisions = [nothing, 1])
        a = toy_assembly(nchrom = 1, len = 400)
        res = simulate_cnas(t, a, CNAModel(rate = PerDivision(0.0),
                                           wgd = ScheduledWGD(2 => 1; mode = :increment));
                            seed = 42)
        path = joinpath(mktempdir(), "e.tsv")
        write_events(path, res)
        rows = readtsv(path)
        @test length(rows) == 2
        @test rows[2][4] == "wgd"
        @test rows[2][11] == "increment"
    end

    @testset "write_bins" begin
        t, a, res, g = setup()
        path = joinpath(mktempdir(), "bins.tsv")
        write_bins(path, g)
        rows = readtsv(path)
        @test rows[1] == ["bin_index", "chrom", "start", "stop"]
        @test length(rows) - 1 == nbins(g)
        @test rows[2] == ["1", "chr1", "1", "100"]
    end

    @testset "write_medicc2 shape and coordinates" begin
        t, a, res, g = setup()
        m = CNMatrix(res, g)
        path = joinpath(mktempdir(), "medicc.tsv")
        write_medicc2(path, m)
        rows = readtsv(path)
        @test rows[1] == ["sample_id", "chrom", "start", "end", "cn_a", "cn_b"]
        autosomal = sum(length(bins_of(g, c)) for c in autosomes(a))
        @test length(rows) - 1 == (ncells(m) + 1) * autosomal
        # BED convention: 0-based start, exclusive end
        @test rows[2][2:4] == ["chr1", "0", "100"]
        # the reference rows come first and are all 1/1
        diploid_rows = [r for r in rows[2:end] if r[1] == "diploid"]
        @test length(diploid_rows) == autosomal
        @test all(r[5] == "1" && r[6] == "1" for r in diploid_rows)
        # sample ids match cellname, so they agree with an exported newick
        @test Set(r[1] for r in rows[2:end]) == Set(vcat("diploid", m.names))
    end

    @testset "write_medicc2 excludes sex chromosomes by default" begin
        t, a, res, g = setup()
        m = CNMatrix(res, g)
        path = joinpath(mktempdir(), "auto.tsv")
        write_medicc2(path, m)
        chroms = Set(r[2] for r in readtsv(path)[2:end])
        @test !("chrX" in chroms) && !("chrY" in chroms)
        path2 = joinpath(mktempdir(), "withxy.tsv")
        write_medicc2(path2, m; include_xy = true)
        chroms2 = Set(r[2] for r in readtsv(path2)[2:end])
        @test "chrX" in chroms2 && "chrY" in chroms2
    end

    @testset "write_medicc2 warns above MEDICC2's copy-number ceiling" begin
        t = phylotree([nothing, 1]; edge_divisions = [nothing, 1])
        a = toy_assembly(nchrom = 1, len = 400)
        start = diploid(a)
        apply!(start, SegmentalCNA(1, 1, 1, 400, 11, :chromosome))   # cn 12
        res = simulate_cnas(t, a, CNAModel(rate = PerDivision(0.0), initial = Given(start));
                            seed = 43)
        m = CNMatrix(res, BinGrid(a, 100))
        @test max_cn(m) > 8
        path = joinpath(mktempdir(), "high.tsv")
        @test_logs (:warn, r"MEDICC2") write_medicc2(path, m)
    end

    @testset "write_medicc2 needs allele tracks" begin
        t = phylotree([nothing, 1]; edge_divisions = [nothing, 1])
        a = toy_assembly(nchrom = 1, len = 400)
        res = simulate_cnas(t, a, CNAModel(rate = PerDivision(0.0)); seed = 44)
        m = CNMatrix(res, BinGrid(a, 100); allele = false)
        @test_throws ArgumentError write_medicc2(joinpath(mktempdir(), "x.tsv"), m)
    end

    @testset "an exported newick and matrix agree on names" begin
        t, a, res, g = setup()
        m = CNMatrix(res, g)
        dir = mktempdir()
        write_newick(joinpath(dir, "t.nwk"), t; branchlength = :divisions)
        write_medicc2(joinpath(dir, "m.tsv"), m)
        nwk = read(joinpath(dir, "t.nwk"), String)
        for nm in m.names
            @test occursin(nm, nwk)
        end
    end

    @testset "writers accept an IO as well as a path" begin
        t, a, res, g = setup()
        buf = IOBuffer()
        write_bins(buf, g)
        @test occursin("bin_index", String(take!(buf)))
    end
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Add `include("test_io.jl")` to `test/runtests.jl`, then run:
`julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL with `UndefVarError: write_profiles not defined`.

- [ ] **Step 3: Write `src/io.jl`**

```julia
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
```

- [ ] **Step 4: Add the exports**

```julia
export
    # Output
    write_profiles, write_events, write_bins, write_medicc2
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/io.jl src/CopyNumberEvolution.jl test/test_io.jl test/runtests.jl
git commit -m "feat: add tabular writers and MEDICC2 TSV export"
```

---

### Task 14: The `MutationLoadDynamics.jl` package extension

Implements spec §4.4. The weak dependency is what keeps the simulator out of the
inference dependency chain.

**Files:**
- Create: `ext/CopyNumberEvolutionMutationLoadDynamicsExt.jl`
- Create: `test/test_ext.jl`
- Modify: `test/runtests.jl` (include)

**Interfaces:**
- Consumes: `PhyloNode`, `PhyloTree` (Task 5).
- Produces: a `CopyNumberEvolution.PhyloTree(::BinaryNode{NonMarkovCell})` method and
  `CopyNumberEvolution.founder_mutations(::BinaryNode{NonMarkovCell})`, both available
  only when `MutationLoadDynamics` is loaded.

To exercise the extension locally:
`julia --project=. -e 'using Pkg; Pkg.develop(path = "../MutationLoadDynamics.jl")'`
The tests skip themselves with an informative message when it is absent, so CI stays
green without an unregistered dependency.

- [ ] **Step 1: Declare the stubs the extension will fill**

`PhyloTree` already exists as a type, so the extension can add a constructor method to
it without a stub. `founder_mutations` needs a function to add a method to, so add
this to `src/tree.jl` (and to the `# Trees` export list):

```julia
"""
    founder_mutations(root) -> Int

Number of mutations the founder cell of a `MutationLoadDynamics.jl` lineage tree
acquired at its own birth.

The founder has no incoming edge, so those mutations cannot be attributed to one, and
[`PhyloTree`](@ref) leaves the root's `edge_mutations` as `nothing`. If you want them
translated into copy-number alterations, feed this to
`CNAModel(initial = TruncalCNAs(founder_mutations(root)))`.

Requires `MutationLoadDynamics` to be loaded; it is provided by a package extension.
"""
function founder_mutations end
```

- [ ] **Step 2: Write the failing tests**

`test/test_ext.jl`:

```julia
@testset "package extension" begin
    @testset "the extension is wired up in Project.toml" begin
        proj = read(joinpath(@__DIR__, "..", "Project.toml"), String)
        @test occursin("[weakdeps]", proj)
        @test occursin("MutationLoadDynamics = \"7b855ee6-6887-412f-a571-26d20a5a92d7\"", proj)
        @test occursin("CopyNumberEvolutionMutationLoadDynamicsExt = \"MutationLoadDynamics\"", proj)
        # and never a hard dependency: the simulator must not be reachable from an
        # inference-only install
        deps = match(r"\[deps\](.*?)\n\["s, proj)
        @test deps !== nothing
        @test !occursin("MutationLoadDynamics", deps.captures[1])
        @test isfile(joinpath(@__DIR__, "..", "ext",
                              "CopyNumberEvolutionMutationLoadDynamicsExt.jl"))
    end

    mld_loaded = try
        @eval using MutationLoadDynamics
        true
    catch
        false
    end

    if !mld_loaded
        @info """MutationLoadDynamics.jl is not available, so the conversion tests are skipped.
                 Enable them with: julia --project=. -e 'using Pkg; Pkg.develop(path = "../MutationLoadDynamics.jl")'"""
    else
        @testset "converting a lineage tree" begin
            # Build a small BinaryNode tree by hand: founder -> two daughters,
            # the left one dividing again.
            root = MutationLoadDynamics.BinaryNode(
                MutationLoadDynamics.NonMarkovCell(1, 0.0, 3, 1.0))
            MutationLoadDynamics.leftchild!(root,
                MutationLoadDynamics.NonMarkovCell(2, 1.5, 4, 1.0))
            MutationLoadDynamics.rightchild!(root,
                MutationLoadDynamics.NonMarkovCell(3, 1.5, 1, 1.0))
            MutationLoadDynamics.leftchild!(root.left,
                MutationLoadDynamics.NonMarkovCell(4, 2.25, 2, 1.0))
            MutationLoadDynamics.rightchild!(root.left,
                MutationLoadDynamics.NonMarkovCell(5, 2.75, 0, 1.0))

            t = PhyloTree(root)
            @test nnodes(t) == 5
            @test isroot(t, treeroot(t))
            @test node(t, treeroot(t)).source_id == 1
            @test node(t, treeroot(t)).birthtime == 0.0
            @test node(t, treeroot(t)).edge_divisions === nothing
            @test node(t, treeroot(t)).edge_mutations === nothing

            # every non-root edge is exactly one division
            @test all(node(t, i).edge_divisions == 1 for i in 1:nnodes(t) if !isroot(t, i))

            i2 = node_by_source_id(t, 2)
            @test node(t, i2).edge_mutations == 4
            @test edge_time(t, i2) ≈ 1.5
            @test node(t, node_by_source_id(t, 5)).edge_mutations == 0
            @test edge_time(t, node_by_source_id(t, 5)) ≈ 1.25

            @test Set(node(t, l).source_id for l in leaves(t)) == Set([3, 4, 5])
            @test founder_mutations(root) == 3
        end

        @testset "all three rate rules work on a converted tree" begin
            root = MutationLoadDynamics.BinaryNode(
                MutationLoadDynamics.NonMarkovCell(1, 0.0, 2, 1.0))
            MutationLoadDynamics.leftchild!(root,
                MutationLoadDynamics.NonMarkovCell(2, 1.0, 5, 1.0))
            MutationLoadDynamics.rightchild!(root,
                MutationLoadDynamics.NonMarkovCell(3, 2.0, 7, 1.0))
            t = PhyloTree(root)
            rng = Random.Xoshiro(1)
            i2 = node_by_source_id(t, 2)
            @test n_cnas(PerDivision(1.0), t, i2, rng) isa Int
            @test n_cnas(PerTime(1.0), t, i2, rng) isa Int
            @test n_cnas(FromEdgeMutations(), t, i2, rng) == 5
        end

        @testset "a unary chain, as pruning leaves behind, converts unchanged" begin
            root = MutationLoadDynamics.BinaryNode(
                MutationLoadDynamics.NonMarkovCell(1, 0.0, 0, 1.0))
            MutationLoadDynamics.leftchild!(root,
                MutationLoadDynamics.NonMarkovCell(2, 1.0, 1, 1.0))
            MutationLoadDynamics.leftchild!(root.left,
                MutationLoadDynamics.NonMarkovCell(3, 2.0, 1, 1.0))
            t = PhyloTree(root)
            @test nnodes(t) == 3
            @test leaves(t) == [node_by_source_id(t, 3)]
            @test depth(t, node_by_source_id(t, 3)) == 2
        end

        @testset "an end-to-end run on a converted tree" begin
            root = MutationLoadDynamics.BinaryNode(
                MutationLoadDynamics.NonMarkovCell(1, 0.0, 0, 1.0))
            MutationLoadDynamics.leftchild!(root,
                MutationLoadDynamics.NonMarkovCell(2, 1.0, 6, 1.0))
            MutationLoadDynamics.rightchild!(root,
                MutationLoadDynamics.NonMarkovCell(3, 1.0, 6, 1.0))
            t = PhyloTree(root)
            a = toy_assembly(nchrom = 2, len = 1000)
            res = simulate_cnas(t, a, CNAModel(rate = FromEdgeMutations(0.5)); seed = 61)
            for i in 1:nnodes(t)
                @test check_invariants(profile(res, i))
            end
            @test replay(res) == [profile(res, i) for i in 1:nnodes(t)]
        end
    end
end
```

- [ ] **Step 3: Run the tests to verify they fail**

Add `include("test_ext.jl")` to `test/runtests.jl`, then run:
`julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `founder_mutations` is not defined, and the `ext/` file does not
exist. (The conversion tests will be skipped unless MutationLoadDynamics is dev'd in.)

- [ ] **Step 4: Write `ext/CopyNumberEvolutionMutationLoadDynamicsExt.jl`**

```julia
"""
    CopyNumberEvolutionMutationLoadDynamicsExt

Bridge from `MutationLoadDynamics.jl`'s pointer-based lineage trees to
`CopyNumberEvolution.PhyloTree`.

This is a **package extension**, loaded only when both packages are present. Loading
`CopyNumberEvolution` alone gives the copy-number modeller with no simulator in the
dependency chain — which is the point, because the downstream inference package must
be installable and runnable against real patient data with no simulator anywhere in
its dependencies.
"""
module CopyNumberEvolutionMutationLoadDynamicsExt

using CopyNumberEvolution
using CopyNumberEvolution: PhyloNode, PhyloTree, founder_mutations
using MutationLoadDynamics: BinaryNode, NonMarkovCell

"""
    PhyloTree(root::BinaryNode{NonMarkovCell}) -> PhyloTree

Convert a `MutationLoadDynamics.jl` lineage tree, full or sampled.

Field mapping:

| `PhyloNode` field | source |
|:---|:---|
| `birthtime` | `cell.birthtime` |
| `edge_divisions` | `1` — one lineage-tree edge is exactly one division |
| `edge_mutations` | `cell.mutations`, the mutations acquired at this cell's birth |
| `source_id` | `cell.id`, so `node_by_source_id` keeps working after leaf sampling |
| `label` | `nothing` |

All three rate rules therefore apply to a converted tree.

Two things this deliberately does **not** do. It does not sample: sampling is
`MutationLoadDynamics.jl`'s own operation, and this converts whatever tree it is
handed. And it does not prune or collapse: unary nodes are preserved, because a
sampled cell's root-to-leaf path must keep one alteration-drawing opportunity per real
division. Call `MutationLoadDynamics.prune_tree!` first if you want dead lineages
gone.

The root's `edge_mutations` is `nothing`, since the founder has no incoming edge — see
[`founder_mutations`](@ref) if you want those mutations translated into truncal
alterations.
"""
function CopyNumberEvolution.PhyloTree(root::BinaryNode{NonMarkovCell})
    par = Union{Int,Nothing}[]
    kids = Vector{Int}[]
    bt = Union{Float64,Nothing}[]
    divs = Union{Int,Nothing}[]
    muts = Union{Int,Nothing}[]
    sids = Union{Int,Nothing}[]

    # Iterative preorder. Push the right child first so the left is popped first and
    # children end up in left-to-right order.
    stack = Tuple{BinaryNode{NonMarkovCell},Union{Int,Nothing}}[(root, nothing)]
    while !isempty(stack)
        nd, p = pop!(stack)
        push!(par, p)
        push!(kids, Int[])
        i = length(par)
        p === nothing || push!(kids[p], i)
        cell = nd.data
        push!(bt, Float64(cell.birthtime))
        push!(divs, p === nothing ? nothing : 1)
        push!(muts, p === nothing ? nothing : Int(cell.mutations))
        push!(sids, Int(cell.id))
        nd.right === nothing || push!(stack, (nd.right, i))
        nd.left === nothing || push!(stack, (nd.left, i))
    end

    nodes = [PhyloNode(i, par[i], kids[i], bt[i], divs[i], muts[i], nothing, sids[i])
             for i in eachindex(par)]
    return PhyloTree(nodes)
end

"""
    founder_mutations(root::BinaryNode{NonMarkovCell}) -> Int

Mutations the founder cell acquired at its own birth. See
`CopyNumberEvolution.founder_mutations`.
"""
CopyNumberEvolution.founder_mutations(root::BinaryNode{NonMarkovCell}) =
    Int(root.data.mutations)

end # module
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS, with the conversion tests reported as skipped unless
MutationLoadDynamics is available.

Then verify the extension really loads, with the simulator dev'd in:

```bash
julia --project=. -e 'using Pkg; Pkg.develop(path = "../MutationLoadDynamics.jl")'
julia --project=. -e 'using Pkg; Pkg.test()'
```
Expected: PASS, with the conversion tests running. **Revert the develop afterwards**
so `Project.toml` keeps MutationLoadDynamics out of `[deps]`:
`julia --project=. -e 'using Pkg; Pkg.rm("MutationLoadDynamics")'`, then confirm with
`git diff Project.toml` that nothing changed.

- [ ] **Step 6: Commit**

```bash
git add ext src/tree.jl src/CopyNumberEvolution.jl test/test_ext.jl test/runtests.jl
git commit -m "feat: add MutationLoadDynamics package extension for tree conversion"
```

---

### Task 15: Cross-cutting scientific tests

Implements spec §15 items 7 and the end-to-end coverage the per-file suites cannot
give. These are the tests that guard the claims the science rests on.

**Files:**
- Create: `test/test_science.jl`
- Modify: `test/fixtures.jl` (add `induced_subtree`, `deep_lineage`)
- Modify: `test/runtests.jl` (include)

**Interfaces:**
- Consumes: everything.
- Produces: no new package API — a test-only `induced_subtree` helper in
  `test/fixtures.jl`.

- [ ] **Step 1: Add the fixture helpers**

Append to `test/fixtures.jl`:

```julia
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
```

- [ ] **Step 2: Write the failing tests**

`test/test_science.jl`:

```julia
@testset "scientific properties" begin
    A() = toy_assembly(nchrom = 3, len = 2000)

    @testset "sampling commutes exactly under rng_mode = :per_node" begin
        full = binary_lineage(4)                     # 31 nodes, 16 leaves
        keep = [17, 20, 25, 31]
        sub = induced_subtree(full, keep)
        @test nnodes(sub) < nnodes(full)
        # pruning, not collapsing: a kept leaf's path length is unchanged
        for l in keep
            sid = node(full, l).source_id
            @test depth(sub, node_by_source_id(sub, sid)) == depth(full, l)
        end

        m = CNAModel(rate = PerDivision(1.5),
                     extent = ExtentMixture(p_chromosome = 0.2, p_arm = 0.2),
                     initial = TruncalCNAs(2),
                     wgd = NoWGD())
        rf = simulate_cnas(full, A(), m; seed = 777, rng_mode = :per_node)
        rs = simulate_cnas(sub, A(), m; seed = 777, rng_mode = :per_node)

        # the root state is the same
        @test profile(rs, treeroot(sub)) == profile(rf, treeroot(full))
        # and every kept leaf's profile is identical, not merely similar
        for l in keep
            sid = node(full, l).source_id
            @test profile(rs, node_by_source_id(sub, sid)) == profile(rf, l)
        end
        # so are the events on every retained edge
        for i in 1:nnodes(sub)
            sid = node(sub, i).source_id
            j = node_by_source_id(full, sid)
            @test [e.event for e in events_on(rs, i)] == [e.event for e in events_on(rf, j)]
        end
    end

    @testset "sampling commutes distributionally under rng_mode = :global" begin
        # With one shared stream the draws cannot line up edge-for-edge, so the claim
        # is about distributions. The number of alterations on the retained edges is a
        # sum of independent Poissons with a known mean, so we can bound the tolerance
        # analytically instead of guessing one.
        full = binary_lineage(3)
        keep = [9, 12, 15]
        sub = induced_subtree(full, keep)
        λ = 1.25
        m = CNAModel(rate = PerDivision(λ), viability = AllowAll())

        retained_sids = Set(node(sub, i).source_id for i in 1:nnodes(sub) if !isroot(sub, i))
        divisions = sum(node(full, node_by_source_id(full, s)).edge_divisions
                        for s in retained_sids)
        expected = λ * divisions

        N = 400
        full_total = 0
        sub_total = 0
        for s in 1:N
            rf = simulate_cnas(full, A(), m; seed = s, retain_internal = false)
            rs = simulate_cnas(sub, A(), m; seed = s, retain_internal = false)
            full_total += count(e -> node(full, e.node).source_id in retained_sids, rf.events)
            sub_total += length(rs.events)
        end
        se = sqrt(expected / N)              # standard error of the mean
        @test abs(full_total / N - expected) < 4 * se
        @test abs(sub_total / N - expected) < 4 * se
    end

    @testset "an end-to-end run on hg38, invariants throughout" begin
        a = hg38(:female)
        tree = binary_lineage(7)             # 255 nodes, 128 leaves
        model = CNAModel(rate = PerDivision(0.6),
                         target = CNWeighted(1.0),
                         extent = ExtentMixture(p_chromosome = 0.05, p_arm = 0.15),
                         kind = GainLoss(0.55),
                         wgd = ScheduledWGD(mrca(tree, [128, 191]) => 1),
                         viability = RejectAndRedraw(),
                         initial = TruncalCNAs(3))
        res = simulate_cnas(tree, a, model; seed = 20260904)

        @test nevents(res) > 100
        for i in 1:nnodes(tree)
            @test check_invariants(profile(res, i))
        end
        @test replay(res) == [profile(res, i) for i in 1:nnodes(tree)]

        # the scheduled doubling reached exactly its descendants
        wgd_node = mrca(tree, [128, 191])
        below = Set(descendant_leaves(tree, wgd_node))
        wgd_events = [e for e in res.events if e.event isa WholeGenomeDoubling]
        @test length(wgd_events) == 1
        @test only(wgd_events).node == wgd_node

        # every scale class the model enables actually occurs
        scales = Set(e.event.scale for e in res.events if e.event isa SegmentalCNA)
        @test :focal in scales
        @test :arm in scales

        # total copy number equals the sum over haplotypes, everywhere, on real data
        p = profile(res, first(leaves(tree)))
        for c in 1:nchromosomes(a)
            tot = total_cn(p, c)
            for pos in (1, chromlength(a, c) ÷ 3, chromlength(a, c))
                @test cn_at(tot, pos) ==
                      sum(cn_at(p.segments[s], pos) for s in slots_of(a, c); init = 0)
            end
        end

        # and the whole output pipeline runs
        grid = BinGrid(a, 500_000)
        mat = CNMatrix(res, grid)
        @test ncells(mat) == length(leaves(tree))
        @test mat.total == mat.allele[1] .+ mat.allele[2]
        dir = mktempdir()
        write_medicc2(joinpath(dir, "cells.tsv"), mat)
        write_profiles(joinpath(dir, "truth.tsv"), res)
        write_events(joinpath(dir, "events.tsv"), res)
        write_bins(joinpath(dir, "bins.tsv"), grid)
        write_newick(joinpath(dir, "tree.nwk"), tree; branchlength = :divisions)
        for f in ("cells.tsv", "truth.tsv", "events.tsv", "bins.tsv", "tree.nwk")
            @test filesize(joinpath(dir, f)) > 0
        end
        rt = read_newick(joinpath(dir, "tree.nwk"); branchlength = :divisions)
        @test nnodes(rt) == nnodes(tree)
    end

    @testset "per-division and per-time diverge when timing is non-exponential" begin
        # The package's reason to exist: on a tree where division count and elapsed
        # time are decoupled, the two rate rules put alterations in different places.
        n = 31
        parents = Vector{Union{Int,Nothing}}(undef, n)
        bt = Vector{Float64}(undef, n)
        parents[1] = nothing
        bt[1] = 0.0
        for i in 2:n
            p = i ÷ 2
            parents[i] = p
            # left children divide fast, right children slowly: divisions and time
            # carry different information
            bt[i] = bt[p] + (iseven(i) ? 0.1 : 5.0)
        end
        t = phylotree(parents; birthtimes = bt,
                      edge_divisions = vcat(nothing, fill(1, n - 1)),
                      source_ids = collect(1:n))

        perdiv = CNAModel(rate = PerDivision(2.0), viability = AllowAll())
        pertime = CNAModel(rate = PerTime(2.0 / 2.55), viability = AllowAll())

        fast_leaf = 16      # all-left path: few elapsed time units
        slow_leaf = 31      # all-right path: many elapsed time units
        N = 200
        div_fast = div_slow = time_fast = time_slow = 0
        for s in 1:N
            rd = simulate_cnas(t, A(), perdiv; seed = s)
            rt = simulate_cnas(t, A(), pertime; seed = s)
            div_fast += length(nonroot_events_to(rd, fast_leaf))
            div_slow += length(nonroot_events_to(rd, slow_leaf))
            time_fast += length(nonroot_events_to(rt, fast_leaf))
            time_slow += length(nonroot_events_to(rt, slow_leaf))
        end
        # per division: both leaves are 4 divisions deep, so the burdens match
        @test div_fast / N ≈ div_slow / N rtol = 0.15
        # per time: the slow path accumulates far more
        @test time_slow / N > 3 * (time_fast / N)
    end
end
```

Add this helper at the top of `test/test_science.jl`, before the `@testset`:

```julia
# Alterations on the path from the root to `leaf`, excluding truncal (root) events.
function nonroot_events_to(res, leaf::Int)
    path = Set(ancestors(res.tree, leaf))
    delete!(path, treeroot(res.tree))
    return [e for e in res.events if e.node in path]
end
```

- [ ] **Step 3: Run the tests to verify they fail**

Add `include("test_science.jl")` to `test/runtests.jl`, then run:
`julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL with `UndefVarError: induced_subtree not defined` until Step 1's
fixtures are in place, then pass.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS.

If the `:global` distributional test fails, do **not** widen the tolerance without
checking the mean first: `abs(observed - expected)` should be well under `4·se`. A
systematically wrong mean means the rate rule or the traversal is at fault, which is
exactly what this test exists to catch.

- [ ] **Step 5: Commit**

```bash
git add test/test_science.jl test/fixtures.jl test/runtests.jl
git commit -m "test: add sampling-commutes, end-to-end and rate-rule-divergence tests"
```

---

### Task 16: Comprehensive documentation

Implements spec §14's docstring requirement plus the manual. Docstrings were written
alongside each task; this task adds the manual, the README, doctest verification, and
a test that nothing exported is undocumented.

**Files:**
- Create: `docs/Project.toml`
- Create: `docs/make.jl`
- Create: `docs/src/index.md`
- Create: `docs/src/concepts.md`
- Create: `docs/src/trees.md`
- Create: `docs/src/model.md`
- Create: `docs/src/output.md`
- Create: `docs/src/interop.md`
- Create: `docs/src/limitations.md`
- Create: `docs/src/api.md`
- Create: `README.md`
- Create: `test/test_docs.jl`
- Modify: `.github/workflows/CI.yml` (docs job)
- Modify: `test/runtests.jl` (include)

**Interfaces:**
- Consumes: the whole public API.
- Produces: no new API.

- [ ] **Step 1: Write the failing docstring-coverage test**

`test/test_docs.jl`:

```julia
@testset "documentation" begin
    @testset "every exported symbol is documented" begin
        undocumented = String[]
        for name in names(CopyNumberEvolution)
            name === :CopyNumberEvolution && continue
            txt = string(Base.Docs.doc(Base.Docs.Binding(CopyNumberEvolution, name)))
            (isempty(strip(txt)) || occursin("No documentation found", txt)) &&
                push!(undocumented, string(name))
        end
        @test isempty(undocumented)
    end

    @testset "the module itself is documented" begin
        txt = string(Base.Docs.doc(CopyNumberEvolution))
        @test occursin("copy-number", lowercase(txt))
    end

    @testset "the manual mentions every exported symbol at least once" begin
        # api.md uses @autodocs, so this checks the narrative pages, not the reference.
        srcdir = joinpath(@__DIR__, "..", "docs", "src")
        @test isdir(srcdir)
        prose = join([read(joinpath(srcdir, f), String)
                      for f in readdir(srcdir) if endswith(f, ".md")], "\n")
        missing_ = [string(n) for n in names(CopyNumberEvolution)
                    if n !== :CopyNumberEvolution && !occursin(string(n), prose)]
        @test isempty(missing_)
    end
end
```

- [ ] **Step 2: Run the test to verify it fails**

Add `include("test_docs.jl")` to `test/runtests.jl`, then run:
`julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `docs/src` does not exist yet, and any symbol whose docstring was
skipped shows up in the first testset.

- [ ] **Step 3: Write `docs/Project.toml`**

```toml
[deps]
CopyNumberEvolution = "a7fad464-da36-487c-8111-789b46084000"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Documenter = "e30172f5-a6a5-5a46-863b-614d45cd2de4"

[compat]
Documenter = "1"
```

- [ ] **Step 4: Write `docs/make.jl`**

```julia
using Documenter
using CopyNumberEvolution

DocMeta.setdocmeta!(CopyNumberEvolution, :DocTestSetup,
                    :(using CopyNumberEvolution); recursive = true)

makedocs(
    sitename = "CopyNumberEvolution.jl",
    modules = [CopyNumberEvolution],
    authors = "Alexander Stein",
    pages = [
        "Home" => "index.md",
        "Concepts" => "concepts.md",
        "Input: trees" => "trees.md",
        "The alteration model" => "model.md",
        "Output" => "output.md",
        "MEDICC2 interoperability" => "interop.md",
        "Limitations and open questions" => "limitations.md",
        "API reference" => "api.md",
    ],
    doctest = true,
    checkdocs = :exports,
    format = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true"),
)
```

- [ ] **Step 5: Write `docs/src/index.md`**

````markdown
# CopyNumberEvolution.jl

Forward simulation of somatic copy-number alterations along a cell-lineage tree.

Give it a tree — simulated by
[`MutationLoadDynamics.jl`](https://github.com/alexander-stein/MutationLoadDynamics.jl)
or read from a newick file — and it draws copy-number alterations along the edges from
a diploid or given root state, returning the allele-specific profile of **every** node
together with a **complete log** of the events that produced it. Profiles then project
onto a fixed bin grid, which is the form real low-coverage single-cell DNA data arrives
in and the form every inference method consumes.

This package is an **observation model**. It does not infer trees, estimate parameters,
or compute distances between profiles.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/alexander-stein/CopyNumberEvolution.jl")
```

Dependencies are deliberately minimal — `Random`, `Distributions`, `StatsBase` — so
that depending on this package for its types stays cheap.
`MutationLoadDynamics.jl` is a *weak* dependency: load it alongside and the tree
converter appears; leave it out and nothing is missing but the converter.

## Quickstart

```julia
using CopyNumberEvolution

assembly = hg38(:female)
tree = read_newick("lineage.nwk"; branchlength = :divisions)

model = CNAModel(
    rate      = PerDivision(0.5),                              # Poisson(λ · divisions)
    target    = CNWeighted(1.0),                               # gains beget gains
    extent    = ExtentMixture(p_chromosome = 0.05, p_arm = 0.15),
    kind      = GainLoss(0.6),                                 # 60% gains
    wgd       = ScheduledWGD(mrca(tree, [12, 34]) => 1),       # one doubling, exactly there
    viability = RejectAndRedraw(min_total_cn = 1),
    initial   = TruncalCNAs(4),                                # four clonal alterations
)

res = simulate_cnas(tree, assembly, model; seed = 20260904)

# Every node's profile, and every event
profile(res, treeroot(tree))          # the truncal state
leaf_profiles(res)                    # the observable cells
res.events                            # the complete record
res.rejections                        # how often viability bit

# Project and export
grid = BinGrid(assembly, 500_000)
mat  = CNMatrix(res, grid)
write_medicc2("cells.tsv", mat)       # input for the reference method
write_profiles("truth.tsv", res)      # the ground truth to compare against
write_events("events.tsv", res)
```

## Where to go next

- [Concepts](concepts.md) — how a genome is represented, and the four things that can
  happen to it.
- [Input: trees](trees.md) — `PhyloTree`, the three meanings of a newick branch
  length, and the `MutationLoadDynamics.jl` bridge.
- [The alteration model](model.md) — every injectable component, and the scientific
  consequences of the defaults.
- [Output](output.md) — profiles, the event log, replay, and bin projection.
- [MEDICC2 interoperability](interop.md) — exporting for the reference method, and the
  comparison that makes possible.
- [Limitations and open questions](limitations.md) — what is deliberately out of
  scope, and what is still undecided.

Detailed notes on MEDICC2, the reference method this model is calibrated against, live
in `literature/MEDICC2.md` in the repository.
````

- [ ] **Step 6: Write `docs/src/concepts.md`**

````markdown
# Concepts

## A genome is a set of haplotype segmentations

A [`GenomeAssembly`](@ref) lists chromosomes — name, length, centromere — and, from a
sex mode, the number of copies of each in the normal karyotype. That determines the
**slot layout**: one slot per haplotype of per chromosome.

```julia
a = hg38(:female)                        # hg19(:male) works the same way
nchromosomes(a)                          # 24 — chrY is in the table with ploidy 0
nslots(a)                                # 46: 22 autosome pairs + 2 X
chromname(a, 1), chromlength(a, 1)       # "chr1", 248956422
centromere(a, 1)                         # 1-based inclusive
ploidy(a, chromindex(a, "chrY"))         # 0 — present, but no slots
slot(a, 1, 2)                            # linear index of chr1 haplotype 2
slots_of(a, chromindex(a, "chrX"))       # both X slots
slot_chrom(a, 3), slot_haplotype(a, 3)   # and back again
arms(a, 1)                               # p and q, flanking the centromere
```

[`hg38`](@ref) and [`hg19`](@ref) ship real chromosome lengths and centromere positions,
transcribed from UCSC and asserted in the test suite against a checked-in reference
file. Real lengths matter: segment sizes then have physical meaning, which is required
both for comparison with real data and for a realistic size distribution. A non-human
or deliberately synthetic genome is a [`GenomeAssembly`](@ref) built from your own
[`ChromosomeSpec`](@ref) list, with the ploidy vector given explicitly if the sex-mode
rules do not apply.

Sex mode **is** the slot layout rather than a separate switch. A male assembly has one
`chrX` and one `chrY` slot; hemizygosity is therefore representable rather than
special-cased. `chrY` stays in a female table with ploidy 0 so chromosome indices
compare across sexes, and [`eligible_chromosomes`](@ref) is what the proposal
distributions actually draw from. [`autosomes`](@ref) names the 22 non-sex chromosomes,
which is what MEDICC2 export uses by default.

A [`CNProfile`](@ref) holds one segmentation per slot. Each is a sorted, gapless,
non-overlapping list of [`Segment`](@ref)s tiling the chromosome — a piecewise-constant
step function of position:

```julia
p = diploid(hg38(:female))
slot_segments(p, 1, 1)        # [Segment(1:248956422, cn=1)]
nsegments(p)                  # 46 — one per slot, to start
```

`cn` counts copies of *that haplotype*, so `cn = 3` on one slot means three copies of
that parental chromosome. This is exactly MEDICC2's `cn_a`/`cn_b` convention.

### Why a segmentation, and why slot-indexed

A dictionary keyed by copy number cannot represent two disjoint segments that happen
to share one — an independent gain at 3p and at 3q, both landing at copy number 3.
That is the common case after a handful of alterations, and such a representation
would silently merge or lose one of them. A segmentation represents everything it can
plus everything it cannot, and it is **exactly what real copy-number callers emit**, so
the same type ingests real and simulated data.

Storage is a `Vector` indexed by slot rather than a `Dict` keyed by
`(chromosome, haplotype)` for a specific reason: `Dict` iteration order is unspecified
in Julia, so any pass over haplotypes that consumes the random number generator —
length-weighted or copy-number-conditioned target choice, doubling, viability checks —
would be irreproducible across Julia versions and insertion histories. Reproducibility
under a fixed seed is a tested property here, not an aspiration.

### Canonical form

Every segmentation is kept canonical: sorted, gapless, covering `1:length`, with **no
two adjacent segments sharing a copy number**, and no negative copy number. That makes
`==` meaningful — two profiles are equal exactly when their segmentations are — and it
is what [`check_invariants`](@ref) enforces:

```julia
check_invariants(p)            # true, or throws naming the slot and segment
canonicalize!(segs)            # merge adjacent equal-cn neighbours after an edit
```

The failure mode of this package is not a crash, it is a plausible-looking wrong
number. A silently non-canonical segmentation would poison everything downstream, so
the invariant checker is called liberally throughout the test suite.

Position lookups are `segment_index(segs, pos)` and [`cn_at`](@ref), both binary
searches.

## Allele-specific is the representation; total is a view

Simulation is *always* allele-specific. Total copy number is derived:

```julia
total_cn(p, 1)      # summed over chr1's slots, re-canonicalised
total_cn(p)         # every chromosome
mean_cn(segs, len)  # length-weighted mean of one slot
```

Two simulation paths would be twice the code and twice the test burden, and they could
disagree. So the "total copy number mode" affects output and comparison only, never
the model. This is also the right call scientifically: everything interesting — loss
of heterozygosity, mirrored subclonal allelic imbalance, parallel evolution on distinct
haplotypes — is invisible in total copy number.

## Four things can happen

The event taxonomy follows MEDICC2's evolutionary model, which is the closest published
statement of which copy-number aberrations matter.

| Event | Effect | Extent |
|:---|:---|:---|
| Segmental gain | `+δ` over a run | any contiguous interval within one chromosome, one haplotype |
| Segmental loss | `−δ` over a run | as above |
| Loss of heterozygosity | a loss reaching `cn = 0` | not a separate type — see below |
| Whole-genome doubling | every slot | the whole genome, crossing chromosome boundaries |

[`CNAEvent`](@ref) has two concrete subtypes, [`SegmentalCNA`](@ref) and
[`WholeGenomeDoubling`](@ref), and [`event_span`](@ref) gives a segmental event's
length in base pairs. Everything is applied through one method:

```julia
apply!(profile, event)
```

Keeping that interface narrow is deliberate: a karyotype backend could later be added
behind it without touching tree traversal or the output layer.

Three consequences of the table worth stating plainly:

**Arbitrary length means arm- and chromosome-level events are not a different kind of
thing.** A focal event and a whole-chromosome event are the same event type at
different extents. They still get their own probabilities in
[`ExtentMixture`](@ref), because no continuous length distribution produces them at a
realistic rate — but nothing downstream treats them specially. The `scale` field
records which class of draw produced an event so it can be tallied afterwards.

**Loss of heterozygosity needs no separate event type.** It is simply a loss that
reaches copy number 0.

**Copy number 0 is absorbing.** A segment at 0 is *absent DNA* and can never be
regained. A gain spanning a run that contains zeroed sub-segments raises the non-zero
parts and leaves the zeros at zero:

```julia
p = diploid(toy)
apply!(p, SegmentalCNA(1, 1, 21, 40, -1, :focal))       # 21:40 -> 0
apply!(p, SegmentalCNA(1, 1, 1, 100, +1, :chromosome))  # 21:40 stays 0
```

This is unconditionally true and is **not** part of the viability policy. Viability is
about states that are *unobserved*; absorption is about states that are *impossible*.
So absorption is never switchable off, and a proposal it blocks costs no rejection
attempt.
````

- [ ] **Step 7: Write `docs/src/trees.md`**

````markdown
# Input: trees

## `PhyloTree`

[`PhyloTree`](@ref) is flat-vector storage of [`PhyloNode`](@ref)s where
`nodes[i].id == i`. Dense integer ids are cheap to serialise, hash and compare, and
tree-similarity metrics want integer leaf labels anyway.

The package owns this type rather than reusing a simulator's node type for three
reasons: it is what a newick file parses into; it is what an *inferred* tree — one with
no simulator behind it — needs to be; and it keeps the simulator out of the dependency
chain of anything that only needs the types.

```julia
t = phylotree([nothing, 1, 1, 2, 2];
              birthtimes     = [0.0, 1.0, 1.5, 2.0, 2.5],
              edge_divisions = [nothing, 1, 1, 1, 1],
              edge_mutations = [nothing, 4, 7, 2, 0],
              labels         = [nothing, nothing, nothing, "a", "b"],
              source_ids     = [10, 20, 30, 40, 50])

nnodes(t); treeroot(t); leaves(t); internal_nodes(t)
node(t, 4); parentof(t, 4); childrenof(t, 2)
isleaf(t, 4); isroot(t, 1); depth(t, 4)
preorder(t); postorder(t); ancestors(t, 4); descendant_leaves(t, 2)
```

Arity is arbitrary and unary nodes are allowed, because newick input and inferred trees
are not guaranteed binary and a pruned lineage tree contains unary nodes. Every
traversal is iterative, so a 10⁵-deep caterpillar tree is safe.

### Naming an edge

Three ways, all resolving to a dense id — needed because you have to be able to say
"the doubling happens on *that* edge":

```julia
node_by_source_id(t, 40)     # the upstream numbering, e.g. a simulator's cell id
node_by_label(t, "a")        # a newick taxon name
mrca(t, [4, 5])              # the most recent common ancestor of a leaf set
```

`source_id` matters because leaf sampling upstream leaves a *sparse* subset of the
original ids, at which point index-equals-id would break. Dense ids keep flat storage;
`source_id` keeps "cell 40" meaningful. [`cellname`](@ref) is the stable output name — a
node's label if it has one, else `cell_<id>` — and both newick writing and MEDICC2
export go through it, so a cell's `sample_id` in an exported matrix always matches its
leaf label in the exported tree.

## The three meanings of a branch length

A newick file carries **one** number per edge. This package needs both real time and a
division count, so [`read_newick`](@ref) and [`parse_newick`](@ref) make you say which
it is. There is deliberately no default.

| `branchlength` | field populated | rate rules enabled |
|:---|:---|:---|
| `:divisions` | `edge_divisions` | [`PerDivision`](@ref) |
| `:mutations` | `edge_mutations` | [`FromEdgeMutations`](@ref) |
| `:time` | `birthtime` by cumulative sum, plus `edge_divisions = 1` | [`PerTime`](@ref), and [`PerDivision`](@ref) treating each edge as one division |

```julia
t = read_newick("lineage.nwk"; branchlength = :divisions)
write_newick("out.nwk", t; branchlength = :divisions)     # explicit on write too
newick_string(t; branchlength = :time, labels = :source_id)
```

The three [`PhyloNode`](@ref) edge fields are held separately and are honestly
`nothing` when unknown, rather than defaulting to a sentinel. Each rate rule requires
exactly one of them and throws a named error, naming the fix, when it is absent — so
`PerTime` on a divisions-encoded tree fails loudly instead of inventing times.
[`edge_time`](@ref) behaves the same way, and throws on the root, which has no incoming
edge.

Parsing handles named and unnamed internal nodes, arbitrary arity, single-quoted
labels, `[...]` comments, and missing branch lengths. Under `:divisions` and
`:mutations` a fractional branch length is rounded with a warning, since a fractional
count usually means the file should have been read as `:time`.

## The `MutationLoadDynamics.jl` bridge

Load both packages and a converter appears:

```julia
using CopyNumberEvolution, MutationLoadDynamics

tree = PhyloTree(root)               # root::BinaryNode{NonMarkovCell}
founder_mutations(root)              # the founder's own mutations
```

Mapping: `birthtime` from the cell; `edge_divisions = 1`, because one lineage-tree edge
is exactly one division; `edge_mutations = cell.mutations`; `source_id = cell.id`. All
three rate rules therefore work on a converted tree.

This is a **package extension**. Loading `CopyNumberEvolution` alone gives the
copy-number modeller with no simulator anywhere in the dependency chain — which
matters, because the downstream inference package has to be installable and runnable
against real patient data, and a hard dependency here would make a simulator
transitively required to analyse a clinical dataset.

The founder has no incoming edge, so its own mutations cannot be attributed to one and
the root's `edge_mutations` is `nothing`. If you want them translated, feed
[`founder_mutations`](@ref) into `TruncalCNAs`.

## Sampling happens upstream

Leaf sampling is `MutationLoadDynamics.jl`'s operation, not this package's. The
converter takes whatever tree it is handed, full or sampled.

The property that matters is that the upstream sampler **prunes but never collapses**.
Every division ancestral to a sampled cell remains a node, so a sampled cell's
root-to-leaf path has the same number of edges as in the full tree — one
alteration-drawing opportunity per real division. That is what makes alterations
simulated on a sampled tree identical in distribution to alterations simulated on the
full tree and then subset. Collapsing unary nodes would turn divisional depth into "a
count of bifurcations that survived sampling", a property of the sample rather than of
the cell.

Two consequences. The root of a sampled tree is the original **founder**, not the most
recent common ancestor of the sample — which is why truncal state is expressed as the
root's [`InitialState`](@ref) rather than as an MRCA special case. And with
`rng_mode = :per_node` the commuting property holds *exactly*, not just
distributionally; see [The alteration model](model.md).
````

- [ ] **Step 8: Write `docs/src/model.md`**

````markdown
# The alteration model

[`CNAModel`](@ref) bundles seven independently replaceable components. Each of the
three draws also accepts a plain function. That is deliberate: downstream inference has
to *fit* these parameters, so every one must be addressable and cheap to vary.

```julia
model = CNAModel(
    rate      = PerDivision(1.0),
    target    = UniformChromosome(),
    extent    = ExtentMixture(),
    kind      = GainLoss(0.5),
    wgd       = NoWGD(),
    viability = RejectAndRedraw(),
    initial   = Diploid(),
)
```

## Rate: the question the package exists to ask

Whether alterations accrue **per division** or **per unit real time** is the
Markov-versus-non-Markov question transposed from point mutations to copy number.
Under exponential division timing the two are hard to tell apart; under non-exponential
timing they are not, because division count and elapsed time decouple. The input tree
carries both, so both are computable — and the difference between them is the signal,
not a nuisance.

| [`CNARate`](@ref) | draws | needs |
|:---|:---|:---|
| [`PerDivision`](@ref)`(λ)` | `Poisson(λ · edge_divisions)` | `edge_divisions` |
| [`PerTime`](@ref)`(μ)` | `Poisson(μ · Δt)` | birthtimes |
| [`FromEdgeMutations`](@ref)`()` | exactly `edge_mutations` | `edge_mutations` |
| [`FromEdgeMutations`](@ref)`(p)` | `Binomial(edge_mutations, p)` | `edge_mutations` |
| [`CustomRate`](@ref)`(f)` | `f(tree, node, rng)` | whatever `f` uses |

`PerDivision` is the default. Note it *multiplies* by the division count rather than
applying per edge: a simulated lineage edge is one division, so it reduces to
`Poisson(λ)` there, but a newick edge can stand for many.

`FromEdgeMutations()` is exact identity — one recorded mutation becomes one alteration,
with no extra randomness — which is the natural reading when the upstream simulator's
per-edge mutation count was recorded with this translation in mind. `p < 1` thins
binomially, keeping a fitness-coupled mutation process intact while lowering the
realised alteration rate.

[`n_cnas`](@ref) is the single method to implement for a new rule.

## Target: which chromosome and haplotype

[`TargetDraw`](@ref) rules are callables `(profile, rng) -> (chrom, haplotype)`;
[`draw_target`](@ref) dispatches.

- [`UniformChromosome`](@ref) — uniform over eligible chromosomes. Note: over
  *chromosomes*, not base pairs.
- [`LengthWeighted`](@ref) — proportional to chromosome length, i.e. uniform over base
  pairs.
- [`CNWeighted`](@ref)`(β)` — slot weight proportional to mean copy number to the power
  `β`, so already-gained material keeps being gained. This is the rule that conditions
  the proposal on the mother cell's copy-number state, and it is what produces
  realistic ploidy skew.

Chromosomes with zero ploidy are never eligible. Haplotype choice is uniform over a
chromosome's slots unless a rule says otherwise, which is what lets mirrored allelic
imbalance arise on its own rather than being injected. A slot at mean copy number 0
gets weight 0 under `CNWeighted` for *every* `β`, including 0, consistent with copy
number 0 being absorbing.

## Extent: where, and how far

[`ExtentDraw`](@ref) rules are callables
`(profile, chrom, haplotype, rng) -> (start, stop, scale)`; [`draw_extent`](@ref)
dispatches. [`ExtentMixture`](@ref) is the one provided:

```julia
ExtentMixture(p_chromosome = 0.05, p_arm = 0.15)     # p_focal = 0.80
ExtentMixture()                                      # focal only
ExtentMixture(lengthdist = LogUniform(1e6, 5e7))     # narrower focal sizes
```

Whole-arm and whole-chromosome events dominate real karyotypes and cannot be produced
at a realistic rate by any continuous length distribution, so they get their own
probabilities. Setting both to zero ignores large-scale events entirely without
changing the code path — which is how early analyses will typically run.

Focal events draw a length, then a uniform start, then **truncate** at the chromosome
end. Truncation rather than rejection, matching MEDICC2, where an event terminates at
the boundary. Arm events pick the p or q arm with equal probability and need the
assembly's centromere positions.

## Kind: gain or loss

[`KindDraw`](@ref) rules are callables
`(profile, chrom, haplotype, start, stop, rng) -> delta`; [`draw_kind`](@ref)
dispatches. [`GainLoss`](@ref)`(p_gain, delta = 1)` returns `+delta` or `−delta`.

## Whole-genome doubling

A [`WGDPolicy`](@ref) resolves, once, before traversal, into a schedule mapping a node
id to the number of doublings on its incoming edge — that is [`prepare_wgd`](@ref).
The traversal then only *reads* the schedule, so no `Dict` iteration ever consumes the
random number generator.

```julia
NoWGD()
ScheduledWGD(node_by_source_id(tree, 42) => 1)        # exactly there
ScheduledWGD(mrca(tree, metastatic_leaves) => 1)      # a subclonal doubling
ScheduledWGD(Dict(7 => 2))                            # two successive doublings
ExactlyNWGD(1)                                        # one doubling, position random
RateWGD(PerDivision(0.01))                            # a rate; RateWGD(0.01) is the same
RateWGD(PerTime(0.005))
```

[`ScheduledWGD`](@ref) is what makes statements like *"one whole-genome doubling in the
dataset, on the edge into node i"* expressible. The root cannot be scheduled — it has
no incoming edge; set the root's state with `TruncalCNAs` or `Given` instead.
[`ExactlyNWGD`](@ref) fixes the count but not the position;
[`RateWGD`](@ref) reuses the [`CNARate`](@ref) machinery, so the
per-division-versus-per-time question applies to doublings on the same footing as to
segmental events.

### `:multiply` versus `:increment`

[`wgd_mode`](@ref) reports which arithmetic a policy uses.

- `:multiply` — every copy number doubles (`cn → 2cn`). **The default.** This is what
  tetraploidization means, and it preserves zeros for free.
- `:increment` — every *non-zero* copy number gains one (`cn → cn + 1`). This is
  MEDICC2's own definition.

They coincide while all copy numbers are 0 or 1 and diverge as soon as any segment is
2 or more — exactly the interesting regime, which is why the choice is explicit rather
than assumed. Use `:increment` when you want profiles generated on MEDICC2's own terms;
otherwise a `:multiply` event will be scored by MEDICC2 as one `+1` doubling plus extra
gains.

### Ordering on an edge is defined, not incidental

Doublings are applied **before** that edge's segmental alterations, and the realised
order is recorded in the event log. So "gained then doubled" is always recoverable from
the output rather than reconstructed by guesswork.

## Viability

An alteration can drive a region, a whole chromosome, or the single X of a male
karyotype to copy number 0. Real data contains no cells with whole-chromosome
nullisomy, so an unconstrained process generates profiles that could not exist.

```julia
AllowAll()                                        # no constraint
RejectAndRedraw(min_total_cn = 1, max_attempts = 100)
AllRules([RejectAndRedraw(), my_rule])
```

[`RejectAndRedraw`](@ref) rejects any alteration that would push a chromosome's total
copy number, summed over haplotypes, below `min_total_cn` at any position it touches.
Rejected proposals are **redrawn** — not skipped, and not fatal to the cell. Killing a
cell was rejected as an option outright: the tree is an *input* with its own
birth–death history, so killing here would contradict the given tree and silently
change the sampled population size. Exceeding `max_attempts` throws rather than
skipping, because a silent skip would bias the realised rate with no signal.

The whole interface is one method, [`violation`](@ref), returning a reason symbol or
`nothing`; [`isviable`](@ref) is the predicate on top and [`max_attempts`](@ref) tells
the traversal how many redraws to allow. A new class of impossible state is therefore a
new struct and no change to the traversal.

!!! warning "This conditions the model"
    Rejection sampling makes the alteration process **conditional on viability**, so
    the realised distribution is not the proposal distribution. That is a modelling
    assumption, and any analysis built on these simulations has to state it. The
    rejection tally is returned in the result so the size of the effect is visible
    rather than hidden.

!!! note "The default forbids all homozygous deletions"
    `min_total_cn = 1` forbids biallelic loss **of any size**, not merely
    whole-chromosome nullisomy. Real tumours do contain small homozygous deletions, so
    set `min_total_cn = 0` — or use [`AllowAll`](@ref) — if focal biallelic loss should
    be permitted. See [Limitations and open questions](limitations.md).

## Root state

Because the upstream sampler keeps the founder as the root, the root is generally *not*
the most recent common ancestor of the sample. Truncal state is therefore the root's
[`InitialState`](@ref), not an MRCA special case.

- [`Diploid`](@ref) — a normal karyotype, sex from the assembly. The default.
- [`Given`](@ref)`(profile)` — start from a called ancestral or consensus profile. It
  must be on the same assembly and sex, and is copied rather than mutated.
- [`TruncalCNAs`](@ref)`(n)` — apply `n` alterations from diploid, drawn from the same
  model and **logged against the root**, so they appear in the event record like any
  others.

## Alterations here are neutral by construction

The tree is an input that already encodes whatever selection produced it. This package
therefore never kills or reweights a cell — which is also why "mark the cell dead" was
rejected as a viability option. If you want selection on copy number, it belongs in the
process that generates the tree.
````

- [ ] **Step 9: Write `docs/src/output.md`**

````markdown
# Output

## Running the simulation

```julia
res = simulate_cnas(tree, assembly, model; seed = 1)
res = simulate_cnas(tree, assembly, model; rng = Xoshiro(1))
res = simulate_cnas(tree, assembly, model; seed = 1, retain_internal = false)
res = simulate_cnas(tree, assembly, model; seed = 1, rng_mode = :per_node)
```

[`simulate_cnas`](@ref) descends depth-first, carrying one profile down the current
path and copying it per child, so memory scales with tree **depth** rather than tree
size. The descent is iterative: a 10⁵-deep tree cannot overflow the stack.

### `rng_mode`

`:global` threads one stream through the whole traversal — the straightforward choice.

`:per_node` gives each edge its own stream, derived from the seed and the node's
`source_id` (falling back to its dense id). An edge then draws *identically* no matter
which other edges exist, which means simulating on a sampled tree yields **exactly** the
same alterations as simulating on the full tree and subsetting — not merely the same
distribution. That property is what the sampling analysis rests on, so being able to
test it exactly rather than statistically is worth the option. It requires a seed. The
doubling schedule is still drawn from the global stream.

## What comes back

[`CNAEvolution`](@ref) holds the tree, the assembly, the model, the profiles, the
complete event log, the rejection tally and the seed.

```julia
profile(res, i)            # one node's profile; throws if it was not retained
leaf_profiles(res)         # the observable cells, in leaves(res.tree) order
res.profiles               # indexed by node id; entries may be nothing
```

Profiles are retained for **every** node by default — tips and internal nodes alike —
because the ancestral states are the ground truth an inference method's reconstruction
gets compared against. `retain_internal = false` keeps only the leaves for very large
trees.

### The event log is the primitive

```julia
res.events                 # Vector{LoggedEvent}, complete, always
nevents(res)
events_on(res, i)          # alterations on the edge into node i
events_below(res, i)       # everything strictly below node i
replay(res)                # reconstruct every profile from the log
```

A [`LoggedEvent`](@ref) records the edge (by its child), the event's `order` within
that edge's sequence, and the [`CNAEvent`](@ref) itself. Truncal alterations are logged
against the root. Events are stored in preorder of node and then by `order`, so
[`replay`](@ref) is a single forward pass.

The log is complete regardless of `retain_internal`, and the retained profiles are a
cache: `replay` reconstructs every profile from the root state plus the log. Nothing is
lost by running lean, and the equality of the two is a tested property.

### Rejections

```julia
res.rejections             # Dict{Symbol,Int}, keyed by the constraint broken
rejection_count(res)       # the total
```

A non-empty tally means the realised alteration distribution is conditioned on
viability. Report it.

## Bin projection

Real low-coverage single-cell data is called in fixed bins, and every inference method
consumes a cells × bins integer matrix.

```julia
grid = BinGrid(assembly, 500_000)        # 500 kb, the DLP+ default
nbins(grid)
bins_of(grid, 1)                         # chr1's column range
```

Bins run consecutively within each chromosome, chromosomes appear in assembly order,
and chromosomes with zero ploidy contribute none — so a female grid has no `chrY`
columns. The final bin of each chromosome is short whenever the bin size does not
divide the chromosome length; that is documented rather than padded.

```julia
tot, alleles = project(profile(res, i), grid)
mat = CNMatrix(res, grid)                            # leaves by default
truth = CNMatrix(res, grid; cells = internal_nodes(res.tree))
mat = CNMatrix(res, grid; rule = AreaWeightedMean())
mat = CNMatrix(res, grid; allele = false)
ncells(mat); max_cn(mat)
```

[`CNMatrix`](@ref) carries `cells` (the node ids of its rows), `names` (from
[`cellname`](@ref), so rows match an exported newick's leaf labels), `total`, and
`allele` — one matrix per haplotype index. A [`Bin`](@ref) on a chromosome whose ploidy
is below a haplotype index gets 0 on that track, which is how a male `chrX` exports
with `cn_b = 0`.

`total` is defined as the **sum of the haplotype tracks**, not as a reprojection of the
summed segmentation. Projection is non-linear, so the two differ at straddling bins;
defining it additively guarantees `total == A + B`, which every consumer of an
allele-specific matrix relies on.

### The straddling-bin rule

A bin containing a breakpoint has no unambiguously correct copy number. Two
[`BinRule`](@ref)s are provided:

- [`LengthWeightedMajority`](@ref) — the copy number covering the most base pairs in
  the bin, ties going to the lower value. The default.
- [`AreaWeightedMean`](@ref) — the length-weighted mean, rounded with halves up.

Either introduces a small systematic difference from a real caller's own binning.
Which one best matches a given caller is **unresolved** — see
[Limitations and open questions](limitations.md).

## Files

```julia
write_profiles("truth.tsv", res)                # node_id, name, chrom, haplotype, start, stop, cn
write_events("events.tsv", res)                 # node_id, name, order, type, chrom, haplotype, start, stop, delta, scale, mode
write_bins("bins.tsv", grid)                    # bin_index, chrom, start, stop
write_medicc2("cells.tsv", mat)                 # see the interoperability page
write_newick("tree.nwk", tree; branchlength = :divisions)
```

All writers accept an `IO` as well as a path. Tables are tab-separated with one header
line and `NA` where a field does not apply — a doubling has no chromosome or span, a
segmental alteration has no mode. [`write_profiles`](@ref) writes every node by
default, so the file contains the ancestral truth as well as the tips;
[`write_events`](@ref) writes the whole log; [`write_bins`](@ref) says what a matrix's
columns mean.

These are deliberately plain tables. The on-disk layout of a whole dataset directory is
specified downstream, where the many-methods-many-datasets requirement lives, and
inventing a second format here would guarantee a mismatch. Newick is the interchange
format for trees.
````

- [ ] **Step 10: Write `docs/src/interop.md`**

````markdown
# MEDICC2 interoperability

[MEDICC2](https://bitbucket.org/schwarzlab/medicc2) (Kaufmann et al., *Genome Biology*
2022) is the reference method for inferring phylogenies and ancestral genomes from
allele-specific copy-number profiles. Its evolutionary model is the closest published
statement of which copy-number aberrations matter, and it is what this package's event
taxonomy is calibrated against. Detailed notes live in `literature/MEDICC2.md` in the
repository.

## Exporting

```julia
grid = BinGrid(hg38(:female), 500_000)
mat  = CNMatrix(res, grid)                     # the leaves — the observable cells
write_medicc2("cells.tsv", mat)
write_medicc2("cells_xy.tsv", mat; include_xy = true)
write_medicc2("cells.tsv", mat; normal_name = "normal")
```

[`write_medicc2`](@ref) emits the long TSV MEDICC2 expects —
`sample_id, chrom, start, end, cn_a, cn_b` — and handles four conversions:

- **0-based half-open (BED) coordinates.** Internal segments are 1-based inclusive;
  the conversion happens at this boundary and nowhere else.
- **Identical segmentation across every sample**, which MEDICC2 requires and which
  holds by construction, since every cell is projected onto the same grid.
- **A reference sample** named by `normal_name` (default `"diploid"`) with
  `cn_a = cn_b = 1` in every bin — the root MEDICC2 measures distances from.
- **Autosomes only** by default, matching MEDICC2's own bulk analyses. With
  `include_xy = true`, a hemizygous chromosome exports as `cn_b = 0`.

It warns if any copy number exceeds 8, which MEDICC2's alphabet cannot represent.

`sample_id` values come from [`cellname`](@ref), the same function
[`write_newick`](@ref) uses, so the matrix and an exported tree always agree on
identity.

Then, from the shell:

```bash
medicc2 cells.tsv medicc2_out --events -j 8
```

## Only the tips go in

The internal-node profiles are the **ground truth** you compare MEDICC2's ancestral
reconstruction *against*. They must never be fed in as input. Write them separately:

```julia
write_medicc2("cells.tsv", mat)                          # input: leaves only
write_profiles("truth_profiles.tsv", res)                # truth: every node
write_events("truth_events.tsv", res)                    # truth: every event
```

Reading MEDICC2's output back — `_final_cn_profiles.tsv`, `_final_tree.new` — is not
this package's job. That belongs downstream, with the estimators, which must not depend
on a simulator.

## What the comparison buys you

MEDICC2's branch length is the number of copy-number events, inferred by parsimony.
This package knows the **true** per-edge event count and the true ancestral profile at
every internal node. That makes two things directly measurable that the published work
could only bound:

**How far parsimony falls short of the truth.** MEDICC2's minimum-event distance is
established as a *lower bound* on the true number of events, and the authors note that
a doubling followed by many chromosome losses inflates the count because each loss is
counted separately. With ground truth in hand, the size and shape of that gap is
measurable rather than assumed.

**How the gap depends on the timing model.** Parsimony has no notion of divisions or
elapsed time. Simulating the same tree under [`PerDivision`](@ref) and
[`PerTime`](@ref) and scoring both with MEDICC2 asks whether the inferred branch
lengths can distinguish them at all — which is the same question the whole
non-Markovian programme asks, transposed to copy number.

MEDICC2's own finding that copy-number trunks are short where point-mutation trunks are
long is a statement about the same gap, seen from the data side.

## Matching MEDICC2's assumptions

Two settings make simulated data directly comparable on MEDICC2's own terms:

```julia
# MEDICC2 defines a doubling as +1 on every non-zero segment, not ×2
wgd = ScheduledWGD(i => 1; mode = :increment)

# and its alphabet caps at 8, so keep copy numbers in range
max_cn(mat) <= 8 || @warn "profiles exceed what MEDICC2 can represent"
```

Simulated profiles are also **phased by construction**, which makes them a clean test
set for MEDICC2's evolutionary phasing: the truth is known.

Three things MEDICC2 does not model, and neither does this package: copy-number-neutral
events, breakage–fusion–bridge cycles, and chromothripsis. MEDICC2's contiguity stress
test is the published argument that omitting them is tolerable for tree inference.
````

- [ ] **Step 11: Write `docs/src/limitations.md`**

````markdown
# Limitations and open questions

## Deliberately out of scope

**General structural variants and a karyotype backend.** Ordered lists of genomic
fragments with orientation would express translocations, inversions and derivative
chromosomes. That is strictly more expressive and strictly more work, and low-coverage
single-cell data cannot resolve most of it anyway — the observable is a bin-level
copy-number profile. The idea is kept, not discarded: all state changes go through
`apply!(profile, event)`, so a karyotype backend can be added behind that interface
without touching tree traversal or the output layer. MEDICC2 makes the same omission.

**Copy-number-neutral events**, breakage–fusion–bridge cycles, chromothripsis. Same
reasoning.

**Read-depth noise.** Real data passes through read counts, GC bias and a hidden Markov
model caller before becoming an integer matrix. That belongs behind a clean boundary —
a `simulate_readcounts(cnmatrix, depth)` in its own file — and is deliberately not
entangled with alteration simulation. Not implemented yet.

**Tree inference, distances, estimators, approximate Bayesian computation.** A
different package. Keeping them apart is what lets the estimators run on real patient
data with no simulator in their dependency chain.

**Leaf sampling.** `MutationLoadDynamics.jl`'s operation. See
[Input: trees](trees.md).

**Selection on copy number.** The tree is an input that already encodes whatever
selection produced it, so alterations here are neutral by construction.

## Open questions

These are genuinely undecided. The defaults are documented placeholders, not settled
positions.

### The straddling-bin rule

A bin containing a breakpoint has no unambiguously correct copy number.
[`LengthWeightedMajority`](@ref) is the default and [`AreaWeightedMean`](@ref) is
available, but which better matches a given caller's behaviour is unresolved, and
either introduces a small systematic difference from a real caller's own binning.

### The copy-number ceiling

MEDICC2 cannot represent copy numbers above 8. This package imposes no internal cap and
warns only on export ([`max_cn`](@ref) tells you where you stand). Whether a cap
belongs in the simulation itself — and if so, what should happen to a gain that would
breach it — is unsettled.

### `min_total_cn` forbids all homozygous deletions

[`RejectAndRedraw`](@ref)'s default `min_total_cn = 1` forbids biallelic loss **of any
size**, not merely whole-chromosome nullisomy — which was the concern that motivated
the rule. Real tumours do contain small homozygous deletions. Until a size- or
region-aware rule exists, the options are `min_total_cn = 0`, [`AllowAll`](@ref), or a
custom [`ViabilityRule`](@ref); the one-method [`violation`](@ref) interface exists
precisely so that further classes of impossible state are cheap to add.

### `PerDivision` on an inferred tree

An internal edge of an inferred phylogeny is not one cell division, so
`edge_divisions` read from a `:divisions` newick file is an estimate rather than a
count. The rate rule is correct given the field; what the field *means* for real data
is a study-level question.

## Things that are true and easy to forget

- **Rejection sampling conditions the model.** The realised alteration distribution is
  not the proposal distribution whenever `rejection_count(res) > 0`. State it.
- **`:multiply` and `:increment` doublings are not the same event** above copy number 1,
  and MEDICC2 assumes the latter.
- **The root of a sampled tree is the founder**, not the sample's most recent common
  ancestor.
- **No real patient data belongs in this repository.** Fixtures are synthetic and
  small; `data/` is in `.gitignore`. Real data lives only in study repositories, and
  only where the relevant data agreement permits.
````

- [ ] **Step 12: Write `docs/src/api.md`**

````markdown
# API reference

```@index
```

```@autodocs
Modules = [CopyNumberEvolution]
Order = [:module, :type, :function]
```
````

- [ ] **Step 13: Write `README.md`**

````markdown
# CopyNumberEvolution.jl

Forward simulation of somatic copy-number alterations along a cell-lineage tree.

Give it a tree — simulated by `MutationLoadDynamics.jl` or read from a newick file — and
it draws copy-number alterations along the edges from a diploid or given root state,
returning the allele-specific profile of **every** node plus a **complete log** of the
events that produced it, then projects those profiles onto a fixed bin grid.

This is an **observation model**: it does not infer trees, estimate parameters, or
compute distances between profiles.

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/alexander-stein/CopyNumberEvolution.jl")
```

## Quickstart

```julia
using CopyNumberEvolution

assembly = hg38(:female)
tree = read_newick("lineage.nwk"; branchlength = :divisions)

model = CNAModel(
    rate      = PerDivision(0.5),
    target    = CNWeighted(1.0),
    extent    = ExtentMixture(p_chromosome = 0.05, p_arm = 0.15),
    kind      = GainLoss(0.6),
    wgd       = ScheduledWGD(mrca(tree, [12, 34]) => 1),
    initial   = TruncalCNAs(4),
)

res  = simulate_cnas(tree, assembly, model; seed = 20260904)
grid = BinGrid(assembly, 500_000)
mat  = CNMatrix(res, grid)

write_medicc2("cells.tsv", mat)     # input for the reference method
write_profiles("truth.tsv", res)    # the ground truth to compare against
write_events("events.tsv", res)
```

## What it models

Four event types, following MEDICC2's evolutionary model: segmental **gains** and
**losses** of arbitrary extent on a named haplotype, **loss of heterozygosity** as a
loss reaching zero, and **whole-genome doubling**. Whole-arm and whole-chromosome
events get their own probabilities, because no continuous length distribution produces
them at a realistic rate. Copy number 0 is **absorbing** — absent DNA is never
regained.

Alteration counts per edge come from an injectable rule: **per division**, **per unit
real time**, or **from the edge's mutation count**. That choice is the point of the
package — whether copy-number alterations accrue per division or per unit time is the
Markov-versus-non-Markov question transposed from point mutations, and the input tree
carries both quantities so both are computable.

Whole-genome doublings can be pinned to an **exact edge**
(`ScheduledWGD(mrca(tree, leaves) => 1)`), fixed in number with random placement, or
drawn from a rate.

## Scope

| | |
|:---|:---|
| Here | tree types, copy-number profiles, the alteration process, bin projection, MEDICC2 export |
| `MutationLoadDynamics.jl` | lineage-tree simulation and leaf sampling (a **weak** dependency) |
| Downstream | tree inference, distances, estimators — these must never depend on a simulator |
| Study repositories | parameter grids, file naming, figures |

Out of scope for v1, deliberately: a karyotype backend for general structural variants
(kept behind `apply!` so it can be added later), copy-number-neutral events,
breakage–fusion–bridge cycles, read-depth noise.

## Documentation

Build locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate(); include("docs/make.jl")'
```

The manual covers the concepts, the input layer, every model component and the
scientific consequences of its defaults, the output and projection rules, MEDICC2
interoperability, and the open questions. `literature/MEDICC2.md` holds detailed notes
on the reference method.

## Open questions

Documented rather than silently settled — see the manual's Limitations page:

- The **straddling-bin projection rule** is unresolved; the default is a placeholder.
- The **copy-number ceiling** (MEDICC2 caps at 8) is warned about on export but not
  capped in simulation.
- `RejectAndRedraw`'s default `min_total_cn = 1` forbids homozygous deletions of *any*
  size, not just whole-chromosome nullisomy.
- **Rejection sampling conditions the model**: the realised alteration distribution is
  not the proposal distribution whenever rejections occur.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The package-extension tests skip themselves unless `MutationLoadDynamics.jl` is
available:

```bash
julia --project=. -e 'using Pkg; Pkg.develop(path = "../MutationLoadDynamics.jl")'
```

**No real patient data belongs in this repository.** Fixtures are synthetic and small;
`data/` is in `.gitignore`.

## License

See [LICENSE](LICENSE).
````

- [ ] **Step 14: Add the docs job to `.github/workflows/CI.yml`**

Append to the `jobs:` mapping:

```yaml
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: '1'
      - uses: julia-actions/cache@v2
      - name: Install docs dependencies
        run: |
          julia --project=docs -e '
            using Pkg
            Pkg.develop(PackageSpec(path = pwd()))
            Pkg.instantiate()'
      - name: Build and run doctests
        run: julia --project=docs docs/make.jl
```

- [ ] **Step 15: Run the docs build and the doctests**

Run:
```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path = pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```
Expected: the site builds under `docs/build/`, with **no doctest failures and no
`checkdocs` warnings about missing docstrings**. Doctest output is compared verbatim,
so if a printed value differs, fix the docstring to match reality rather than the
reverse — the doctest is telling you what the code actually does.

- [ ] **Step 16: Run the full test suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS, including all three documentation testsets.

- [ ] **Step 17: Commit**

```bash
git add docs README.md test/test_docs.jl test/runtests.jl .github/workflows/CI.yml
git commit -m "docs: add Documenter manual, README and docstring coverage test"
```

---

## Self-review

Run after the plan is complete, before execution.

### Spec coverage

| Spec section | Task |
|:---|:---|
| §2 dependency rule, weak dep | 1, 14 |
| §3 settled decisions | throughout; recorded in `docs/src/limitations.md` (16) |
| §4.1 `PhyloTree`, dense ids, `source_id` | 5 |
| §4.2 newick, three branch-length meanings | 6 |
| §4.3 sampling is upstream | 14 (converter), 15 (`induced_subtree` fixture), 16 (docs) |
| §4.4 package extension | 14 |
| §5 `GenomeAssembly`, hg38/hg19, sex modes, provenance test | 2 |
| §6.1 segmentation, slot-indexed | 3 |
| §6.2 canonical form, invariant checker | 3 |
| §6.3 absorbing zero | 4 |
| §6.4 allele-specific with total as a view | 3 |
| §6.5 `apply!` as the sole interface | 4 |
| §7.1 `CNAModel` | 11 |
| §7.2 rate rules | 7 |
| §7.3 target draws | 8 |
| §7.4 extent, arm and chromosome events | 8 |
| §7.5 kind | 8 |
| §7.6 event types | 4 |
| §8 WGD policies, exact placement, modes, ordering | 9, 11 |
| §9 viability, reason-keyed rejections | 10 |
| §10 initial state | 11 |
| §11 traversal, result, replay | 11 |
| §12 bin grid, projection rules | 12 |
| §13 tabular writers, MEDICC2 export | 13 |
| §14 layout, dependency cap, `data/` ignored | 1 |
| §15 tests 1–9 | 3, 4, 6, 11, 12, 15 |
| §15 tests 10–17 (added by the design) | 4, 7, 9, 10, 11, 12, 13 |
| §16 out of scope | 16 (`docs/src/limitations.md`, README) |
| §17 open problems | 12 (`BinRule` docstring), 10 (`RejectAndRedraw` docstring), 16 |

No spec section is unimplemented.

### Deviations recorded

The five additions to the spec are listed under "Additions to the spec made by this
plan" at the top. Two smaller ones, noted here for completeness:

- Tree accessors are `treeroot`, `parentof`, `childrenof` rather than `root`, `parent`,
  `children`, to avoid clashing with `Base.parent` and with likely names in consumer
  packages (Task 5).
- `founder_mutations` is added to the extension surface, so a converted tree's
  founder-cell mutations are not silently dropped (Task 14).

### Verification checklist

- [ ] Every task's tests were run and failed before its implementation, then passed
      after.
- [ ] `julia --project=. -e 'using Pkg; Pkg.test()'` passes with no warnings.
- [ ] `git diff Project.toml` shows `MutationLoadDynamics` only under `[weakdeps]`.
- [ ] `julia --project=docs docs/make.jl` builds with no doctest failures and no
      `checkdocs` warnings.
- [ ] `data/` is in `.gitignore` and no data file is tracked: `git ls-files data/` is
      empty.
