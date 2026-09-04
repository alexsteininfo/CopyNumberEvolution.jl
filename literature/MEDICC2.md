# MEDICC2 — whole-genome doubling aware copy-number phylogenies

**Reference.** Kaufmann TL, Petkovic M, Watkins TBK, Colliver EC, Laskina S, Thapa N,
Minussi DC, Navin N, Swanton C, Van Loo P, Haase K, Tarabichi M, Schwarz RF.
*MEDICC2: whole-genome doubling aware copy-number phylogenies for cancer evolution.*
**Genome Biology** 23:241 (2022). doi:[10.1186/s13059-022-02794-9](https://doi.org/10.1186/s13059-022-02794-9)
· PDF in this folder (`MEDICC2.pdf`, 27 pp.) · PMC: [PMC9661799](https://pmc.ncbi.nlm.nih.gov/articles/PMC9661799/)

**Code.** Python 3, GPLv3, <https://bitbucket.org/schwarzlab/medicc2>.
Predecessor: MEDICC (Schwarz et al. 2014, ref. [6] in the paper) — same MED idea,
context-free-grammar implementation, no WGD.

**Why it is in this repo.** MEDICC2 is the reference *inference* method for
allele-specific copy-number phylogenies, and its evolutionary model is the closest
published statement of "which copy-number aberrations matter". This note is the
source for the CNA event taxonomy `CopyNumberEvolution.jl` simulates and for the
output format that has to be consumable by MEDICC2. See
[§13 Implications](#13-implications-for-copynumberevolutionjl).

---

## 1. What the method does

Given allele-specific (haplotype-specific) integer copy-number profiles for a set of
taxa — bulk samples, subclones, or single cells — MEDICC2:

1. computes pairwise **minimum-event distances** (MED) between all profiles,
2. infers a tree topology by **neighbor joining** on that distance matrix,
3. reconstructs **ancestral copy-number profiles** at internal nodes so the total
   number of events along the tree is minimal (this also sets the branch lengths),
4. **extracts individual copy-number events** and assigns each to the branch where it
   occurred (including WGD), and
5. reports a per-patient summary plus a tree-plus-CN-profile plot.

It deliberately drops three simplifications that most competing methods make:

| Simplification MEDICC2 rejects | Consequence of rejecting it |
|---|---|
| Infinite-sites assumption | Multiple hits per locus, back-mutation and **parallel / convergent evolution** are representable |
| Independence of adjacent loci | One event may span an arbitrary run of adjacent segments |
| Minimum-spanning-tree shortcut | Solves the (NP-complete) Steiner-tree problem — ancestral, unsampled states are inferred, not assumed to be observed |

---

## 2. The evolutionary model — the event taxonomy

This section is the one that matters most for simulation.

A genome is a vector of non-negative integer copy numbers `(k_1 … k_n)`, `0 ≤ k ≤ 8`,
one entry per genomic segment, **per haplotype**. Both haplotypes are concatenated
into one sequence; chromosome boundaries and the haplotype boundary are marked by a
separator symbol `X`. Alphabet: `Σ = {0,…,8,X}`.

**Four elementary event types** (each is one event, i.e. costs 1):

| Event | Effect | Extent | Stops at chromosome boundary? |
|---|---|---|---|
| **Segmental gain** | `+1` on a run of segments | any contiguous run within one chromosome, one haplotype | yes |
| **Segmental loss** | `−1` on a run of segments (not reaching 0) | as above | yes |
| **LOH** | a loss that takes a haplotype's copy number to **0** | as above | yes |
| **WGD** | `+1` to **every non-zero segment in the genome**, both haplotypes | whole genome | **no** — crosses chromosomes, leaves `X` unchanged |

Deliberate modelling choices inside that table:

- **Arbitrary event length.** A gain or loss covers any contiguous run of segments.
  A focal event and a whole-chromosome event are the *same* event type at different
  lengths, and both cost exactly 1. There is no separate "arm-level" primitive —
  arm- and chromosome-level events fall out as long runs.
- **Copy number is capped at 8.** `maxcn` is an alphabet size, not a soft prior.
- **Zero is absorbing.** A segment at copy number 0 can never be re-gained: lost
  DNA is physically gone. Subsequent gains and losses *skip over* zero-copy segments
  as if that DNA were absent (so a run of segments can be interrupted by a zero and
  still be a single event). This is the physical constraint that makes the MED
  asymmetric.
- **WGD is `+1` on non-zero segments, not `×2`.** For profiles with copy number ≤ 1
  everywhere the two coincide; they differ once any segment is already ≥ 2. `×2` is
  available as the expert flag `--wgd-x2`.
- **Event ordering is forced.** LOH must be resolved first (lost material cannot be
  regained); WGD must precede segmental gains and losses (so material gained by a WGD
  can subsequently be deleted). Among gains and losses after that, the MED is
  order-oblivious. The composed transducer is therefore
  `T = T_LOH ∘ T_WGD ∘ T_L ∘ T_G`.
- **Tetraploidization → near-triploidy is emergent**, not a primitive: one WGD event
  followed by several independent whole-chromosome losses. Each such loss is counted
  separately, so the MED can *overestimate* the event count for that trajectory —
  the authors flag this explicitly.

**Explicitly out of the model:** copy-number-neutral events (inversions, balanced
translocations), breakage–fusion–bridge cycles, chromothripsis, and any
non-contiguity with respect to the reference. MEDICC2 approximates complex events by
combinations of the four elementary operations, and the paper shows this stays
accurate — but the model has no karyotype and no segment order beyond the reference.

---

## 3. Minimum-event distance (MED) and the FST machinery

- Profiles are unweighted finite-state **acceptors**; events are weighted finite-state
  **transducers** over the **tropical semiring** (weights add along a path, the path
  score is the minimum over all paths).
- One-step FSTs `T1_LOH`, `T1_G`, `T1_L`, `T1_WGD` each allow a single-step change per
  position (`1→2`, `2→3`, … but not `1→3`). They are self-composed `|Σ|−1` times (LOH)
  or `|Σ|−2` times (gain/loss/WGD) to reach the full event FSTs.
- `MED(x,y) = ShortestDistance(x ∘ T ∘ y)`; **linear in the number of segments**.
- The MED is **asymmetric** (because of the zero-absorbing constraint). The symmetric
  distance used for tree building goes *through a common ancestor*, minimised over all
  possible ancestors: `S(x,y) = ShortestDistance(x ∘ T⁻¹ ∘ T ∘ y)`.
- The composition `T⁻¹ ∘ T` is huge if built explicitly. MEDICC2's main engineering
  contribution is **lazy (on-demand) composition** with a shortest-first queue,
  expanding only the visited path — roughly an order of magnitude faster.
- Overall complexity: linear in segments, **quadratic in taxa**. Parallelised by
  splitting the `N×N` matrix into `p²+p` groups of size `p` (smallest prime with
  `p² ≥ N`), giving a `p²+p` speed-up. Thousands of cells on 32 cores in under an hour.

---

## 4. WGD detection

Compute the MED to a diploid normal `d` twice, with and without the WGD transducer:

```
s_i = MED_noWGD(d, t_i) − MED_WGD(d, t_i)      # WGD evidence score, s_i ≥ 0
```

`s_i ≥ 1` means a WGD makes the history more parsimonious. Substituting an `n`-step
WGD transducer tests for multiple WGDs: `MED_1WGD > MED_2WGD = MED_3WGD` ⇒ two WGDs.

Performance on 2778 PCAWG tumours against the PCAWG consensus "gold standard"
(818 WGD-positive, 1960 negative): **96.0 %** accuracy single-shot, all 110 errors
false negatives; **98.8 %** with 100 bootstrap replicates and a "WGD if ≥ 5 % of
replicates show one" rule (6 FP, 27 FN). 27 tumours called as two successive WGDs.
Errors are enriched for PCAWG's own "WGD uncertain" label. AUC 0.99 over the
bootstrap threshold, i.e. the threshold barely matters.

---

## 5. Evolutionary phasing

Allele-specific callers report **major/minor** copy number per segment, which is not
phased: the assignment of the two values to the two parental haplotypes is unknown
and may flip between segments.

- Preferred fix when several samples per patient exist: **Refphase** multi-sample
  reference phasing (Watkins et al.).
- Single-sample fallback: **evolutionary phasing** — choose the assignment minimising
  the summed MED from a diploid reference over both haplotypes (minimum-evolution
  criterion). Encoded as a linear "phasing FST" `P` with two mirrored transitions per
  segment, so each of the `2ⁿ` paths is one phasing choice; composed with
  `u = (d ∘ T)↓` on both sides and solved by shortest path. Exact, and linear-time —
  the MEDICC1 version used a weighted context-free grammar and was far slower.

Phasing matters because unphased profiles hide **mirrored subclonal allelic imbalance
(MSAI)** and haplotype-resolved parallel evolution, which are exactly what a
non-infinite-sites model can exploit.

---

## 6. Tree inference, ancestral reconstruction, event extraction

- **Topology:** neighbor joining on the symmetric distance matrix. (A user-supplied
  Newick tree can be passed with `--tree` to skip this step — useful for scoring a
  known/simulated topology.)
- **Ancestral states:** chosen to minimise total events along the tree; this
  *defines* the branch lengths (branch length = number of copy-number events).
- **Event extraction:** post-order traversal, relative copy-number changes per
  segment, each event counted once at the branch where it appears — so parallel
  events on different branches are counted separately and a shared event is not
  double-counted across descendants. Events can be intersected with a BED file of
  regions of interest (genes, chromosome arms); an event counts as hitting a region
  at ≥ 90 % overlap. Caveat from the docs: the reconstructed event *path* is minimal
  and deterministic but **not unique**.

---

## 7. Robustness / resampling

Standard phylogenetic bootstrap assumes independent sites, which copy-number profiles
violate. Two alternatives:

1. **Chromosome-wise bootstrap** — draw whole chromosomes with replacement. Safe
   (gains/losses stop at chromosome boundaries; WGD ignores chromosome identity) so
   no false events are introduced, but coarse.
2. **Segment-wise natural jackknife** — draw `N` of `N` segments with replacement and
   discard duplicates, i.e. drop ≈ `1/e` of segments. Faster, less faithful.

Branch support is the percentage of replicates recovering the split.

---

## 8. Benchmarking and validation

**Their simulator (`Methods → Simulating genome evolution`)** — note it is deliberately
*not* a copy-number-level simulator, to avoid favouring MEDICC2's own model:

- Random topology by randomly joining sample labels, rooted at the diploid.
- Events per branch `~ Poisson(λ = Δt · S · μ)`, `Δt = 1`, genome size `S = 440`
  segments = `2 × 22` chromosomes (two haplotype sets) × 10 equal-size segments,
  `μ ∈ {0.01, 0.025, 0.05}`.
- Evolution simulated **at the genome level**: whole-chromosome gains/losses, focal
  losses, insertions, **breakage–fusion–bridge (BFB)**, **WGD**, and copy-number-neutral
  balanced/unbalanced translocations and inversions. Copy-number profiles are then
  *read off* the simulated genome by counting copies per segment. Unbalanced
  translocations propagate: move a chr1 segment onto chr2, then gain chr2, and the
  chr1 segment is gained too.
- All events equally likely except BFB (10 % of the others) and WGD
  (`0.000125` for large single-cell-like trees; `0`/`0.0125`/`0.065` for
  no/low/high-WGD medium trees).
- Homozygous deletions suppressed by forbidding losses on haplotype 2 — which also
  lowers the effective loss:gain ratio.
- Grid: 25 trees per cell; large scenario `N ∈ {5,10,15,20,50,100,250,500}`,
  medium scenario `N ∈ {5,10,15,20}` × three WGD levels.
- Separate contiguity-stress runs push the translocation+inversion : gain+loss ratio
  up to 25.

**Accuracy of the MED itself** (profiles simulated *under* the MEDICC2 model:
Poisson `μ = 10`, 5 % WGD / 47.5 % gain / 47.5 % loss, uniform start over non-zero
positions, geometric lengths `p = 0.2`, 5 chromosomes × 10 segments): MED recovered
exactly, linear in sequence length, and forms a **lower bound** on the true event
count. Euclidean distance does not (`r² = 0.17`).

**Tree accuracy:** generalized Robinson–Foulds (`TreeDist`), plus plain RF (`ape`)
and quartet distance as controls. MEDICC2 beats Euclidean/Manhattan + NJ,
Euclidean/Manhattan + minimum evolution (`fastme.bal`), **MEDALT** (MST-based) and
**Sitka** (breakpoint/perfect-phylogeny) at every mutation rate and tree size, most
clearly when WGDs are present. It is slower than MEDALT and Sitka but tractable to
~1000 taxa.

**Event validation against orthogonal SVs (PCAWG, 2778 tumours):** take MEDICC2
events whose start or end lies within 100 kb of an SV breakpoint, ask how often the
*other* end matches too, versus two null models (next segment boundary; random
segment boundary on the same chromosome). MEDICC2 event boundaries agree with SV
breakpoints more often than segment boundaries do, for every SV size class
(100 kb / 1 Mb / 10 Mb) and type.

**Real data.**
- *Gundem et al.* 10 metastatic prostate cancers (Battenberg + Refphase): exact
  topological concordance (RF = 0) with the published SNV clone phylogenies in 6/10,
  partial in 4/10 (the mismatches are the polyclonal-seeding cases where a single
  dominant-subclone profile cannot represent the sample). 4 WGDs found (2 clonal,
  1 subclonal, 1 terminal); in A31 the subclonal WGD sits at the ancestor of all
  metastases (`s = 22`), followed by an 8p gain and multiple chromosome losses;
  parallel LOH on chr6 and chr13 plus several MSAI events correctly split across
  branches. Arm-level `#gains − #losses` correlates with the Davoli OG–TSG score
  (`r = 0.59`); gene-level `r = 0.09` overall, `0.25` for the top 100 genes.
- *Minussi et al.* triple-negative breast cancer single cells, TN1 (1100 cells) and
  TN2 (1023 cells), run directly on allele-specific profiles with **no clustering and
  no consensus profiles**: recovers the published super-/subclone structure (TN2 all
  superclones, TN1 merges two), and detects the truncal WGDs without exome data.
- Recurring biological finding: **truncal branches are short in SCNA trees but long in
  SNV trees** (A31: 11/54 SCNA events vs 2056/3700 SNVs; TN1 42/164, TN2 71/238) —
  few founder SCNAs, many founder SNVs, i.e. most copy-number diversification happens
  *after* the MRCA. Root-to-leaf lengths still correlate between the two data types
  (Spearman ρ = 0.57).

---

## 9. Software: interface and file formats

```
medicc2  input_file  output_directory  [options]
```

**Input (TSV, the recommended format).** Columns:

| column | meaning |
|---|---|
| `sample_id` | taxon name (cell / sample / subclone) |
| `chrom` | chromosome |
| `start`, `end` | BED convention — 0-based, half-open |
| `cn_a`, `cn_b` | integer allele-specific copy numbers (diploid normal = 1 and 1) |

Requirements: **identical segmentation across every sample** (same breakpoints);
integer copy numbers only; ≥ 2 non-diploid samples; allele-specific input should be
phased (Refphase, or accept major/minor). A reference row set named `diploid` by
default (`--normal-name`) supplies the root. Gaps in the segmentation are silently
treated as contiguous. FASTA input is also supported (`--input-type f`) with a
description file.

**Selected options.**

| flag | effect |
|---|---|
| `-a, --input-allele-columns` | rename the CN columns (default `cn_a,cn_b`) |
| `--total-copy-numbers` | run on total CN instead of allele-specific |
| `-n, --normal-name` | id of the reference/root sample (default `diploid`) |
| `--maxcn` | copy-number cap (default and maximum 8) |
| `--no-wgd` | disable the WGD transducer |
| `--wgd-x2` | treat WGD as `×2` instead of `+1` on non-zero segments |
| `--tree` | supply a Newick topology and skip tree inference |
| `-s, --topology-only` | topology without ancestral reconstruction |
| `--events` | run event extraction (adds WGD status to the summary) |
| `--regions-bed`, `--chromosomes-bed` | regions of interest / chromosome definitions |
| `--bootstrap-method {chr-wise,segment-wise}`, `--bootstrap-nr` | resampling |
| `--filter-segment-length` | drop segments below a bp length |
| `-j, --n-cores` | parallelism |
| `-x, --exclude-samples`, `-p, --prefix`, `--plot`, `-v/-vv` | misc |

**Output files** (prefixed): `_final_tree.new` / `.xml` / `.png`,
`_pairwise_distances.tsv` (`N×N` symmetric), `_final_cn_profiles.tsv` (input taxa
**plus inferred ancestors**), `_branch_lengths.tsv`, `_summary.tsv`,
`_cn_profiles.pdf`, and with `--events`: `_copynumber_events_df.tsv`,
`_events_overlap.tsv`.

Install: `conda install -c bioconda -c conda-forge medicc2` (pip needs
`openfst=1.8.2` and **Cython 0.29** — 3.0 is incompatible). Unix only.

**Documented limitations of the tool:** non-unique event paths; own phasing only
adequate for single samples; noisy small segments distort distances (mitigate with
larger bins / `--filter-segment-length`); severe taxon imbalance at 100s–1000s of
samples can produce wrong trees; segmentation gaps treated as contiguous; multiple
WGDs at one node can break event reconstruction when using total copy numbers.

---

## 10. Stated limitations of the method

1. Only copy-number-**changing** events are modelled; everything is assumed
   contiguous with respect to the reference.
2. No explicit representation of BFB cycles or chromothripsis — approximated by
   elementary operations.
3. WGD followed by many chromosome losses inflates the event count (each loss is a
   separate event), so the MED may overestimate the true number of events for that
   trajectory.
4. It does not call copy number; results inherit the resolution and noise of the
   input caller.
5. Symmetric MED is NP-hard, and the pipeline is quadratic in taxa.

---

## 11. Glossary

- **SCNA** — somatic copy-number alteration.
- **CIN** — chromosomal instability.
- **MED / MED-WGD** — minimum-event distance, with or without WGD events.
- **WGD** — whole-genome doubling.
- **LOH** — loss of heterozygosity; here specifically a haplotype's copy number
  reaching 0.
- **MSAI** — mirrored subclonal allelic imbalance: opposite haplotypes are the
  higher-copy one in different samples of the same tumour.
- **Homoplasy** — the same state arising independently on separate branches
  (parallel/convergent evolution); the thing infinite-sites models cannot express.
- **FST / FSA** — weighted finite-state transducer / unweighted finite-state acceptor.
- **GRF** — generalized Robinson–Foulds tree distance.
- **OG–TSG score** — Davoli et al. per-gene / per-arm oncogene vs tumour-suppressor
  enrichment score.

---

## 12. Related methods named in the paper

| Method | Basis | Why MEDICC2 differs |
|---|---|---|
| MEDICC (2014) | MED via weighted CFG | no WGD, much slower |
| MEDALT | minimum **spanning** tree over observed profiles | ignores unsampled ancestors; polynomial but not a phylogeny |
| Sitka | breakpoints + perfect phylogeny | infinite-sites on breakpoints; places breakpoints as internal nodes |
| NJ / minimum evolution on Euclidean or Manhattan distance | generic metric | no evolutionary model; Euclidean distance is not a lower bound on event count |
| Refphase | multi-sample reference phasing | upstream input to MEDICC2, not a competitor |
| Battenberg | bulk allele-specific CN caller | upstream caller |

---

## 13. Implications for `CopyNumberEvolution.jl`

What this paper settles, or constrains, for our forward simulator.

**Event taxonomy to simulate.** MEDICC2's four elementary operations are the minimal
set our generative model must be able to produce: haplotype-specific segmental
**gain**, **loss**, **LOH** (loss to 0), and **WGD**. Because MEDICC2 treats an
arbitrary-length run as one event, our whole-arm and whole-chromosome events are the
long tail of the same distribution rather than a different kind of thing — but we
still draw them from their own distributions, since a continuous length distribution
will not produce them at realistic frequency.

**Zero is absorbing — build it in from the start.** "Gains of zero-copy segments are
not permitted" is a physical constraint, not a convenience. Our `apply!` must never
resurrect a segment at copy number 0. This is a stronger and cleaner statement of the
viability question in the handoff (§6): the minimum-CN rule is about *whole-region or
whole-chromosome* nullisomy being unobserved in real data, while zero-absorption is
about a single haplotype's segment and is unconditionally true.

**Copy number effectively caps at 8.** MEDICC2 cannot represent more. If we simulate
profiles with CN > 8 they are not round-trippable through the reference inference
method. Worth a documented ceiling (or at least a warning) on the output side.

**WGD semantics need a choice.** MEDICC2's default is `+1` to every non-zero segment;
`×2` is the alternative (`--wgd-x2`). Biologically, tetraploidization is `×2`. These
differ once any segment is ≥ 2, i.e. exactly in the non-truncal cases we care about.
We should implement `×2` as the biological default and be explicit about it, because
a `×2` event will be scored by MEDICC2 as a `+1` WGD plus extra gains.

**Allele-specific is the right internal representation**, and total CN a projection —
independently confirmed here: MEDICC2 models both haplotypes, and everything
interesting (LOH, MSAI, parallel evolution on distinct haplotypes) is invisible in
total CN. Our simulated profiles are *phased by construction*, which makes them a
clean test set for MEDICC2's evolutionary phasing: we know the truth.

**Output format target.** The long-form TSV `sample_id, chrom, start, end, cn_a, cn_b`
with **identical segmentation across all cells** is exactly what a bin-grid projection
produces, so `CNMatrix` → MEDICC2 input is a formatting exercise. Keep 0-based
half-open (BED) coordinates in the writer even if internal `Segment` coordinates are
1-based inclusive, and keep chrX/chrY out of the default MEDICC2 export (their bulk
analysis used autosomes only) while keeping them in the simulation.

**The comparison the papers set up for us.** MEDICC2 branch length = number of
copy-number events, inferred by parsimony. Our simulator knows the *true* per-edge
event count and the true ancestral profiles at every internal node. That makes two
directly measurable quantities: (i) how far the parsimony branch length falls below
the true event count — the paper only establishes it is a lower bound; and (ii) how
that gap depends on whether CNAs accrue per division or per unit time (the
non-Markovian question), since parsimony has no notion of either. Their own finding
that SCNA trunks are short relative to SNV trunks is a statement about the same gap
seen from the data side.

**Retaining internal-node profiles and the event log is not optional for us.**
MEDICC2's deliverables *are* ancestral profiles and per-branch events, so the truth
set we compare against has to contain both.

**What we are choosing not to model** (and MEDICC2 agrees is out of scope): inversions,
translocations, BFB, chromothripsis — i.e. the karyotype backend. Their contiguity
stress test is the argument that this omission is tolerable for tree inference.
Their own benchmark simulator *does* include these, which is worth remembering if we
ever want to reproduce their benchmark rather than just consume their method.
