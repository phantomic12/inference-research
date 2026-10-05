# 25 — Citation integrity: when a source does not contain the number it is cited for

<!-- written 2026-10-05 against 3,848 records. Every citation resolves to a data/ record;
     verified with an explicit checker against the committed tree. -->

**The decision this document serves:** you are about to trust a number in this repo, or
write one. A record whose value resolves to a `sources` id is **not** the same as a
record whose value was *found in* that source. This document is about the gap between
those two things, because it is the failure mode AGENTS.md calls "the worst failure mode
in this repo, because **it looks sourced**" — and a full sweep of the accelerator FLOPs
citations has now measured how often it actually happens, and what it looks like when
it does.

**The short version: three source records were found to back figures they did not
contain.** One was never corrected. Two were. Both remaining cases are still open in the
records as this is written, and §4 says what to do about them.

---

## 0. The decision rule, short form

| situation | what it means | what to do |
|---|---|---|
| value found in the cited source, under the same spelling | **cited** | quote it |
| value found, but the source prints the **2× or ÷2** partner | **cited, dense/sparse** | correct as stored; check both spellings before flagging |
| value found, but the cited **column is a 2-card pair** | **cited, wrong column** | the per-card figure lives on a product page, not the datasheet |
| value **not in the source at all**, and no source has it | **unsourced** | `status: contested`, `basis: unverified`, `vendor_claim: false` |
| source **could not be fetched** | **unverified, not passing** | treat as unresolved; never as cleared |

**The rule that matters most is the last one.** An unreadable source and a
correctly-read source produce the same empty result in a checker, and only one of them is
evidence.

---

## 1. The measured result: 72 sources, 395 rows, 3 unsourced figures

The sweep in [[sources/w5a-part2-full-sweep-method-and-cleared-artefacts]] is the
foundation here, and it is worth reading because **its first pass was wrong and the
correction is the lesson.** Every one of the 395 `flops` rows across the 108 accelerator
records was mapped to its `source_id`, giving **72 distinct sources** to check. Each was
fetched, its text normalised (thousands separators stripped, smart quotes folded,
whitespace collapsed), and each attributed value searched under every plausible printed
spelling.

**The first pass flagged 25 of 72 sources as containing at least one value they do not
print. Twenty-two of those 25 were not failures.** The headline for any future pass, in
the record's own words:

> "**72 sources probed, 395 rows, 4 genuine citation failures, 2
> wrong-or-misleading citations now corrected, 1 wrong negative finding deposed.**"

So: **25 reported, 3 real.** An over-reporting rate of 88% on the first pass. A reader
who took the first pass at face value would have "corrected" 22 correct records.

**What this means for a decision:** *a citation audit that reports a failure rate is
reporting its own method's false-positive rate until proven otherwise.* Any future pass
must clear the dense/sparse and thousands-separator classes **before** counting a
failure, or it will repeat this.

---

## 2. The three artefacts that manufacture false failures

These are the classes that produced 22 of the 25 first-pass flags. They are not
hypothetical; each one has a named source and, in two cases, a record that was nearly
broken by it.

### 2.1 Dense vs sparse — the largest class, and the repo is *right*

**The repo correctly records the DENSE member; the vendor prints the SPARSE one under a
footnote.** In every confirmed case the stored value is exactly half a printed figure:

| source | dense/sparse pairs that were "missing" |
|---|---|
| `nv-h100-datasheet` | 378, 494.5, 756.5, 989.5, 1670.5 |
| `nv-h200-product-page` | 417.5, 494.5, 835.5, 989.5 |
| `nv-l4-product-page` | 121, 242.5 |
| `nv-l40-datasheet` | 728.5 (RTX 6000 Ada row) |
| `nv-a30-datasheet` | 82.5, 330.5 |
| `nv-blackwell-datasheet` | 1100 |
| `nv-gb200-nvl72-product-page` | 2500 |
| `cn-huawei-atlas-650e-spec` | 803.75 |

**Every one of these is CORRECT as stored.** The record states the corollary, and it is
the sentence to memorise:

> "**a naive citation check reports a correct dense value as missing, so 'the number is
> absent from the cited page' is NOT by itself a finding — it must be tested against the
> 2x and /2 spellings first.**"
> — [[sources/w5a-part2-full-sweep-method-and-cleared-artefacts]]

### 2.2 Thousands separators — the cheapest way to manufacture a failure

`asic-rebellions-rebel100` prints **'1,024 TFLOPS (FP16)'**. A search for `1024` returns
**zero**. In the record's words this is "**the single easiest way to manufacture a false
citation failure and it nearly produced one on a record that is entirely correct.**"

**Rule: strip separators before matching, always.** A checker that does not is not
measuring the citation.

### 2.3 Layout loss — PDF extraction destroys the table

`nv-h100-datasheet`'s spec table collapses under plain extraction to a run of
concatenated cells — `'...FP3267teraFLOPS51teraFLOPS134teraFLOPSTF32TensorCore989teraFLOPS2...'`
— which destroys the column and row structure. **Reading the PDF with
`pdftotext -layout` recovers it exactly.** Any check of a PDF-based source must use
`-layout`, or it is reading a different document than the one a human would see.

### 2.4 The H100 NVL column is a 2-card pair — dense/sparse *plus* a pair

A fifth case, distinct from all three above, and the reason a whole class of "missing"
values turned out to be missing
([[sources/w5a-part2-h100-datasheet-nvl-column-is-a-2x-pair]]). With layout preserved
the datasheet reads `H100 NVL(1)`: FP64 68, FP32 134, TF32 1,979, BF16/FP16 3,958, FP8
7,916, INT8 7,916. Footnote 1 verbatim: "**Specifications shown for 2x H100 NVL PCIe
cards paired with NVLink Bridge**" — and footnote `(2)` is the sparsity marker on every
tensor row.

**The traps stack:** divide by 2 for sparsity, then **divide by 2 again** for the pair.
The stored `nvidia-h100-nvl` values — fp64 30, fp32 60, tf32 417.5, bf16/fp16 835.5,
fp8/int8 1670.5 — are **per card**, and they match NVIDIA's H100/H200 **product** page
rather than the datasheet column they are cited to. **The values are correct; the
citation names a column that is a pair.**

---

## 3. The three source records that backed figures they did not contain

These are the real findings. Two are corrected in the records; **one is still open**, and
§4.2 says why it is acceptable.

### 3.1 oneDNN README — the worst case, because the citation looks first-party

**What the records claimed.** `xpu-intel-dc-gpu-flex-170` carried fp32 20, fp16 80,
int8 160 sourced to `xpu-onednn-github-readme` with `basis: vendor-docs` and
`vendor_claim: true`. `xpu-intel-dc-gpu-max-1550` carried fp32 52, fp16 409, bf16 409,
int8 819 to the same README.

**What the document actually contains.** **Nothing.** A substring sweep over the full
raw README (22,267 characters) returns: `409` → 0 occurrences, `819` → 0, `160` → 0,
`80` → 0, `52` → 0. "The strings TFLOPS, teraFLOPS and TOPS do not occur as a
specification of any product." The GPU section is a bulleted list of **supported
architectures** with no numbers at all. The `52` that a naive search finds is inside a
**Base64-encoded camo.githubusercontent.com image URL** in the OpenSSF Scorecard badge — a
badge asset id, not a spec
([[sources/w5a-part2-onednn-readme-contains-no-tflops-figure]]).

**Why this one mattered more than a missing figure.** The record puts it exactly right:

> "A README that names 'Intel Data Center GPU Flex Series' and 'Intel Data Center GPU Max
> Series' is the **single most plausible-looking citation for an Intel GPU's compute
> figure**: same vendor ecosystem, same product family names, machine-readable, present
> in the record's sources list. **It looks sourced in a way that no aggregator citation
> does.** ... and here it also carried `vendor_claim: true`."

**A vendor-ecosystem README is a worse citation than a random aggregator**, because the
ecosystem match makes it read as first-party. That is the whole hazard in one sentence.

**A third retrieval path nearly deepened the error.** A plain-text reader proxy "returned
the shape names and the GPU counts but **DROPPED the price column entirely**; had only
that path been used, this record would have been filed as another negative and the repo's
error would have been **deepened rather than corrected**." Two methods agreed on the
figure (raw fetch + rendered DOM read); a third agreed with itself and disagreed with both.
**One retrieval path is not a verification.**

### 3.2 NVIDIA L40 datasheet — right document, wrong part

`nvidia-l40s` attributed all six compute rows to `nv-l40-datasheet`. **That datasheet is
for a different part and contains none of the six numbers**
([[sources/w5a-part2-nvidia-l40s-product-page-holds-the-l40s-figures]]). Read with
`-layout`, the L40 prints FP32 90.5, TF32 90.5, BF16 181.05, FP16 362, FP8 362, INT8 724
at 300W. The stored values are fp32 91.6, tf32 183, bf16 362, fp16 362, fp8 733, int8 733.
**"A grep for '91.6' and '733' over the extracted text returns ZERO matches."**

**The trap is that the two parts look like the same table.**
[[sources/w5a-part2-l40-ada-datasheet-vs-l40s-two-different-parts]] documents why: L40 and
L40S share the Ada architecture, 18,176 CUDA cores, 142 RT cores, 568 Tensor Cores, 48GB
GDDR6 and 864 GB/s, and **differ only in the compute table and the power** (300W vs
350W). "An auditor reading only the shape of the table — same page furniture, same
header, same `N | N**` pair notation — and carrying 91.6/183/362/733 across **would find
every one of those strings absent here**."

**One document on one page prints three different conventions.** The L40S product page
carries a full table in dense|sparse form *and* a second inline summary table that
"prints only the **SPARSE** member of each pair" — so a single page is both the §2.1 trap
and a fresh instance of it.

**Status: values unchanged, citation corrected.** All six rows now cite
`w5a-part2-nvidia-l40s-product-page-holds-the-l40s-figures`, still `vendor_claim: true`
and `vendor-spec-sheet`, which is now true.

### 3.3 The wrong negative — a failure in a *negative* finding

`amd-radeon-pro-vii-product-page` **does** carry 13.1 TFLOPS. The failure was not in the
record; it was in a negative finding *about* the page
([[sources/w5a-part2-amd-radeon-pro-vii-page-does-contain-the-figure]]). The earlier
check reported zero matches for `TFLOPS`, `TFLOP`, `TOPS`, `Peak` and `sparsit` across
18,346 characters. Re-extraction of the same URL yields **22,831 characters containing
the full spec table, with TFLOPs ×2 and Peak ×2.**

**Two generalisable errors, and the second is the important one:**

1. The extract was **shorter** (18,346 vs 22,831), so the earlier pull **did not reach
   the specifications section at all.** A truncated fetch reads as an empty page.
2. "**A negative finding needs a control.**" A zero-match sweep for common tokens is only
   informative "**if the same sweep is positive on a page known to contain them**."

**Recommendation: every negative citation finding must run a positive control on the same
fetch path.** Without one, "I searched and found nothing" is indistinguishable from "I
searched a truncated page." This is the same discipline as §4.1's single-path problem,
inverted: there, one path agreed with itself and disagreed with reality; here, the
absence was produced by a short read and would have been believed.

---

## 4. The two open items

### 4.1 The two Intel parts still have no Intel-published compute figure

**The honest state for `xpu-intel-dc-gpu-flex-170` and `xpu-intel-dc-gpu-max-1550` is
`contested`, and it should stay that way.** Three separate Intel documents have now been
checked and none carries the stored figures:

| document | what it says |
|---|---|
| Intel ARK, Flex 170 | **no compute row at all** — no TFLOPS row, no TOPS row |
| Intel ARK, Max 1550 | **no TFLOPS row and no TOPS row anywhere** |
| Flex Series product brief | the 170 as "**16 TFLOPS (FP32) / 250 TOPS (INT8)**" — a third, different pair |

The stored pairs (20/80/160 and 52/409/819) "match **neither** the 170 nor the 140."
Both records are now `contested`, their `basis` demoted to `unverified` /
`secondary-aggregator`, `source_id` null, and `vendor_claim: false`.

**Recommendation: do not plan on these three numbers.** The Max 1550's own record is blunt
about why this is not cosmetic: "**409 TFLOPS is the figure a reader would size a cluster
on. It is not traceable to Intel. Treat it as an unverified community estimate, not a
spec.**"

### 4.2 Two sources were unreadable, and unreadable is not passing

`vb-cerebras-cs3-datasheet-125pf-sparse` (a `cdn.sanity.io` PDF that failed to scrape)
and `cn-huawei-atlas-650e-spec` **remained unreadable rather than wrong**, and are
recorded as such. The sweep's conclusion is the rule:

> "**The remaining risk is concentrated in sources that could not be fetched at all, which
> is the argument for treating an unreadable source as unverified rather than as
> passing.**"

For the Huawei source there is a further reason to check the arithmetic rather than the
page: its four values are already marked `basis: derived` rather than transcribed, "so the
arithmetic — **cabinet total divided by 8 NPUs** — is the thing to check, not the page."

**Recommendation: record retrieval outcome as a field, not as an absence.** A source that
was never successfully fetched is in a different epistemic state from one that was
fetched and searched, and right now the corpus does not distinguish them.

---

## 5. What would change these recommendations

| change | which recommendation flips | record to re-read |
|---|---|---|
| Intel publishes a compute row for Flex 170 or Max 1550 in ARK | §4.1's `contested` becomes a sourced figure, or a confirmed absence | [[sources/w5a-part2-onednn-readme-contains-no-tflops-figure]] |
| the Cerebras PDF becomes fetchable | §4.2's unverified item becomes checkable; until then it stays open | [[sources/w5a-part2-full-sweep-method-and-cleared-artefacts]] |
| a citation checker lands in `tools/` with the three artefacts handled | §1's "25 reported, 3 real" becomes a standing invariant instead of a one-off sweep, and the 22 false positives stop costing a pass | [[sources/w5a-part2-full-sweep-method-and-cleared-artefacts]] |
| a retrieval-path field is added to `sources` | §4.2's unreadable/checked distinction becomes machine-readable | [[sources/w5a-part2-amd-radeon-pro-vii-page-does-contain-the-figure]] |

---

## 6. What the records cannot answer

1. **The sweep covered accelerator `flops` rows only.** 395 rows across 108 records and
   72 sources is a real denominator, and it is **not** a statement about supply prices,
   benchmark values, memory figures, interconnect bandwidth or any other numeric class.
   **Do not read "3 unsourced figures" as "this repo has 3 unsourced figures."**
2. **The sweep is one pass, one date (2026-10-05), one method.** Nothing here establishes
   a trend, and the pass that found 25-and-then-3 is evidence that a single method's raw
   output is not trustworthy without adjudication.
3. **Nothing re-verifies a citation after the fact.** Every figure in the corpus carries
   an `updated` date, but no record states when its *source* was last confirmed to
   contain it. A vendor page can change under a valid citation, and nothing in the schema
   detects that.
4. **No `sources` record states its own retrieval outcome** — see §4.2. This is the one
   structural gap in the sweep's own conclusions.

---

## Related documents

- [14-audit-findings.md](14-audit-findings.md) — the repo-wide mechanical audit; §4's
  "provenance integrity" check class is the ancestor of this document.
- [07-benchmarking-methodology.md](07-benchmarking-methodology.md) — the same
  discipline applied to measurement rather than to citation.
- [01-hardware-selection.md](01-hardware-selection.md) — the consumer of the figures whose
  citations §3 found unsound; §4.1's "do not size a cluster on 409 TFLOPS" is a
  hardware-selection constraint.
- [02-flop-map.md](02-flop-map.md) — the crossovers computed from the rows this sweep
  audited. A wrong `flops` value propagates into every ridge point downstream.
- [25 — this document] — see §6.1 for what this audit does **not** cover.