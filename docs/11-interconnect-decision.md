# 11 — Interconnect decision table: when is tensor parallel worth it over a given fabric?

<!-- Authored 2026-10-03 against the interconnect audit of the same date. Every number below is computed from records in data/interconnect/ and data/accelerators/; none is introduced here. -->

**What this document is for.** [[flops/tensor-parallel-allreduce]] derives one number —
the batch at which the tensor-parallel all-reduce stops being free:

> **B\* = P_layer · BW_link / (2 · d · BW_hbm)**

B\* is the crossover batch. Below it, the all-reduce costs less than the weight read it
competes with, and tensor parallelism is essentially free. Above it, communication dominates
and every layer boundary is paid for in fabric time. That record computes B\* for two points
on NVLink 4 — 3,318 on a full 450 GB/s per-GPU domain versus 184 on a single 25 GB/s link —
an 18× spread on *identical silicon*. That 18× is the argument for NVSwitch, stated as
arithmetic.

This document extends that to every fabric in the slice, on the corrected interconnect
records. It is the connective-tissue document: every accelerator attaches to memory on one
tier and to its peers on another, and the **ratio between those two tiers** is what decides
TP degree.

---

## 1. Why B\* and not a throughput benchmark

Three properties make B\* decision-useful in a way a measured tok/s is not.

**It is a property of the fabric, not the model.** The model enters only through P_layer/d,
roughly the feed-forward expansion. Llama-2-7B has P_layer/d = 49,408; a 70B-class model has
104,448 — within 2.1×, because d doubles with the model. A 70B model does *not* have a
materially worse TP problem than a 7B. This is the opposite of the received wisdom that bigger
models are harder to tensor-parallel.

**It is linear in batch while its competitor is constant.** The all-reduce payload is
4·B·d bytes per rank per layer, linear in B. The weight read it competes with is
2·P_layer bytes, independent of B. TP therefore gets monotonically worse as you batch more —
exactly the regime continuous batching pushes you into. A fabric that is fine at batch 32
degrades at batch 256 with no change in hardware.

**It is analytically exact given the two bandwidths.** No measurement to interpret, so it can
be recomputed for any part, fabric and geometry, and it moves only when a bandwidth record
changes. That is why it survived this audit better than any other number in the slice.

---

## 2. Inputs, all cited

**The fabric side**, `BW_link` — per-GPU, **per direction**. Every value is a per-direction
rate from a record whose `bandwidth_basis` states that convention:

| record | per direction | derivation held in the record |
|---|---|---|
| [[interconnect/nvswitch-5]] | 900 GB/s | 1,800 GB/s bidirectional per GPU halved; 18 × 50 |
| [[interconnect/nvlink-c2c]] | 900 GB/s | 1.8 TB/s bidirectional coherent CPU–GPU, halved |
| [[interconnect/nvswitch-4]] / [[interconnect/nvswitch-3]] | 450 GB/s | 900 GB/s bidirectional per GPU halved; 18 × 25 |
| [[interconnect/amd-infinity-fabric-cdna4]] | 537.6 GB/s | 7 links × 76.8, CDNA 4 |
| [[interconnect/cxl-4-0]] | 472 GB/s | x16 at 128 GT/s PAM4, 236B/256B framing |
| [[interconnect/amd-infinity-fabric-link]] | 448 GB/s | 7 links × 64, CDNA 3 |
| [[interconnect/nvswitch-2]] | 300 GB/s | 600 GB/s bidirectional per GPU halved |
| [[interconnect/nvswitch-1]] | 150 GB/s | 300 GB/s bidirectional per GPU halved |
| [[interconnect/nvlink-2]] | 150 GB/s | 300 GB/s bidirectional per GPU, 6 × 25 |
| [[interconnect/nvlink-1]] | 80 GB/s | 160 GB/s bidirectional per GPU, 4 × 20 |
| [[interconnect/pcie-gen6]] | 236 GB/s | x16, 64 GT/s PAM4 |
| [[interconnect/aws-neuronlink]] | 192 GB/s | published flat, no direction given |
| [[interconnect/cambricon-mlu-link]] | 200 GB/s | published per chip, no direction given |
| [[interconnect/moore-threads-mtlink]] | 120 GB/s | 240 GB/s, direction reading flagged in record |
| [[interconnect/infiniband-xdr]] | 100 GB/s/port, 25/lane | 4 lanes at 200 Gb/s |
| [[interconnect/huawei-ub-link]] | 392 (8-NPU), 840 (16-NPU) | 784 / 1.68 TB/s bidirectional halved |
| [[interconnect/pcie-gen5]] / [[interconnect/cxl-2-0]] | 63 GB/s | x16, 32 GT/s NRZ |
| [[interconnect/amd-infinity-fabric-link]] single link | 64 GB/s | CDNA 3 per direction |
| [[interconnect/huawei-hccs]] | ~90 GB/s/part | **contested**, secondary source only |
| [[interconnect/pcie-gen4]] | 31.5 GB/s | x16, 16 GT/s NRZ |
| [[interconnect/nvlink-3]] / [[interconnect/nvlink-4]] single link | 25 GB/s | 50 GB/s bidirectional per link halved |
| [[interconnect/ualink-200g-1-0]] | 25 GB/s/lane | 200 Gb/s per lane |
| [[interconnect/google-tpu-ici]] | 200 (v5e), 400 (v6e) | 400 / 800 GBps bidirectional halved |

**The memory side**, `BW_hbm`, per accelerator record — shown in the table, because using
each fabric on its own native part is the comparison that matters.

**The model side.** Llama-2-7B geometry from [[flops/tensor-parallel-allreduce]]: d = 4,096,
intermediate 11,008, P_layer = 202,375,168. The 70B column uses d = 8,192,
P_layer = 855,638,016.

**Excluded, and why.** [[interconnect/metaxlink]] and [[interconnect/biren-blink]] have no
sourced per-direction rate — both are nulls with the failed search documented in their
`bandwidth_basis` — so no B\* is computed. [[interconnect/ethernet-uec]] and
[[interconnect/roce]] are nulls by design: neither has a link rate of its own, inheriting
port rates from Ethernet. [[interconnect/ufi-ultra-express-fabric]] has no published
specification. [[interconnect/cerebras-wafer-mesh]] and [[interconnect/groq-lpu-interconnect]]
publish aggregates in units this field cannot hold (Pb/s on-wafer; on-chip SRAM only), so
they have no per-link rate either. Inventing a B\* for any of these would be the fabrication
this repo's contract forbids.

---

## 3. The table

Each row pairs a fabric with the part it actually ships on, so the ratio is realistic.

| Fabric / configuration | Part (BW_hbm GB/s) | BW/dir | **B\* (7B)** | B\* (70B) |
|---|---|---:|---:|---:|
| Huawei UB Link, 16-NPU full mesh | 950DT (500) | 840 | **41,503** | 87,736 |
| Huawei UB Link, 8-NPU server | 950DT (500) | 392 | **19,368** | 40,944 |
| Cambricon MLU-Link | MLU370 X8 (614) | 200 | **8,042** | 17,000 |
| NVLink-C2C coherent | H100 SXM (3,350) | 900 | **6,637** | 14,030 |
| TPU v6e ICI, 4 ports | TPU v6e (1,638) | 400 | **6,033** | 12,753 |
| TPU v5e ICI, 4 ports | TPU v5e (819) | 200 | **6,033** | 12,753 |
| AWS NeuronLink | Inferentia2 (820) | 192 | **5,784** | 12,228 |
| NVSwitch 1 domain | V100 SXM2 (900) | 150 | **4,117** | 8,704 |
| NVLink 2 aggregate per GPU | V100 SXM2 (900) | 150 | **4,117** | 8,704 |
| Moore Threads MTLink | MTT S4000 (768) | 120 | **3,860** | 8,160 |
| NVSwitch 2 / NVSwitch 3 domain | A100 SXM4 (2,039) | 300 | **3,635** | 7,684 |
| CXL 4.0 x16 | H100 SXM (3,350) | 472 | **3,481** | 7,358 |
| **NVSwitch 4 domain** | **H100 SXM (3,350)** | **450** | **3,318** | 7,015 |
| NVLink 5 / NVSwitch 5 domain | B200 (8,000) | 900 | **2,779** | 5,875 |
| NVLink 1 aggregate per GPU | V100 SXM2 (900) | 80 | **2,196** | 4,642 |
| Tenstorrent QSFP-DD port | Wormhole n150 (288) | 25 | **2,144** | 4,533 |
| MI300X Infinity Fabric domain, 7×64 | MI300X (5,300) | 448 | **2,088** | 4,414 |
| PCIe 6.0 x16 host attach | H100 SXM (3,350) | 236 | **1,740** | 3,679 |
| CXL 3.x x16 | H100 SXM (3,350) | 236 | **1,740** | 3,679 |
| MI350 Infinity Fabric domain, 7×76.8 | MI355X (8,000) | 537.6 | **1,660** | 3,509 |
| Huawei HCCS (~90/part, contested) | Ascend 910B (1,400) | 90 | **1,588** | 3,357 |
| InfiniBand XDR, one port | A100 SXM4 (2,039) | 100 | **1,212** | 2,561 |
| PCIe 5.0 x16 host attach | H100 SXM (3,350) | 63 | **465** | 982 |
| CXL 2.0 x16 | H100 SXM (3,350) | 63 | **465** | 982 |
| AMD xGMI | MI250X (3,200) | 50 | **386** | 816 |
| NVLink 3, one link | A100 SXM4 (2,039) | 25 | **303** | 640 |
| InfiniBand XDR, one lane | A100 SXM4 (2,039) | 25 | **303** | 640 |
| MI300X single IF link | MI300X (5,300) | 64 | **298** | 631 |
| MI350 single IF link | MI355X (8,000) | 76.8 | **237** | 501 |
| PCIe 4.0 x16 host attach | H100 SXM (3,350) | 31.5 | **232** | 491 |
| **NVLink 4, one link** | **H100 SXM (3,350)** | **25** | **184** | 390 |
| UALink 200G, one lane | H100 SXM (3,350) | 25 | **184** | 390 |

**The spread is 226×** — from 184 on a single NVLink 4 link to 41,503 on Huawei's 16-NPU
mesh — on the same model geometry, from fabrics in the same repo.

---

## 4. Four things the table says that the folklore gets wrong

**Tensor parallelism is not expensive on a fabric-rich part.** On an NVSwitch 4 domain B\* is
3,318 (7B) and 7,015 (70B); on NVLink-C2C it is 6,637 and 14,030. All far beyond any real
serving batch. The folklore that "TP is expensive and should be minimised" is folklore about
*PCIe*, over-generalised.

**The dominant multiplier is the fabric, not the model.** Going 7B → 70B raises B\* by 2.1×.
Going from one NVLink 4 link to a full NVSwitch 4 domain raises it 18×, and the full range of
the table is 226×. Fabric beats model by two orders of magnitude.

**AMD has more fabric per GPU than NVIDIA; NVIDIA has a much better fabric-to-memory ratio.**
An 8-way MI355X node has 537.6 GB/s per direction of scale-up against an 8-way H100 node's
450 — **1.19× more** — and needs no switch silicon, while NVIDIA reaches the same 8-way shape
only by adding NVSwitch. Yet B\* on MI355X (1,660) is *half* B\* on H100 (3,318), because
MI355X also has 8 TB/s of HBM against 3.35 TB/s. Since decode is memory-bound at low batch, it
is the **fabric-to-memory ratio** that sets the crossover. Per-GPU fabric bandwidth is the
wrong axis to shop on. The per-direction ratios: **Ascend 950DT 0.78, MTT S4000 0.16,
A100 0.15, H100 0.13, MI355X 0.067.** AMD's advantage on one axis and disadvantage on the
other is the single most misread comparison in this domain.

**Switch silicon buys topology, not bandwidth, and topology is worth a lot.** NVSwitch 1
delivers exactly the 300 GB/s bidirectional per GPU the direct-attached V100 link already had
while raising the domain from a partial mesh to 16 non-blocking GPUs — and the table shows it
scoring *identically* to direct-attached NVLink 2 (4,117) because B\* cannot see topology.
NVIDIA reports up to 1.5× higher throughput for Llama 3.1 70B on H200 with NVSwitch versus
point-to-point ([[interconnect/nvswitch-4]]) — the same effect as raising BW_link, obtained
without raising it. **This is the table's main blind spot, and it is worth stating plainly:**
B\* is a bandwidth calculation, so it cannot distinguish a non-blocking fabric from a partial
mesh at equal per-GPU bandwidth. On a partial mesh an all-reduce routing over a 2-hop path pays
2× the per-hop cost on that pair, and the weakest link sets collective time for the whole
group. Read the NVSwitch rows as a floor on achievable performance, not a ceiling.

---

## 5. The practical rule

**Use tensor parallelism over a fabric when your serving batch stays below B\*.**
Operationally: **B\* > 1,000 — TP is free, use it freely.** **300 < B\* < 1,000 — fine at
latency-serving batch, degrades under continuous batching.** **B\* < 300 — TP is already
costing you at moderate batch; prefer pipeline parallelism.**

- **B\* > 1,000 (22 rows). TP is free across the entire realistic serving range.** Huawei
  UB Link, Cambricon MLU-Link, NVLink-C2C, TPU ICI, NeuronLink, NVSwitch 1–5, CXL 4.0, MTLink,
  PCIe 6.0 x16, both AMD Infinity Fabric *domains*, HCCS (contested), InfiniBand XDR per port.
  There is no batch at which you should avoid TP for bandwidth reasons. Choose TP degree to fit
  the model and fill the machine, not to protect the fabric.

- **300 < B\* < 1,000 (5 rows).** PCIe 5.0 x16 and CXL 2.0 (465), AMD xGMI (386), single
  NVLink 3 link and single InfiniBand lane (303). Fine at batch 1–64; the collective becomes
  the dominant cost somewhere in the low hundreds. If you run continuous batching at batch in
  the hundreds, do not put a TP boundary on this fabric.

- **B\* < 300 (5 rows).** Single MI350 IF link (237), PCIe 4.0 x16 (232), single NVLink 4
  link (184), single UALink lane (184), single MI300X IF link (298). Use TP only as far as
  capacity forces, and prefer **pipeline parallelism**, which needs point-to-point transfers at
  stage boundaries rather than a collective at every layer.

The rule is a function of the **fabric**; the model enters only through P_layer/d.

### Where the advice inverts

**Inter-node is never a TP fabric.** The best inter-node figure here is InfiniBand XDR at
100 GB/s per direction per port — **4.5× below NVLink 4 per GPU**. Pipeline or expert
parallelism across nodes; tensor parallel inside.

**CXL is a memory tier, not a TP fabric, despite its bandwidth.** CXL 4.0 at 472 GB/s per
direction is *above* NVLink 4's 450 — and it is still the wrong tool for collectives, because
the obstacle is architectural rather than bandwidth: CXL's coherence model is CPU-memory-centric
and it has no in-network reduction, so collectives pay full latency with no SHARP equivalent.
Its correct use is pooled KV cache and weight offload for large-batch decode, where bandwidth
is amortised across many concurrent requests rather than paid per token. Note also that
[[interconnect/cxl-4-0]]'s corrected rate means CXL is no longer "4× below a real fabric" as
that record previously claimed — it is at parity on raw rate and is kept off this table's TP
tier by architecture alone.

**Vendor fabrics with no published rate cannot be evaluated at all.**
[[interconnect/metaxlink]] is the worked example: the accelerator record lists an
interconnect, the only source in the repo is a secondary aggregator saying "multi-GPU
interconnect" and nothing more, and therefore no B\* exists. Absence of a number is itself a
procurement-risk finding.

---

## 6. What this audit changed, and what is still open

Every B\* above was recomputed on **corrected** records. Four error classes were found; all
four moved B\* values:

| Error class | Records | Effect |
|---|---|---|
| Gb/s read as GB/s | [[interconnect/ualink-200g-1-0]], [[interconnect/infiniband-xdr]] | 8× overstatement; XDR port read as 800 GB/s/dir instead of 100 |
| PAM4 2-bits/symbol ignored | [[interconnect/pcie-gen6]] | 242 → 236 GB/s/dir |
| Per-direction halved as if an aggregate | [[interconnect/cxl-2-0]], [[interconnect/cxl-3-x]], [[interconnect/cxl-4-0]] | up to **4× understatement**; CXL 4.0 121 → 472 |
| Aggregate stated with no direction | [[interconnect/amd-infinity-fabric-link]] | MI300X's 128 GB/s resolved as bidirectional → 64 per direction |

The PCIe/CXL family had been audited three times and was still wrong in the same direction.
The cause is now pinned: **PCI-SIG quotes bandwidth per lane per direction, while NVIDIA and
AMD quote bidirectional aggregates** — opposite conventions. Each affected record's
`bandwidth_basis` now states which it uses. Two records moved *up* — [[interconnect/nvlink-3]]
off Wikipedia to the A100 datasheet, and the AMD record off `contested` to `verified` once the
CDNA 4 white paper stated the convention outright ("76.8 GB/s in each direction" against a
"153.6 GB/s peak total aggregate"). [[interconnect/nvswitch-3]] was added as **contested** on
a pure naming disagreement: NVIDIA's developer blog calls the Hopper switch third-generation
while NVIDIA's own spec table labels it "NVLink 4 Switch". The bandwidth is identical either
way; only the name is in dispute.

**Still open, honestly.** Every NVLink 5/6 and NVSwitch 5 figure is NVIDIA's own and marked
preliminary. [[interconnect/nvlink-6]] remains contested between NVIDIA's 3,000 GB/s per GPU
over 36 links and a secondary 3,600 GB/s. [[interconnect/huawei-hccs]] is still contested on a
secondary source with no Huawei datasheet in existence. Huawei UB Link and Moore Threads MTLink
figures are vendor-theoretical and unmeasured by any third party here. And **no B\* in this
document has been validated against a measured collective time** — every one is analytic. The
highest-value missing record is a third-party measurement of TP all-reduce time versus batch on
at least one slow fabric, because that would convert the bottom five rows from arithmetic into
evidence.