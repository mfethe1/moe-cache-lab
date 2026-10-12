# exp356 VERDICT — WC H2D on an RTX 5060 Ti host (2026-09-16)

Run: local python + cudart (system v12.9 fallback torch-lib 12.8), both GPUs, 3 sizes × 4 arms, 10 warmup + 50 trials, medians. Prereg: a prereg frozen before the run.

## Frozen-rule verdicts
- **GPU0: KILL (negative)** — pinning not governing (0/3 sizes ≥1.5x). WC numbers recorded as descriptive.
- **GPU1: CONFIRM-governing** (2/3 ≥1.5x: 576KiB 1.54x, 13.5MiB 1.53x) → WC verdict applies:
  - **WC_FLAT** (0.85–1.15x at all sizes: 1.04x / 0.98x / 1.01x) — write-combined adds nothing over default-pinned on the healthy GPU.
  - **SPLIT verdict: regression** — wc_split 6.20 vs wc 8.76 GB/s at 576KiB = **0.71x, >10% regression** (split-cost datapoint recorded per prereg).
- **GPU0-vs-GPU1 asymmetry REPLICATED** (free control): GPU1 pinned 8.4–13.2 GB/s vs GPU0 1.4–1.5 GB/s — the degraded-GPU0-link finding from exp211 (9x) replicates (~6–8x here, pinned-vs-pinned).

## Read-through to the exp87 anchor
GPU0 pageable ~1.53 GB/s at 13.5MiB ≈ exp87's 1.546 GB/s pinned — GPU0's link is capped near ~1.5 GB/s regardless of host-allocation strategy. GPU1 is the healthy 13+ GB/s path.

## Implication
WC + prefix/payload split (the moe-cache-host.cu pattern) does **not** rescue GPU0 and does not improve GPU1. The pattern is not the bottleneck; the degraded GPU0 link is hardware/platform-level. Recorded as a negative for the WC-pattern claim; asymmetry replication strengthens the degraded-link finding.

Raw data: `exp356_results_run1.json` and `exp356_results_replication.json` in this directory.
