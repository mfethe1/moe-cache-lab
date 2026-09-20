# H2D transport negatives — write-combined memory does not help expert streaming (v1)

A **negative result**, published because it closes off a plausible and expensive
optimization path for MoE expert-cache implementations.

Pre-registered before the run, then **independently re-run ~5.3 hours later** with the
same frozen harness (SHA256-verified). Every load-bearing cell agreed.

## The hypothesis that failed

Streaming cold MoE experts host→device is the dominant cost in CPU-offloaded MoE
inference. A natural fix, and the pattern used in some expert-cache host code, is:

1. allocate host staging memory **write-combined** (`cudaHostAllocWriteCombined`), and
2. **split** the copy into a small metadata prefix plus a large payload.

Both were tested at 576 KiB (one q4_K expert), 2.5 MiB, and 13.5 MiB, 10 warmup +
50 trials, medians, on two GPUs in one RTX 5060 Ti-class host.

## Result: write-combined is flat

On the GPU where pinning **is** governing (so the test is meaningful — 1.54x and 1.53x
pinned-vs-pageable at 576 KiB and 13.5 MiB):

| payload | WC / pinned (run 1) | WC / pinned (replication) |
|---|---|---|
| 576 KiB | 1.04x | 0.94x |
| 2.5 MiB | 0.98x | 1.00x |
| 13.5 MiB | 1.01x | 1.00x |

**Write-combined host memory adds nothing over ordinary pinned memory.** Flat in both
runs, at every size.

The prefix/payload split never helped either: run 1 measured 0.71x at 576 KiB (a
regression), the replication 0.935x. The magnitude straddles the 10% threshold and the
cell is noise-sensitive (a 64 B second copy launch against a small median), so the
honest statement is **"split ≤ WC, never better"** — do not cite the split magnitude as
a point estimate.

## The finding that outlived the hypothesis

The same runs replicated a **6–8x host→device asymmetry between two GPUs in the same
machine** (degraded GPU 1.4–1.5 GB/s pinned vs healthy GPU 8.4–13.3 GB/s), matching a
~9x asymmetry seen in an earlier independent experiment.

On the degraded device, pinning is **not** governing at any size (0.94x / 1.04x /
1.01x), and its *pageable* throughput (~1.53 GB/s) matches an older *pinned* anchor of
1.546 GB/s — i.e. that link is capped near ~1.5 GB/s regardless of host allocation
strategy. No host-side allocation trick rescued it, because the limit is not host-side.

**Practical consequence.** At ~1.55 GB/s, one 576 KiB q4_K expert costs ~0.38 ms of
pure transfer; with 8 active experts per step that alone consumes most of a 15 tok/s
budget. If you are hunting expert-streaming throughput, measure your actual per-device
H2D bandwidth **first** — a degraded link will dominate every software-side choice you
make downstream, and two GPUs in the same box can differ by nearly an order of
magnitude.

## Scope and limits

- One host, two GPUs, RTX 5060 Ti class, CUDA 12.8. Not generalized across platforms.
- Synthetic H2D copies, not an end-to-end decode benchmark.
- This tests **transport mechanics only**. It says nothing about cache admission or
  eviction policy.
- The degraded-link root cause (platform, slot, or hardware) was **not** isolated.

## Files

- `exp356_VERDICT.md` — frozen-rule verdicts
- `exp356_REPLICATION_NOTE.md` — independent re-run, including the one disclosed divergence
- `exp356_results_run1.json`, `exp356_results_replication.json` — raw medians
- `SHA256SUMS`
