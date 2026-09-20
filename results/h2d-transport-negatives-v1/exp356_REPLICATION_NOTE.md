# exp356 independent replication (Lenny t1 cron, 2026-09-17T01:4xZ)

Second execution of the SAME frozen prereg + harness (SHA256 adef42d12a8c6cca3b8ad19cbaa5f4c964c47cea6203a3879a7a501d22cf7893, byte-verified transferred), ~5.3h after the 20:18Z first run. Harness transferred to the host and byte-verified before the run.

## Agreement with first run (all load-bearing cells)
- GPU0 KILL replicated: pinned/pageable 0.94x/1.04x/1.01x — pinning not governing, 0/3 >= 1.5x.
- GPU1 CONFIRM-governing replicated: 1.59x (576KiB), 1.26x (2.5MiB), 1.65x (13.5MiB) — 2/3 >= 1.5x.
- GPU1 WC_FLAT replicated: wc/pinned 0.94x / 0.998x / 0.998x (first run 1.04/0.98/1.01).
- GPU0-vs-GPU1 asymmetry replicated: GPU0 all arms ~1.28-1.55 GB/s vs GPU1 pinned 8.55/11.91/13.33 GB/s (first run 8.4-13.2).

## Divergence (disclosed)
- SPLIT cell: this run wc_split 7.51 vs wc 8.04 GB/s = 0.935x (FLAT within +-10%); first run 6.20 vs 8.76 = 0.71x (regression). Same direction (split never helps), magnitude straddles the 10% prereg threshold => SPLIT cell is NOISY (64B second memcpy launch overhead ~ small-median sensitivity; max_ms 0.78 in first run indicates noise). Do not cite split magnitude as a point estimate; cite "split <= WC, never helps".

GPU1 absolute pinned medians moved 8.55 vs 8.76 (576KiB) and 13.33 vs 13.2 (13.5MiB) — run-to-run stability good on the healthy GPU.

Raw: `exp356_results_replication.json`
