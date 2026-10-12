# exp227 POOLED REFIT #4 — host-class term (optimizer v2) (2026-09-16)

**Inputs (arm A, chronological, 19 cells):** identical to refit #3 — 227b (A100, 09-14) 6 cells; 227c (A100, 09-15) block 1 only, 7 cells (block 2 = contamination, excluded per refit #2 rule); 227d run-51243200 (RTX 5090 exclusive, 09-16) 6 cells.

**Host-class fit (log-linear: tps = intercept[class] · N^slope; shared slope, per-class intercept; least squares over cell means; slope in refit #2/#3 convention d ln(tps)/d log2(N)):**
- **shared slope = 0.268 (power-law exponent 0.387), host-class R² = 0.901 (log scale)** — vs pooled 0.795 (Δ +0.106)
- intercept a100 = 161.4 tok/s · intercept rtx5090_exclusive = 267.5 tok/s (tps at N=1)

**Reference pooled fit (refit #3 method, reproduced live as a gate):** slope 0.310, intercept 166.8, R² 0.795 — matches PooledRefit3.md.

**Per-class own fits (contamination rule applied):** a100 0.246/170.6 (13 cells, R² 0.929) · rtx5090_exclusive 0.332/206.9 (6 cells, R² 0.761)
**Per-source own fits:** 227b 0.274 · 227c 0.230 · 227d 0.332/206.9 — note refit #3 quoted 227c 0.184, a figure that included the contaminated block-2 cells; under the contamination rule 227c block-1-only is 0.230.

**Findings:**
1. Per-host-class intercepts absorb the cross-host level shift: log-space R² rises 0.795 → 0.901. Heterogeneity between hosts no longer dominates the pooled residual; what remains is within-class scatter.
2. Shared slope 0.268 sits between the class slopes (a100 0.246, 5090-exclusive 0.332): the class term captures LEVEL, not slope. The 5090's steeper N-scaling is not modeled — keep a per-class slope split flagged for v3 if more 5090 cells land.
3. Level separation at N=1: rtx5090_exclusive 1.66× a100 — and the gap grows with N under the class-specific slopes.
4. Capacity planning must use class-specific intercepts; do NOT port the a100-derived pooled prior to 5090-exclusive hosts (confirms refit #3 finding).

**Artifacts:** inputs = evidence/exp227b-n16-replication, exp227c-a100-full-sweep, exp227d-5090-cross-cohort-sweep/run-51243200 cells.jsonl (read-only, sha-pinned in receipt); fit + receipt = historical-optimizer-v1/data/hostclass-refit-v1/; this summary = PooledRefit4-hostclass.md. Fit method identical to refit #3 (log-linear least squares over chronological cell means; contamination rule applied).
