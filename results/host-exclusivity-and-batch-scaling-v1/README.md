# Host exclusivity and batch scaling — two measured results (v1)

Two independent measured-hardware results from rented-GPU benchmarking, published
because both correct a modelling error that is easy to make and expensive to keep.

Both were pre-registered before acquisition. Both report the gates they failed.

| | exp227d | exp382 |
|---|---|---|
| Question | Does an A100-derived concurrency-scaling model transfer to other hosts? | How does MoE decode step time scale with batch? |
| Answer | No — not unless GPU exclusivity is a covariate | `b^0.089` over b in [1,32] |
| Class | measured hardware | measured hardware |
| Cost | $0.16 | $0.44 |

---

## Result 1 — `gpu_frac` is a required covariate in rented-GPU scaling models

**Claim (level only):** a concurrency-scaling model fitted on a mixed population of
rented hosts under-predicts throughput on an *exclusive* host, because the pooled
population's ceiling is set partly by co-tenancy rather than by the accelerator.

Measured on one exclusive RTX 5090 (`gpu_frac=1.0`, sm_120, 32 GB), engine/model/client
SHA-pinned, gates G1/G2/G3 PASS:

| N (concurrent) | predicted band | observed tok/s | verdict |
|---|---|---|---|
| 4 | [103, 539] | 439.2 | PASS |
| 16 | [151, 789] | 761.4 | PASS |
| 64 | [221, 1154] | **1616.4** | **upper bound breached** |

The N=64 cell broke the *upper* pre-registered bound — a generous 2.28x multiplicative
prediction interval could not contain the observation.

**Root cause.** Five earlier hosts in the fitting population died mid-run. All five were
**shared-GPU** hosts; one was `gpu_frac=0.125` (eight tenants on a single 5090). A model
fitted across that population encodes co-tenancy contention as if it were accelerator
behaviour.

**Refit with a host-class term** (shared slope, per-class intercept):

- log-space R^2 **0.795 -> 0.901**
- intercepts: a100 **161.4** tok/s, rtx5090_exclusive **267.5** tok/s at N=1 (1.66x level separation)

### What is NOT claimed

The attractive headline — *"the 5090 has a steeper scaling exponent"* — **is not
supported and is not made here.** The slope estimate is 0.4794 (SE 0.1344, 95% CI
[0.2159, 0.7429]) and that interval **still contains** the pooled slope 0.2747. With
3 N-levels / 6 block-cells (df=4) the slope is not identified. Only the **level** claim
is citable.

The confirmatory high-N extension designed to settle the slope returned **zero usable
rows across 7 acquisition attempts**; it is unresolved, and is reported here as
unresolved rather than omitted.

**Practical guidance:** when fitting a scaling model across rented GPU hosts, record
`gpu_frac` (or equivalent exclusivity) per cell and include it. Otherwise the fit
silently averages hardware behaviour with contention from other tenants, and will
under-predict on the exclusive hosts you actually want to buy.

---

## Result 2 — measured batch-scaling exponent for MoE decode

**sigma = 0.0894, R^2 = 0.9876** over batch {1,2,4,8,16,32}.

`step_time(b) = 28.480 * b^0.0894 ms` — step time grew **1.379x** while batch grew
**32x**. Throughput 34.9 -> 809.2 tok/s (**23.2x**).

Rig: RTX 4090 (sm_89, 24 GB), OLMoE-1B-7B-0924 fp16 at pinned revision
`6d84c48581ece794365f2b8e9cfb043c68ade9c5` (64 experts, 8 active, 16 layers),
torch 2.6.0+cu124, transformers 5.17.0. Synthetic inputs, prompt length 128,
24 decode steps, 3 repeats, medians. ABBA ladder (ascending then descending);
arms agreed to within **3.39%**, so the fit is not drift.

### A gate this experiment failed

Per-module CUDA-event attribution (splitting "spine" from "expert" time) was
pre-registered with a **3% profiler-overhead bound**. Measured overhead was **32.8%**.
Per the pre-registration, **all attribution outputs are discarded** — including an
apparent ~0.96 "spine fraction", which is an artifact of hook overhead inflating the
non-expert bucket, not a measurement.

`sigma` above is therefore fitted on **total** step time from the unprofiled arms.
This is a weaker statement than isolated-spine scaling, and it is the honest one.

### Extrapolation is not supported

The same ladder refit with a conjugate Normal-Inverse-Gamma Bayesian linear model,
sweeping the slope prior as a sensitivity check:

| slope prior scale | posterior slope | predicted tok/s at b=64 |
|---|---|---|
| 0.1 | 0.110 | 738 |
| 0.5 | 0.329 | 13,906 |
| 1.0 | 0.351 | 18,654 |

**The b=64 prediction moves 25x on prior choice alone.** Six batch points do not
dominate the prior. Inside the measured range the fit is tight; outside it, nothing
is established.

`sigma ~= 0.09` is the signature of decode being memory-bandwidth bound and far from
compute saturation — a regime that **must** end at some batch size. The ladder stopped
at 32 and **did not find that knee**. Any capacity plan quoting a b=64 number from this
data is quoting the prior, not the hardware.

### Scope limits

One rig, one model, HF `transformers` eager decode — **not** an optimized serving stack
(vLLM/TRT-LLM with CUDA graphs and paged attention will differ). No spine/expert
decomposition survived. **No economics, $/token, ROI, or payback claim is licensed by
this result.**

---

## Reproduction

```
python exp382_sigma_bench.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
  --batches 1,2,4,8,16,32 --prompt-len 128 --steps 24 --warmup 3 --repeats 3 \
  --out exp382_raw.json
```

## Files

- `exp382_sigma_bench.py` — exact benchmark script used
- `exp382_raw.json` — raw timings, both ABBA arms, including the discarded attribution fields
- `exp382_pooled.json`, `exp382_sigma_fit.json`, `exp382_bayes_prior_sensitivity.json` — derived fits
- `exp227d_PooledRefit4-hostclass.md` — host-class refit detail
- `SHA256SUMS`

## Provenance and honesty notes

exp382 pre-registration was written and hashed
(`fb6d3aa77861d3b8cbfdac2ae2c83eaca4b74a29d94e09cb7f436f5104bb4cd7`) **before** the
instance was acquired, and that hash is unchanged. The originally specified model
(Qwen1.5-MoE-A2.7B) OOM'd at load on 24 GB and was replaced by OLMoE within the
pre-registration's stated "or smaller if VRAM-constrained" latitude; the
pre-registration's named single-expert ablation fallback was **not** run, which is why
the quantity is labelled `sigma_total` rather than an isolated-spine exponent
throughout.

Both instances were destroyed after their runs and zero-instance state was verified.
