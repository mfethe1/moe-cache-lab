#!/usr/bin/env python3
"""exp382 — measured batch-spine scaling exponent (sigma) for MoE decode.

Per frozen PRE_REGISTRATION.md (sha256 fb6d3aa77861d3b8cbfdac2ae2c83eaca4b74a29d94e09cb7f436f5104bb4cd7).

Measures decode step time across batch ladder b in {1,2,4,8,16,32}, separating
spine (attention/router/non-expert) from expert-GEMM time via per-module CUDA
event hooks. Runs the ladder ASCENDING then DESCENDING (ABBA) in the same
session to expose thermal/clock drift.

Emits raw JSON only. No sigma fit here -- fitting happens off-node so the
raw measurements stay auditable.
"""
import argparse, json, os, statistics, sys, time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------- attribution
# We attribute time to "expert" if the module path looks like an MoE expert
# projection, else "spine". Attribution is recorded so the verdict can state
# which method produced the numbers (prereg method 4).
EXPERT_HINTS = ("expert", "mlp.experts", ".experts.")


def is_expert_module(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in EXPERT_HINTS)


class EventTimer:
    """Per-module CUDA-event timer. Accumulates elapsed ms per bucket."""

    def __init__(self, model, enabled=True):
        self.enabled = enabled
        self.handles = []
        self.pending = []          # (bucket, start_evt, end_evt)
        self.totals = {"expert": 0.0, "spine": 0.0}
        if not enabled:
            return
        for name, mod in model.named_modules():
            if not name or len(list(mod.children())):
                continue           # leaf modules only
            bucket = "expert" if is_expert_module(name) else "spine"
            self.handles.append(mod.register_forward_pre_hook(self._pre(bucket)))
            self.handles.append(mod.register_forward_hook(self._post()))

    def _pre(self, bucket):
        def hook(_m, _i):
            s = torch.cuda.Event(enable_timing=True)
            s.record()
            self.pending.append([bucket, s, None])
        return hook

    def _post(self):
        def hook(_m, _i, _o):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            for rec in reversed(self.pending):
                if rec[2] is None:
                    rec[2] = e
                    break
        return hook

    def reset(self):
        self.pending.clear()
        self.totals = {"expert": 0.0, "spine": 0.0}

    def gather(self):
        torch.cuda.synchronize()
        for bucket, s, e in self.pending:
            if e is None:
                continue
            self.totals[bucket] += s.elapsed_time(e)
        self.pending.clear()
        return dict(self.totals)

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------- measurement
@torch.inference_mode()
def decode_steps(model, input_ids, n_steps, timer=None):
    """Run n_steps of true decode (1 token at a time) with KV cache."""
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1:].argmax(-1)

    torch.cuda.synchronize()
    if timer:
        timer.reset()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    buckets = timer.gather() if timer else None
    return wall_ms / n_steps, buckets


def measure_batch(model, tok, b, prompt_len, n_steps, warmup, timer):
    ids = torch.randint(
        low=10, high=min(2000, tok.vocab_size - 1),
        size=(b, prompt_len), device=model.device,
    )
    for _ in range(warmup):
        decode_steps(model, ids, 2, None)
    per_step_ms, buckets = decode_steps(model, ids, n_steps, timer)
    rec = {"batch": b, "per_step_ms": per_step_ms}
    if buckets:
        tot = buckets["expert"] + buckets["spine"]
        rec["expert_ms_total"] = buckets["expert"]
        rec["spine_ms_total"] = buckets["spine"]
        if tot > 0:
            rec["spine_frac"] = buckets["spine"] / tot
            rec["spine_ms_per_step"] = per_step_ms * (buckets["spine"] / tot)
            rec["expert_ms_per_step"] = per_step_ms * (buckets["expert"] / tot)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    batches = [int(x) for x in a.batches.split(",")]

    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision, trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, revision=a.revision, torch_dtype=torch.float16, device_map="cuda:0",
    ).eval()

    result = {
        "model": a.model,
        "revision": a.revision,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "prompt_len": a.prompt_len,
        "steps_per_measure": a.steps,
        "repeats": a.repeats,
        "config": {
            "architectures": getattr(cfg, "architectures", None),
            "num_experts": getattr(cfg, "num_experts", None) or getattr(cfg, "n_routed_experts", None),
            "num_experts_per_tok": getattr(cfg, "num_experts_per_tok", None),
            "hidden_size": getattr(cfg, "hidden_size", None),
            "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        },
        "arms": {},
    }

    # --- profiler-overhead control (prereg P4): b=1 with and without hooks ---
    ctrl_no = measure_batch(model, tok, 1, a.prompt_len, a.steps, a.warmup, None)
    t = EventTimer(model, enabled=True)
    ctrl_yes = measure_batch(model, tok, 1, a.prompt_len, a.steps, a.warmup, t)
    overhead = abs(ctrl_yes["per_step_ms"] - ctrl_no["per_step_ms"]) / ctrl_no["per_step_ms"]
    result["profiler_overhead_check"] = {
        "b1_no_profiler_ms": ctrl_no["per_step_ms"],
        "b1_with_profiler_ms": ctrl_yes["per_step_ms"],
        "relative_overhead": overhead,
        "p4_pass": overhead <= 0.03,
    }

    # --- ABBA ladder: ascending then descending, same session ---
    for arm, order in (("ascending", batches), ("descending", list(reversed(batches)))):
        arm_rows = []
        for b in order:
            reps = []
            for _ in range(a.repeats):
                try:
                    reps.append(measure_batch(model, tok, b, a.prompt_len, a.steps, a.warmup, t))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    reps.append({"batch": b, "oom": True})
                    break
            ok = [r for r in reps if not r.get("oom")]
            if ok:
                row = {
                    "batch": b,
                    "per_step_ms_median": statistics.median(r["per_step_ms"] for r in ok),
                    "n_reps": len(ok),
                }
                if "spine_ms_per_step" in ok[0]:
                    row["spine_ms_per_step_median"] = statistics.median(r["spine_ms_per_step"] for r in ok)
                    row["expert_ms_per_step_median"] = statistics.median(r["expert_ms_per_step"] for r in ok)
                    row["spine_frac_median"] = statistics.median(r["spine_frac"] for r in ok)
            else:
                row = {"batch": b, "oom": True}
            arm_rows.append(row)
            print(f"[{arm}] b={b}: {json.dumps(row)}", flush=True)
        result["arms"][arm] = arm_rows

    t.close()
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
