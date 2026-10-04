"""Write the disaggregation fixtures from Disaggregated_Inference_Sim at the vendored commit.

The live simulator in chapters 11-14 runs the simulator's own JavaScript port
(`src/lib/disagg/vendor/sim_engine.js`, vendored by `pnpm vendor:sim`). This
script runs the Python package at the same commit and writes three things:

* `tests/unit/fixtures/disagg_parity.json`: engine parity. Thirteen
  configurations (colocated and disaggregated, heterogeneous pools, power caps
  and DVFS, KV compression in transit and at the GPU, FFT-mixing models on the
  optical transform engine), each with its workload and every request's six
  timestamps, every instance's energy counters and the link's counters, as the
  simulator's own JS-parity tests compare them.
* `tests/unit/fixtures/disagg_results.json`: the results.md rows the chapters
  quote. Each entry holds the configuration, the Python summary and the
  matching line of `examples/results.md`; this script checks that every value,
  formatted the way `examples/results.py` formats it, appears in that line.
* `public/disagg/workloads/seed1-rate<r>.json`: the recorded workloads (seed 1,
  800 requests, prompts 2,048 tokens cv 0.5, outputs 256 cv 0.5), which the
  browser loads so its runs use the very requests Python used. Python's
  `random` is Mersenne Twister; a JavaScript port reproduces every length but
  V8's Math.log differs from glibc's in the last bit for some inputs, which
  moves arrival times by an ulp, so the workloads are recorded instead.

Run with the simulator's virtualenv, the simulator checked out at the commit
recorded in `src/lib/disagg/vendor/VENDORED.json`:

    ../Disaggregated_Inference_Sim/.venv/bin/python scripts/disagg_reference.py [path/to/sim]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "Disaggregated_Inference_Sim"
sys.path.insert(0, str(SIM / "src"))

from disagg_sim.hardware import ACCELERATORS, KV_PRESETS, LINKS, MODELS, KVTransit  # noqa: E402
from disagg_sim.metrics import percentile, summarise  # noqa: E402
from disagg_sim.sim import SimConfig, simulate  # noqa: E402
from disagg_sim.workload import LengthDist, poisson_workload  # noqa: E402

VENDORED = json.loads((ROOT / "src/lib/disagg/vendor/VENDORED.json").read_text())
FIXTURES = ROOT / "tests/unit/fixtures"
WORKLOADS = ROOT / "public/disagg/workloads"
RATES = [2, 3, 4, 5, 6, 8, 10, 12, 14]


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(SIM), *args], capture_output=True, text=True, check=True).stdout.strip()


def py_config(c: dict) -> SimConfig:
    """The Python SimConfig a JS config means (the JS port's keys are camelCase).
    The same mapping as the simulator's tests/test_optical.py `py_config`, plus link
    channels and per-pool power caps."""
    def dev(key):
        if key is None:
            return None
        d = ACCELERATORS[key]
        e = c.get("engine", {})
        if d.transform is not None and e:
            names = {"enob": "enob", "maskRate": "mask_rate_hz", "detection": "detection", "overlap": "overlap"}
            over = {names[k]: v for k, v in e.items() if k in names}
            if "staticW" in e:
                over["laser_w"] = over["tuning_w"] = e["staticW"] / 2
            d = replace(d, transform=replace(d.transform, **over))
        if "fftEff" in c:
            d = replace(d, fft_efficiency=c["fftEff"])
        return d

    model = replace(MODELS[c.get("model", "llama3-70b")], prefill_lm_head=c.get("lmHead", "all"))
    tr = None
    if "kvCompress" in c:
        tr = KVTransit(KV_PRESETS[c["kvCompress"]], where=c.get("kvCompressAt", "transit"))
    return SimConfig(model=model, device=dev(c.get("device", "h100")), devices_per_instance=c.get("devicesPerInstance", 4),
                     mode=c.get("mode", "disagg"), n_prefill=c.get("nPrefill", 1), n_decode=c.get("nDecode", 1),
                     n_colocated=c.get("nColocated", 2),
                     link=replace(LINKS[c.get("link", "ib-ndr")], channels=c.get("linkChannels", 1)),
                     power_cap_w=c.get("powerCap"), prefill_power_cap_w=c.get("prefillPowerCap"),
                     decode_power_cap_w=c.get("decodePowerCap"), dvfs=c.get("dvfs", False),
                     prefill_device=dev(c.get("prefillDevice")), decode_device=dev(c.get("decodeDevice")),
                     prefill_devices_per_instance=c.get("prefillDevicesPerInstance"),
                     decode_devices_per_instance=c.get("decodeDevicesPerInstance"), kv_transit=tr,
                     fast_forward=c.get("fast", False))


def js_config(c: dict) -> dict:
    """The full JS config: defaults made explicit, Python-only keys dropped."""
    out = {"model": "llama3-70b", "device": "h100", "devicesPerInstance": 4, "mode": "disagg", "nPrefill": 1,
           "nDecode": 1, "nColocated": 2, "link": "ib-ndr", "ttftSlo": 1.0, "tpotSlo": 0.025}
    out.update({k: v for k, v in c.items() if k != "fast"})
    return out


# ───────────────────────────────────────────────────────────── 1. parity ──
PARITY = {
    "70B 1P1D, IB": {},
    "70B 2P2D, 25 GbE x2 channels": dict(nPrefill=2, nDecode=2, link="eth-25g", linkChannels=2),
    "70B colocated x2": dict(mode="colocated"),
    "70B 1P1D, cap 350 W + DVFS": dict(powerCap=350.0, dvfs=True),
    "70B colocated, cap 300 W": dict(mode="colocated", powerCap=300.0),
    "70B 1P1D, decode cap 250 W + DVFS": dict(decodePowerCap=250.0, dvfs=True),
    "8B H100 prefill + A100 decode": dict(model="llama3-8b", devicesPerInstance=1, prefillDevice="h100",
                                          decodeDevice="a100"),
    "8B 2x A100 prefill + H100 decode": dict(model="llama3-8b", devicesPerInstance=1, prefillDevice="a100",
                                             decodeDevice="h100", nPrefill=2),
    "8B fp8 in transit, 25 GbE": dict(model="llama3-8b", devicesPerInstance=1, link="eth-25g", kvCompress="fp8"),
    "8B fp4-block at the GPU, 25 GbE": dict(model="llama3-8b", devicesPerInstance=1, link="eth-25g",
                                            kvCompress="fp4-block", kvCompressAt="endpoint"),
    "8B co-packaged optics link": dict(model="llama3-8b", devicesPerInstance=1, link="cpo-optical"),
    "circulant last-token, optical prefill (optimistic)": dict(
        model="llama3-8b-hyena-circ", lmHead="last", devicesPerInstance=1, prefillDevice="optical-fft",
        engine=dict(enob=11, maskRate=20000.0, overlap=True)),
    "Hyena distilled decode, colocated": dict(model="llama3-8b-hyena-dist", devicesPerInstance=1, mode="colocated"),
}


def parity() -> list[dict]:
    cases = []
    for name, c in PARITY.items():
        wl = poisson_workload(5.0, 200, LengthDist(2048, 0.6), LengthDist(128, 0.6), seed=5)
        rows = [[r.arrival, r.prompt_len, r.output_len] for r in wl]
        res = simulate(py_config(c), wl)
        m = summarise(res)
        cases.append({
            "name": name, "cfg": js_config(c), "rows": rows,
            "stamps": [[r.prefill_start, r.first_token, r.kv_start, r.kv_ready, r.decode_start, r.finish] for r in wl],
            "inst": [[i.compute_j, i.memory_j, i.optical_j, i.busy, i.peak_power] for i in res.instances],
            "link": [res.link.energy, res.link.transit_j, res.link.bytes, res.link.busy, res.link.wait],
            "capped": any(i.power_bound_time > 0 for i in res.instances),
            "summary": {"ttft_p99": m["latency_s"]["ttft"]["p99"], "tpot_p99": m["latency_s"]["tpot"]["p99"],
                        "itl_p99": m["latency_s"]["itl"]["p99"], "slo": m["throughput"]["slo_attainment"],
                        "j_tok": m["energy"]["J_per_output_token"]},
        })
    return cases


# ─────────────────────────────────────────────────── 2. results.md rows ──
def ms(x):            # examples/results.py ms()
    return f"{1e3 * x:,.1f} ms" if x < 1 else f"{1e3 * x:,.0f} ms"


def ms2(x):           # examples/results.py render_optical ms2()
    return f"{1e3 * x:,.1f} ms" if x < 10 else f"{x:,.1f} s"


def pct(x):
    return f"{100 * x:.1f}%"


FMT = {"ms": ms, "ms2": ms2, "pct": pct, "W": lambda x: f"{x:,.0f} W", "f2": lambda x: f"{x:.2f}",
       "f3": lambda x: f"{x:.3f}"}

L8 = dict(model="llama3-8b", devicesPerInstance=1, fast=True)
CIRC = dict(model="llama3-8b-hyena-circ", lmHead="last", devicesPerInstance=1, fast=True)
OPTIMISTIC = dict(enob=11, maskRate=20000.0, overlap=True)


def sweep_rows():
    for r in [2, 3, 4, 5, 6, 8]:
        yield (f"sweep-{r}", "4", None, f"| {r} |", r,
               {"colocated": dict(mode="colocated"), "disagg": {}},
               [("colocated", "tpot_p99", "ms"), ("colocated", "slo", "pct"), ("disagg", "tpot_p99", "ms"),
                ("disagg", "ttft_p99", "ms"), ("disagg", "slo", "pct")])


REFERENCE = [
    *sweep_rows(),
    ("itl-4", "4", None, "* Inter-token latency at 4 req/s", 4,
     {"colocated": dict(mode="colocated"), "disagg": {}},
     [("colocated", "itl_p50", "ms"), ("colocated", "itl_p99", "ms"), ("colocated", "itl_max", "ms"),
      ("disagg", "itl_p50", "ms"), ("disagg", "itl_p99", "ms"), ("disagg", "itl_max", "ms")]),
    ("energy-colocated", "3", None, "| Colocated, 2 instances |", 4, {"run": dict(mode="colocated")},
     [("run", "ttft_p99", "ms"), ("run", "tpot_p99", "ms"), ("run", "slo", "pct"), ("run", "avg_w", "W"),
      ("run", "j_tok", "f2")]),
    ("energy-1p1d", "3", None, "| Disaggregated 1P1D |", 4, {"run": {}},
     [("run", "ttft_p99", "ms"), ("run", "tpot_p99", "ms"), ("run", "slo", "pct"), ("run", "avg_w", "W"),
      ("run", "j_tok", "f2")]),
    ("hetero-hh", "10", None, "| H100 prefill + H100 decode (--device h100) |", 8, {"run": dict(L8)},
     [("run", "ttft_p99", "ms2"), ("run", "tpot_p99", "ms"), ("run", "goodput", "f2"), ("run", "slo", "pct"),
      ("run", "j_tok", "f3"), ("run", "avg_w", "W")]),
    ("hetero-ha", "10", None, "| H100 prefill + A100 decode |", 8,
     {"run": dict(L8, prefillDevice="h100", decodeDevice="a100")},
     [("run", "ttft_p99", "ms2"), ("run", "tpot_p99", "ms"), ("run", "goodput", "f2"), ("run", "slo", "pct"),
      ("run", "j_tok", "f3"), ("run", "avg_w", "W")]),
    ("hetero-ah", "10", None, "| A100 prefill + H100 decode |", 8,
     {"run": dict(L8, prefillDevice="a100", decodeDevice="h100")},
     [("run", "ttft_p99", "ms2"), ("run", "tpot_p99", "ms"), ("run", "slo", "pct"), ("run", "j_tok", "f3")]),
    ("hetero-2ah", "10", None, "| 2x A100 prefill + H100 decode |", 8,
     {"run": dict(L8, prefillDevice="a100", decodeDevice="h100", nPrefill=2)},
     [("run", "ttft_p99", "ms2"), ("run", "tpot_p99", "ms"), ("run", "slo", "pct"), ("run", "j_tok", "f3")]),
    ("t11-colocated", "11", "**Transformer (Llama-3-8B)**", "| Colocated x2, H100 |", 8,
     {"run": dict(L8, mode="colocated")},
     [("run", "ttft_p99", "ms2"), ("run", "tpot_p99", "ms"), ("run", "slo", "pct"), ("run", "j_tok", "f3")]),
    ("t11-1p1d", "11", "**Transformer (Llama-3-8B)**", "| 1P1D H100 |", 8, {"run": dict(L8)},
     [("run", "ttft_p99", "ms2"), ("run", "tpot_p99", "ms"), ("run", "slo", "pct"), ("run", "j_tok", "f3")]),
    ("circ-gpu", "11", "**Hyena-2 + circulant 256, last-token head**", "| 1P1D H100 |", 8, {"run": dict(CIRC)},
     [("run", "ttft_p99", "ms2"), ("run", "tpot_p99", "ms"), ("run", "slo", "pct"), ("run", "j_tok", "f3")]),
    ("circ-optical", "11", "**Hyena-2 + circulant 256, last-token head**",
     "| optical-fft prefill (optimistic) + H100 decode |", 8,
     {"run": dict(CIRC, prefillDevice="optical-fft", engine=OPTIMISTIC)},
     [("run", "ttft_p99", "ms2"), ("run", "tpot_p99", "ms"), ("run", "slo", "pct"), ("run", "j_tok", "f3")]),
    *[(f"link-{lk}", "14", "**GQA KV cache**", f"| {lk} |", 8, {"run": dict(L8, link=lk)},
       [("run", "kv_share", "pct"), ("run", "link_util", "pct"), ("run", "ttft_p99", "ms2"), ("run", "tpot_p99", "ms"),
        ("run", "slo", "pct")])
      for lk in ["nvlink4", "ib-ndr", "cpo-optical", "eth-100g", "eth-25g"]],
    ("ib-fp8", "15", "**GQA KV cache**", "| ib-ndr | fp8 |", 8,
     {"run": dict(L8, kvCompress="fp8")},
     [("run", "handoff_p99", "ms2"), ("run", "tpot_p99", "ms"), ("run", "slo", "pct")]),
    *[(f"eth25-{r}-{pr}", "15", "**Load sweep: GQA KV over 25 GbE**", f"| {r} req/s | {pr} |", r,
       {"run": dict(L8, link="eth-25g", **({} if pr == "none" else {"kvCompress": pr}))},
       [("run", "link_util", "pct"), ("run", "handoff_p99", "ms2"), ("run", "tpot_p99", "ms"), ("run", "slo", "pct")])
      for r in [8, 10, 12, 14] for pr in ["none", "fp8"]],
]


# Lines of results.md the chapters quote as text (bounds, bullets, headers):
# (id, section, sub-heading or None, line prefix). Stored verbatim; the chapter
# tests check each quoted string is in its line.
QUOTES = [
    ("capacity", "8", None, "* Prefill 7.49 req/s"),
    ("2p1d", "4", None, "* 2P1D, rates"),
    ("proportionality-0.5", "6", None, "| 0.5 req/s |"),
    ("handoff-bytes", "14", None, "Photonic interconnect is *not* Fourier optics"),
    ("eth25-saturation", "15", None, "**Load sweep: GQA KV over 25 GbE**"),
]


def metrics(res) -> dict:
    m = summarise(res)
    lat, ss = m["latency_s"], m["stage_share"]
    hand = [r.kv_ready - r.first_token for r in res.requests if r.kv_ready is not None]
    return {"ttft_p99": lat["ttft"]["p99"], "tpot_p99": lat["tpot"]["p99"], "itl_p50": lat["itl"]["p50"],
            "itl_p99": lat["itl"]["p99"], "itl_max": lat["itl"]["max"], "slo": m["throughput"]["slo_attainment"],
            "goodput": m["throughput"]["goodput_req_per_s"], "avg_w": m["energy"]["avg_power_W"],
            "j_tok": m["energy"]["J_per_output_token"],
            "kv_share": ss.get("kv_wait", 0.0) + ss.get("kv_transfer", 0.0),
            "link_util": m["utilisation"].get("kv-link", 0.0),
            "handoff_p99": percentile(hand, 99) if hand else 0.0,
            "hotspot": f"{m['hotspots']['stage']} -> {m['hotspots']['resource']}"}


def results_line(md: list[str], section: str, sub: str | None, prefix: str) -> str:
    start = next(i for i, l in enumerate(md) if l.startswith(f"## {section}. "))
    end = next((i for i in range(start + 1, len(md)) if md[i].startswith("## ")), len(md))
    if sub is not None:
        start = next(i for i in range(start, end) if md[i].startswith(sub))
        end = next((i for i in range(start + 1, end) if md[i].startswith("**")), end)
    hits = [l for l in md[start:end] if l.startswith(prefix)]
    if not hits:
        raise SystemExit(f"results.md §{section} {sub or ''}: no line starts with {prefix!r}")
    return hits[0]


def reference(workloads: dict[int, list]) -> list[dict]:
    md = (SIM / "examples/results.md").read_text().splitlines()
    out = []
    for rid, section, sub, prefix, rate, runs, cells in REFERENCE:
        line = results_line(md, section, sub, prefix)
        wl = poisson_workload(float(rate), 800, LengthDist(2048, 0.5), LengthDist(256, 0.5), seed=1)
        assert [[r.arrival, r.prompt_len, r.output_len] for r in wl] == workloads[rate]
        fresh = lambda: poisson_workload(float(rate), 800, LengthDist(2048, 0.5), LengthDist(256, 0.5), seed=1)
        py = {k: metrics(simulate(py_config(c), fresh())) for k, c in runs.items()}
        formatted = []
        for run, key, fmt in cells:
            s = FMT[fmt](py[run][key])
            if s not in line:
                raise SystemExit(f"{rid}: {run}.{key} = {s!r} not in results.md line {line!r}")
            formatted.append({"run": run, "key": key, "fmt": fmt, "text": s})
        out.append({"id": rid, "section": section, "rate": rate, "line": line,
                    "runs": {k: js_config(c) for k, c in runs.items()}, "python": py, "cells": formatted})
        print(f"  {rid:28s} ok  {line[:70]}")
    return out


def main() -> None:
    commit = git("rev-parse", "HEAD")
    if commit != VENDORED["commit"]:
        raise SystemExit(f"simulator is at {commit[:7]}, the vendored engine at {VENDORED['commit'][:7]}:"
                         f" check out {VENDORED['commit'][:7]} (or re-vendor) first")
    if git("status", "--porcelain", "--", "src", "web", "examples/results.md"):
        raise SystemExit("simulator checkout has local changes in src/, web/ or results.md")
    header = {"simulator": VENDORED["repository"], "commit": commit}

    WORKLOADS.mkdir(parents=True, exist_ok=True)
    workloads = {}
    for r in RATES:
        wl = poisson_workload(float(r), 800, LengthDist(2048, 0.5), LengthDist(256, 0.5), seed=1)
        workloads[r] = [[q.arrival, q.prompt_len, q.output_len] for q in wl]
        (WORKLOADS / f"seed1-rate{r}.json").write_text(json.dumps(
            {**header, "rate": r, "n": 800, "prompt": [2048, 0.5], "output": [256, 0.5], "seed": 1,
             "rows": workloads[r]}, separators=(",", ":")) + "\n")

    (FIXTURES / "disagg_parity.json").write_text(json.dumps({**header, "cases": parity()}) + "\n")
    print("results.md rows:")
    md = (SIM / "examples/results.md").read_text().splitlines()
    quotes = {qid: results_line(md, sec, sub, prefix) for qid, sec, sub, prefix in QUOTES}
    (FIXTURES / "disagg_results.json").write_text(
        json.dumps({**header, "rows": reference(workloads), "quotes": quotes}, indent=1) + "\n")
    print(f"wrote fixtures and {len(RATES)} workloads at {commit[:7]}")


if __name__ == "__main__":
    main()
