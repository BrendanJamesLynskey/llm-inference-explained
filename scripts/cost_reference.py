"""Write tests/unit/fixtures/cost_model.json from Disaggregated_Inference_Sim.

The TypeScript roofline (src/lib/inference/costModel.ts) is a port of the
simulator's transformer cost model. This script runs the Python original on a
grid of steps and records every result, so the unit tests can compare the port
against it exactly (the way scripts/reference.py records PyTorch fixtures for
the vendored transformer).

Run manually, with the simulator installed or checked out next to this repo:

    python3 scripts/cost_reference.py [path/to/Disaggregated_Inference_Sim]

The simulator commit is recorded in the fixture. Re-run and commit the JSON
whenever the simulator's cost model changes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "Disaggregated_Inference_Sim"
sys.path.insert(0, str(SIM / "src"))

from disagg_sim.hardware import (  # noqa: E402
    A100_SXM, H100_SXM, LINKS, LLAMA3_70B, LLAMA3_8B, CostModel,
)

MODELS = {"llama3-8b": LLAMA3_8B, "llama3-70b": LLAMA3_70B}
DEVICES = {"h100": H100_SXM, "a100": A100_SXM}


def step(s) -> dict:
    return {"flops": s.flops, "bytes": s.bytes, "time": s.time, "bound": s.bound}


def main() -> None:
    commit = subprocess.run(["git", "-C", str(SIM), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    cases = []
    for mk, m0 in MODELS.items():
        for wb, kb in [(2.0, 2.0), (1.0, 2.0), (1.0, 1.0), (0.5, 2.0), (0.5, 1.0)]:
            m = replace(m0, weight_bytes=wb, kv_bytes=kb)
            for dk, dev in DEVICES.items():
                for n in (1, 2, 4, 8):
                    cm = CostModel(m, dev, n_devices=n)
                    base = {"model": mk, "weight_bytes": wb, "kv_bytes": kb, "device": dk, "n_devices": n}
                    for p in ([128], [2048], [512, 1024], [8192]):
                        cases.append({**base, "kind": "prefill", "prompt_lens": p, **step(cm.prefill(p))})
                    for b in (1, 8, 64, 256):
                        for ctx in (512, 2048, 8192):
                            cases.append({**base, "kind": "decode", "ctx": ctx * b, "batch": b,
                                          **step(cm.decode([ctx] * b))})
    # A step slow enough on few devices to be power-bound exercises the TDP branch.
    for mk, m in MODELS.items():
        for dk, dev in DEVICES.items():
            cm = CostModel(m, dev)
            cases.append({"model": mk, "weight_bytes": 2.0, "kv_bytes": 2.0, "device": dk, "n_devices": 1,
                          "kind": "prefill", "prompt_lens": [4096] * 4, **step(cm.prefill([4096] * 4))})
    kv = {mk: m.kv_bytes_per_token for mk, m in MODELS.items()}
    ridge = {dk: d.ridge_point for dk, d in DEVICES.items()}
    out = {"simulator_commit": commit, "kv_bytes_per_token": kv, "ridge_point": ridge,
           "params": {mk: m.params for mk, m in MODELS.items()},
           "links": {k: {"name": l.name, "bandwidth": l.bandwidth, "latency": l.latency} for k, l in LINKS.items()},
           "cases": cases}
    dest = ROOT / "tests" / "unit" / "fixtures" / "cost_model.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    bounds = {}
    for c in cases:
        bounds[c["bound"]] = bounds.get(c["bound"], 0) + 1
    print(f"wrote {len(cases)} cases to {dest.relative_to(ROOT)} (simulator {commit[:7]}; bounds {bounds})")


if __name__ == "__main__":
    main()
