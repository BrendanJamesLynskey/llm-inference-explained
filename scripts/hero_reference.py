"""Write the brief 41 hero fixtures from independent Python references.

    python scripts/hero_reference.py path/to/Disaggregated_Inference_Sim

The simulator checkout must be at the commit tests/unit/fixtures/cost_model.json
records (its hardware.py is the cost model's reference); numpy and torch must
be installed (transformer-explainer's .venv-fixtures has both). Writes:

  tests/unit/fixtures/heroes.json   the chapter heroes' states:
      timeline  chapter 1: one request's prefill then decode steps, priced by
                the simulator's CostModel (Llama-3-8B, one H100);
      roofline  chapter 3: the decode step's intensity as the batch grows;
      kv        chapter 2: the trained toy model filling its KV cache, from a
                float64 numpy forward pass recomputed at every step (no cache:
                an independent route to the same keys, values and weights);
      flash     chapter 6: FlashAttention's tile schedule with each query
                row's running max and sum, from a plain-Python online softmax.
  public/disagg/hero.json           the landing hero's two configurations and
                rows, cut from tests/unit/fixtures/disagg_parity.json (cases
                "70B colocated x2" and "70B 1P1D, IB").

Every existing fixture is left as it is. Re-run and commit the JSON whenever a
hero's model or parameters change.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SIM = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "Disaggregated_Inference_Sim"
sys.path.insert(0, str(SIM / "src"))

from disagg_sim.hardware import H100_SXM, LLAMA3_8B, CostModel  # noqa: E402

FIX = ROOT / "tests" / "unit" / "fixtures"


def sim_commit() -> str:
    return subprocess.run(["git", "-C", str(SIM), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


# ---------------------------------------------------------------- chapter 1
TIMELINE = {"model": "llama3-8b", "device": "h100", "nDevices": 1, "prompt": 2048, "output": 16}


def timeline() -> dict:
    cm = CostModel(LLAMA3_8B, H100_SXM, n_devices=1)
    dev = H100_SXM
    fr = dev.peak_flops * dev.flops_eff * 1  # derated FLOP/s and bytes/s, as the TS forms them
    br = dev.mem_bw * dev.bw_eff * 1
    p = cm.prefill([TIMELINE["prompt"]])
    steps = [{"k": 0, "kind": "prefill", "t0": 0.0, "dt": p.time, "flops": p.flops,
              "bytes": p.bytes, "bound": p.bound, "tCompute": p.flops / fr, "tMemory": p.bytes / br,
              "ctx": TIMELINE["prompt"], "out": 1}]
    t = p.time
    for k in range(1, TIMELINE["output"]):
        ctx = TIMELINE["prompt"] + k - 1
        d = cm.decode([ctx])
        steps.append({"k": k, "kind": "decode", "t0": t, "dt": d.time, "flops": d.flops,
                      "bytes": d.bytes, "bound": d.bound, "tCompute": d.flops / fr,
                      "tMemory": d.bytes / br, "ctx": ctx, "out": k + 1})
        t += d.time
    return {"opts": TIMELINE, "steps": steps}


# ---------------------------------------------------------------- chapter 3
ROOF = {"model": "llama3-8b", "device": "h100", "nDevices": 1, "ctx": 2048,
        "batches": [1, 2, 4, 8, 16, 32, 64, 128, 256]}


def roofline() -> dict:
    cm = CostModel(LLAMA3_8B, H100_SXM, n_devices=1)
    dev = H100_SXM
    peak = dev.peak_flops * dev.flops_eff
    bw = dev.mem_bw * dev.bw_eff

    def point(batch, s, tokens):
        inten = s.flops / s.bytes
        return {"batch": batch, "ctx": ROOF["ctx"], "flops": s.flops, "bytes": s.bytes,
                "time": s.time, "bound": s.bound, "intensity": inten,
                "attainable": min(peak, inten * bw), "tokensPerS": tokens / s.time}

    return {"opts": ROOF, "peak": peak, "bandwidth": bw, "ridge": peak / bw,
            "points": [point(b, cm.decode([ROOF["ctx"]] * b), b) for b in ROOF["batches"]],
            "prefill": point(1, cm.prefill([ROOF["ctx"]]), ROOF["ctx"])}


# ---------------------------------------------------------------- chapter 2
ALPHABET = "abcdefghijklmnopqrstuvwxyz" "0123456789" " .,!?;:'\"()-\n" "+-*/=<>[]{}@#$%"
CHAR = {}
for i, c in enumerate(ALPHABET):
    CHAR[c] = i  # '-' appears twice; the later index wins, as in the TypeScript Map
KV_PROMPT = "said"
KV_ROUNDS = 4


def trained() -> dict:
    p = json.loads((ROOT / "public" / "data" / "trained-weights.json").read_text())["params"]
    return {k: np.array(v, dtype=np.float64) for k, v in p.items()}


def ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * g + b


def pe(S, D):
    out = np.zeros((S, D))
    for pos in range(S):
        for i in range(D // 2):
            ang = pos * math.exp(2 * i * (-math.log(10000.0) / D))
            out[pos, 2 * i] = math.sin(ang)
            out[pos, 2 * i + 1] = math.cos(ang)
    return out


def forward(ids, w, H=2):
    """Full forward pass; returns logits, each block's K and V, and each block's
    per-head attention weights of the last row."""
    S = len(ids)
    x = w["tok_emb"][ids] + pe(S, 16)
    Ks, Vs, last = [], [], []
    for b in range(2):
        q_ = f"block_{b}."
        h1 = ln(x, w[q_ + "ln1.gamma"], w[q_ + "ln1.beta"])
        Q, K, V = h1 @ w[q_ + "attn.W_q"], h1 @ w[q_ + "attn.W_k"], h1 @ w[q_ + "attn.W_v"]
        Ks.append(K.tolist())
        Vs.append(V.tolist())
        dh = 16 // H
        heads, lw = [], []
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            sc = Q[:, sl] @ K[:, sl].T / math.sqrt(dh)
            sc = sc + np.triu(np.full((S, S), -np.inf), 1)
            pr = np.exp(sc - sc.max(-1, keepdims=True))
            pr = pr / pr.sum(-1, keepdims=True)
            heads.append(pr @ V[:, sl])
            lw.append(pr[-1].tolist())
        last.append(lw)
        x = x + np.concatenate(heads, -1) @ w[q_ + "attn.W_o"]
        h2 = ln(x, w[q_ + "ln2.gamma"], w[q_ + "ln2.beta"])
        u = h2 @ w[q_ + "ffn.W1"] + w[q_ + "ffn.b1"]
        a = 0.5 * u * (1 + np.tanh(math.sqrt(2 / math.pi) * (u + 0.044715 * u ** 3)))
        x = x + a @ w[q_ + "ffn.W2"] + w[q_ + "ffn.b2"]
    xf = ln(x, w["ln_final.gamma"], w["ln_final.beta"])
    return xf @ w["tok_emb"].T, Ks, Vs, last


def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


def kv() -> dict:
    w = trained()
    ids = [CHAR.get(c, 0) for c in KV_PROMPT]
    logits, K, V, _ = forward(ids, w)
    p = softmax(logits[-1])
    nxt = int(np.argmax(p))
    states = [{"phase": "prefill", "round": 0, "ids": list(ids), "cached": len(ids),
               "pos": len(ids) - 1, "K": K, "V": V, "attn": None, "next": nxt,
               "prob": float(p[nxt])}]
    for r in range(1, KV_ROUNDS + 1):
        before_K, before_V = K, V
        ids = ids + [nxt]
        pos = len(ids) - 1
        logits, K, V, attn = forward(ids, w)
        base = {"round": r, "ids": list(ids), "pos": pos, "next": None, "prob": None}
        states.append({**base, "phase": "attend", "cached": pos, "K": before_K, "V": before_V,
                       "attn": attn})
        states.append({**base, "phase": "append", "cached": pos + 1, "K": K, "V": V, "attn": attn})
        p = softmax(logits[-1])
        nxt = int(np.argmax(p))
        states.append({**base, "phase": "pick", "cached": pos + 1, "K": K, "V": V, "attn": attn,
                       "next": nxt, "prob": float(p[nxt])})
    return {"prompt": KV_PROMPT, "rounds": KV_ROUNDS, "states": states}


# ---------------------------------------------------------------- chapter 6
U32 = 0xFFFFFFFF


def mulberry32(seed):
    state = seed & U32

    def rng():
        nonlocal state
        state = (state + 0x6D2B79F5) & U32
        t = state
        t = ((t ^ (t >> 15)) * (t | 1)) & U32
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & U32)) & U32
        return ((t ^ (t >> 14)) & U32) / 4294967296

    return rng


def randn(rows, cols, seed, scale):
    rng = mulberry32(seed)

    def sample():
        u1 = max(rng(), 1e-12)
        u2 = rng()
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    return [[sample() * scale for _ in range(cols)] for _ in range(rows)]


FLASH = {"n": 16, "d": 8, "br": 4, "bc": 4, "seeds": [11, 12, 13]}


def fold(q, K, V, a, b, st):
    """foldTile in plain Python, the TypeScript's operation order."""
    scale = 1 / math.sqrt(len(q))
    s = []
    m_tile = -math.inf
    for j in range(a, b):
        x = 0.0
        for i in range(len(q)):
            x += q[i] * K[j][i]
        x *= scale
        s.append(x)
        if x > m_tile:
            m_tile = x
    m_new = max(st["m"], m_tile)
    alpha = 0.0 if st["m"] == -math.inf else math.exp(st["m"] - m_new)
    o = [v * alpha for v in st["o"]]
    l = st["l"] * alpha
    for i, x in enumerate(s):
        p = math.exp(x - m_new)
        l += p
        for c, v in enumerate(V[a + i]):
            o[c] += p * v
    return {"m": m_new, "l": l, "o": o}


def flash() -> dict:
    n, d, br, bc = FLASH["n"], FLASH["d"], FLASH["br"], FLASH["bc"]
    Q = randn(n, d, FLASH["seeds"][0], 1.0)
    K = randn(n, d, FLASH["seeds"][1], 1.0)
    V = randn(n, d, FLASH["seeds"][2], 1.0)
    st = [{"m": -math.inf, "l": 0.0, "o": [0.0] * d} for _ in range(n)]
    out, written, hbm, t = [], [], 0, 0
    for i0 in range(0, n, br):
        i1 = min(n, i0 + br)
        hbm += (i1 - i0) * d
        for j0 in range(0, n, bc):
            j1 = min(n, j0 + bc)
            hbm += (j1 - j0) * (d + d)
            for i in range(i0, i1):
                st[i] = fold(Q[i], K, V, j0, j1, st[i])
            if j1 == n:
                hbm += (i1 - i0) * d
                written.extend(range(i0, i1))
            out.append({"t": t, "qi": i0 // br, "kj": j0 // bc, "rows": [i0, i1], "cols": [j0, j1],
                        "m": [None if s["m"] == -math.inf else s["m"] for s in st],
                        "l": [s["l"] for s in st], "written": list(written), "hbm": hbm})
            t += 1
    # the output, normalised, against textbook attention in numpy
    O = np.array([[v / s["l"] for v in s["o"]] for s in st])
    Qn, Kn, Vn = np.array(Q), np.array(K), np.array(V)
    S = Qn @ Kn.T / math.sqrt(d)
    P = np.exp(S - S.max(-1, keepdims=True))
    P /= P.sum(-1, keepdims=True)
    return {"opts": FLASH, "Q0": Q[0], "states": out,
            "maxDiffVsNaive": float(np.abs(O - P @ Vn).max())}


def main() -> None:
    commit = sim_commit()
    cm = json.loads((FIX / "cost_model.json").read_text())
    if commit != cm["simulator_commit"]:
        raise SystemExit(f"simulator at {commit}, cost_model.json was made at {cm['simulator_commit']}")
    out = {"note": "Brief 41 hero states from scripts/hero_reference.py.", "simulator_commit": commit,
           "timeline": timeline(), "roofline": roofline(), "kv": kv(), "flash": flash()}
    (FIX / "heroes.json").write_text(json.dumps(out, allow_nan=False, separators=(",", ":")) + "\n")
    print("wrote tests/unit/fixtures/heroes.json")

    par = json.loads((FIX / "disagg_parity.json").read_text())
    by = {c["name"]: c for c in par["cases"]}
    co, dis = by["70B colocated x2"], by["70B 1P1D, IB"]
    assert co["rows"] == dis["rows"]
    hero = {"simulator": par["simulator"], "commit": par["commit"], "rows": dis["rows"],
            "modes": [{"key": "colocated", "name": co["name"], "cfg": co["cfg"]},
                      {"key": "disagg", "name": dis["name"], "cfg": dis["cfg"]}]}
    (ROOT / "public" / "disagg" / "hero.json").write_text(
        json.dumps(hero, allow_nan=False, separators=(",", ":")) + "\n")
    print("wrote public/disagg/hero.json")


if __name__ == "__main__":
    main()
