"""
reference.py — Reference PyTorch implementation of the toy decoder.

PURPOSE
=======
Generates numerical fixtures that the TypeScript implementation in
src/lib/transformer/ must match to within 1e-5. This is the authoritative
numerical source of truth for the project. Run it manually whenever the
toy-model definition in SPEC.md §4 changes; commit the regenerated fixtures.

WHY A SEPARATE PYTHON SCRIPT?
=============================
Two reasons:
  1. PyTorch is the de facto reference for "what the right answer is"
     in transformer maths. By generating fixtures from it we make sure
     our hand-written TS isn't just self-consistent but actually correct.
  2. It documents the toy model in a second, executable form. If our
     TS and this script disagree, one of them has a bug.

USAGE
=====
    pip install torch numpy
    python scripts/reference.py

Outputs JSON files into tests/unit/fixtures/. These are checked into git.

The TS unit tests load the JSON and assert max-abs-error < 1e-5 against
their own forward passes with the same seed and inputs.

CONVENTIONS — must match SPEC.md §4 and src/lib/transformer/* exactly
=====================================================================
- Pre-norm decoder block: x -> x + Attn(LN(x)); h -> h + FFN(LN(h))
- GELU: torch's "tanh" approximation (matches our TS impl)
- Positional encoding: sinusoidal (Vaswani 2017 §3.5)
- Causal mask: upper triangular -inf above the main diagonal
- Softmax: numerically stable (subtract row-max before exp)
- LayerNorm: epsilon = 1e-5
- Char tokenizer over the 64-char alphabet defined below
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants — keep in sync with SPEC.md §4 and src/lib/transformer/tokenizer.ts
# ---------------------------------------------------------------------------
ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"  # 26
    "0123456789"  # 10
    " .,!?;:'\"()-\n"  # 13
    "+-*/=<>[]{}@#$%^&_|"  # 19 (extras to round out to >=64)
)[:64]
assert len(ALPHABET) == 64, f"alphabet must be 64 chars, got {len(ALPHABET)}"
CHAR_TO_ID = {c: i for i, c in enumerate(ALPHABET)}

VOCAB_SIZE = 64
SEED = 42
EPS = 1e-5

# Default toy-model config (SPEC §4).
DEFAULT_CONFIG = dict(
    seq_len=8,
    d_model=16,
    n_heads=2,
    d_ff=32,
    n_blocks=2,
    vocab_size=VOCAB_SIZE,
    seed=SEED,
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def encode(text: str, seq_len: int) -> list[int]:
    """Encode `text` to token ids, truncate or pad with id 0 to `seq_len`."""
    ids = [CHAR_TO_ID.get(c, 0) for c in text[:seq_len]]
    while len(ids) < seq_len:
        ids.append(0)
    return ids


# ---------------------------------------------------------------------------
# Positional encoding (sinusoidal, Vaswani 2017)
# ---------------------------------------------------------------------------
def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


# ---------------------------------------------------------------------------
# Weight init — deterministic from a seed.
# ---------------------------------------------------------------------------
def init_weights(config: dict) -> dict[str, torch.Tensor]:
    """Initialise all weights from a single seed.

    Uses normal(0, 0.02) like GPT-2 for the matrices; zeros for biases;
    γ=1, β=0 for layernorm. Order of draws is significant — keep in sync
    with src/lib/transformer/model.ts.
    """
    g = torch.Generator().manual_seed(config["seed"])
    d_model = config["d_model"]
    d_ff = config["d_ff"]
    vocab_size = config["vocab_size"]
    n_blocks = config["n_blocks"]

    def normal(*shape: int) -> torch.Tensor:
        return torch.empty(*shape).normal_(mean=0.0, std=0.02, generator=g)

    weights: dict[str, torch.Tensor] = {
        "tok_emb": normal(vocab_size, d_model),
    }
    for i in range(n_blocks):
        prefix = f"block_{i}"
        weights[f"{prefix}.ln1.gamma"] = torch.ones(d_model)
        weights[f"{prefix}.ln1.beta"] = torch.zeros(d_model)
        weights[f"{prefix}.attn.W_q"] = normal(d_model, d_model)
        weights[f"{prefix}.attn.W_k"] = normal(d_model, d_model)
        weights[f"{prefix}.attn.W_v"] = normal(d_model, d_model)
        weights[f"{prefix}.attn.W_o"] = normal(d_model, d_model)
        weights[f"{prefix}.ln2.gamma"] = torch.ones(d_model)
        weights[f"{prefix}.ln2.beta"] = torch.zeros(d_model)
        weights[f"{prefix}.ffn.W1"] = normal(d_model, d_ff)
        weights[f"{prefix}.ffn.b1"] = torch.zeros(d_ff)
        weights[f"{prefix}.ffn.W2"] = normal(d_ff, d_model)
        weights[f"{prefix}.ffn.b2"] = torch.zeros(d_model)
    weights["ln_final.gamma"] = torch.ones(d_model)
    weights["ln_final.beta"] = torch.zeros(d_model)
    # We tie the output head to the token embedding (common in tiny GPTs).
    return weights


# ---------------------------------------------------------------------------
# Ops — written explicitly (no nn.Module) so each op maps 1:1 to a TS file.
# ---------------------------------------------------------------------------
def layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + EPS) * gamma + beta


def gelu(x: torch.Tensor) -> torch.Tensor:
    # tanh approximation — matches our TS impl (see src/lib/transformer/gelu.ts).
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
            )
        )
    )


def causal_mask(seq_len: int) -> torch.Tensor:
    """Upper triangular -inf above the main diagonal; 0 elsewhere."""
    m = torch.zeros(seq_len, seq_len)
    m.masked_fill_(torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(), float("-inf"))
    return m


def multi_head_attention(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    n_heads: int,
) -> dict[str, torch.Tensor]:
    """Causal multi-head self-attention. Returns the output and a trace dict.

    Shapes:
        x:   [seq_len, d_model]
        W_*: [d_model, d_model]
        out: [seq_len, d_model]
    """
    seq_len, d_model = x.shape
    d_head = d_model // n_heads
    assert d_model % n_heads == 0

    Q = x @ W_q  # [S, D]
    K = x @ W_k
    V = x @ W_v

    # Reshape to [n_heads, S, d_head]
    Q_h = Q.view(seq_len, n_heads, d_head).transpose(0, 1)
    K_h = K.view(seq_len, n_heads, d_head).transpose(0, 1)
    V_h = V.view(seq_len, n_heads, d_head).transpose(0, 1)

    scores = Q_h @ K_h.transpose(-2, -1) / math.sqrt(d_head)  # [H, S, S]
    mask = causal_mask(seq_len)
    masked = scores + mask  # broadcasts over heads
    weights = F.softmax(masked, dim=-1)
    head_out = weights @ V_h  # [H, S, d_head]

    concat = head_out.transpose(0, 1).contiguous().view(seq_len, d_model)
    out = concat @ W_o

    return {
        "Q": Q,
        "K": K,
        "V": V,
        "scores": scores,
        "mask": mask,
        "weights": weights,
        "head_out": head_out,
        "out": out,
    }


def ffn(
    x: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    return gelu(x @ W1 + b1) @ W2 + b2


def block_forward(x: torch.Tensor, w: dict, prefix: str, n_heads: int) -> torch.Tensor:
    """Pre-norm decoder block (SPEC §4)."""
    h = x + multi_head_attention(
        layernorm(x, w[f"{prefix}.ln1.gamma"], w[f"{prefix}.ln1.beta"]),
        w[f"{prefix}.attn.W_q"],
        w[f"{prefix}.attn.W_k"],
        w[f"{prefix}.attn.W_v"],
        w[f"{prefix}.attn.W_o"],
        n_heads,
    )["out"]
    h = h + ffn(
        layernorm(h, w[f"{prefix}.ln2.gamma"], w[f"{prefix}.ln2.beta"]),
        w[f"{prefix}.ffn.W1"],
        w[f"{prefix}.ffn.b1"],
        w[f"{prefix}.ffn.W2"],
        w[f"{prefix}.ffn.b2"],
    )
    return h


def forward(token_ids: list[int], config: dict, weights: dict) -> dict:
    """Full forward pass. Returns a dict with logits and intermediates."""
    seq_len = config["seq_len"]
    d_model = config["d_model"]
    n_blocks = config["n_blocks"]
    n_heads = config["n_heads"]

    ids = torch.tensor(token_ids, dtype=torch.long)
    tok_emb = weights["tok_emb"][ids]  # [S, D]
    pos_emb = positional_encoding(seq_len, d_model)
    x = tok_emb + pos_emb

    for i in range(n_blocks):
        x = block_forward(x, weights, f"block_{i}", n_heads)

    x_final = layernorm(x, weights["ln_final.gamma"], weights["ln_final.beta"])
    logits = x_final @ weights["tok_emb"].T  # tied head

    return {
        "tok_emb": tok_emb,
        "pos_emb": pos_emb,
        "x_after_emb": tok_emb + pos_emb,
        "x_final": x_final,
        "logits": logits,
    }


# ---------------------------------------------------------------------------
# Fixture writer
# ---------------------------------------------------------------------------
def to_jsonable(t: torch.Tensor) -> list:
    return t.detach().cpu().to(torch.float64).tolist()


def write_fixture(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"wrote {path}")


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "tests" / "unit" / "fixtures"
    config = DEFAULT_CONFIG
    weights = init_weights(config)

    # Per-op fixtures with small fixed inputs ----------------------------------
    g = torch.Generator().manual_seed(123)
    x_small = torch.empty(config["seq_len"], config["d_model"]).normal_(generator=g)

    write_fixture(
        out_dir / "softmax.json",
        {
            "input": to_jsonable(x_small),
            "output_dim_neg1": to_jsonable(F.softmax(x_small, dim=-1)),
        },
    )

    write_fixture(
        out_dir / "layernorm.json",
        {
            "input": to_jsonable(x_small),
            "gamma": to_jsonable(torch.ones(config["d_model"])),
            "beta": to_jsonable(torch.zeros(config["d_model"])),
            "epsilon": EPS,
            "output": to_jsonable(
                layernorm(x_small, torch.ones(config["d_model"]), torch.zeros(config["d_model"]))
            ),
        },
    )

    write_fixture(
        out_dir / "gelu.json",
        {
            "input": to_jsonable(x_small),
            "output": to_jsonable(gelu(x_small)),
        },
    )

    # Matmul: [S,D] @ [D,D]
    A = x_small
    g2 = torch.Generator().manual_seed(7)
    B = torch.empty(config["d_model"], config["d_model"]).normal_(generator=g2)
    write_fixture(
        out_dir / "matmul.json",
        {
            "A": to_jsonable(A),
            "B": to_jsonable(B),
            "output": to_jsonable(A @ B),
        },
    )

    write_fixture(
        out_dir / "causal_mask.json",
        {
            "seq_len": config["seq_len"],
            "output": to_jsonable(causal_mask(config["seq_len"])),
        },
    )

    # Single attention forward
    attn = multi_head_attention(
        x_small,
        weights["block_0.attn.W_q"],
        weights["block_0.attn.W_k"],
        weights["block_0.attn.W_v"],
        weights["block_0.attn.W_o"],
        config["n_heads"],
    )
    write_fixture(
        out_dir / "attention.json",
        {
            "input": to_jsonable(x_small),
            "W_q": to_jsonable(weights["block_0.attn.W_q"]),
            "W_k": to_jsonable(weights["block_0.attn.W_k"]),
            "W_v": to_jsonable(weights["block_0.attn.W_v"]),
            "W_o": to_jsonable(weights["block_0.attn.W_o"]),
            "n_heads": config["n_heads"],
            "Q": to_jsonable(attn["Q"]),
            "K": to_jsonable(attn["K"]),
            "V": to_jsonable(attn["V"]),
            "scores": to_jsonable(attn["scores"]),
            "weights": to_jsonable(attn["weights"]),
            "output": to_jsonable(attn["out"]),
        },
    )

    # Full-model forward fixture
    sample_text = "hello!"
    token_ids = encode(sample_text, config["seq_len"])
    fwd = forward(token_ids, config, weights)
    write_fixture(
        out_dir / "forward.json",
        {
            "config": config,
            "input_text": sample_text,
            "token_ids": token_ids,
            "tok_emb": to_jsonable(fwd["tok_emb"]),
            "pos_emb": to_jsonable(fwd["pos_emb"]),
            "x_after_emb": to_jsonable(fwd["x_after_emb"]),
            "x_final": to_jsonable(fwd["x_final"]),
            "logits": to_jsonable(fwd["logits"]),
            "weights": {k: to_jsonable(v) for k, v in weights.items()},
        },
    )

    # Manifest so the TS side knows what to expect
    write_fixture(
        out_dir / "manifest.json",
        {
            "version": 1,
            "tolerance": 1.0e-5,
            "config": config,
            "alphabet": ALPHABET,
            "files": [
                "softmax.json",
                "layernorm.json",
                "gelu.json",
                "matmul.json",
                "causal_mask.json",
                "attention.json",
                "forward.json",
            ],
        },
    )

    print("\nAll fixtures written.")
    print("Commit tests/unit/fixtures/ to git.")


if __name__ == "__main__":
    main()
