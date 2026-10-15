"""A small synthetic layer export in the spec/INTERFACES.md format, for tests and the offline demo.

Random bf16-valued weights and inputs with OLMoE's structure (much smaller sizes). Not a real model: results on it
say the tooling works, never anything about OLMoE.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 values to the nearest bf16 value (RNE), returned as float32."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.astype(np.uint32).view(np.float32)


def make_export(directory: str | Path, *, hidden: int = 64, inter: int = 32, experts: int = 16, top_k: int = 4,
                tokens: int = 64, seed: int = 0) -> Path:
    d = Path(directory)
    (d / "weights").mkdir(parents=True, exist_ok=True)
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    shapes = {"o_proj": (hidden, hidden), "router": (experts, hidden),
              "gate_up_proj": (experts, 2 * inter, hidden), "down_proj": (experts, hidden, inter)}
    for name, shape in shapes.items():
        np.save(d / "weights" / f"{name}.npy", bf16(rng.normal(0, 1 / np.sqrt(shape[-1]), shape)))
    np.save(d / "weights" / "post_attention_layernorm.npy", bf16(1 + 0.1 * rng.normal(size=hidden)))
    np.save(d / "inputs" / "o_proj_in.npy", bf16(rng.normal(size=(tokens, hidden))))
    np.save(d / "inputs" / "residual.npy", bf16(rng.normal(size=(tokens, hidden))))
    np.save(d / "inputs" / "token_ids.npy", np.arange(tokens, dtype="<i4"))
    meta = [{"prompt_id": t // 8, "position": t % 8, "split": "calibration" if (t // 8) % 2 == 0 else "evaluation"}
            for t in range(tokens)]
    (d / "inputs" / "token_meta.json").write_text(json.dumps(meta))
    (d / "config.json").write_text(json.dumps({
        "model_id": "synthetic", "layer": 0, "hidden_size": hidden, "intermediate_size": inter,
        "num_experts": experts, "top_k": top_k, "norm_topk_prob": False, "rms_norm_eps": 1e-5,
        "model_dtype": "bfloat16", "checkpoint_dtype": "bfloat16", "experts_implementation": "eager",
        "provenance": {"note": "synthetic export for tests and the offline demo", "seed": seed},
    }, indent=2))
    return d
