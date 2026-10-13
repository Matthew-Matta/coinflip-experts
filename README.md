# Coin-Flip Experts

**Same prompt, different batch, different experts.** Mixture-of-Experts models pick the top-k experts for every token. When the k-th and (k+1)-th scores are nearly tied, tiny rounding differences — batch size, other requests in the batch, precision, chip count — decide which experts run. This repo measures how often that happens inside a production engine (SGLang), traces each flip to its cause, prices the fixes, and ships `routediff`, a tool that finds the first op responsible when porting an MoE model to new hardware.

> Status: in progress (Oct 2026). Hypotheses, baselines and method: [`PREREG.md`](PREREG.md).

## Layout
| Dir | What |
|---|---|
| `routetrace/` | Router hooks (HF + SGLang), teacher-forced driver |
| `census/` | Flip census runners and configs |
| `attribution/` | Activation patching, replay residual |
| `routediff/` | Cross-backend bisector + planted-bug suite |
| `port/` | C++ MoE block with configurable accumulator, reduction order, collectives |
| `budget/` | Determinism cost/benefit sweeps |
| `kaggle/`, `modal/` | GPU notebooks and job scripts |
| `spec/` | Route-flip tolerance budget |

## Quick start
```bash
pip install -e ".[dev]" && pytest                        # Python utilities
cmake -S port -B port/build && cmake --build port/build && ./port/build/test_moe   # C++ port
bash tools/sglang_audit.sh                                # where SGLang numerics depend on batch size
```
GPU work: upload `kaggle/00_gate.ipynb` to Kaggle (GPU T4 x2, Internet on).
