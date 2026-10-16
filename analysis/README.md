# Descriptive analyses

## Communication model

```sh
python -m analysis.collective_cost --elements 1048576 --alpha-us 1 \
  --bandwidth-gbytes-per-second 25 --out data/analysis/cost.csv
```

The example assumptions (1 microsecond and 25 decimal GB/s) are illustrative,
not fitted to a device or fabric. Latency is seconds per step; beta is seconds per
byte (the reciprocal of supplied bytes/second). Reduction compute, overlap,
contention and bandwidth saturation are omitted.

For payload S bytes and n ranks:

| Algorithm | Critical-path steps | Critical-path bytes | Aggregate bytes sent |
|---|---:|---:|---:|
| Ring reduce-scatter + all-gather | 2(n-1) | 2(n-1)S/n | 2(n-1)S |
| Full-buffer binary tree reduce + broadcast | 2 ceil(log2 n) | 2 ceil(log2 n)S | 2(n-1)S |

Time = steps * alpha + critical-path bytes / bandwidth. Ring traffic is uniform;
tree per-rank traffic is not, so a uniform per-rank value is intentionally absent.
These model two explicit communication schedules; the numerical emulator's
summation orders are not implementations of a measured NCCL runtime.

For o_proj, one logical output tensor has T*H elements. Expert-down communication
depends on when reductions happen: use actual per-expert message sizes/counts, or
a documented combined tensor. Do not silently substitute T*H for all T*K*H expert
outputs. BF16 halves bytes versus fp32, but does not halve latency or guarantee a
2x runtime speedup. CSV outputs cover n=2..64, both dtypes/topologies, and carry a
provenance sidecar.

## Real backend pairs

```sh
python -m reference.run_pairs --exports data/export/olmoe-1b-7b-0924/layer0 \
  data/export/olmoe-1b-7b-0924/layer7 --out data/dumps/real-pairs
python -m analysis.backend_pairs --pairs data/dumps/real-pairs \
  --out data/analysis/real-pairs.json
```

MPS uses the actual Apple GPU with CPU fallback disabled. CPU grouped_mm uses
Transformers' dispatcher; its selected implementation is recorded. Every dump's
mirror must equal its actual Transformers module bit-exactly. Unsupported paths
write FAIL status instead of a successful manifest.

The summary measures route-set differences, reference cutoff ties, per-op maximum
absolute error and relative L2 with bounded-memory reads. Slot-aligned expert ops
are compared only where route sets match and are reordered by expert ID. Combined
output is compared on all tokens. These are descriptive differences, not proof
of a causal site or claims about other hardware.
