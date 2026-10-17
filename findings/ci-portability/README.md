# Eager mirror portability: layout-dependent elementwise kernel selection observed

Two GitHub Linux runs failed three fp32 eager-adapter tests with `eager mirror !=
actual Transformers module`: [39059895380](https://github.com/Matthew-Matta/coinflip-experts/actions/runs/39059895380)
and [39065786553](https://github.com/Matthew-Matta/coinflip-experts/actions/runs/39065786553).
The maintainer identified [39007858464](https://github.com/Matthew-Matta/coinflip-experts/actions/runs/39007858464)
as the passing baseline. These runs used torch 2.14.1 and Transformers 5.12.1.
The failing logs do not contain per-op traces or sufficient CPU diagnostics to
identify the first differing operation or attribute the failure to a CPU model.
The new controlled run below supplies the missing per-op evidence.

## Source discrepancy and correction

The actual eager module applies SiLU to the strided gate half returned by
`F.linear(...).chunk(2, dim=-1)` (Transformers 5.12.1,
`models/olmoe/modeling_olmoe.py:332–333`). The adapter had copied the gate into a
slot tensor and gathered contiguous rows before calling SiLU. Its input values
were preserved, but its strides differed. Layout can change the selected
elementwise kernel; therefore preserving values alone is insufficient to claim
an exact numerical mirror.

`reference/backend_pairs.py` now reconstructs the fused gate/up row layout before
SiLU. The production mirror still checks equality with the actual module and
raises on any discrepancy. No tolerances were loosened, no `allclose` check was
substituted, and no production-path skip was added.

## Measured Linux result (CI 39066770369)

[Run 39066770369](https://github.com/Matthew-Matta/coinflip-experts/actions/runs/39066770369) is green. Its artifact records AMD EPYC 7763, AVX2, torch **2.14.1+cpu**, GCC 13.3, oneAPI MKL 2024.2, MKL-DNN v3.12.0, and OpenMP 4.5. At **both one and two threads**, the legacy contiguous layout first differs at **expert_act**: **3 elements**, maximum absolute difference **9.313225746154785e-10**. Every earlier op is bit-exact. The corrected fused-strided layout passes the strict module comparison at both thread counts.

**Layout-dependent elementwise kernel selection observed.** This controlled comparison changes the activation input layout while retaining the same values and module. We do not identify or claim which specific kernel executes. The full Torch build and CPU output are retained in `linux-39066770369.json`; per-op counts and activation input strides are in `linux-39066770369-ops.json`.

## Additional checks

On the local Apple CPU, both old and corrected layouts pass bit-exactly at one
and two threads (`macos-layout-check.json`). All 25 reference tests pass locally.
A controlled test simulates stride-dependent activation dispatch: the old layout
fails first at `expert_act`, saves the two paths, and the corrected layout passes.
This test validates the diagnostic and correction mechanism; it is **not** a
measurement of Linux dispatch or proof that SiLU caused the historical failures.
Previously recorded Mac experiment results remain unchanged.

The CI workflow now logs `lscpu`, `torch.__config__.show()`, Torch CPU capability
and thread settings. `python -m reference.ci_portability --out ci-portability`
uses the exact tiny fixture from the failing tests, compares old/corrected
layouts at one/two threads, and records descriptive old-layout failures. A
corrected-layout failure remains an error. The always-uploaded `ci-portability`
artifact contains summaries and, on failure, `diagnostic.json` plus all 13
`mirror/*.npy` and `module/*.npy` arrays. The diagnostic traces the real module's
functional calls without replacing its arithmetic or changing input layouts.
Both paths receive the same router input; upstream values are copied for alignment
and are not independently validated by this trace.

The measured layout effect occurs at both tested thread counts. It does not identify a particular kernel, and does not establish behaviour on every x86 CPU.
