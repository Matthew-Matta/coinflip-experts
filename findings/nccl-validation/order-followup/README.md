# Four-rank order follow-up: missing saved element arrays

**No observed order-match shares or new size-aware error numbers are reported.**
The local `data/nccl/remote/` contains four NCCL debug logs only. The original
worker saved aggregate mismatch counts and max ULP, but did not save the NCCL
result tensors. The partial arrays and port expectations remain available;
aggregate error statistics do not determine the missing per-element values.
No GPU/Modal measurement has been rerun to fill this gap.

`analysis.nccl_orders` is ready for actual saved outputs if another copy is
located. It tests all 24 permutations under sequential and balanced pairwise
wire-rounded addition (48 labelled candidates, including equivalent orders),
reports each candidate's exact-match share, union coverage and ambiguity, and
breaks shares into 64 contiguous position bins. These bins are not inferred
NCCL channels; a correlation alone cannot identify channel or chunk assignment.

The requested size-aware column is defined explicitly as
`abs(nccl-emulation) / sum(abs(partials)) / eps_wire`, with eps=2^-23 for fp32
and 2^-7 for bf16. This measures error in relative wire-precision units and
avoids dividing by the spacing of a near-zero cancelled output. Zero partial
scale gives zero only if the error is also zero; otherwise inputs are rejected.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python -m analysis.nccl_orders --partials PARTIALS.npy --actual SAVED_NCCL.npy --emulation EXPECTED.npy --wire bf16 --out NEW_RESULT.json
```

Label retained: **4xL4 on Modal, SHM transport**. No PCIe/NVLink topology claim.
