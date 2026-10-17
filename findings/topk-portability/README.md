# Ties change across CPU builds

**In one line:** Linux matched macOS on only **19/38** tied expert choices.

- **What I found:** I gave both CPU builds the same saved probabilities and saw different selections.
- **Why it matters:** A model port needs an explicit rule for exact ties.
- **Caveat:** This small fixture does not establish general portability or identify the library responsible.

**Type:** descriptive   ·   **Reproduce:** `python -m analysis.topk_portability --out NEW_TOPK.json`

---

## Details

The small committed fixture `reference/tests/fixtures/bf16_topk_ties.json`
contains the lossless logits and macOS fp32 probabilities for all **38** bf16
8th/9th-boundary ties: 13 layer-0 and 25 layer-7 rows identified from the inputs
to `findings/backend-pairs-v1/bf16_ties.json`. It also records source SHA256 hashes
and the original macOS CPU expert sets. No prompt text or weights are included.

I ran two descriptive probes on Ubuntu CI with torch 2.14.1:

1. `torch.topk` on saved macOS probabilities isolates the selection algorithm.
2. fp32 `torch.softmax(logits)` then `torch.topk` includes platform softmax math.

Each row prints the chosen sets and comparison against the macOS set. Any
mismatch is an observation, not a failing expectation. Structural fixture checks
still fail on malformed data. Compare sets, not the order within equal scores.
A match on 38 rows is not a general portability guarantee.

Local macOS arm64, torch 2.14.1: **38/38 sets match** in both probes; all
recomputed probability rows are bit-exact. Full row results: `macos.json`.
**Linux result recorded:** [CI run 38991546221](https://github.com/Matthew-Matta/coinflip-experts/actions/runs/38991546221),
artifact `topk-linux`, Linux x86_64, torch **2.14.1+cpu**: **19/38** selected
sets match macOS in both probes. Full artifact: `linux.json`; hashes and counts:
`comparison.json`.

A stable lower-expert-ID rule matches **16/38 Linux** sets and **11/38 macOS**
sets, using the saved fixed probabilities. Thus neither tested CPU build behaves
as that simple tie rule on all rows. Differences already occur when both builds
receive identical saved probabilities, independently of softmax recomputation.
I treat this as descriptive evidence of platform/build-dependent tie choices; I make **no
attribution to libstdc++ or libc++** without checking the actual builds
and selection implementation.

```sh
python -m analysis.topk_portability --out topk-linux.json
```
