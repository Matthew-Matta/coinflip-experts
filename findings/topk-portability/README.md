# CPU top-k portability probe

The small committed fixture `reference/tests/fixtures/bf16_topk_ties.json`
contains the lossless logits and macOS fp32 probabilities for all **38** bf16
8th/9th-boundary ties: 13 layer-0 and 25 layer-7 rows identified from the inputs
to `findings/backend-pairs-v1/bf16_ties.json`. It also records source SHA256 hashes
and the original macOS CPU expert sets. No prompt text or weights are included.

Two descriptive probes run on Ubuntu CI with torch 2.14.1:

1. `torch.topk` on saved macOS probabilities isolates the selection algorithm.
2. fp32 `torch.softmax(logits)` then `torch.topk` includes platform softmax math.

Each row prints the chosen sets and comparison against the macOS set. Any
mismatch is an observation, not a failing expectation. Structural fixture checks
still fail on malformed data. Compare sets, not the order within equal scores.
A match on 38 rows is not a general portability guarantee.

Local macOS arm64, torch 2.14.1: **38/38 sets match** in both probes; all
recomputed probability rows are bit-exact. Full row results: `macos.json`.
**Linux result pending the next CI run**; inspect the `topk-linux` artifact and
step logs. No Linux outcome is claimed from the macOS run.

```sh
python -m analysis.topk_portability --out topk-linux.json
```
