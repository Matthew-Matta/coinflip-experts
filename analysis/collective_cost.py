"""Alpha-beta all-reduce model; predictions, never measured interconnect timings.

Ring: reduce-scatter + all-gather, 2(n-1) steps, 2(n-1)/n * payload bytes sent per rank.
Tree: full-buffer binary reduction + broadcast, 2 ceil(log2(n)) critical-path steps.
Both send 2(n-1)*payload aggregate network bytes; tree's max-rank traffic differs.
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path


def allreduce_cost(elements: int, ranks: int, dtype: str, topology: str,
                   alpha_seconds: float, bandwidth_bytes_per_second: float) -> dict:
    if elements < 1 or ranks < 1 or dtype not in ('bf16','fp32') or topology not in ('ring','tree'):
        raise ValueError('invalid payload, ranks, dtype or topology')
    if not math.isfinite(alpha_seconds) or alpha_seconds < 0 or not math.isfinite(bandwidth_bytes_per_second) or bandwidth_bytes_per_second <= 0:
        raise ValueError('latency must be nonnegative and bandwidth positive, both finite')
    payload = elements * (2 if dtype == 'bf16' else 4)
    if ranks == 1:
        steps = path_bytes = aggregate = per_rank = 0
    elif topology == 'ring':
        steps = 2*(ranks-1)
        path_bytes = per_rank = 2*(ranks-1)/ranks * payload
        aggregate = 2*(ranks-1)*payload
    else:
        steps = 2*math.ceil(math.log2(ranks))
        path_bytes = steps * payload
        per_rank = None  # rank-dependent; avoid pretending a uniform full-buffer tree
        aggregate = 2*(ranks-1)*payload
    return dict(model='alpha-beta prediction (not measured)', topology=topology, ranks=ranks,
                dtype=dtype, elements=elements, payload_bytes=payload, sequential_steps=steps,
                critical_path_bytes=path_bytes, per_rank_sent_bytes=per_rank, aggregate_sent_bytes=aggregate,
                alpha_seconds=alpha_seconds, bandwidth_bytes_per_second=bandwidth_bytes_per_second,
                predicted_seconds=steps*alpha_seconds + path_bytes/bandwidth_bytes_per_second)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--elements', type=int, required=True, help='elements in one reduced tensor; e.g. tokens * hidden_size')
    p.add_argument('--alpha-us', type=float, required=True, help='assumed latency per communication step')
    p.add_argument('--bandwidth-gbytes-per-second', type=float, required=True, help='assumed decimal GB/s per link, not gigabits/s')
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args()
    rows=[allreduce_cost(a.elements,n,d,t,a.alpha_us*1e-6,a.bandwidth_gbytes_per_second*1e9)
          for n in (2,4,8,16,32,64) for d in ('bf16','fp32') for t in ('ring','tree')]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    from coinflip.provenance import collect
    from reference.format import write_json, sha256
    write_json(a.out.with_suffix('.provenance.json'),dict(
        kind='descriptive communication model, not measured', csv_sha256=sha256(a.out),
        assumptions='Ideal links; no contention, compute overlap, bandwidth saturation or measured interconnect fit. Full-buffer tree, ring reduce-scatter/all-gather.',
        provenance=collect(dict(elements=a.elements,alpha_us=a.alpha_us,
                                bandwidth_gbytes_per_second=a.bandwidth_gbytes_per_second))))
    print(f'Wrote {len(rows)} model predictions to {a.out}; parameters supplied, not fitted.')

if __name__=='__main__': main()
