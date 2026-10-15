"""routediff command line.

  python -m routediff calibrate REF CAND [REF CAND ...] --out tolerances.json [--margin 1.0]
  python -m routediff compare REF CAND --out REPORT_DIR [--tolerances tolerances.json] [--rerun 'CMD {patch} ... {out}']
"""
from __future__ import annotations

import argparse
import sys

from .dumps import load_run
from .localize import calibrate, load_tolerances, localize, save_tolerances
from .report import write_reports


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="routediff", description="Find the first op that changes MoE routing.")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="learn per-op tolerances from benign (reference, candidate) pairs")
    c.add_argument("runs", nargs="+", help="dump directories, alternating reference and candidate")
    c.add_argument("--out", required=True)
    c.add_argument("--margin", type=float, default=1.0)
    d = sub.add_parser("compare", help="compare a candidate against a reference and localise the first bad op")
    d.add_argument("reference")
    d.add_argument("candidate")
    d.add_argument("--out", required=True, help="directory for report.json and report.md")
    d.add_argument("--tolerances", help="from `routediff calibrate`; without it, any difference is flagged")
    d.add_argument("--rerun", help="command that reruns the candidate; must contain {patch} and {out}")
    a = p.parse_args(argv)

    if a.cmd == "calibrate":
        if len(a.runs) % 2:
            p.error("calibrate expects pairs of runs")
        pairs = [(load_run(a.runs[i]), load_run(a.runs[i + 1])) for i in range(0, len(a.runs), 2)]
        save_tolerances(calibrate(pairs, a.margin), a.out)
        print(f"wrote tolerances from {len(pairs)} benign pair(s) to {a.out}")
        return 0

    ref, cand = load_run(a.reference), load_run(a.candidate)
    tol = load_tolerances(a.tolerances) if a.tolerances else {}
    if a.rerun and ("{patch}" not in a.rerun or "{out}" not in a.rerun):
        p.error("--rerun must contain {patch} and {out}")
    result = localize(ref, cand, tol, a.rerun)
    out = write_reports(result, ref, cand, a.out, a.tolerances)
    print(result["verdict"])
    print(f"report: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
