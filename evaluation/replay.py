"""Opaque patch rerun client. It has no access to injection labels or spec paths."""
from __future__ import annotations
import argparse
import json
from urllib.request import Request, urlopen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint', required=True)
    p.add_argument('--dump', required=True)
    p.add_argument('--patch', action='append', default=[])
    p.add_argument('--patch-ids')
    a = p.parse_args()
    payload = json.dumps(dict(dump=a.dump, patches=a.patch, patch_ids=a.patch_ids)).encode()
    request = Request(a.endpoint, data=payload, headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=330) as response:
        result = json.load(response)
    if not result['ok']:
        raise SystemExit('Backend replay failed; private diagnostic retained by evaluator')


if __name__ == '__main__':
    main()
