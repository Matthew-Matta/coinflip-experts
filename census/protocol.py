"""Census inputs, scheduler evidence and commit-before-measurement checks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess

PREFILL=re.compile(r'Prefill batch(?: \[\d+\])?, #new-seq:\s*(\d+), #new-token:\s*(\d+), #cached-token:\s*(\d+)')


def parse_prefills(text):
    return [dict(sequences=int(m[1]),new_tokens=int(m[2]),cached_tokens=int(m[3])) for m in PREFILL.finditer(text)]


def batches(rows, size, seed=20261017):
    order=list(range(len(rows)));random.Random(seed).shuffle(order)
    return [order[i:i+size] for i in range(0,len(order),size)]


def validate_tokens(rows):
    from collections import Counter
    if len(rows)!=200 or Counter(r['source'] for r in rows)!=dict.fromkeys(('GSM8K','HumanEval','ShareGPT','FLORES'),50):
        raise ValueError('need exactly 50 prompts from each source')
    if len({r['prompt_id'] for r in rows})!=200:raise ValueError('duplicate prompt ids')
    for r in rows:
        if len(r['continuation_ids'])!=256 or not 1<=len(r['prompt_token_ids'])<=256:
            raise ValueError('expected <=256 prompt tokens and exactly 256 reference continuation tokens')
        if r['token_ids']!=r['prompt_token_ids']+r['continuation_ids'] or len(r['token_ids'])>512:
            raise ValueError('unaligned frozen token ids')


def require_committed_tokens(path: Path, marker='census/olmoe_tokens_v1.sha256'):
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    committed=subprocess.check_output(['git','show',f'HEAD:{marker}'],text=True).split()[0]
    if committed!=digest:raise ValueError('token SHA-256 must match committed HEAD before any census run')
    rows=json.loads(path.read_text());validate_tokens(rows)
    return digest,rows
