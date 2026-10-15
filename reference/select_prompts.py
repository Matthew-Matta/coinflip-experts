"""Build a revision-pinned prompt pool for the block capture (not census data).

Only the first --pool-size eligible rows per source form the candidate pool;
export_block performs the seeded selection and records prompt-level splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_url

from reference.format import write_json

# Dataset revisions are resolved to immutable SHAs and written with every row.
DATASETS = (
    ('GSM8K', 'openai/gsm8k', 'main', 'test', 'question'),
    ('HumanEval', 'openai/openai_humaneval', None, 'test', 'prompt'),
    ('ShareGPT', 'anon8231489123/ShareGPT_Vicuna_unfiltered', None, 'train', 'conversations'),
    ('FLORES', 'facebook/flores', 'eng_Latn', 'devtest', 'sentence'),
)


def first_human_turn(value):
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        # Arrow can expose a sequence-of-struct column as a struct-of-sequences.
        value = [dict(zip(value, items)) for items in zip(*value.values(), strict=True)]
    value = [json.loads(m) if isinstance(m, str) else m for m in (value or [])]
    return next((m.get('value') for m in (value or [])
                 if isinstance(m, dict) and m.get('from') == 'human'), None)


def parse_json_array(chunks):
    """Incrementally parse an array without materialising a large conversation file."""
    decoder, buffer, started = json.JSONDecoder(), '', False
    for chunk in chunks:
        buffer += chunk
        if not started:
            buffer = buffer.lstrip()
            if not buffer:
                continue
            if buffer[0] != '[':
                raise ValueError('expected JSON array')
            buffer, started = buffer[1:], True
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(','):
                buffer = buffer[1:].lstrip()
            if not buffer:
                break
            if buffer.startswith(']'):
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                break
            yield value
            buffer = buffer[end:]
    raise ValueError('truncated JSON array')


def stream_conversations(repo: str, revision: str):
    import httpx
    url = hf_hub_url(repo, 'ShareGPT_V3_unfiltered_cleaned_split.json',
                     repo_type='dataset', revision=revision)
    with httpx.stream('GET', url, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        yield from parse_json_array(response.iter_text(chunk_size=16384))


def flores_rows(cache: Path):
    """Read the original public FLORES-200 archive, without dataset scripts."""
    import httpx
    url = 'https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz'
    if not cache.exists():
        partial = cache.with_suffix('.partial')
        with httpx.stream('GET', url, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with partial.open('wb') as output:
                for chunk in response.iter_bytes():
                    output.write(chunk)
        partial.replace(cache)
    revision = 'sha256:' + hashlib.sha256(cache.read_bytes()).hexdigest()
    with tarfile.open(cache, 'r:gz') as archive:
        names = [n for n in archive.getnames() if n.endswith('/devtest/eng_Latn.devtest')]
        if len(names) != 1:
            raise ValueError('FLORES archive lacks unique English devtest file')
        with archive.extractfile(names[0]) as stream:
            sentences = stream.read().decode('utf-8').splitlines()
    return [dict(id=str(i+1), sentence=text) for i, text in enumerate(sentences)], revision


def main() -> None:
    from datasets import load_dataset
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('data/prompts/block_pool.jsonl'))
    p.add_argument('--pool-size', type=int, default=128)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows, sources = [], []
    for source, repo, config, split, field in DATASETS:
        if source == 'FLORES':
            ds, revision = flores_rows(a.out.parent / 'flores200_dataset.tar.gz')
        else:
            revision = HfApi().dataset_info(repo).sha
        if source == 'ShareGPT':
            ds = stream_conversations(repo, revision)
        elif source != 'FLORES':
            ds = load_dataset(repo, config, split=split, revision=revision, streaming=True)
        count = 0
        for index, row in enumerate(ds):
            text = row.get(field)
            if source == 'ShareGPT':
                text = first_human_turn(text)
            if not isinstance(text, str) or not text.strip():
                continue
            rows.append(dict(id=str(row.get('task_id', row.get('id', index))), source=source,
                             text=text, dataset=repo, dataset_revision=revision, split=split, row=index))
            count += 1
            if count >= a.pool_size:
                break
        if count < 8:
            raise RuntimeError(f'{source}: only {count} eligible prompts')
        sources.append(dict(source=source, dataset=repo, revision=revision, config=config, split=split, count=count))
        print(f'PASS {source}: {count} candidate prompts, revision={revision}', flush=True)
    a.out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    write_json(a.out.with_suffix('.sources.json'), sources)


if __name__ == '__main__':
    main()
