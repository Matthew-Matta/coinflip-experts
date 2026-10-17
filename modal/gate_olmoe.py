"""Bounded L4 compatibility gate. CPU preparation and GPU execution are separate.

    modal run modal/gate_olmoe.py                 # image build + CPU weight cache
    modal run modal/gate_olmoe.py --gpu-approved  # only after explicit spend approval
    python modal/gate_olmoe.py --dry-run          # local plan, no remote calls
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid

import modal

# Backport of sgl-project/sglang#40980 (merged 2026-09-29, after 0.5.21): OLMoE's TopK lacks
# layer_id, so --enable-return-routed-experts writes buffer[:, None, :] and crashes the scheduler.
# layer_id only addresses the route-capture buffer here (TP=1, no EPLB/LPLB); routing is unchanged.
OLMOE_TOPK_OLD = '        self.topk = TopK(\n            top_k=top_k,\n            renormalize=False,\n        )\n'
OLMOE_TOPK_NEW = ('        self.topk = TopK(\n            top_k=top_k,\n            layer_id=layer_id,\n'
                  '            renormalize=False,\n        )\n')
# One line: Modal turns each run_commands entry into a Dockerfile RUN, which cannot hold raw newlines.
BACKPORT_40980 = '; '.join([
    'import importlib.util, pathlib',
    "p = pathlib.Path(importlib.util.find_spec('sglang.srt.models.olmoe').origin); s = p.read_text()",
    f"assert s.count({OLMOE_TOPK_OLD!r}) == 1, 'backport 40980: OLMoE TopK block not found exactly once'",
    f'p.write_text(s.replace({OLMOE_TOPK_OLD!r}, {OLMOE_TOPK_NEW!r}))',
    "print('backport 40980 applied to', p)",
])

MODEL = 'allenai/OLMoE-1B-7B-0924'
VOLUME_PATH = '/weights'
BASE = 'http://127.0.0.1:30000'
PINS = ('sglang==0.5.21', 'torch==2.13.0', 'flashinfer-python==0.6.18',
        'sglang-kernel==0.4.7')
app = modal.App('coinflip-olmoe-gate')
volume = modal.Volume.from_name('coinflip-models', create_if_missing=True)
cpu_image = modal.Image.debian_slim(python_version='3.12').pip_install('huggingface_hub==1.33.0')
gpu_image = (
    modal.Image.from_registry('nvidia/cuda:13.0.0-devel-ubuntu24.04', add_python='3.12')
    .env({'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
          'LIBRARY_PATH': '/usr/local/cuda/lib64/stubs'})
    .pip_install(*PINS, 'pybase64', 'requests', 'numpy', 'pandas', 'pyarrow')
    # SGLang pins an older torchaudio ABI; text-only OLMoE does not need it.
    .run_commands('python -m pip uninstall -y torchaudio',
                  'python -c "import torch, transformers; from transformers import AutoConfig"')
    .run_commands(f'python -c {shlex.quote(BACKPORT_40980)}')
    .add_local_python_source('coinflip')
)


def driver_major(smi: str) -> int:
    match = re.search(r'(?:Driver|KMD) Version:\s*(\d+)', smi)
    if not match:
        raise RuntimeError('FAIL: cannot parse NVIDIA driver')
    return int(match.group(1))


def server_command(model_path: str) -> list[str]:
    return [sys.executable, '-m', 'sglang.launch_server', '--model-path', model_path,
            '--port', '30000', '--dtype', 'bfloat16', '--tp', '1',
            '--disable-cuda-graph', '--enable-return-routed-experts',
            # Not -1: RoutedExpertsCapturer sizes its buffer max(chunked_prefill_size, max_running_requests)
            # rows (state_capturer/routed_experts.py:79), so -1 gives 32 rows and crashes on longer prefills.
            # Equal to --max-prefill-tokens, every gate/census batch (<= 32 x 512) runs as one unchunked prefill.
            '--disable-radix-cache', '--chunked-prefill-size', '32768',
            '--max-prefill-tokens', '32768', '--max-running-requests', '32',
            '--mem-fraction-static', '0.80']


def log_errors(path: Path) -> str:
    if not path.exists():
        return 'no server log'
    lines = path.read_text(errors='replace').splitlines()
    errors = [line for line in lines if re.search(r'error|exception|traceback|failed|out of memory|not supported', line, re.I)]
    return '\n'.join(errors[-40:] + ['--- log tail ---'] + lines[-40:])


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # Kill any workers left in the server's process group, even if its parent exited.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=10)


def decode_routes(item: dict, length: int, decode_fn):
    import numpy as np
    arr = np.asarray(decode_fn(item))
    if arr.size % (16*8):
        raise RuntimeError(f'FAIL malformed route array: {arr.size} values')
    arr = arr.reshape(-1, 16, 8)
    if arr.shape != (length, 16, 8):
        raise RuntimeError(f'FAIL expected {length} prefill route rows, got {arr.shape}')
    if np.any(arr < 0) or np.any(arr >= 64) or np.any(np.diff(np.sort(arr, axis=-1), axis=-1) == 0):
        raise RuntimeError('FAIL invalid or duplicate expert IDs')
    return arr


@app.function(image=cpu_image, volumes={VOLUME_PATH: volume},
              secrets=[modal.Secret.from_name('huggingface')], timeout=3600, retries=0)
def prepare_weights(revision: str = 'main') -> dict:
    """No GPU allocated; resolve and download an immutable model snapshot."""
    from huggingface_hub import HfApi, snapshot_download
    start = time.perf_counter()
    volume.reload()
    sha = HfApi().model_info(MODEL, revision=revision).sha
    directory = Path(VOLUME_PATH) / 'olmoe' / sha
    snapshot_download(MODEL, revision=sha, local_dir=directory,
                      allow_patterns=['*.json', '*.safetensors', 'tokenizer*', '*.model', '*.txt'])
    summary = dict(model_id=MODEL, revision=sha, path=str(directory),
                   bytes=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()),
                   download_seconds=time.perf_counter()-start)
    (Path(VOLUME_PATH) / 'prepared.json').write_text(json.dumps(summary, indent=2))
    volume.commit()
    print(f'PASS CPU weight cache: {summary["bytes"] / 1024**3:.2f} GiB, {summary["download_seconds"]:.1f}s, revision={sha}')
    return summary


@app.function(image=gpu_image, gpu='L4', volumes={VOLUME_PATH: volume},
              timeout=1800, retries=0, scaledown_window=2)
def run_gate(prepared: dict) -> dict:
    """One bounded GPU job. Always persist a summary and stop the server."""
    import importlib.metadata as md
    import numpy as np
    import requests
    from coinflip.provenance import collect

    volume.reload()
    run_id = f'gate-{uuid.uuid4().hex[:12]}'
    directory = Path(VOLUME_PATH) / 'gate-results' / run_id
    directory.mkdir(parents=True)
    log_path = directory / 'server.log'
    summary = dict(run_id=run_id, status='FAIL', model=prepared, checks=[],
                   phase='prefill', cuda_graph=False, radix_cache=False)
    proc = None
    log_handle = None
    monitor_stop = threading.Event()
    samples: list[dict] = []
    monitor = None
    start = time.perf_counter()
    def sample_memory():
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, check=True, timeout=5)
            samples.append(dict(seconds=time.perf_counter()-start, used_mib=int(result.stdout.strip().splitlines()[0])))
        except Exception:
            pass
    def memory_monitor():
        while not monitor_stop.is_set():
            sample_memory()
            monitor_stop.wait(0.1)
    def check(label: str, /, **values):
        summary['checks'].append(dict(check=label, status='PASS', **values))
        print('PASS', label, json.dumps(values), flush=True)
    try:
        smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True, timeout=10).stdout
        print(smi, flush=True)
        summary['nvidia_smi'] = smi
        if driver_major(smi) < 580:
            raise RuntimeError('FAIL CUDA 13 environment requires NVIDIA driver >=580')
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('FAIL expected exactly one usable CUDA device')
        check('GPU capability', name=torch.cuda.get_device_name(0),
              capability=list(torch.cuda.get_device_capability(0)))
        from transformers import AutoConfig, AutoTokenizer
        from sglang.srt.state_capturer.routed_experts import extract_routed_experts_from_meta_info
        for pin in PINS:
            package, version = pin.split('==')
            actual = md.version(package)
            if actual.split('+')[0] != version:
                raise RuntimeError(f'FAIL version mismatch: {pin}, installed={actual}')
        import hashlib
        import sglang.srt.models.olmoe as sglang_olmoe
        olmoe_source = Path(sglang_olmoe.__file__).read_text()
        if OLMOE_TOPK_NEW not in olmoe_source:
            raise RuntimeError('FAIL backport 40980 (OLMoE TopK layer_id) not present in image')
        summary['sglang_patches'] = [dict(backport='sgl-project/sglang#40980', file='sglang/srt/models/olmoe.py',
                                          sha256=hashlib.sha256(olmoe_source.encode()).hexdigest())]
        cfg = AutoConfig.from_pretrained(prepared['path'], local_files_only=True)
        if (cfg.num_hidden_layers, cfg.num_experts_per_tok, cfg.num_experts) != (16, 8, 64):
            raise RuntimeError('FAIL unexpected model dimensions')
        tokenizer = AutoTokenizer.from_pretrained(prepared['path'], local_files_only=True)
        monitor = threading.Thread(target=memory_monitor, daemon=True)
        monitor.start()
        cmd = server_command(prepared['path'])
        summary['server_command'] = cmd
        log_handle = log_path.open('w')
        proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
        ready = time.monotonic() + 600
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f'FAIL server exited with {proc.returncode}')
            if time.monotonic() >= ready:
                raise TimeoutError('FAIL health timeout (600s)')
            try:
                if requests.get(f'{BASE}/health', timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        check('server health', launch_seconds=time.perf_counter()-start)
        def generate(batch: list[list[int]], max_new_tokens: int = 1):
            began = time.perf_counter()
            response = requests.post(f'{BASE}/generate', json=dict(
                input_ids=batch, sampling_params=dict(temperature=0, max_new_tokens=max_new_tokens,
                                                      ignore_eos=True),
                return_routed_experts=True), timeout=120)
            response.raise_for_status()
            items = response.json()
            if not isinstance(items, list):
                items = [items]
            if len(items) != len(batch):
                raise RuntimeError('FAIL response count mismatch')
            return items, time.perf_counter()-began
        prompts = ['The capital of France is', 'def fibonacci(n):',
                   'Explain why the sky is blue.', 'What is 17 plus 25?',
                   'Translate to German: The weather is nice today.']
        prompt_ids = [tokenizer.encode(p) for p in prompts]
        items, seconds = generate(prompt_ids, max_new_tokens=8)
        for ids, item in zip(prompt_ids, items, strict=True):
            if not item.get('text', '').strip():
                raise RuntimeError('FAIL empty greedy output')
            # Smoke generates extra tokens; prefill-only assertions are in R0 below.
            route = np.asarray(extract_routed_experts_from_meta_info(item)).reshape(-1, 16, 8)
            if len(route) < len(ids) or np.any(route < 0) or np.any(route >= 64):
                raise RuntimeError('FAIL smoke route capture')
        check('five-prompt greedy smoke', prompts=5, seconds=seconds)
        r0a, _ = generate([prompt_ids[0]])
        r0b, _ = generate([prompt_ids[0]])
        a = decode_routes(r0a[0], len(prompt_ids[0]), extract_routed_experts_from_meta_info)
        b = decode_routes(r0b[0], len(prompt_ids[0]), extract_routed_experts_from_meta_info)
        mismatches = int(np.any(np.sort(a, -1) != np.sort(b, -1), -1).sum())
        if mismatches:
            raise RuntimeError(f'FAIL R0: {mismatches}/{len(a)*16} token-layer sets differ')
        check('R0 prefill route sets identical', positions=len(a), token_layers=len(a)*16, mismatches=0)
        seed_ids = tokenizer.encode('A short story about numerical precision and expert routing. ' * 100)
        if len(seed_ids) < 512:
            raise RuntimeError('FAIL insufficient benchmark tokens')
        # Warmup then timed requests. Requested HTTP batch != proven realised scheduler batch.
        for size in (1, 8, 32):
            batch = [seed_ids[j:j+512] for j in range(size)]
            if any(len(ids) != 512 for ids in batch):
                raise RuntimeError('FAIL benchmark sequence length')
            generate(batch)
            # Bracket the timed request with synchronous samples: a short prefill can finish between monitor ticks.
            sample_start = len(samples)
            sample_memory()
            items, seconds = generate(batch)
            sample_memory()
            for ids, item in zip(batch, items, strict=True):
                decode_routes(item, len(ids), extract_routed_experts_from_meta_info)
            current = samples[sample_start:]
            if not current:
                raise RuntimeError('FAIL no GPU memory samples during timed request')
            check('512-token prefill benchmark', requested_batch_size=size, tokens_per_sequence=512,
                  seconds=seconds, sampled_peak_device_memory_mib=max(s['used_mib'] for s in current),
                  realised_batch_verified=False)
        summary['status'] = 'PASS'
    except Exception as exc:
        summary['error'] = f'{type(exc).__name__}: {exc}'
        print(summary['error'], flush=True)
        print(log_errors(log_path), flush=True)
    finally:
        stop_server(proc)
        if log_handle:
            log_handle.close()
        monitor_stop.set()
        if monitor:
            monitor.join(timeout=6)
        summary['wall_seconds'] = time.perf_counter()-start
        summary['memory_sampling_interval_seconds'] = 0.1
        summary['peak_device_memory_mib'] = max((s['used_mib'] for s in samples), default=None)
        summary['provenance'] = collect({'gate': True, 'revision': prepared['revision']})
        summary['provenance'].update(prepared.get('code_git', {}))
        summary['provenance']['gate_source_sha256'] = prepared.get('gate_source_sha256')
        (directory / 'memory.json').write_text(json.dumps(samples))
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
        volume.commit()
    print(summary['status'], 'gate summary:', str(directory / 'summary.json'), flush=True)
    return summary


@app.function(image=gpu_image.add_local_python_source('census'), gpu='L4',
              volumes={VOLUME_PATH: volume}, timeout=3600, retries=0, scaledown_window=2)
def run_census_job(prepared: dict, rows: list, mode: str):
    from census.worker import execute
    volume.reload()
    result = execute(prepared, rows, mode, server_command(prepared['path']), OLMOE_TOPK_NEW, PINS)
    volume.commit()
    return result


@app.function(image=gpu_image.add_local_python_source('reference'), gpu='L4:4',
              volumes={VOLUME_PATH: volume}, timeout=900, retries=0, scaledown_window=2)
def run_nccl_job(input_hashes: dict):
    import hashlib
    import torch
    volume.reload()
    directory = Path(VOLUME_PATH)/'nccl-results'/uuid.uuid4().hex[:12]
    directory.mkdir(parents=True)
    topology = subprocess.run(['nvidia-smi','topo','-m'],capture_output=True,text=True,timeout=15)
    pci = subprocess.run(['nvidia-smi','--query-gpu=name,pci.bus_id','--format=csv,noheader'],
                         capture_output=True,text=True,timeout=15)
    result = dict(status='FAIL', input_sha256=input_hashes, numerical_mismatch_is_valid=True,
                  directory=str(directory), selection='64 captured positions, first two per original prompt',
                  nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),
                  interconnect_topology=dict(command='nvidia-smi topo -m',returncode=topology.returncode,
                                            stdout=topology.stdout,stderr=topology.stderr,
                                            limitation='Do not infer PCIe/NVLink topology if this command fails.'),
                  pci_bus_ids=dict(returncode=pci.returncode,stdout=pci.stdout,stderr=pci.stderr),
                  nccl_version=torch.cuda.nccl.version(), runs=[])
    began=time.perf_counter()
    try:
        for name,digest in input_hashes.items():
            if hashlib.sha256((Path('/weights/nccl-input')/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('input hash mismatch')
        for n in (2,4):
            for algo in ('Ring','Tree'):
                target=directory/f'n{n}-{algo}'
                env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,range(n))),
                     'NCCL_ALGO':algo,'NCCL_DEBUG':'INFO'}
                cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={n}',
                     '-m','reference.nccl_worker','--input','/weights/nccl-input','--out',str(target)]
                with (directory/f'n{n}-{algo}.log').open('w') as log:
                    subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=150)
                result['runs'].extend(json.loads(p.read_text()) for p in sorted(target.glob('rank*.json')))
        result['status']='PASS'
    except Exception as exc:result['error']=f'{type(exc).__name__}: {exc}'
    finally:
        result['wall_seconds']=time.perf_counter()-began
        (directory/'summary.json').write_text(json.dumps(result,indent=2));volume.commit()
    print(json.dumps(result,indent=2),flush=True)
    return result


@app.function(image=gpu_image, volumes={VOLUME_PATH: volume}, timeout=600, retries=0,
              cpu=2, memory=4096)
def archive_census(directory: str):
    import numpy as np
    import tarfile
    import hashlib
    volume.reload()
    root=Path(directory)
    if root.parent != Path('/weights/census-results') or not root.name.startswith('census-'):
        raise ValueError('expected a completed census result directory')
    summary=json.loads((root/'summary.json').read_text())
    if summary['status']!='PASS':raise ValueError('census failed')
    records=json.loads((root/'batch_composition.json').read_text())
    ids={label:[] for label in ('r0a','r0b','bs8','bs32')}
    for row in records:
        assert row['cached_tokens']==0 and row['sequences']==len(row['prompt_ids'])
        ids[row['config']].extend(row['prompt_ids'])
    for label,values in ids.items():assert len(values)==200 and len(set(values))==200
    recalc=[]
    for i in range(200):
        routes={label:np.sort(np.load(root/label/f'{i:03d}.npy'),axis=-1) for label in ids}
        assert all(v.shape==routes['r0a'].shape for v in routes.values())
        for label in ('r0b','bs8','bs32'):
            flipped=np.any(routes['r0a']!=routes[label],axis=-1)
            recalc.append(dict(index=i,config=label,flips=int(flipped.sum()),units=flipped.size))
    expected={(r['config'],r['prompt_id']):r for r in summary['counts']}
    # Worker writes records prompt-major and config-minor; bind by the frozen record order.
    for actual,record in zip(recalc,summary['counts'],strict=True):
        assert actual['config']==record['config'] and actual['flips']==record['flips'] and actual['units']==record['units']
    for label in ids:
        run=json.loads((root/label/'run.json').read_text())
        assert run['patched_file_sha256']==summary['patched_file_sha256']
        assert run['token_ids_sha256']==summary['model']['token_ids_sha256']
    validation=dict(status='PASS',raw_array_recompute_matches_counts=True,
                    batch_records=len(records),sequences_per_config=200,
                    totals={label:sum(r['flips'] for r in recalc if r['config']==label) for label in ('r0b','bs8','bs32')})
    (root/'independent_validation.json').write_text(json.dumps(validation,indent=2))
    archive=Path(str(root)+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(root,arcname=root.name)
    volume.commit()
    return dict(archive=str(archive).removeprefix('/weights'),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),validation=validation)


@app.local_entrypoint()
def main(gpu_approved: bool = False, revision: str = 'main', mode: str = 'gate',
         pool: str = 'data/prompts/block_pool_v1.jsonl',
         tokens: str = 'data/census/olmoe_tokens_v1.json',
         prepared_file: str = 'data/gate/gate-0769297bc8dd/summary.json',
         out: str = 'data/census'):
    if mode == 'archive':
        result=archive_census.remote(out)
        Path('data/census/archive.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2));return
    if mode == 'nccl':
        if not gpu_approved:
            print('DRY RUN NCCL: 4xL4; n=2/4, Ring/Tree, fp32/bf16, o_proj/down; timeout900s; no GPU invoked.')
            return
        from reference.format import sha256, write_json
        directory = Path('data/nccl/partials')
        paths = sorted(directory.glob('*.npy'))
        if len(paths) != 24: raise ValueError('expected 24 frozen partial/expectation arrays')
        with volume.batch_upload(force=True) as batch:
            for path in paths: batch.put_file(path, '/nccl-input/'+path.name)
        result = run_nccl_job.remote({p.name:sha256(p) for p in paths})
        write_json(Path('data/nccl/summary.json'), result)
        if result['status'] != 'PASS': raise RuntimeError('NCCL validation execution failed (numerical mismatch is valid)')
        return
    if mode in ('corpus', 'census'):
        from reference.format import sha256, write_json
        from coinflip.provenance import git_info
        if not gpu_approved:
            print(f'DRY RUN {mode}: L4, timeout 3600s; no GPU invoked. Explicit --gpu-approved required.')
            return
        gate_summary = json.loads(Path(prepared_file).read_text())
        if gate_summary['status'] != 'PASS': raise ValueError('a successful gate summary is required')
        prepared = dict(gate_summary['model'])
        prepared['code_git'] = git_info()
        prepared['gate_source_sha256'] = sha256(Path(__file__))
        prepared['census_source_sha256'] = {str(p):sha256(p) for p in sorted(Path('census').glob('*.py'))}
        if mode == 'corpus':
            from transformers import AutoTokenizer
            from reference.export_block import select_prompts
            tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=prepared['revision'],
                         cache_dir='data/hf-cache/hub', local_files_only=True)
            with Path(pool).open() as stream:
                prompt_pool = [json.loads(x) for x in stream if x.strip()]
            selected = select_prompts(prompt_pool,
                                      tokenizer, seed=42, per_source=50, max_tokens=256)
            rows = [dict(prompt_id=r['prompt_id'], source=r['source'],
                         dataset_revision=r['dataset_revision'], prompt_token_ids=r['token_ids']) for r in selected]
            prepared['pool_sha256'] = sha256(Path(pool))
        else:
            from census.protocol import require_committed_tokens
            digest, rows = require_committed_tokens(Path(tokens))
            prepared['token_ids_sha256'] = digest
        result = run_census_job.remote(prepared, rows, mode)
        destination = Path(out) / Path(result['directory']).name
        destination.mkdir(parents=True, exist_ok=False)
        write_json(destination/'summary.json', result)
        if result['status'] != 'PASS':
            raise RuntimeError(f'{mode} failed; saved {destination}')
        if mode == 'corpus':
            if Path(tokens).exists(): raise FileExistsError(tokens)
            write_json(Path(tokens), result['rows'])
            marker = Path('census/olmoe_tokens_v1.sha256')
            marker.write_text(sha256(Path(tokens)) + '  olmoe_tokens_v1.json\n')
            write_json(Path('census/olmoe_tokens_v1.manifest.json'), dict(
                token_ids_sha256=sha256(Path(tokens)), prompts=200, per_source=50,
                continuation_tokens=256, total_length_cap=512, selection_seed=42,
                model_revision=prepared['revision'], pool_sha256=prepared['pool_sha256'],
                generation='SGLang greedy, temperature 0, ignore_eos=True; output token ids retained verbatim'))
            print('STOP: Matthew must commit census/olmoe_tokens_v1.sha256 before census measurement.')
        return
    if mode != 'gate': raise ValueError('mode must be gate, corpus or census')
    # Resolving the function builds the GPU image without allocating a GPU.
    run_gate.hydrate()
    prepared = prepare_weights.remote(revision)
    if not gpu_approved:
        print('PASS CPU preparation complete. GPU NOT RUN. Request approval before --gpu-approved.')
        return
    import hashlib
    from coinflip.provenance import git_info
    prepared['code_git'] = git_info()
    prepared['gate_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = run_gate.remote(prepared)
    print(json.dumps(summary, indent=2))
    if summary['status'] != 'PASS':
        raise RuntimeError('L4 gate failed; inspect the persisted summary and server.log')


if __name__ == '__main__':
    if sys.argv[1:] != ['--dry-run']:
        raise SystemExit('Use modal run, or python modal/gate_olmoe.py --dry-run')
    print(json.dumps(dict(model=MODEL, gpu='L4', python='3.12', pins=PINS,
                          timeout_seconds=1800, cpu_preparation_first=True,
                          server_command=server_command('/weights/olmoe/REVISION'), gpu_invoked=False), indent=2))
