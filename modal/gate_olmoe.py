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
import signal
import subprocess
import sys
import threading
import time
import uuid

import modal

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
    .add_local_python_source('coinflip')
)


def driver_major(smi: str) -> int:
    match = re.search(r'Driver Version:\s*(\d+)', smi)
    if not match:
        raise RuntimeError('FAIL: cannot parse NVIDIA driver')
    return int(match.group(1))


def server_command(model_path: str) -> list[str]:
    return [sys.executable, '-m', 'sglang.launch_server', '--model-path', model_path,
            '--port', '30000', '--dtype', 'bfloat16', '--tp', '1',
            '--disable-cuda-graph', '--enable-return-routed-experts',
            '--disable-radix-cache', '--chunked-prefill-size', '-1',
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
    def memory_monitor():
        while not monitor_stop.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                        capture_output=True, text=True, check=True, timeout=5)
                samples.append(dict(seconds=time.perf_counter()-start, used_mib=int(result.stdout.strip().splitlines()[0])))
            except Exception:
                pass
            monitor_stop.wait(0.1)
    def check(name: str, **values):
        summary['checks'].append(dict(check=name, status='PASS', **values))
        print('PASS', name, json.dumps(values), flush=True)
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
            sample_start = len(samples)
            items, seconds = generate(batch)
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


@app.local_entrypoint()
def main(gpu_approved: bool = False, revision: str = 'main'):
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
