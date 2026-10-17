"""ONE approved allocation: saved NCCL tensors and descriptive CPU/CUDA pairs.

Default entrypoint is a dry run; --gpu-approved invokes exactly one function.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import uuid
import modal

# Reuse the pinned, cached environment from the original NCCL experiment.
sys.path.insert(0,str(Path(__file__).parent))
from gate_olmoe import gpu_image,volume

app=modal.App('coinflip-numerics-followup')
image=gpu_image.add_local_python_source('gate_olmoe','reference','routediff')
TIMEOUT=900
# Published base rates checked 2026-10-17; allowance includes CPU/RAM and startup.
# 4 L4 * $0.000222/s + 2 CPU * $0.0000131/s + 48 GiB * $0.00000222/s.
RESOURCE_RATE=4*.000222+2*.0000131+48*.00000222

@app.function(image=image,gpu='L4:4',cpu=(2,2),memory=(32768,49152),
              volumes={'/weights':volume},timeout=TIMEOUT,retries=0,
              max_containers=1,scaledown_window=2)
def execute(remote_input: str,prepared: dict):
    import torch
    from reference.cuda_pairs_job import run
    from reference.format import sha256,write_json
    volume.reload();inputs=Path('/weights')/remote_input
    root=Path('/weights/numerics-followup-results')/uuid.uuid4().hex[:12];root.mkdir(parents=True)
    began=time.perf_counter()
    import importlib,inspect
    modules=['reference.backend_pairs','reference.patch_hooks','reference.nccl_worker','reference.cuda_pairs_job','routediff.localize']
    actual_sources={name:sha256(Path(inspect.getsourcefile(importlib.import_module(name)))) for name in modules}
    result=dict(source_sha256=actual_sources,status='FAIL',directory=str(root),input_sha256=prepared,
                nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),
                gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,cuda_version=torch.version.cuda,
                nccl_version=torch.cuda.nccl.version(),timeout_seconds=TIMEOUT,retries=0,
                rate_usd_per_second=RESOURCE_RATE,resource_timeout_estimate_usd=TIMEOUT*RESOURCE_RATE,
                transport_label='4xL4 on Modal, SHM transport (verify NCCL logs)',runs=[])
    try:
        from reference.restore_cached_weights import restore
        restore(inputs)
        for name,digest in prepared['input_sha256'].items():
            if sha256(inputs/name)!=digest:raise ValueError('CUDA input SHA mismatch: '+name)
        for name,digest in prepared['nccl_input_sha256'].items():
            if sha256(Path('/weights/nccl-input')/name)!=digest:raise ValueError('NCCL input SHA mismatch: '+name)
        for n in (2,4):
            for algo in ('Ring','Tree'):
                target=root/'nccl'/f'n{n}-{algo}';target.parent.mkdir(exist_ok=True)
                env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,range(n))),'NCCL_ALGO':algo,'NCCL_DEBUG':'INFO',
                     'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
                cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={n}',
                     '-m','reference.nccl_worker','--input','/weights/nccl-input','--out',str(target)]
                with (target.parent/f'n{n}-{algo}.log').open('w') as log:
                    subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
                result['runs'].extend(json.loads(p.read_text()) for p in sorted(target.glob('rank*.json')))
                write_json(root/'summary.json',result);volume.commit()
        result['cuda_pairs']=run(inputs,root/'cuda-pairs')
        result['status']='PASS'
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    finally:
        result['wall_seconds']=time.perf_counter()-began
        result['execution_resource_estimate_usd']=result['wall_seconds']*RESOURCE_RATE
        result['output_sha256']={str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file() and p.name!='summary.json'}
        write_json(root/'summary.json',result)
        archive=Path(str(root)+'.tar.gz')
        with tarfile.open(archive,'w:gz',compresslevel=1) as tar:tar.add(root,arcname=root.name)
        volume.commit()
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','output_sha256','runs','cuda_pairs')},indent=2),flush=True)
    return dict(status=result['status'],archive=str(archive).removeprefix('/weights'),sha256=sha256(archive),
                wall_seconds=result['wall_seconds'],error=result.get('error'))

@app.local_entrypoint()
def main(gpu_approved: bool=False,inputs: str='data/followup/input-v2',out: str='data/followup/launch.json'):
    root=Path(inputs);prepared=json.loads((root/'prepared.json').read_text())
    plan=dict(mode='ONE combined allocation',gpu='L4:4',timeout_seconds=TIMEOUT,retries=0,
              max_spend_usd=2,resource_timeout_estimate_usd=TIMEOUT*RESOURCE_RATE,
              tasks=['NCCL unchanged n2/n4 Ring/Tree fp32/bf16 save every rank tensor','CUDA eager L0/L7 fp32/bf16, null patch and CPU comparison with full patch-forward'],
              selection=prepared['selection'],files=len(prepared['input_sha256']),cached_weights='Rebuild pinned L0/L7 weights from existing volume, then require original file SHA-256')
    print(json.dumps(plan,indent=2),flush=True)
    if not gpu_approved:
        print('DRY RUN PASS: no remote calls, no GPU allocation.');return
    if TIMEOUT*RESOURCE_RATE>1.1:raise ValueError('resource guard exceeds budget allowance')
    prepared['launch_code']=dict(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),git_dirty=True,
          source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('reference/nccl_worker.py'),Path('reference/cuda_pairs_job.py'),Path('reference/backend_pairs.py'),Path('reference/patch_hooks.py'),Path('reference/restore_cached_weights.py')]})
    remote='numerics-followup-input/'+uuid.uuid4().hex[:12]
    with volume.batch_upload() as batch:
        for name,digest in prepared['input_sha256'].items():
            if '/export/weights/' in name:continue
            p=root/name
            if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('local input changed: '+name)
            batch.put_file(p.resolve(),'/'+remote+'/'+name)
            print('Queued input '+name,flush=True)
    result=execute.remote(remote,prepared)
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']!='PASS':raise RuntimeError('Follow-up failed; saved partial results, no retry')
