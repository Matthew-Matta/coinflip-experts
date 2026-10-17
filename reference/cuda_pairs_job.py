"""Run descriptive CPU/CUDA pairs inside the single bounded follow-up allocation."""
import gc
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import time
import numpy as np
import torch
from reference.backend_pairs import dump_backend
from reference.format import OPS, sha256, write_json


def run(inputs: Path,out: Path):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    out.mkdir(parents=True,exist_ok=True); rows=[]
    for layer in (0,7):
        export=inputs/f'layer{layer}/export'
        for dtype in ('float32','bfloat16'):
            began=time.perf_counter();tag=f'layer{layer}-{dtype}'
            ref=out/tag/'cpu'
            ref.parent.mkdir(parents=True)
            shutil.copytree(inputs/f'layer{layer}/cpu-{dtype}',ref)
            manifest=json.loads((ref/'manifest.json').read_text())
            manifest['export']['path']=str(export)
            write_json(ref/'manifest.json',manifest)
            candidate=out/tag/'cuda';candidate.parent.mkdir(parents=True,exist_ok=True)
            torch.cuda.reset_peak_memory_stats()
            dump_backend(export,candidate,dtype,'cuda','eager')
            # Same-backend null patch: logits replacement recomputes all later ops.
            null=out/tag/'null-patch'
            dump_backend(export,null,dtype,'cuda','eager',{'router_logits':candidate/'router_logits.npy'})
            null_exact=all(np.array_equal(np.load(candidate/f'{op}.npy').view('u4'),np.load(null/f'{op}.npy').view('u4')) for op in OPS)
            if not null_exact:raise RuntimeError('CUDA null patch changed computation')
            tolerance=inputs/f'tolerances-{dtype}.json'
            command=[sys.executable,'-m','reference.backend_pairs','--export',str(export),'--dtype',dtype,
                     '--device','cuda','--implementation','eager','--threads','2','--out','{out}','{patch}']
            subprocess.run([sys.executable,'-m','routediff','compare',str(ref),str(candidate),'--out',str(out/tag/'report'),
                            '--tolerances',str(tolerance),'--rerun',shlex.join(command)],check=True,timeout=180)
            report=json.loads((out/tag/'report/report.json').read_text())
            rows.append(dict(layer=layer,dtype=dtype,tokens=manifest['tokens'],mirror_equals_module=True,null_patch_all_ops_bit_exact=null_exact,
                             route_mismatch_tokens=report['route_mismatch_tokens'],localized_op=report['localized_op'],
                             verdict=report['verdict'],patch_checks=report['patch_checks'],tolerance_sha256=sha256(tolerance),
                             max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.perf_counter()-began))
            gc.collect();torch.cuda.empty_cache()
    write_json(out/'summary.json',dict(kind='Descriptive real CPU/CUDA backend pairs; not blinded evaluation',
               tf32=False,torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name(0),rows=rows))
    return rows
