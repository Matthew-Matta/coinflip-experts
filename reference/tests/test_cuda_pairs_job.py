"""CPU plumbing check only; GPU numerical claims come from the actual job."""
import json
from pathlib import Path
import shutil
import torch
from reference.tests.test_dump_block import tiny_export
from reference.format import sha256,write_json


def test_cuda_job_preserves_inputs_and_checks_null_patches(tiny_export,tmp_path,monkeypatch):
    import reference.cuda_pairs_job as job
    from reference.backend_pairs import dump_backend
    inputs=tmp_path/'input';inputs.mkdir()
    for layer in (0,7):
        export=inputs/f'layer{layer}/export';shutil.copytree(tiny_export,export)
        for dtype in ('float32','bfloat16'):
            dump_backend(export,inputs/f'layer{layer}/cpu-{dtype}',dtype,'cpu','eager')
    for dtype in ('float32','bfloat16'):write_json(inputs/f'tolerances-{dtype}.json',{})
    before={str(p.relative_to(inputs)):sha256(p) for p in inputs.rglob('*') if p.is_file()}
    def cpu_only(*args,**kwargs):
        args=list(args);assert args[3]=='cuda';args[3]='cpu'
        return dump_backend(*args,**kwargs)
    monkeypatch.setattr(job,'dump_backend',cpu_only)
    for name in ('reset_peak_memory_stats','empty_cache'):monkeypatch.setattr(torch.cuda,name,lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _: 'CPU plumbing test, not CUDA')
    rows=job.run(inputs,tmp_path/'output')
    assert len(rows)==4
    assert all(row['mirror_equals_module'] and row['null_patch_all_ops_bit_exact'] for row in rows)
    assert all(row['route_mismatch_tokens']==0 and row['patch_checks']==[] for row in rows)
    assert before=={str(p.relative_to(inputs)):sha256(p) for p in inputs.rglob('*') if p.is_file()}
