"""Bounded remote corpus preparation or census; no router-margin capture."""
from __future__ import annotations
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import os
import signal
import subprocess
import threading
import time
import uuid


def save_census(records, directory, summary):
    """Snapshot successful measurements without marking the run done before I/O."""
    import pandas as pd
    from coinflip.provenance import save_run
    save_run(pd.DataFrame(records),directory,{**summary,'status':'PASS'},name='census')


def execute(prepared, rows, mode, command, expected_patch, pins, root='/weights'):
    import numpy as np
    import requests
    from coinflip.provenance import collect
    from census.protocol import batches,parse_prefills,validate_tokens
    from sglang.srt.state_capturer.routed_experts import extract_routed_experts_from_meta_info
    import sglang.srt.models.olmoe as olmoe
    directory=Path(root)/'census-results'/f'{mode}-{uuid.uuid4().hex[:12]}'
    directory.mkdir(parents=True)
    log=directory/'server.log';proc=None;handle=None;start=time.perf_counter()
    summary=dict(status='FAIL',mode=mode,directory=str(directory),phase='prefill',model=prepared)
    stop=threading.Event();samples=[]
    def memory():
        while not stop.is_set():
            try:
                s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5)
                samples.append(int(s.strip()))
            except Exception:pass
            stop.wait(.2)
    monitor=threading.Thread(target=memory,daemon=True)
    try:
        for pin in pins:
            pkg,version=pin.split('==')
            if md.version(pkg).split('+')[0]!=version:raise ValueError(f'pin mismatch {pin}')
        source=Path(olmoe.__file__).read_text()
        if expected_patch not in source:raise ValueError('OLMoE capture backport missing')
        summary.update(patched_file='sglang/srt/models/olmoe.py',patched_file_sha256=hashlib.sha256(source.encode()).hexdigest(),
                       sglang_backport='sgl-project/sglang#40980',server_command=command,
                       nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),provenance=collect(prepared))
        handle=log.open('w');monitor.start()
        proc=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+600
        while True:
            if proc.poll() is not None:raise RuntimeError('server exited before health')
            try:
                if requests.get('http://127.0.0.1:30000/health',timeout=2).status_code==200:break
            except requests.RequestException:pass
            if time.monotonic()>deadline:raise TimeoutError('server health timeout')
            time.sleep(2)
        def generate(ids,new,capture):
            response=requests.post('http://127.0.0.1:30000/generate',json=dict(input_ids=ids,
                sampling_params=dict(temperature=0,max_new_tokens=new,ignore_eos=True),return_routed_experts=capture),timeout=300)
            response.raise_for_status();items=response.json()
            if not isinstance(items,list):items=[items]
            if len(items)!=len(ids):raise ValueError('response count mismatch')
            return items
        if mode=='corpus':
            frozen=[]
            for selection in batches(rows,8):
                items=generate([rows[i]['prompt_token_ids'] for i in selection],256,False)
                for i,item in zip(selection,items,strict=True):
                    continuation=item.get('output_ids')
                    if continuation is None or len(continuation)!=256:raise ValueError('missing exact generated output_ids; refusing text re-tokenization')
                    frozen.append(dict(**rows[i],continuation_ids=continuation,token_ids=rows[i]['prompt_token_ids']+continuation))
                print(f'PASS reference continuations {len(frozen)}/200',flush=True)
            by_id={r['prompt_id']:r for r in frozen};frozen=[by_id[r['prompt_id']] for r in rows]
            validate_tokens(frozen)
            (directory/'tokens.json').write_text(json.dumps(frozen,indent=2)+'\n')
            summary['rows']=frozen
        else:
            validate_tokens(rows)
            saved={};batch_records=[]
            for label,size in [('r0a',1),('r0b',1),('bs8',8),('bs32',32)]:
                run_dir=directory/label;run_dir.mkdir();config_begin=time.perf_counter()
                for number,selection in enumerate(batches(rows,size)):
                    before=log.stat().st_size
                    ids=[rows[i]['token_ids'] for i in selection]
                    items=generate(ids,1,True)
                    # No next request is submitted until this log evidence is available.
                    deadline=time.monotonic()+5
                    while True:
                        with log.open('rb') as stream:stream.seek(before);text=stream.read().decode(errors='replace')
                        realised=parse_prefills(text)
                        if realised or time.monotonic()>deadline:break
                        time.sleep(.05)
                    expected=dict(sequences=len(selection),new_tokens=sum(map(len,ids)),cached_tokens=0)
                    if realised!=[expected]:raise ValueError(f'realised batch mismatch {realised} != {[expected]}')
                    batch_records.append(dict(config=label,batch=number,prompt_ids=[rows[i]['prompt_id'] for i in selection],**realised[0]))
                    for i,item in zip(selection,items,strict=True):
                        a=np.asarray(extract_routed_experts_from_meta_info(item)).reshape(-1,16,8)
                        if a.shape!=(len(rows[i]['token_ids']),16,8) or np.any(a<0) or np.any(a>=64) or np.any(np.diff(np.sort(a,axis=-1),axis=-1)==0):raise ValueError('invalid captured route rows')
                        np.save(run_dir/f'{i:03d}.npy',a.astype('<i4'))
                        saved[(label,i)]=np.sort(a,axis=-1)
                config=dict(config=label,requested_batch_size=size,seconds=time.perf_counter()-config_begin,
                            patched_file_sha256=summary['patched_file_sha256'],token_ids_sha256=prepared['token_ids_sha256'],
                            versions=summary['provenance']['packages'],gpu_and_driver=summary['nvidia_smi'],
                            server_command=command,model_revision=prepared['revision'])
                (run_dir/'run.json').write_text(json.dumps(config,indent=2))
                print(f'PASS {label}: 200 prompts, scheduler batches verified',flush=True)
            records=[]
            for i,row in enumerate(rows):
                r0=np.any(saved[('r0a',i)]!=saved[('r0b',i)],axis=-1)
                for label in ('r0b','bs8','bs32'):
                    flips=np.any(saved[('r0a',i)]!=saved[(label,i)],axis=-1)
                    records.append(dict(prompt_id=row['prompt_id'],source=row['source'],config=label,
                                        units=flips.size,flips=int(flips.sum()),r0_flips=int(r0.sum()),any_flip=bool(flips.any())))
            save_census(records,directory,summary)
            (directory/'batch_composition.json').write_text(json.dumps(batch_records,indent=2))
            summary['counts']=records
        summary['status']='PASS'
    except Exception as exc:summary['error']=f'{type(exc).__name__}: {exc}';print(summary['error'],flush=True)
    finally:
        if proc is not None:
            try:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=15)
            except (ProcessLookupError,subprocess.TimeoutExpired):pass
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        stop.set()
        if monitor.is_alive():monitor.join(timeout=6)
        if handle:handle.close()
        summary.update(wall_seconds=time.perf_counter()-start,peak_device_memory_mib=max(samples,default=None))
        (directory/'summary.json').write_text(json.dumps(summary,indent=2))
    print(summary['status'],mode,directory,summary['wall_seconds'],flush=True)
    return summary
