"""Download only index-selected embedding/layer-0 shards; capture one Qwen layer."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from huggingface_hub import HfApi,hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig,AutoTokenizer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel,Qwen3MoeRotaryEmbedding
from reference.export_block import capture_and_export,peak_rss_gib
from reference.dump_block import require_version,dump_block
from reference.format import write_json,sha256

MODEL='Qwen/Qwen3-30B-A3B'


def required_keys(weight_map):
    keys=[k for k in weight_map if k=='model.embed_tokens.weight' or k.startswith('model.layers.0.')]
    if 'model.embed_tokens.weight' not in keys:raise ValueError('missing embedding')
    return keys,sorted({weight_map[k] for k in keys})


def run(out: Path,cache: Path,selection: Path):
    require_version();torch.set_num_threads(4);start=time.perf_counter()
    out.mkdir(parents=True,exist_ok=False)
    revision=HfApi().model_info(MODEL).sha
    fetch=lambda name:hf_hub_download(MODEL,name,revision=revision,cache_dir=cache)
    index_path=Path(fetch('model.safetensors.index.json'))
    mapping=json.loads(index_path.read_text())['weight_map'];keys,shards=required_keys(mapping)
    write_json(out/'download_plan.json',dict(model=MODEL,revision=revision,index_sha256=sha256(index_path),
                  shards=shards,required_tensor_keys=keys))
    paths={name:fetch(name) for name in shards}
    config=AutoConfig.from_pretrained(MODEL,revision=revision,cache_dir=cache)
    config.num_hidden_layers=1;config.use_cache=False
    config._experts_implementation='eager';config._attn_implementation='eager'
    config.intermediate_size=config.moe_intermediate_size
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=revision,cache_dir=cache)
    original=json.loads(selection.read_text())['prompts']
    selected=[]
    for row in original:
        r=dict(row)
        r['token_ids']=tokenizer(r['text'],add_special_tokens=True,truncation=True,max_length=64)['input_ids']
        selected.append(r)
    write_json(out/'selection.json',dict(prompts=selected,source_selection_sha256=sha256(selection),
               note='Same prompt IDs/text and prompt-level split; Qwen tokenizer determines token IDs.'))
    with torch.device('meta'):model=Qwen3MoeModel(config)
    model.norm=torch.nn.Identity()  # final model norm is after the captured block and is not needed
    model.rotary_emb=Qwen3MoeRotaryEmbedding(config=config,device='cpu')
    state={}
    for shard in shards:
        with safe_open(paths[shard],framework='pt',device='cpu') as stream:
            for key in keys:
                if mapping[key]==shard:state[key.removeprefix('model.')]=stream.get_tensor(key)
    prefix='layers.0.mlp.experts.'
    fused=[];downs=[]
    for e in range(config.num_experts):
        fused.append(torch.cat([state.pop(f'{prefix}{e}.gate_proj.weight'),state.pop(f'{prefix}{e}.up_proj.weight')],dim=0))
        downs.append(state.pop(f'{prefix}{e}.down_proj.weight'))
    state[prefix+'gate_up_proj']=torch.stack(fused);state[prefix+'down_proj']=torch.stack(downs)
    del fused,downs
    model.load_state_dict(state,assign=True,strict=True);del state;model.eval()
    if any(p.device.type=='meta' for p in model.parameters()):raise RuntimeError('uninitialised meta parameter')
    capture_and_export(model,selected,[0],out,revision,model_id=MODEL)
    del model;gc.collect()
    dumps=[]
    for dtype in ('bfloat16','float32'):
        dest=out/f'torch-{dtype}';dump_block(out/'layer0',dest,dtype);dumps.append(str(dest));gc.collect()
    result=dict(status='PASS',revision=revision,downloaded_shards=shards,shard_bytes=sum(Path(p).stat().st_size for p in paths.values()),
                peak_rss_gib=peak_rss_gib(),seconds=time.perf_counter()-start,dumps=dumps,mirror_equals_module=True,
                final_norm_omitted=True,scope='Single truncated layer, reference-only rectangular o_proj; not full-model generation.')
    write_json(out/'summary.json',result);print(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('data/export/qwen3-30b-a3b-layer0'))
    p.add_argument('--cache',type=Path,default=Path('data/hf-cache/hub'))
    p.add_argument('--selection',type=Path,default=Path('data/export/olmoe-1b-7b-0924/selection.json'))
    a=p.parse_args();run(a.out,a.cache,a.selection)
if __name__=='__main__':main()
