"""Restore exported weights from the already cached, pinned checkpoint; verify SHA externally."""
import gc
import json
from pathlib import Path
import torch
from transformers import AutoConfig,OlmoeModel
from reference.format import save_array


def restore(inputs: Path):
    configs=[json.loads((inputs/f'layer{layer}/export/config.json').read_text()) for layer in (0,7)]
    revision=configs[0]['hf_revision']
    if any(c['hf_revision']!=revision for c in configs):raise ValueError('mixed checkpoint revisions')
    cached=Path('/weights/olmoe')/revision
    if not (cached/'config.json').is_file():raise FileNotFoundError('pinned checkpoint absent; no download allowed')
    torch.set_num_threads(2)
    cfg=AutoConfig.from_pretrained(cached,local_files_only=True)
    cfg.num_hidden_layers=8;cfg.use_cache=False
    model,loading=OlmoeModel.from_pretrained(cached,config=cfg,local_files_only=True,dtype=torch.bfloat16,
                      experts_implementation='eager',attn_implementation='eager',output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')):
        raise RuntimeError(f'incomplete cached weight load: {loading}')
    for layer in (0,7):
        decoder=model.layers[layer]
        weights=dict(o_proj=decoder.self_attn.o_proj.weight,post_attention_layernorm=decoder.post_attention_layernorm.weight,
                     router=decoder.mlp.gate.weight,gate_up_proj=decoder.mlp.experts.gate_up_proj,down_proj=decoder.mlp.experts.down_proj)
        root=inputs/f'layer{layer}/export/weights';root.mkdir(parents=True,exist_ok=False)
        for name,value in weights.items():save_array(root/f'{name}.npy',value)
    del weights,decoder,model;gc.collect()
    print('PASS restored cached pinned weights; full frozen SHA verification follows',flush=True)
