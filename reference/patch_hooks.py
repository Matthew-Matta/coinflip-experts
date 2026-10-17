"""Forward-hook taps for the validated backend mirror, with expert-ID alignment."""
from pathlib import Path
import numpy as np
import torch
from reference.format import OPS

SLOT_OPS={'topk_weights','expert_gate','expert_up','expert_act','expert_down','combine_terms'}


class PatchHooks:
    def __init__(self, patches=None, patch_ids=None, demo_scale_combine=None):
        self.taps=torch.nn.ModuleDict({name:torch.nn.Identity() for name in OPS})
        self.patches=patches or {};self.ids=None;self.skipped=0;self.handles=[]
        self.source_ids=np.load(patch_ids,allow_pickle=False) if patch_ids is not None else None
        if demo_scale_combine is not None:
            if not np.isfinite(demo_scale_combine) or demo_scale_combine <= 0:
                raise ValueError('demo combine scale must be finite and positive')
            # The known demo fault runs before replacement, so a patch can repair it.
            self.handles.append(self.taps['combine_terms'].register_forward_hook(
                lambda _module, _args, output: output * demo_scale_combine))
        for name,path in self.patches.items():
            if name not in OPS:raise ValueError(f'unknown patch op {name}')
            values=np.load(Path(path),allow_pickle=False)
            if not np.isfinite(values).all():raise ValueError('nonfinite patch')
            if name in SLOT_OPS and self.source_ids is None:raise ValueError('slot patch requires --patch-ids')
            self.handles.append(self.taps[name].register_forward_hook(self.hook(name,values)))

    def hook(self,name,values):
        def replace(_module,_args,output):
            if tuple(values.shape)!=tuple(output.shape):raise ValueError(f'patch shape mismatch {name}')
            replacement=torch.from_numpy(values.copy()).to(device=output.device,dtype=output.dtype)
            if name in SLOT_OPS:
                current=self.ids.detach().cpu().numpy()
                if self.source_ids.shape!=current.shape:raise ValueError('patch ids shape mismatch')
                same=np.all(np.sort(current,axis=-1)==np.sort(self.source_ids,axis=-1),axis=-1)
                order=np.argmax(current[:,:,None]==self.source_ids[:,None,:],axis=-1)
                index=torch.tensor(order,device=output.device)
                if output.ndim==3:index=index[:,:,None].expand_as(output)
                replacement=torch.gather(replacement,1,index)
                valid=torch.tensor(same,device=output.device)
                while valid.ndim<output.ndim:valid=valid.unsqueeze(-1)
                self.skipped+=int((~same).sum())
                replacement=torch.where(valid,replacement,output)
            return replacement
        return replace

    def __call__(self,name,value,ids=None):
        if ids is not None:self.ids=ids
        result=self.taps[name](value)
        if name=='topk_ids':
            a=result.detach().cpu().numpy()
            if np.any(a<0) or np.any(np.diff(np.sort(a,axis=-1),axis=-1)==0):raise ValueError('invalid patched expert ids')
            self.ids=result
        return result
