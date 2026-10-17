import numpy as np
from analysis.nccl_orders import rounded,reduction
from analysis.nccl_followup import run
from reference.format import sha256,write_json


def test_saved_rank_followup_checks_hashes_and_all_rank_groups(tmp_path):
    source=tmp_path/'source';source.mkdir();parts=tmp_path/'parts';parts.mkdir()
    for n in (2,4):
        for target in ('oproj','down'):
            for wire in ('fp32','bf16'):
                prefix=f'{target}-n{n}-{wire}'
                values=rounded(np.array([[1e8,1],[1,2],[-1e8,3],[1,4]][:n],np.float32),wire)
                np.save(parts/f'{prefix}-partials.npy',values)
                expected=reduction(values,(0,1,2,3),'sequential',wire) if n==4 else rounded(values[0]+values[1],wire)
                for algo in ('Ring','Tree'):
                    np.save(parts/f'{prefix}-{algo.lower()}.npy',expected)
                    folder=source/'nccl'/f'n{n}-{algo}';folder.mkdir(parents=True,exist_ok=True)
                    for rank in range(n):
                        actual=expected.copy()
                        if n==4 and rank==3:actual=reduction(values,(0,2,1,3),'sequential',wire)
                        np.save(folder/f'{prefix}-rank{rank}.npy',actual)
    hashes={p.name:sha256(p) for p in parts.glob('*.npy')}
    write_json(source/'summary.json',dict(input_sha256=dict(nccl_input_sha256=hashes),
               output_sha256={str(p.relative_to(source)):sha256(p) for p in source.rglob('*.npy')},
               gpu='synthetic fixture',torch_version='fixture',cuda_version='fixture',nccl_version='fixture',wall_seconds=0,source_sha256={}))
    result=run(source,parts,tmp_path/'output')
    assert len(result['order_summary'])>8  # Distinct rank outputs analysed separately.
    assert all(row['any_order_share']==1 for row in result['order_summary'])
    assert (tmp_path/'output/order_shares.csv').is_file()
    assert (tmp_path/'output/position_shares.csv').is_file()
