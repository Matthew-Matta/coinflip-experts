import json
from dataclasses import dataclass

import pandas as pd

from coinflip.provenance import collect, save_run


@dataclass
class Cfg:
    model: str = "allenai/OLMoE-1B-7B-0924"
    dtype: str = "float16"
    batch: int = 8


def test_collect_has_core_fields():
    p = collect(Cfg())
    for key in ("timestamp_utc", "git_sha", "packages", "config", "config_hash"):
        assert key in p
    assert p["config"]["batch"] == 8


def test_config_hash_is_stable():
    assert collect(Cfg())["config_hash"] == collect(Cfg())["config_hash"]
    assert collect(Cfg())["config_hash"] != collect(Cfg(batch=16))["config_hash"]


def test_save_run_writes_parquet_and_sidecar(tmp_path):
    df = pd.DataFrame({"token": [0, 1], "layer": [0, 0], "flip": [False, True]})
    pq = save_run(df, tmp_path, Cfg(), name="smoke")
    assert pq.exists()
    assert pd.read_parquet(pq).equals(df)
    side = json.loads((tmp_path / "smoke.provenance.json").read_text())
    assert side["config"]["model"].startswith("allenai/")
