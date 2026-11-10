"""Recorded paths (write relative with the root named; read both that and the old absolute form)
and the HF_HUB_OFFLINE default. No model, no data."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hs_rag_classifier import offline
from hs_rag_classifier import paths as P


def roots(tmp_path):
    data, repo = tmp_path / "data", tmp_path / "repo"
    data.mkdir(), repo.mkdir()
    return data, repo


def test_record_writes_relative_with_the_root(tmp_path):
    data, repo = roots(tmp_path)
    assert P.record(data / "eval" / "x.json", data, repo) == {"path": "eval/x.json", "root": "data"}
    assert P.record(repo / "configs" / "r.json", data, repo) == {
        "path": "configs/r.json",
        "root": "repo",
    }
    out = P.record(tmp_path / "elsewhere.json", data, repo)
    assert out == {"path": str(tmp_path / "elsewhere.json"), "root": None}
    # a data root inside the repo root is the more specific one
    assert P.record(repo / "d" / "f", repo / "d", repo) == {"path": "f", "root": "data"}
    assert json.loads(json.dumps(P.record(data / "a" / "b", data, repo)))["path"] == "a/b"


def test_resolve_reads_the_new_relative_form(tmp_path):
    data, repo = roots(tmp_path)
    for p in (data / "eval" / "x.json", repo / "configs" / "r.json", tmp_path / "out.json"):
        assert P.resolve(P.record(p, data, repo), data_root=data, repo_root=repo) == p
    assert P.resolve("eval/x.json", root="data", data_root=data) == data / "eval" / "x.json"


def test_resolve_reads_the_old_absolute_form(tmp_path):
    data, repo = roots(tmp_path)
    old = str(data / "mlflow" / "predictions.parquet")
    assert P.resolve(old, data_root=data, repo_root=repo) == Path(old)
    assert P.resolve({"path": old, "sha256": "x"}, data_root=data) == Path(old)  # no root key
    # The frozen rule file now records its input in the record() form (no home path in the tree).
    rule = json.loads((P.REPO_ROOT / "configs" / "operating_point_rule.json").read_text())
    assert rule["input"]["root"] == "data" and not Path(rule["input"]["path"]).is_absolute()
    assert P.resolve(rule["input"]) == P.DATA_ROOT / rule["input"]["path"]
    with pytest.raises(ValueError, match="without a root"):
        P.resolve("eval/x.json")
    with pytest.raises(ValueError, match="unknown root"):
        P.resolve({"path": "x", "root": "home"})


def test_offline_default_is_set_only_when_unset():
    env = {}
    assert offline.set_default(env) == "1" and env == {"HF_HUB_OFFLINE": "1"}
    for explicit in ("0", "1", "false"):
        env = {"HF_HUB_OFFLINE": explicit}
        assert offline.set_default(env) == explicit and env["HF_HUB_OFFLINE"] == explicit


def test_entry_points_get_the_default_on_import(tmp_path):
    code = "import hs_rag_classifier, os; print(os.environ['HF_HUB_OFFLINE'])"
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
    run = [sys.executable, "-c", code]
    got = subprocess.run(run, env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert got.stdout.strip() == "1"
    env["HF_HUB_OFFLINE"] = "0"
    got = subprocess.run(run, env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert got.stdout.strip() == "0"
