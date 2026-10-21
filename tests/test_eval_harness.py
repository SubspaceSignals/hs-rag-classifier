"""Loaders, the test-split guard, the test-run log and an MLflow smoke test, on tiny data in
tmp_path (never the real data root or the real MLflow store)."""

import json
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from hs_rag_classifier.eval import items as I
from hs_rag_classifier.eval import runner as R
from hs_rag_classifier.eval.__main__ import main
from hs_rag_classifier.eval.arms import Candidate, OracleArm, Prediction, RandomArm

EVAL_SRC = Path(__file__).resolve().parents[1] / "src" / "hs_rag_classifier" / "eval"
VALID8 = {"01012100", "82119100"}
VALID10 = {"0101210000", "8211910000"}
CLEAN = {"commit": "abc123", "dirty": False}


@pytest.fixture()
def data(tmp_path):
    """A data root and splits dir holding tiny rulings, A to Z and hand sets."""
    root, splits = tmp_path / "data", tmp_path / "repo" / "splits"
    rulings = root / "processed" / "rulings" / "v1"
    rulings.mkdir(parents=True)
    val = pd.DataFrame({
        "ref": ["10", "20"], "member_refs": [["10", "11"], ["20"]],
        "description": ["Steel table knives", "Live horses"],
        "keywords": [["KNIVES", "TABLE"], []], "code10": ["8211910000", "0101210000"],
        "cn8": ["82119100", "01012100"], "hs6": ["821191", "010121"], "chapter": ["82", "01"],
        "start_date": [None, None], "expiry_date": [None, None], "cluster_id": ["10", "10"],
        "code_status": ["current", "current"], "label_depth": pd.array([10, 8], dtype="Int64"),
        "flags": [[], []],
    })  # fmt: skip
    val.to_parquet(rulings / "val.parquet", index=False)
    (rulings / "justifications.parquet").write_bytes(b"not parquet: reading this must fail")
    (splits / "rulings_v1").mkdir(parents=True)
    (splits / "rulings_v1" / "val_refs.txt").write_text("10\n11\n20\n")
    atoz = root / "processed" / "atoz" / "v1"
    atoz.mkdir(parents=True)
    pd.DataFrame({
        "id": ["5", "7"], "member_ids": [["5"], ["7", "9"]], "title": ["horses", "knives"],
        "cn8": ["01012100", "82119100"], "code10": [None, "8211910000"],
        "label_depth": pd.array([8, 10], dtype="Int64"),
    }).to_parquet(atoz / "atoz_test8.parquet", index=False)  # fmt: skip
    (splits / "atoz_v1").mkdir()
    (splits / "atoz_v1" / "test_refs.txt").write_text("5\n7\n9\n")
    (root / "hand").mkdir()
    (root / "hand" / "hand_v1.csv").write_text(
        "id,description,cn8,code10,label_confidence,source_url,category,notes\n"
    )
    return root, splits


def test_loaders_never_read_justifications(data, monkeypatch):
    root, splits = data
    read = []
    real = pd.read_parquet

    def spy(path, *args, **kwargs):
        read.append(Path(path).name)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(I.pd, "read_parquet", spy)
    I.load("rulings", "val", root=root, splits_dir=splits)
    I.load("rulings", "val", with_keywords=True, root=root, splits_dir=splits)
    I.load("atoz", "test", root=root, splits_dir=splits)
    assert read == ["val.parquet", "val.parquet", "atoz_test8.parquet"]
    for path in EVAL_SRC.glob("*.py"):  # and no eval module names the file or the column
        text = path.read_text(encoding="utf-8")
        assert "justifications.parquet" not in text and '"justification"' not in text, path


def test_rulings_keywords_only_with_flag(data):
    root, splits = data
    plain = I.load("rulings", "val", root=root, splits_dir=splits)
    assert [i.text for i in plain.items] == ["Steel table knives", "Live horses"]
    assert plain.ablation == "none" and plain.items[0].keywords == ["KNIVES", "TABLE"]
    abl = I.load("rulings", "val", with_keywords=True, root=root, splits_dir=splits)
    assert abl.items[0].text == "Steel table knives\nKeywords: KNIVES; TABLE"
    assert abl.items[1].text == "Live horses" and abl.ablation == "keywords"
    assert [(i.label_code, i.label_depth) for i in plain.items] == [("8211910000", 10),
                                                                    ("01012100", 8)]  # fmt: skip
    assert plain.split_sha256 == I.sha256_file(splits / "rulings_v1" / "val_refs.txt")
    with pytest.raises(ValueError, match="rulings only"):
        I.load("atoz", "test", with_keywords=True, root=root, splits_dir=splits)


def test_loaders_check_frozen_ids_and_splits(data):
    root, splits = data
    (splits / "rulings_v1" / "val_refs.txt").write_text("10\n20\n")
    with pytest.raises(RuntimeError, match="frozen ids"):
        I.load("rulings", "val", root=root, splits_dir=splits)
    with pytest.raises(ValueError, match="only a test split"):
        I.load("atoz", "val", root=root, splits_dir=splits)
    with pytest.raises(ValueError, match="only a test split"):
        I.load("hand", "val", root=root, splits_dir=splits)


def test_atoz_and_hand_loaders(data):
    root, splits = data
    atoz = I.load("atoz", "test", root=root, splits_dir=splits)
    assert [(i.item_id, i.label_code, i.cluster_id, i.chapter) for i in atoz.items] == [
        ("5", "01012100", "5", "01"), ("7", "8211910000", "7", "82")]  # fmt: skip
    assert atoz.split_sha256 == I.sha256_file(splits / "atoz_v1" / "test_refs.txt")
    hand_csv = root / "hand" / "hand_v1.csv"
    assert I.load("hand", "test", root=root, splits_dir=splits).items == []  # empty: 0 items
    with hand_csv.open("a") as fh:
        fh.write("H001,Table knife,82119100,,sure,https://example.org,cutlery,\n")
        fh.write("H002,Horse,01012100,0101210000,unsure,https://example.org,animals,\n")
    hand = I.load("hand", "test", root=root, splits_dir=splits)
    assert [(i.item_id, i.label_code, i.label_depth, i.cluster_id) for i in hand.items] == [
        ("H001", "82119100", 8, "H001"), ("H002", "0101210000", 10, "H002")]  # fmt: skip


class FixedArm:
    """A non-sanity arm for guard tests: always predicts 8211910000."""

    name = "fixed"

    def __init__(self):
        self.config = {"code": "8211910000"}

    def predict(self, items):
        return [Prediction(item_id=i.item_id, candidates=[Candidate(code="8211910000", score=1)],
                           confidence=0.7) for i in items]  # fmt: skip


def make_set(data, split="test"):
    root, splits = data
    evalset = I.load("atoz", "test", root=root, splits_dir=splits)
    return evalset.model_copy(update={"split": split})


def run(arm, evalset, tmp_path, git=CLEAN, **kw):
    return R.run_eval(arm, evalset, VALID8, VALID10, mlflow_dir=tmp_path / "mlflow",
                      test_log=tmp_path / "eval" / "test_runs.jsonl", git=git, n_resamples=20,
                      **kw)  # fmt: skip


def test_guard_refuses_dirty_tree_limit_and_sanity_arms(data, tmp_path):
    evalset = make_set(data)
    with pytest.raises(R.GuardError, match="dirty"):
        run(FixedArm(), evalset, tmp_path, git={"commit": "abc", "dirty": True})
    with pytest.raises(R.GuardError, match="limit"):
        run(FixedArm(), evalset, tmp_path, limit=1)
    for arm in (OracleArm(), RandomArm(sorted(VALID8), k=2)):
        with pytest.raises(R.GuardError, match="sanity"):
            run(arm, evalset, tmp_path)
    assert not (tmp_path / "eval").exists() and not (tmp_path / "mlflow").exists()
    assert R.git_info()["commit"]  # the real repo is readable


def test_test_run_logged_and_listed(data, tmp_path, capsys):
    result = run(FixedArm(), make_set(data), tmp_path)
    runs = R.read_test_runs(tmp_path / "eval" / "test_runs.jsonl")
    assert len(runs) == 1
    line = runs[0]
    assert set(line) == {"time", "arm", "config_hash", "commit", "dataset", "split_sha256",
                         "mlflow_run_id"}  # fmt: skip
    assert (line["arm"], line["commit"], line["mlflow_run_id"]) == ("fixed", "abc123",
                                                                     result["run_id"])  # fmt: skip
    run(FixedArm(), make_set(data, split="val"), tmp_path)  # validation: not logged
    assert len(R.read_test_runs(tmp_path / "eval" / "test_runs.jsonl")) == 1
    assert main(["test-runs", "--log", str(tmp_path / "eval" / "test_runs.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "1 test-set runs" in out and "fixed: 1" in out


def test_mlflow_smoke(data, tmp_path):
    root, splits = data
    evalset = I.load("rulings", "val", root=root, splits_dir=splits)
    result = run(OracleArm(), evalset, tmp_path)
    assert (tmp_path / "mlflow" / "mlflow.db").exists()
    client = MlflowClient(tracking_uri=R.tracking_uri(tmp_path / "mlflow"))
    r = client.get_run(result["run_id"])
    assert r.data.params["arm"] == "oracle" and len(r.data.params["config_hash"]) == 12
    tags = r.data.tags
    assert tags["dataset"] == "rulings" and tags["split"] == "val"
    assert tags["split_sha256"] == evalset.split_sha256 and tags["ablation"] == "none"
    assert tags["git_commit"] == "abc123" and tags["nomenclature_version"] == "v4.0.1633"
    assert r.data.metrics["top1_8"] == 1.0 and r.data.metrics["n_10"] == 1.0
    assert r.data.metrics["headline"] == 1.0 and r.data.metrics["invalid_rate"] == 0.0
    files = {f.path for f in client.list_artifacts(result["run_id"])}
    assert files == {"predictions.parquet", "metrics.json", "per_chapter.csv",
                     "reliability.png", "coverage.png"}  # fmt: skip
    exp = client.get_experiment(r.info.experiment_id)
    assert (
        exp.name == "eval-rulings"
        and str(tmp_path / "mlflow" / "artifacts") in exp.artifact_location
    )
    metrics = json.loads(Path(client.download_artifacts(result["run_id"], "metrics.json",
                                                        str(tmp_path))).read_text())  # fmt: skip
    assert metrics["bootstrap"]["unit"] == "cluster_id"
    lines = R.summary_lines(result, OracleArm(), evalset)
    assert lines[0].startswith(f"Run {result['run_id']}")


def test_cli_needs_split_test_for_atoz_and_hand(capsys):
    assert main(["run", "--arm", "oracle", "--dataset", "atoz"]) == 2
    assert "only a test split" in capsys.readouterr().err
