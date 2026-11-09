"""Error analysis by error type and chapter, A3 v A4, on saved test predictions (offline).

    python -m hs_rag_classifier.eval.error_analysis [--out DIR] [--no-mlflow]

Reads only saved files: each run's predictions.parquet (MLflow artifacts; the runs are found
from test_runs.jsonl by arm and dataset), the frozen test items for the input text, and the
pinned nomenclature (codes8.parquet, v4.0.1633). Never calls a model, a retriever or an arm,
never writes to test_runs.jsonl and never goes through the test-run guard. It logs one MLflow
run in experiment "error-analysis", tagged analysis=true and test_run=false.

Types: every wrong final answer (final top-1 cn8 != gold cn8) gets exactly one type, checked in
this order (classify):
    retrieval_miss       the gold HS6 is not among the item's stage-1 options
    wrong_heading        the final top-1's HS4 is not the gold HS4 (a wrong chapter included)
    right_hs4_wrong_hs6  HS4 right, HS6 wrong
    right_hs6_wrong_cn8  HS6 right, cn8 wrong
The HS6/HS4 are the final top-1 code's, never the stage-1 argmax. right_hs6_wrong_cn8 is split
by whether an "Other" line is involved (gold only, predicted only, both, neither). Rule
(is_other): a cn8 is "Other" when the nomenclature line that names it, codes8.parquet
description (the cn8's own 8-digit line, or the nearest parent line when the cn8 has none of its
own, node_above_cn8), starts with "Other" after leading dashes, spaces and colons are removed,
compared case-insensitively. Items that cannot be classified (no top-1, a gold or predicted code
shorter than 8 digits, no stage-1 options, a cn8 missing from the nomenclature for the split)
are listed with the reason and never given a type.

Hard checks (HardCheckError): the stage-1 option lists are identical for A3 and A4 on every item
(same retrieval; A4 only adds the adapter), so the retrieval_miss items are identical; and no
"fixed" item (A3 wrong, A4 right) is a retrieval_miss for A4.

Intervals: A4 - A3 type shares of all items by eval.metrics.paired_difference (the set's
clusters, 1,000 resamples, seed 20261021), the scheme of the A4 v A3 test comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hs_rag_classifier.eval import metrics as M
from hs_rag_classifier.eval.items import DATA_ROOT, sha256_file

TYPES = ("retrieval_miss", "wrong_heading", "right_hs4_wrong_hs6", "right_hs6_wrong_cn8")
OTHER_SPLITS = ("gold_only", "pred_only", "both", "neither")
SETS = ("rulings", "atoz")
SET_NAMES = {"rulings": "rulings", "atoz": "A to Z"}
ARMS = ("a3", "a4")
OUT = DATA_ROOT / "eval" / "error_analysis"
SMALL_CHAPTER = 20
N_EXAMPLES = 10
EXPERIMENT = "error-analysis"


class HardCheckError(RuntimeError):
    """A hard check failed; the analysis stops."""


class Unclassifiable(ValueError):
    """An item that cannot be given a type; the message is the reason."""


# --- rules -------------------------------------------------------------------------------


def is_other(description: str | None) -> bool:
    """The "Other" rule (module docstring)."""
    if description is None:
        raise Unclassifiable("no nomenclature line names this cn8")
    return description.lstrip(" -–—:\t").casefold().startswith("other")


def classify(gold: str | None, pred: str | None, options: Sequence[str] | None) -> str | None:
    """The type of a wrong answer, None when right (precedence in the module docstring)."""
    if not gold or len(gold) < 8:
        raise Unclassifiable(f"gold code {gold!r} is shorter than 8 digits")
    if not pred:
        raise Unclassifiable("no final top-1 code")
    if len(pred) < 8:
        raise Unclassifiable(f"final top-1 {pred!r} is shorter than 8 digits")
    if not options:
        raise Unclassifiable("no stage-1 options saved")
    if pred[:8] == gold[:8]:
        return None
    if gold[:6] not in {str(o)[:6] for o in options}:
        return "retrieval_miss"
    if pred[:4] != gold[:4]:
        return "wrong_heading"
    if pred[:6] != gold[:6]:
        return "right_hs4_wrong_hs6"
    return "right_hs6_wrong_cn8"


def other_split(gold: str, pred: str, descriptions: Mapping[str, str]) -> str:
    g, p = is_other(descriptions.get(gold[:8])), is_other(descriptions.get(pred[:8]))
    return {(True, False): "gold_only", (False, True): "pred_only", (True, True): "both",
            (False, False): "neither"}[(g, p)]  # fmt: skip


# --- per item ----------------------------------------------------------------------------


def item_table(frame: pd.DataFrame, descriptions: Mapping[str, str]) -> pd.DataFrame:
    """One row per item: gold, final top-1, stage-1 options, type, "Other" split, reason."""
    rows = []
    for r in frame.itertuples(index=False):
        det = json.loads(r.details) if isinstance(r.details, str) else (r.details or {})
        options = list((det.get("stage1") or {}).get("options") or [])
        gold = str(r.label_code)[:8] if r.label_code else None
        pred = str(r.top1)[:8] if isinstance(r.top1, str) and r.top1 else None
        row = {"item_id": r.item_id, "cluster_id": r.cluster_id, "gold_cn8": gold, "pred_cn8": pred,
               "gold_chapter": gold[:2] if gold else None, "n_options": len(options),
               "options_hs6": tuple(str(o)[:6] for o in options),
               "gold_in_options": bool(gold) and gold[:6] in {str(o)[:6] for o in options},
               "correct": None, "type": None, "other_split": None, "unclassified": None}  # fmt: skip
        try:
            t = classify(gold, pred, options)
            row["correct"] = t is None
            row["type"] = t
            if t == "right_hs6_wrong_cn8":
                row["other_split"] = other_split(gold, pred, descriptions)
        except Unclassifiable as exc:
            row["type"], row["other_split"], row["unclassified"] = None, None, str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def check_correct(t: pd.DataFrame, frame: pd.DataFrame) -> None:
    """The classifier's right/wrong must equal the harness's correct_8 on every item."""
    c8 = frame.set_index("item_id").loc[t["item_id"], "correct_8"].to_numpy(float)
    ok = t["correct"].map(lambda x: math.nan if x is None else float(x)).to_numpy(float)
    keep = ~np.isnan(ok)
    if not np.array_equal(ok[keep], c8[keep]):
        raise HardCheckError("the classifier's right/wrong differs from the harness's correct_8")


def hard_checks(t3: pd.DataFrame, t4: pd.DataFrame) -> dict[str, Any]:
    """Both hard checks (module docstring); raises HardCheckError on a failure."""
    if list(t3["item_id"]) != list(t4["item_id"]):
        raise HardCheckError("A3 and A4 do not cover the same items in the same order")
    same_opts = t3["options_hs6"].to_numpy() == t4["options_hs6"].to_numpy()
    rm3 = set(t3.loc[t3["type"] == "retrieval_miss", "item_id"])
    rm4 = set(t4.loc[t4["type"] == "retrieval_miss", "item_id"])
    fixed = set(t3.loc[t3["correct"].eq(False) & t4["correct"].eq(True).to_numpy(), "item_id"])
    out = {"options_identical": int(same_opts.sum()), "n_items": len(t3),
           "retrieval_miss_a3": len(rm3), "retrieval_miss_a4": len(rm4),
           "retrieval_miss_identical": rm3 == rm4, "a3_only": sorted(rm3 - rm4)[:20],
           "a4_only": sorted(rm4 - rm3)[:20], "fixed": len(fixed),
           "fixed_that_are_a4_retrieval_miss": sorted(fixed & rm4)}  # fmt: skip
    if not same_opts.all():
        raise HardCheckError(f"stage-1 options differ on {int((~same_opts).sum())} items")
    if rm3 != rm4:
        raise HardCheckError(
            f"retrieval_miss differs: A3 only {out['a3_only']}, A4 only {out['a4_only']}"
        )
    if fixed & rm4:
        raise HardCheckError(
            f"fixed items that are A4 retrieval misses: {out['fixed_that_are_a4_retrieval_miss']}"
        )
    return out


# --- tables ------------------------------------------------------------------------------


def _indicator(t: pd.DataFrame, typ: str) -> pd.DataFrame:
    return pd.DataFrame({"item_id": t["item_id"], "cluster_id": t["cluster_id"],
                         "x": (t["type"] == typ).astype(float)})  # fmt: skip


def type_table(t3: pd.DataFrame, t4: pd.DataFrame, n_resamples: int = M.BOOTSTRAP_RESAMPLES,
               seed: int = M.BOOTSTRAP_SEED) -> pd.DataFrame:  # fmt: skip
    """Counts and shares (of all items and of wrong items) per type, A4 - A3 with intervals."""
    rows = []
    wrong3, wrong4 = int(t3["correct"].eq(False).sum()), int(t4["correct"].eq(False).sum())
    n = len(t3)
    for typ in (*TYPES, "wrong_total", "right", "unclassified"):
        if typ == "wrong_total":
            c3, c4 = wrong3, wrong4
        elif typ == "right":
            c3, c4 = int(t3["correct"].eq(True).sum()), int(t4["correct"].eq(True).sum())
        elif typ == "unclassified":
            c3, c4 = int(t3["unclassified"].notna().sum()), int(t4["unclassified"].notna().sum())
        else:
            c3, c4 = int((t3["type"] == typ).sum()), int((t4["type"] == typ).sum())
        row = {"type": typ, "a3_n": c3, "a4_n": c4, "a3_share_all": c3 / n, "a4_share_all": c4 / n,
               "diff_share_all": (c4 - c3) / n,
               "a3_share_wrong": c3 / wrong3 if typ in TYPES and wrong3 else None,
               "a4_share_wrong": c4 / wrong4 if typ in TYPES and wrong4 else None}  # fmt: skip
        if typ in TYPES:
            d = M.paired_difference(
                _indicator(t4, typ), _indicator(t3, typ), "x", n_resamples, seed
            )
            row.update(
                diff_low=d["low"],
                diff_high=d["high"],
                share_resamples_le_0=d["share_resamples_le_0"],
            )
        rows.append(row)
    return pd.DataFrame(rows)


def other_table(t3: pd.DataFrame, t4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in OTHER_SPLITS:
        rows.append({"other_split": s, "a3_n": int((t3["other_split"] == s).sum()),
                     "a4_n": int((t4["other_split"] == s).sum())})  # fmt: skip
    return pd.DataFrame(rows)


def chapter_table(t3: pd.DataFrame, t4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ch, g3 in t3.groupby("gold_chapter", sort=True):
        g4 = t4.loc[g3.index]
        row = {"chapter": ch, "n": len(g3), "small": len(g3) < SMALL_CHAPTER,
               "a3_cn8_acc": float(g3["correct"].eq(True).mean()),
               "a4_cn8_acc": float(g4["correct"].eq(True).mean()),
               "a3_errors": int(g3["correct"].eq(False).sum()),
               "a4_errors": int(g4["correct"].eq(False).sum())}  # fmt: skip
        for typ in TYPES:
            row[f"a3_{typ}"] = int((g3["type"] == typ).sum())
            row[f"a4_{typ}"] = int((g4["type"] == typ).sum())
        rows.append(row)
    f = pd.DataFrame(rows)
    return f.sort_values(["a4_errors", "n", "chapter"], ascending=[False, False, True]).reset_index(
        drop=True
    )


def fixed_broke(t3: pd.DataFrame, t4: pd.DataFrame) -> dict[str, Any]:
    fixed = t3["correct"].eq(False).to_numpy() & t4["correct"].eq(True).to_numpy()
    broke = t3["correct"].eq(True).to_numpy() & t4["correct"].eq(False).to_numpy()
    by_type = {
        "fixed_by_a3_type": t3.loc[fixed, "type"]
        .value_counts()
        .reindex(TYPES, fill_value=0)
        .astype(int)
        .to_dict(),
        "broke_by_a4_type": t4.loc[broke, "type"]
        .value_counts()
        .reindex(TYPES, fill_value=0)
        .astype(int)
        .to_dict(),
    }
    ch = pd.DataFrame(
        {"chapter": t3["gold_chapter"], "fixed": fixed.astype(int), "broke": broke.astype(int)}
    )
    by_ch = ch.groupby("chapter").sum().reset_index()
    by_ch = by_ch[(by_ch["fixed"] > 0) | (by_ch["broke"] > 0)]
    by_ch = (
        by_ch.assign(net=by_ch["fixed"] - by_ch["broke"])
        .sort_values(["net", "chapter"])
        .reset_index(drop=True)
    )
    b4 = t4.loc[broke]
    return {"fixed": int(fixed.sum()), "broke": int(broke.sum()), **by_type,
            "broke_a4_same_hs6_as_gold": int((b4["pred_cn8"].str[:6] == b4["gold_cn8"].str[:6]).sum()),
            "by_chapter": by_ch}  # fmt: skip


def ceiling(t3: pd.DataFrame, t4: pd.DataFrame) -> dict[str, Any]:
    wrong4 = t4["correct"].eq(False)
    wrong3 = t3["correct"].eq(False)
    rm4 = int((t4["type"] == "retrieval_miss").sum())
    rec = float(t4["gold_in_options"].mean())
    return {"hs6_recall_at_m": rec, "m_distribution": t4["n_options"].value_counts().sort_index().astype(int).to_dict(),
            "a4_wrong": int(wrong4.sum()), "a4_retrieval_miss": rm4,
            "a4_share_of_wrong_retrieval_miss": rm4 / int(wrong4.sum()) if wrong4.any() else None,
            "a3_share_of_wrong_retrieval_miss": int((t3["type"] == "retrieval_miss").sum()) / int(wrong3.sum())
            if wrong3.any() else None,
            "a4_model_errors": int(wrong4.sum()) - rm4}  # fmt: skip


def ceiling_sentences(c: Mapping[str, Any], dataset: str) -> list[str]:
    s = c["a4_share_of_wrong_retrieval_miss"]
    where = "retrieval" if s is not None and s > 0.5 else "the model"
    first = (
        f"On {dataset} test the gold HS6 is among the stage-1 options for "
        f"{c['hs6_recall_at_m']:.1%} of items, and {c['a4_retrieval_miss']} of A4's "
        f"{c['a4_wrong']} wrong answers ({s:.1%}) are retrieval misses."
    )
    second = (
        f"So A4's remaining errors lie mainly in {where}: {c['a4_model_errors']} of its wrong "
        f"answers had the gold HS6 among the options and {c['a4_retrieval_miss']} did not."
    )
    return [first, second]


def examples(t4: pd.DataFrame, texts: Mapping[str, str], paths: Mapping[str, str],
             k: int = N_EXAMPLES, seed: int = M.BOOTSTRAP_SEED) -> pd.DataFrame:  # fmt: skip
    """k items per type for A4, drawn with a fixed seed (one generator per type)."""
    rows = []
    for typ in TYPES:
        ids = sorted(t4.loc[t4["type"] == typ, "item_id"], key=lambda x: (len(x), x))
        rng = np.random.default_rng(seed)
        pick = sorted(rng.choice(len(ids), size=min(k, len(ids)), replace=False)) if ids else []
        sub = t4.set_index("item_id")
        for i in pick:
            iid = ids[i]
            r = sub.loc[iid]
            rows.append({"type": typ, "item_id": iid, "input": (texts.get(iid) or "")[:200],
                         "gold_cn8": r["gold_cn8"], "gold_description": paths.get(r["gold_cn8"]),
                         "pred_cn8": r["pred_cn8"], "pred_description": paths.get(r["pred_cn8"]),
                         "other_split": r["other_split"]})  # fmt: skip
    return pd.DataFrame(rows)


# --- inputs and run ----------------------------------------------------------------------


def find_runs(test_log: Path) -> dict[tuple[str, str], tuple[int, str]]:
    """(dataset, arm) -> (test-run number, MLflow run id), for a3/a4 on rulings and A to Z."""
    out = {}
    for n, line in enumerate(test_log.read_text(encoding="utf-8").splitlines(), 1):
        r = json.loads(line)
        if r["arm"] in ARMS and r["dataset"] in SETS:
            if (r["dataset"], r["arm"]) in out:
                raise HardCheckError(
                    f"more than one {r['arm']} {r['dataset']} test run in {test_log}"
                )
            out[(r["dataset"], r["arm"])] = (n, r["mlflow_run_id"])
    return out


def predictions_path(run_id: str) -> Path:
    import mlflow

    uri = mlflow.get_run(run_id).info.artifact_uri
    path = Path(uri.removeprefix("file://")) / "predictions.parquet"
    if not path.is_file():
        raise HardCheckError(f"no saved predictions for run {run_id} at {path}")
    return path


def nomenclature(root: Path = DATA_ROOT) -> tuple[dict[str, str], dict[str, str], Path]:
    from hs_rag_classifier.data.rulings import NOMENCLATURE_VERSION

    path = root / "processed" / "tariff" / NOMENCLATURE_VERSION / "codes8.parquet"
    df = pd.read_parquet(path, columns=["cn8", "description", "path_text"])
    return dict(zip(df["cn8"], df["description"], strict=True)), dict(
        zip(df["cn8"], df["path_text"], strict=True)), path  # fmt: skip


def analyse(f3: pd.DataFrame, f4: pd.DataFrame, descriptions: Mapping[str, str],
            n_resamples: int = M.BOOTSTRAP_RESAMPLES, seed: int = M.BOOTSTRAP_SEED) -> dict[str, Any]:  # fmt: skip
    """Every table for one set (A3 and A4 frames over the same items)."""
    f4 = f4.set_index("item_id").loc[f3["item_id"]].reset_index()
    t3, t4 = item_table(f3, descriptions), item_table(f4, descriptions)
    if "correct_8" in f3:
        check_correct(t3, f3)
        check_correct(t4, f4)
    checks = hard_checks(t3, t4)
    return {"items": {"a3": t3, "a4": t4}, "hard_checks": checks,
            "types": type_table(t3, t4, n_resamples, seed), "other": other_table(t3, t4),
            "chapters": chapter_table(t3, t4), "fixed_broke": fixed_broke(t3, t4),
            "ceiling": ceiling(t3, t4),
            "unclassified": pd.concat([t.loc[t["unclassified"].notna(), ["item_id", "unclassified"]]
                                       .assign(arm=a) for a, t in (("a3", t3), ("a4", t4))])}  # fmt: skip


def run(out: Path = OUT, log_mlflow: bool = True) -> dict[str, Any]:
    import mlflow

    from hs_rag_classifier.eval import runner as R
    from hs_rag_classifier.eval.items import load

    out.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(R.tracking_uri())
    test_log_sha = sha256_file(R.TEST_LOG)
    runs = find_runs(R.TEST_LOG)
    descriptions, paths, nom_path = nomenclature()
    summary: dict[str, Any] = {"test_runs_jsonl_sha256": test_log_sha, "nomenclature": {
        "path": str(nom_path), "sha256": sha256_file(nom_path)}, "inputs": {}, "sets": {},
        "rules": {"types": list(TYPES),
                  "other": ("codes8.parquet description of the cn8 (its own line, or the nearest "
                            "parent line when node_above_cn8) starts with 'Other' after stripping "
                            "leading dashes, spaces and colons, case-insensitive"),
                  "bootstrap": {"resamples": M.BOOTSTRAP_RESAMPLES, "seed": M.BOOTSTRAP_SEED}}}  # fmt: skip
    files = []
    for ds in SETS:
        frames = {}
        for arm in ARMS:
            n, rid = runs[(ds, arm)]
            p = predictions_path(rid)
            summary["inputs"][f"{ds}_{arm}"] = {"test_run": n, "mlflow_run_id": rid, "path": str(p),
                                                "sha256": sha256_file(p)}  # fmt: skip
            frames[arm] = pd.read_parquet(p)
        res = analyse(frames["a3"], frames["a4"], descriptions)
        texts = {i.item_id: i.text for i in load(ds, "test").items}
        ex = examples(res["items"]["a4"], texts, paths)
        fb = res["fixed_broke"]
        tables = {"types": res["types"], "other_split": res["other"], "chapters": res["chapters"],
                  "fixed_broke_by_chapter": fb["by_chapter"], "examples_a4": ex,
                  "unclassified": res["unclassified"],
                  "items_a3": res["items"]["a3"].drop(columns=["options_hs6"]),
                  "items_a4": res["items"]["a4"].drop(columns=["options_hs6"])}  # fmt: skip
        for name, df in tables.items():
            path = out / f"{ds}_{name}.csv"
            df.to_csv(path, index=False)
            files.append(path)
        summary["sets"][ds] = {
            "hard_checks": res["hard_checks"],
            "cn8_top1": {a: float(res["items"][a]["correct"].eq(True).mean()) for a in ARMS},
            "types": res["types"].to_dict(orient="records"),
            "other_split": res["other"].to_dict(orient="records"),
            "fixed_broke": {k: v for k, v in fb.items() if k != "by_chapter"},
            "ceiling": res["ceiling"],
            "ceiling_sentences": ceiling_sentences(res["ceiling"], SET_NAMES[ds]),
            "n_unclassified": len(res["unclassified"]),
        }
    spath = out / "summary.json"
    spath.write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    files.append(spath)
    if log_mlflow:
        from hs_rag_classifier.eval.runner import _experiment_id

        exp = _experiment_id(EXPERIMENT, R.MLFLOW_DIR / "artifacts")
        with mlflow.start_run(experiment_id=exp, run_name="a3_v_a4_error_types") as r:
            mlflow.set_tags({"analysis": "true", "test_run": "false", "comparison": "true",
                             "git_commit": R.git_info()["commit"]})  # fmt: skip
            mlflow.log_params({f"{k}_sha256": v["sha256"][:16] for k, v in summary["inputs"].items()}
                              | {f"{k}_run": v["mlflow_run_id"] for k, v in summary["inputs"].items()})  # fmt: skip
            for ds, s in summary["sets"].items():
                for row in s["types"]:
                    if row["type"] in TYPES:
                        mlflow.log_metrics({f"{ds}_a3_{row['type']}": row["a3_n"],
                                            f"{ds}_a4_{row['type']}": row["a4_n"]})  # fmt: skip
                mlflow.log_metrics({f"{ds}_hs6_recall_at_m": s["ceiling"]["hs6_recall_at_m"]})
            for f in files:
                mlflow.log_artifact(str(f))
            summary["mlflow_run_id"] = r.info.run_id
        spath.write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="error_analysis", description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--no-mlflow", action="store_true")
    a = p.parse_args(argv)
    try:
        s = run(a.out, not a.no_mlflow)
    except HardCheckError as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({ds: {k: v for k, v in x.items() if k in ("cn8_top1", "hard_checks", "ceiling_sentences")}
                      for ds, x in s["sets"].items()}, indent=1, default=str))  # fmt: skip
    print("mlflow run", s.get("mlflow_run_id"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
