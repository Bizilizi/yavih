"""Report view 1 (the lm-evaluation-harness table), the results store (files, merge, joins; one directory
per model name, entries ``<benchmark>:<reading>``), the results-file version check, ``yavih.config.results_root``.
Synthetic records only."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import pytest

from test_report import EH, PH, PH2, QH, make_result, write_result
from yavih import hashing
from yavih.config import results_root
from yavih.report import report, store, tables
from yavih.report.compare import lm_eval_tables


@pytest.fixture(autouse=True)
def _no_site_results(monkeypatch):
    monkeypatch.delenv("YAVIH_RESULTS", raising=False)


# --------------------------------------------------------------------------- config (results root)


def test_results_root_precedence(tmp_path, monkeypatch):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[results]\nroot = "/site/results"\n')
    assert results_root(tmp_path / "runs", env={}, config={}) == (tmp_path / "results")
    assert results_root(tmp_path / "runs", env={}, config={"results": {"root": "/site/results"}}) == Path("/site/results")
    assert results_root(tmp_path / "runs", env={"YAVIH_RESULTS": "/env/res"}, config={"results": {"root": "/x"}}) == Path("/env/res")
    # without an explicit runs root: the configured runs root's sibling
    assert results_root(None, env={"YAVIH_RUNS": str(tmp_path / "r2")}, config={}) == tmp_path / "results"


# --------------------------------------------------------------------------- view 1


def _records(*results):
    return [{"path": f"/x/{i}/result.json", "result": r} for i, r in enumerate(results)]


def test_lm_eval_table_rows_order_floor_and_na():
    box = make_result(bench="vggss_box", readout="box", model="m", reading="av", metrics={"n_eval": 900, "iou_mean": 0.4, "acc@0.5": 0.3},
                      floors={"iou_floor": 0.2}, ci=[0.35, 0.45])
    box["ci95"] = {"iou_mean": [0.35, 0.45]}
    na = make_result(bench="nyuv2_depth", readout="grid", model="m", status="not_applicable")
    k400 = [make_result(model="m", seed=s, acc=a) for s, a in ((0, 0.5), (1, 0.6))]
    rows = lm_eval_tables(_records(box, na, *k400), aliases={})["m"]
    by_bench = {}
    for r in rows:
        by_bench.setdefault(r["benchmark"], []).append(r)
    b = by_bench["vggss_box"]
    # headline first (iou_mean has the floor), disclosures last; the floor comes from floor_of / the name stem
    assert [r["metric"] for r in b] == ["iou_mean", "acc@0.5", "n_eval"]
    assert b[0]["floor"] == 0.2 and b[0]["ci95"] == [0.35, 0.45] and b[1]["floor"] is None and b[2]["value"] == 900
    assert b[0]["reading"] == "av" and b[0]["eval_hash"] == EH[:12]
    k = by_bench["k400"]
    assert k[0]["metric"] == "acc" and k[0]["value"] == pytest.approx(0.55) and k[0]["std"] == pytest.approx(0.0707, abs=1e-3)
    assert k[0]["n_seeds"] == 2 and k[0]["ci95"] == pytest.approx([0.45, 0.55])  # the lowest seed's interval
    assert by_bench["nyuv2_depth"] == [{"benchmark": "nyuv2_depth", "reading": "v", "version": 2, "eval_hash": EH[:12], "readout": "grid",
                                        "metric": "—", "value": None, "std": None, "n_seeds": 1, "ci95": None, "floor": None,
                                        "status": "not_applicable", "reason": "eligibility: missing pos"}]
    text = tables.render_lm_eval({"m": rows})
    # the nine-column table, eval_hash next to the version
    assert "| benchmark | reading | version | eval_hash | readout | metric | value | ci95 | floor |" in text
    assert f"| vggss_box | av | 2 | `{EH[:12]}` | box | **iou_mean** | 0.4 | [0.35, 0.45] | 0.2 |" in text
    assert f"| nyuv2_depth | v | 2 | `{EH[:12]}` | grid | — | _not_applicable_ (eligibility: missing pos) | — | — |" in text
    assert re.search(r"\| k400 \| v \| 2 \| `[0-9a-f]{12}` \| clf \| \*\*acc\*\* \| 0\.55 ± 0\.07071 \(n_seeds 2\) \|", text)


def test_lm_eval_table_separates_eval_hashes_readings_and_nonstandard():
    a = make_result(model="m", protocol_hash=PH, protocol_version=1)
    b = make_result(model="m", protocol_hash=PH2, protocol_version=2)
    c = make_result(model="m", protocol_hash=PH2, protocol_version=2, standard=False, probe_hash="9" * 64)
    d = make_result(model="m", protocol_hash=PH2, protocol_version=2, reading="av")
    rows = lm_eval_tables(_records(a, b, c, d), aliases={})["m"]
    keys = {(r["benchmark"], r["reading"], r["version"], r["eval_hash"]) for r in rows}
    assert keys == {("k400", "v", 1, hashing.eval_hash(PH, QH)[:12]), ("k400", "v", 2, hashing.eval_hash(PH2, QH)[:12]),
                    ("k400 (nonstandard)", "v", 2, hashing.eval_hash(PH2, "9" * 64)[:12]), ("k400", "av", 2, hashing.eval_hash(PH2, QH)[:12])}
    # within one (benchmark, reading): the newest protocol_version first
    vs = [r["version"] for r in rows if r["benchmark"] == "k400" and r["reading"] == "v" and r["metric"] == "acc"]
    assert vs == [2, 1]


def test_report_build_prints_view_1_before_the_columns(tmp_path):
    runs = tmp_path / "runs"
    write_result(runs, make_result(model="m"))
    rep = report([str(runs / "*")], write=False)
    i1 = rep.text.index("## m")
    i2 = rep.text.index("# Comparability columns")
    assert i1 < i2 and f"| k400 | v | 2 | `{EH[:12]}` | clf | **acc** | 0.5 | [0.45, 0.55] | 0.1 |" in rep.text
    assert rep.to_json()["lm_eval"]["m"][0]["metric"] == "acc"


# --------------------------------------------------------------------------- results files


def test_write_invocation_without_a_benchmark_object(tmp_path):
    recs = [make_result(model="m", seed=s, acc=a) for s, a in ((1, 0.6), (0, 0.5))]
    recs[0]["metrics"]["acc"] = float("nan")  # strict JSON: non-finite → null
    now = dt.datetime(2026, 10, 8, 14, 0, 0, 123456, tzinfo=dt.timezone.utc)
    paths = store.write_invocation(recs, results_root=tmp_path, result_paths=["/a", "/b"], samples=[{"clip_id": "c", "row": 0}], now=now)
    assert paths["results_file"] == tmp_path / "m" / "results_2026-10-08T14-00-00.123456.json"
    assert paths["samples_file"] == tmp_path / "m" / "samples_k400_v_2026-10-08T14-00-00.123456.jsonl"
    doc = json.loads(paths["results_file"].read_text())
    assert doc["results_version"] == store.RESULTS_FILE_VERSION == 2
    assert set(doc["results"]) == {"k400:v"}
    r = doc["results"]["k400:v"]
    assert r["acc"] == 0.5 and r["acc_floor"] == 0.1 and r["n_seeds"] == 2 and r["headline"] == "acc"  # NaN ignored in the mean
    cfg = doc["configs"]["k400:v"]
    assert (cfg["benchmark"], cfg["reading"], cfg["eval_hash"]) == ("k400", "v", EH)
    assert cfg["protocol_hash"] == PH and cfg["unit_results"] == ["/a", "/b"]
    assert cfg["bridge"] == recs[0]["bridge"] and cfg["describe_sha256"] == recs[0]["bridge"]["describe_sha256"]
    assert doc["versions"] == {"k400:v": 2} and doc["higher_is_better"]["k400:v"] == {"acc": True}
    assert doc["config"]["model"] == "m" and doc["config"]["bridge"] == recs[0]["bridge"]
    assert doc["config"]["probe"]["seeds"] == [0, 1] and doc["config"]["probe"]["standard"] is True
    assert doc["env"]["host"] == "h" and doc["yavih_version"] and doc["date"].endswith("Z")
    with pytest.raises(ValueError):
        store.write_invocation([make_result(model="a"), make_result(model="b")], results_root=tmp_path)
    with pytest.raises(ValueError):
        store.write_invocation([make_result(reading="a"), make_result(reading="v")], results_root=tmp_path)


def test_two_readings_of_one_model_land_in_one_directory_under_two_keys(tmp_path):
    """One directory per model name; entries ``<benchmark>:<reading>``; per-entry bridge."""
    t0 = dt.datetime(2026, 10, 8, 10, tzinfo=dt.timezone.utc)
    a = make_result(model="avmodel", reading="a", describe_extra={"streams": ["audio"], "layer": 6}, acc=0.3)
    v = make_result(model="avmodel", reading="v", acc=0.6)
    pa = store.write_invocation([a], results_root=tmp_path, now=t0)["results_file"]
    pv = store.write_invocation([v], results_root=tmp_path, now=t0 + dt.timedelta(minutes=1))["results_file"]
    assert pa.parent == pv.parent == tmp_path / "avmodel"
    (merged,) = store.merge_results(tmp_path, now=t0 + dt.timedelta(hours=1))
    doc = json.loads(merged.read_text())
    assert set(doc["results"]) == {"k400:a", "k400:v"}
    assert doc["configs"]["k400:a"]["bridge"]["describe"]["layer"] == 6 and doc["configs"]["k400:v"]["bridge"]["describe"]["layer"] == 12
    assert doc["config"]["bridge"] is None  # two bridges: the file-level block would be ambiguous
    assert doc["config"]["model"] == "avmodel"


def test_bridge_configuration_never_changes_the_results_directory_or_key(tmp_path):
    t0 = dt.datetime(2026, 10, 8, 10, tzinfo=dt.timezone.utc)
    p1 = store.write_invocation([make_result(describe_extra={"layer": 3})], results_root=tmp_path, now=t0)["results_file"]
    p2 = store.write_invocation([make_result(describe_extra={"dtype": "float32", "weights_sha256": None})], results_root=tmp_path,
                                now=t0 + dt.timedelta(minutes=1))["results_file"]
    assert p1.parent == p2.parent == tmp_path / "vjepa"
    assert set(json.loads(p1.read_text())["results"]) == set(json.loads(p2.read_text())["results"]) == {"k400:v"}


def test_metric_info_conventions():
    info = store.metric_info(["d1", "absrel", "rmse", "n_eval", "seg_acc_boundary"], floors=["d1_floor", "seg_acc_boundary_floor"])
    assert info["d1"]["higher_is_better"] is True and info["absrel"]["higher_is_better"] is False and info["rmse"]["higher_is_better"] is False
    assert info["n_eval"]["higher_is_better"] is None and info["n_eval"]["kind"] == "disclosure"
    assert info["d1"]["floor"] == "d1_floor" and info["seg_acc_boundary"]["floor"] == "seg_acc_boundary_floor" and info["absrel"]["floor"] is None


def test_merge_results_newest_wins_and_other_eval_hashes_are_kept(tmp_path):
    t0 = dt.datetime(2026, 10, 8, 10, tzinfo=dt.timezone.utc)
    old = make_result(model="m", acc=0.4, protocol_hash=PH, protocol_version=1)
    new = make_result(model="m", acc=0.5, protocol_hash=PH, protocol_version=1, describe_extra={"layer": 18})  # another configuration
    v2 = make_result(model="m", acc=0.7, protocol_hash=PH2, protocol_version=2)
    ave = make_result(bench="ave", readout="seq", model="m", metrics={"seg_acc": 0.3, "n_eval": 316}, floors={"seg_acc_floor": 0.1})
    for i, rec in enumerate((old, new, v2, ave)):
        store.write_invocation([rec], results_root=tmp_path, now=t0 + dt.timedelta(minutes=i))
    store.write_invocation([make_result(model="solo")], results_root=tmp_path, now=t0)
    written = store.merge_results(tmp_path, now=t0 + dt.timedelta(hours=1))
    assert written == [tmp_path / "m" / "results_2026-10-08T11-00-00.000000.json"]  # solo has nothing to merge
    doc = json.loads(written[0].read_text())
    old_key = f"k400:v@{hashing.eval_hash(PH, QH)[:12]}"
    assert set(doc["results"]) == {"k400:v", old_key, "ave:v"}
    assert doc["results"]["k400:v"]["acc"] == 0.7 and doc["versions"]["k400:v"] == 2  # the newest protocol_version under the plain key
    # the newest entry of the older eval_hash wins across configurations (one entry, never averaged)
    assert doc["results"][old_key]["acc"] == 0.5 and doc["configs"][old_key]["bridge"]["describe"]["layer"] == 18
    assert doc["configs"][old_key]["eval_hash"] == hashing.eval_hash(PH, QH) and doc["configs"][old_key]["reading"] == "v"
    assert doc["results"]["ave:v"]["seg_acc"] == 0.3 and len(doc["config"]["merged_from"]) == 3  # the superseded first file contributes nothing


@pytest.mark.parametrize("standard_first", [True, False])
def test_merge_results_a_standard_entry_holds_the_key_and_provenance_survives(tmp_path, standard_first):
    """A newer non-standard (fast) run of the same protocol must not take ``<benchmark>:<reading>``, and
    each entry keeps its own probe config / seeds / standard flag in ``configs[key].probe``."""
    t0 = dt.datetime(2026, 10, 8, 10, tzinfo=dt.timezone.utc)
    std = make_result(model="m", acc=0.8, probe_hash="p" * 64)
    fast = make_result(model="m", acc=0.4, probe_hash="f" * 64, standard=False)
    fast["probe"]["config"] = {"fast": True}
    order = (std, fast) if standard_first else (fast, std)
    for i, rec in enumerate(order):
        store.write_invocation([rec], results_root=tmp_path, now=t0 + dt.timedelta(hours=i))
    doc = json.loads(store.merge_results(tmp_path, now=t0 + dt.timedelta(hours=2))[0].read_text())
    assert doc["results"]["k400:v"]["acc"] == 0.8 and doc["configs"]["k400:v"]["probe"]["standard"] is True
    other = f"k400:v@{hashing.eval_hash(PH, 'f' * 64)[:12]}"
    assert doc["results"][other]["acc"] == 0.4 and doc["configs"][other]["probe"] == {"config": {"fast": True}, "seeds": [0], "standard": False}
    assert doc["config"]["probe"]["standard"] is True  # the file-level block describes the main-key entries


def test_merge_skips_a_results_file_of_another_version(tmp_path, capsys):
    """A merge reads files of ``RESULTS_FILE_VERSION`` only: an older or newer one is skipped with a notice."""
    t0 = dt.datetime(2026, 10, 8, 10, tzinfo=dt.timezone.utc)
    cur = make_result(model="vjepa", acc=0.6)
    store.write_invocation([make_result(model="vjepa", acc=0.5)], results_root=tmp_path, now=t0)
    path = store.write_invocation([cur], results_root=tmp_path, now=t0 + dt.timedelta(hours=1))["results_file"]
    doc = json.loads(Path(path).read_text())
    assert store.upgrade_results_file(doc) == doc
    for version in (store.RESULTS_FILE_VERSION - 1, store.RESULTS_FILE_VERSION + 1, None):
        with pytest.raises(ValueError):
            store.upgrade_results_file({**doc, "results_version": version})
    (tmp_path / "vjepa" / "results_2026-10-07T10-00-00.000000.json").write_text(
        json.dumps({**doc, "results_version": store.RESULTS_FILE_VERSION - 1, "results": {"k400:v": {"acc": 0.1}}}))
    (merged,) = store.merge_results(tmp_path, now=t0 + dt.timedelta(hours=2))
    assert json.loads(merged.read_text())["results"]["k400:v"]["acc"] == 0.6
    assert "older than this yavih reads" in capsys.readouterr().err


def test_report_merge_results_flag(tmp_path):
    runs = tmp_path / "runs"
    write_result(runs, make_result(model="m"))
    root = tmp_path / "results"
    t0 = dt.datetime(2026, 10, 8, 10, tzinfo=dt.timezone.utc)
    for i in range(2):
        store.write_invocation([make_result(model="m", acc=0.1 * (i + 1))], results_root=root, now=t0 + dt.timedelta(minutes=i))
    rep = report([str(runs / "*")], merge_results=True, write=False)  # default root: `results` next to the runs root
    assert len(rep.merged) == 1 and Path(rep.merged[0]).parent == root / "m"
    assert json.loads(Path(rep.merged[0]).read_text())["results"]["k400:v"]["acc"] == pytest.approx(0.2)
