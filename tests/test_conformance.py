"""Conformance fixtures (``yavih.conformance``): a stable generator, a record/check round trip per readout, drift
and stale detection, the per-metric tolerance, the lint check and the CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from yavih import cli
from yavih import conformance as C
from yavih import testing as T
from yavih.benchmark import Benchmark, packaged_benchmarks
from yavih.benchmarks import packaged_folder


def _bench(tmp_path: Path, readout: str = "clf", **kw) -> Benchmark:
    folder = T.make_synthetic_benchmark(tmp_path, readout, n_train=12, n_eval=8, media="none", **kw)
    return Benchmark.from_dir(folder)


def _edit_yaml(folder: Path, mutate) -> None:
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    mutate(y)
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")


# ----------------------------------------------------------------------------- generator


def test_stream_is_stable_and_uniform():
    """sha256 counter mode: pinned values, independent of numpy's RNG, in [0, 1)."""
    u = C.stream("tag", 1000)
    assert u.shape == (1000,) and u.dtype == np.float64 and (u >= 0).all() and (u < 1).all()
    assert abs(float(u.mean()) - 0.5) < 0.05
    assert np.array_equal(C.stream("tag", 1000), u) and not np.array_equal(C.stream("other", 1000), u)
    assert np.array_equal(C.stream("tag", 10), u[:10])  # a prefix is a prefix
    # the first value is a fixed function of sha256: pin it (any platform, any numpy)
    first = int.from_bytes(hashlib.sha256(b"tag:0").digest()[:8], "big") >> 11
    assert u[0] == first / float(1 << 53)
    assert C.stream("x", 0).shape == (0,)


def test_select_rows_is_a_stable_uniform_draw():
    rows = C.select_rows("m" * 64, "eval", 1000, 50)
    assert rows.dtype == np.int64 and len(rows) == 50 and len(set(rows.tolist())) == 50
    assert np.array_equal(rows, np.sort(rows)) and np.array_equal(C.select_rows("m" * 64, "eval", 1000, 50), rows)
    assert not np.array_equal(C.select_rows("n" * 64, "eval", 1000, 50), rows)
    assert not np.array_equal(C.select_rows("m" * 64, "eval", 1000, 50, salt="train"), rows)
    assert np.array_equal(C.select_rows("m" * 64, "eval", 20, 50), np.arange(20))  # the whole split when it is small
    assert 300 < int(rows.mean()) < 700


def test_derangement_has_no_fixed_point():
    for n in (2, 3, 7, 64, 128):
        p = C._derangement(n, "t")
        assert sorted(p.tolist()) == list(range(n)) and not (p == np.arange(n)).any()
        assert np.array_equal(C._derangement(n, "t"), p)


def test_the_documented_generator_spec_reproduces_the_stream():
    """A re-implementation written only from docs/conformance.md "Generator specification" (hashlib and Python floats;
    numpy only for the documented float32 casts) gives the same stream, arrays, row draw and derangement as
    ``yavih.conformance``."""

    def spec_stream(tag: str, n: int) -> list[float]:
        out: list[float] = []
        i = 0
        while len(out) < n:
            digest = hashlib.sha256(f"{tag}:{i}".encode()).digest()
            for k in range(4):
                out.append((int.from_bytes(digest[8 * k: 8 * k + 8], "big") >> 11) * 2.0 ** -53)
            i += 1
        return out[:n]

    def spec_rows(manifest_hash: str, split: str, role: str, n_rows: int, n: int) -> list[int]:
        if n >= n_rows:
            return list(range(n_rows))
        key = {r: hashlib.sha256(f"{manifest_hash}:conformance:{split}:{role}:{r}".encode()).digest() for r in range(n_rows)}
        return sorted(sorted(range(n_rows), key=key.__getitem__)[:n])

    def spec_derangement(tag: str, n: int) -> list[int]:
        if n < 2:
            return list(range(n))
        u = spec_stream(tag, n)
        perm = list(range(n))
        for i in range(n - 1, 0, -1):
            j = int(u[i] * (i + 1))
            perm[i], perm[j] = perm[j], perm[i]
        for k in [k for k in range(n) if perm[k] == k]:
            perm[k], perm[(k + 1) % n] = perm[(k + 1) % n], perm[k]
        if any(perm[i] == i for i in range(n)):
            perm = [(i - 1) % n for i in range(n)]
        return perm

    mh = hashlib.sha256(b"a manifest").hexdigest()
    T = f"{mh}:conformance:{C.GENERATOR_VERSION}"
    for tag, n in ((f"{T}:outputs", 1027), (f"{T}:fold3:outputs", 5), (f"{T}:train:depth", 64), (f"{T}:eval:holes", 3),
                   (f"{T}:floor:outputs", 1)):
        assert C.stream(tag, n).tolist() == spec_stream(tag, n), tag
    # an array takes the stream in row-major order; logits and boxes as tabled
    assert C._logits(f"{T}:outputs", (3, 7)).ravel().tolist() == [float(np.float32((u - 0.5) * 6)) for u in spec_stream(f"{T}:outputs", 21)]
    u = spec_stream(f"{T}:outputs", 8)
    assert C._boxes(f"{T}:outputs", 2).tolist() == [
        [float(np.float32(a + b * u[4 * r + c])) for c, (a, b) in enumerate(((0.2, 0.6), (0.2, 0.6), (0.1, 0.3), (0.1, 0.3)))] for r in range(2)]
    for split, role, n_rows, n in (("test", "eval", 500, 64), ("train", "train", 300, 256), ("val", "floor", 10, 64)):
        assert C.select_rows(mh, split, n_rows, n, salt=role).tolist() == spec_rows(mh, split, role, n_rows, n)
    for n in (2, 3, 17, 128):
        assert C._derangement(n, f"{T}:eval").tolist() == spec_derangement(f"{T}:eval:derangement", n)


# ----------------------------------------------------------------------------- compute / record / check


def _bump_to_v2(folder: Path) -> None:
    _edit_yaml(folder, lambda y: (y.__setitem__("protocol_version", 2), y["changelog"].append(
        {"version": 2, "date": "2026-10-20", "reason": "bump", "invalidates": True})))


def _shift_acc(monkeypatch, delta: float = 0.05) -> None:
    """Move the registered ``acc`` metric by ``delta``: a scoring change beyond the default tolerance."""
    import dataclasses

    from yavih.metrics import registry

    spec = registry.REGISTRY["acc"]
    original = spec.fn
    shifted = dataclasses.replace(spec, fn=lambda outputs, targets, protocol, split_meta: float(original(outputs, targets, protocol, split_meta)) + delta)
    monkeypatch.setitem(registry.REGISTRY, "acc", shifted)


@pytest.mark.parametrize("readout", ["clf", "seq", "grid", "box"])
def test_record_and_check_round_trip(tmp_path, readout):
    b = _bench(tmp_path, readout)
    res = C.record(b)
    rec = res.record
    assert not res.compared and res.moved == [] and res.reason_dropped is None  # a first record: nothing to compare
    path = C.expected_path(b)
    assert path.is_file() and json.loads(path.read_text()) == rec
    assert rec["schema"] == C.CONFORMANCE_SCHEMA and rec["protocol_version"] == b.spec.protocol_version
    assert rec["generator"] == C.GENERATOR_VERSION and rec["benchmark"] == b.name
    values = rec["values"]["*"]
    assert set(values) == {"metrics", "floors", "controls", "ci95"}
    assert set(values["metrics"]) == set(b.spec.metrics.list) and set(values["floors"]) == set(b.spec.metrics.floors)
    assert rec["tolerance"] == {n: C.DEFAULT_TOLERANCE for n in (*b.spec.metrics.list, *b.spec.metrics.floors, *b.spec.metrics.controls)}
    assert rec["recorded_with"]["protocol_hash"] == b.protocol_hash and rec["recorded_with"]["folder_source"] == b.folder_source
    rep = C.check(Benchmark.from_dir(b.folder))
    assert rep.ok and not rep.skipped and rep.compared == len(values["metrics"]) + len(values["floors"]) + len(values["controls"]) and rep.drifted == []
    # the computation is deterministic across processes: compute twice, identical values
    assert C.compute(b)["values"] == rec["values"]


def test_missing_and_stale_fixture(tmp_path):
    b = _bench(tmp_path)
    rep = C.check(b)
    assert not rep.ok and [f.code for f in rep.errors] == [C.E_MISSING] and "--record" in rep.errors[0].message
    C.record(b)
    _bump_to_v2(b.folder)
    b2 = Benchmark.from_dir(b.folder)
    rep = C.check(b2)
    assert not rep.ok and [f.code for f in rep.errors] == [C.E_STALE] and rep.expected_version == 1 and rep.protocol_version == 2
    C.record(b2)  # a new version records without a question
    assert C.check(b2).ok
    # a malformed file is a schema error, never a crash
    C.expected_path(b2).write_text("{not json", encoding="utf-8")
    assert [f.code for f in C.check(b2).errors] == [C.E_SCHEMA]
    C.expected_path(b2).write_text(json.dumps({"schema": "other"}), encoding="utf-8")
    assert [f.code for f in C.check(b2).errors] == [C.E_SCHEMA]


def test_drift_is_detected_and_resolved_by_a_version_bump(tmp_path, monkeypatch):
    """A scoring change that moves a number beyond tolerance at the same protocol_version is E_CONFORMANCE_DRIFT;
    bumping the version (or --same-version --reason) lets it be recorded. The refusal applies from the second
    changelog entry on (a released folder)."""
    b = _bench(tmp_path)
    _bump_to_v2(b.folder)
    b = Benchmark.from_dir(b.folder)
    C.record(b)
    _shift_acc(monkeypatch)
    rep = C.check(Benchmark.from_dir(b.folder))
    assert not rep.ok and [f.code for f in rep.errors] == [C.E_DRIFT]
    assert rep.drifted and rep.drifted[0]["metric"] == "acc" and abs(rep.drifted[0]["got"] - rep.drifted[0]["expected"] - 0.05) < 1e-9
    with pytest.raises(C.ConformanceError) as ei:
        C.record(Benchmark.from_dir(b.folder))
    assert ei.value.code == C.E_DRIFT
    with pytest.raises(C.ConformanceError):
        C.record(Benchmark.from_dir(b.folder), same_version=True)  # needs a reason
    res = C.record(Benchmark.from_dir(b.folder), same_version=True, reason="fixture bug")
    assert res.record["same_version_reason"] == "fixture bug" and C.check(Benchmark.from_dir(b.folder)).ok
    assert res.compared and [d["metric"] for d in res.moved] == ["acc"] and res.reason_dropped is None
    # the reason explains values that moved: a same-version re-record that changes nothing stores none, and says why
    res = C.record(Benchmark.from_dir(b.folder), same_version=True, reason="nothing moved")
    assert "same_version_reason" not in res.record and "same_version_reason" not in C.read_expected(Benchmark.from_dir(b.folder))
    assert res.compared and res.moved == [] and res.reason_dropped == "unchanged"
    res = C.record(Benchmark.from_dir(b.folder), same_version=True, reason="forced", force=True)  # nothing compared: not stored
    assert "same_version_reason" not in res.record and not res.compared and res.reason_dropped == "force"
    assert C.record(Benchmark.from_dir(b.folder), force=True).reason_dropped is None  # no reason given: nothing dropped
    _edit_yaml(b.folder, lambda y: (y.__setitem__("protocol_version", 3), y["changelog"].append(  # another version: not compared either
        {"version": 3, "date": "2026-10-21", "reason": "bump", "invalidates": True})))
    res = C.record(Benchmark.from_dir(b.folder), same_version=True, reason="x")
    assert not res.compared and res.reason_dropped == "other_version"


def test_unreleased_folder_record_overwrites_without_drift_refusal(tmp_path, monkeypatch):
    """While the changelog has exactly one entry (a new, unreleased folder), ``record`` overwrites a differing record
    at the same protocol_version without asking; the check then passes against the new numbers."""
    b = _bench(tmp_path)
    assert len(b.spec.changelog) == 1
    first = C.record(b).record
    _shift_acc(monkeypatch)
    b = Benchmark.from_dir(b.folder)
    assert [f.code for f in C.check(b).errors] == [C.E_DRIFT]
    res = C.record(b)  # no --same-version, no --force
    rec = res.record
    assert abs(rec["values"]["*"]["metrics"]["acc"] - first["values"]["*"]["metrics"]["acc"] - 0.05) < 1e-9
    assert "same_version_reason" not in rec and C.check(Benchmark.from_dir(b.folder)).ok
    assert not res.compared and res.reason_dropped is None
    res = C.record(b, same_version=True, reason="why")  # a reason on an unreleased folder: dropped, and it says why
    assert "same_version_reason" not in res.record and not res.compared and res.reason_dropped == "unreleased"


def test_scoring_refusal_is_wrapped_as_unsupported(tmp_path, monkeypatch):
    """A ProbeRefused from the scoring path that is not about an external package is E_CONFORMANCE_UNSUPPORTED, with the
    refusal's own code and message kept in the text."""
    from yavih.readouts import base

    b = _bench(tmp_path)

    def refuse(*a, **k):
        raise base.ProbeRefused("E_LABELS", "labels out of range; more detail")

    monkeypatch.setattr(base, "evaluate_metrics", refuse)
    with pytest.raises(C.ConformanceError) as ei:
        C.compute(b)
    assert ei.value.code == C.E_UNSUPPORTED and "E_LABELS: labels out of range" in str(ei.value)


def test_tolerance_from_the_yaml_and_the_comparison_rules(tmp_path):
    b = _bench(tmp_path)
    _edit_yaml(b.folder, lambda y: y["metrics"].__setitem__("tolerance", {"acc": 0.2}))
    b = Benchmark.from_dir(b.folder)
    assert b.spec.metrics.tolerance == {"acc": 0.2} and C.tolerances(b)["acc"] == 0.2
    assert C.tolerances(b)["n_eval"] == C.DEFAULT_TOLERANCE
    # tolerance is not hashed
    _edit_yaml(b.folder, lambda y: y["metrics"].__setitem__("tolerance", {"acc": 0.3}))
    assert Benchmark.from_dir(b.folder).protocol_hash == b.protocol_hash
    with pytest.raises(Exception, match="tolerance"):
        _edit_yaml(b.folder, lambda y: y["metrics"].__setitem__("tolerance", {"nope": 0.1}))
        Benchmark.from_dir(b.folder)
    assert C._within(0.5, 0.5004, 0.001) and not C._within(0.5, 0.502, 0.001)
    assert C._within(None, None, 0.1) and not C._within(None, 0.0, 0.1)
    assert C._within([1.0, 2.0], [1.0005, 2.0], 0.001) and not C._within([1.0], [1.0, 2.0], 0.001)
    assert C._within(True, True, 0.0) and not C._within(True, False, 1.0)
    assert C._within(float("nan"), float("nan"), 0.0) and not C._within(float("nan"), 0.0, 1.0)


def test_find_prose_free_views_and_unsupported_readout(tmp_path):
    b = _bench(tmp_path)

    class Fake:
        spec = type("S", (), {"readout": "nope", "external": []})()
        name = "fake"
        manifest_hash = "m" * 64

    with pytest.raises(C.ConformanceError) as ei:
        C.compute(Fake())
    assert ei.value.code == C.E_UNSUPPORTED
    assert C.compute(b)["population"]["eval"] == 8  # the whole eval split when it is smaller than the population


# ----------------------------------------------------------------------------- packaged folders


@pytest.mark.parametrize("name", packaged_benchmarks())
def test_packaged_folder_fixture_is_current(name):
    """Every packaged folder ships ``conformance/expected.json`` at its protocol_version, and its scoring path still
    reproduces it (skipped, with the reason, when the pinned external package is absent here)."""
    b = Benchmark.load(name)
    rec = C.read_expected(b)
    assert rec is not None and rec["protocol_version"] == b.spec.protocol_version and rec["generator"] == C.GENERATOR_VERSION
    assert rec["recorded_with"]["protocol_hash"] == b.protocol_hash, f"{name}: the fixture was recorded under another hash; re-record"
    rep = C.check(b)
    if rep.skipped:
        pytest.skip(rep.warnings[0].message)
    assert rep.ok, [f.message for f in rep.errors]
    assert rep.compared > 0


def test_population_keeps_one_row_above_tolerance():
    for readout, pop in C.POPULATION.items():
        n = pop["eval"] if readout != "clf" else max(pop["eval"], pop["fold"] * 5)
        assert 1.0 / n > C.DEFAULT_TOLERANCE, readout


# ----------------------------------------------------------------------------- lint and CLI


def test_lint_runs_the_conformance_check(tmp_path):
    from yavih.lint import lint_folder

    b = _bench(tmp_path)
    rep = lint_folder(b.folder, checks=["conformance"])
    assert [f.code for f in rep.errors] == [C.E_MISSING]
    C.record(b)
    assert lint_folder(b.folder, checks=["conformance"]).ok


def test_cli_conformance_and_hash(tmp_path, capsys, monkeypatch):
    b = _bench(tmp_path)

    def run(argv):
        rc = cli.main(["--json", *argv])
        out = capsys.readouterr().out
        return rc, json.loads(out)

    rc, env = run(["bench", "conformance", str(b.folder)])
    assert rc == 2 and env["errors"][0]["code"] == C.E_MISSING
    rc, env = run(["bench", "conformance", str(b.folder), "--record"])
    assert rc == 0 and env["data"][0]["recorded"] is True and env["data"][0]["values"] > 0
    rc, env = run(["bench", "conformance", str(b.folder)])
    assert rc == 0 and env["data"][0]["ok"] is True and env["data"][0]["compared"] > 0
    # a flag that would be silently ignored is a usage error
    for argv in (["--same-version"], ["--reason", "x"], ["--force"], ["--record", "--reason", "x"], ["--record", "--force", "--reason", "x"]):
        rc, env = run(["bench", "conformance", str(b.folder), *argv])
        assert rc == 1 and env["errors"][0]["code"] == "E_USAGE", argv
    # --same-version --reason on an unreleased folder (one changelog entry): nothing is compared, the reason is not
    # stored and the output says which case applied (never "no value moved")
    rc, env = run(["bench", "conformance", str(b.folder), "--record", "--same-version", "--reason", "x"])
    assert rc == 0 and env["data"][0]["recorded"] is True and env["data"][0]["same_version_reason"] is None
    assert env["data"][0]["compared"] is False and env["data"][0]["reason_dropped"] == "unreleased"
    assert "same_version_reason" not in json.loads((b.folder / "conformance" / "expected.json").read_text(encoding="utf-8"))
    rc = cli.main(["bench", "conformance", str(b.folder), "--record", "--same-version", "--reason", "x"])
    out = capsys.readouterr().out
    assert rc == 0 and "--reason not stored: an unreleased folder" in out and "no value moved" not in out
    rc = cli.main(["bench", "conformance", str(b.folder), "--record", "--same-version", "--reason", "x", "--force"])
    assert rc == 0 and "--reason not stored: --force records without comparison" in capsys.readouterr().out
    # a released folder (two changelog entries): the record is compared, and an unmoved value is the third case
    _bump_to_v2(b.folder)
    rc, env = run(["bench", "conformance", str(b.folder), "--record"])
    assert rc == 0 and env["data"][0]["compared"] is False and env["data"][0]["reason_dropped"] is None
    rc, env = run(["bench", "conformance", str(b.folder), "--record", "--same-version", "--reason", "x"])
    assert rc == 0 and env["data"][0]["compared"] is True and env["data"][0]["moved"] == 0 and env["data"][0]["reason_dropped"] == "unchanged"
    rc = cli.main(["bench", "conformance", str(b.folder), "--record", "--same-version", "--reason", "x"])
    assert rc == 0 and "--reason not stored: no value moved at this version" in capsys.readouterr().out
    rc = cli.main(["bench", "conformance", str(b.folder), "--record"])
    assert rc == 0 and "--reason not stored" not in capsys.readouterr().out
    b = Benchmark.from_dir(b.folder)  # the bumped folder's hashes
    rc, env = run(["bench", "hash", str(b.folder), "--explain"])
    assert rc == 0 and env["data"]["protocol_hash"] == b.protocol_hash and env["data"]["folder_source"] == b.folder_source
    ex = env["data"]["explain"]
    for side in ("input", "scoring"):
        for c in ex[side]["components"].values():
            assert hashlib.sha256(c["canon"].encode("utf-8")).hexdigest() == c["digest"]
    assert ex["protocol_hash"] == b.protocol_hash


def test_scaffold_records_a_fixture(tmp_path, monkeypatch):
    from yavih.scaffold import new_benchmark

    monkeypatch.chdir(tmp_path)
    folder = new_benchmark("scaffolded", scorer="clf", dir=tmp_path)
    b = Benchmark.from_dir(folder)
    assert C.read_expected(b) is not None and C.check(b).ok
    from yavih.scaffold import scaffold_conformance

    assert scaffold_conformance(folder) == {"state": "recorded", "copied_from": None, "packages": []}


def test_scaffold_copies_the_like_fixture_when_the_package_is_absent(tmp_path, monkeypatch, capsys):
    """``bench new --like nyuv2_depth`` where Probe3D's ``evals`` is absent: recording is impossible (W_CONFORMANCE_SKIPPED),
    so the scaffold copies nyuv2_depth's fixture with the new folder's identity fields (``recorded_with.copied_from``);
    with the package installed, ``check`` on the copy passes: the copied values are what recording produces here."""
    from yavih import external
    from yavih.scaffold import scaffold_conformance

    real = external.require

    def absent(name, pip, pin):
        if name == "evals":
            raise external.ExternalMissing(name, "E_EXTERNAL_MISSING: evals is not installed (test)")
        return real(name, pip, pin)

    with monkeypatch.context() as m:
        m.setattr(external, "require", absent)
        assert cli.main(["--json", "bench", "new", "new_depth", "--like", "nyuv2_depth", "--dir", str(tmp_path)]) == 0
        env = json.loads(capsys.readouterr().out)
        # the text form says what happened and how to record it here
        assert cli.main(["bench", "new", "new_depth2", "--like", "nyuv2_depth", "--dir", str(tmp_path)]) == 0
        assert "  conformance: copied from nyuv2_depth (record it here once evals is installed: `yavih bench conformance" in capsys.readouterr().out
    folder = tmp_path / "new_depth"
    assert env["data"]["conformance"] == {"state": "copied", "copied_from": "nyuv2_depth", "packages": ["evals"]}
    assert scaffold_conformance(folder) == env["data"]["conformance"]
    rec = json.loads((folder / "conformance" / "expected.json").read_text(encoding="utf-8"))
    like = json.loads((packaged_folder("nyuv2_depth") / "conformance" / "expected.json").read_text(encoding="utf-8"))
    b = Benchmark.from_dir(folder)
    assert rec["benchmark"] == "new_depth" and rec["protocol_version"] == 1 and rec["schema"] == C.CONFORMANCE_SCHEMA
    assert rec["generator"] == like["generator"] and rec["values"] == like["values"] and rec["population"] == like["population"]
    assert rec["tolerance"] == C.tolerances(b)
    rw = rec["recorded_with"]
    assert rw["copied_from"] == "nyuv2_depth" and rw["externals"] == like["recorded_with"]["externals"]
    assert rw["protocol_hash"] == b.protocol_hash and rw["scoring_hash"] == b.scoring_hash and rw["folder_source"] == b.folder_source
    assert C.read_expected(b) == rec  # the identity fields read as this folder's record
    rep = C.check(b)
    if rep.skipped:
        pytest.skip("Probe3D's evals is not installed here: the copy is written, the proof needs the package")
    assert rep.ok and not rep.skipped and rep.compared > 0 and not rep.drifted, [f.message for f in rep.findings]
    fresh = C.record(b, force=True).record
    assert {k: v for k, v in fresh["recorded_with"].items() if k != "commit"} == {k: v for k, v in rw.items() if k not in ("copied_from", "commit")}
    assert {k: v for k, v in fresh.items() if k != "recorded_with"} == {k: v for k, v in rec.items() if k != "recorded_with"}


def test_population_follows_the_scorer_family_or_its_own_builder():
    """A scorer with ``family="clf"`` is checked on clf's population; a scorer declaring its own
    ``conformance_population`` uses that; one with neither has no fixture (E_CONFORMANCE_UNSUPPORTED)."""
    from yavih.readouts.registry import Scorer

    fam = Scorer(name="fam", version=1, module="m", kind="stateless", label_kinds=frozenset({"single"}), family="clf")
    assert C.population_builder(fam) is C._population_clf
    own = lambda ctx, tag: {}  # noqa: E731
    mine = Scorer(name="mine", version=1, module="m", kind="direct", label_kinds=frozenset({"none"}), conformance_population=own)
    assert C.population_builder(mine) is own
    none = Scorer(name="none_at_all", version=1, module="m", kind="direct", label_kinds=frozenset({"none"}))
    assert C.population_builder(none) is None


def test_fixture_records_the_scorer(tmp_path):
    b = _bench(tmp_path)
    rec = C.record(b).record
    assert rec["recorded_with"]["scorer"] == {"name": b.spec.readout, "version": 1, "kind": "probe"}
    assert C.read_expected(b)["recorded_with"]["scorer"] == rec["recorded_with"]["scorer"]
