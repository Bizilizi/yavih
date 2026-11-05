"""The catalog (``yavih.catalog``): the index generated from git history, revision choice by ``core:``, ``yavih bench
fetch`` / ``update`` from a checkout, an archive or ``file://`` URLs, the lock, tree-id verification and the provenance
it gives results. Everything runs offline on a throwaway git repository."""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

from yavih import catalog
from yavih.catalog import CatalogError
from yavih.testing import make_synthetic_benchmark

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

REPO = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> str:
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.org", "GIT_AUTHOR_DATE": "2026-11-05T12:00:00Z",
           "GIT_COMMITTER_DATE": "2026-11-05T12:00:00Z", "PATH": "/usr/bin:/bin", "HOME": str(repo)}
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


def _edit_yaml(folder: Path, **changes) -> None:
    p = folder / "benchmark.yaml"
    d = yaml.safe_load(p.read_text())
    d.update(changes)
    p.write_text(yaml.safe_dump(d, sort_keys=False))


@pytest.fixture
def catrepo(tmp_path):
    """A git repository holding ``benchmarks/catx`` at three commits: c1 (core >=1.1,<2, protocol_version 1), c2
    (protocol_version 2, same core), c3 (core >=9,<10); then c4 adds the generated ``catalog/index.json``."""
    repo = tmp_path / "repo"
    (repo / "benchmarks").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    folder = make_synthetic_benchmark(tmp_path / "gen", "clf", media="none", name="catx")
    shutil.copytree(folder, repo / "benchmarks" / "catx")
    f = repo / "benchmarks" / "catx"
    (f / "tests").mkdir(exist_ok=True)
    t = f / "bridge_template.py"
    t.write_text(t.read_text().replace('"reading": "TODO",', '"reading": "v",'))
    _edit_yaml(f, core=">=1.1,<2")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c1")
    c1 = _git(repo, "rev-parse", "HEAD")
    d = yaml.safe_load((f / "benchmark.yaml").read_text())
    log = d["changelog"] + [{**d["changelog"][-1], "version": 2, "reason": "second version"}]
    _edit_yaml(f, protocol_version=2, changelog=log)
    _git(repo, "commit", "-qam", "c2")
    c2 = _git(repo, "rev-parse", "HEAD")
    (f / "README.md").write_text((f / "README.md").read_text() + "\nA note.\n")
    _git(repo, "commit", "-qam", "c2b: same core and protocol_version (not a new revision)")
    c2b = _git(repo, "rev-parse", "HEAD")
    _edit_yaml(f, core=">=9,<10")
    _git(repo, "commit", "-qam", "c3")
    c3 = _git(repo, "rev-parse", "HEAD")
    assert catalog.main(["--repo", str(repo), "--write"]) == 0
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c4: the index")
    return {"repo": repo, "c1": c1, "c2": c2, "c2b": c2b, "c3": c3, "c4": _git(repo, "rev-parse", "HEAD")}


def test_tree_id_is_gits_for_every_repository_folder():
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    for folder in sorted((REPO / "benchmarks").iterdir()):
        if not (folder / "benchmark.yaml").is_file():
            continue
        rel = f"benchmarks/{folder.name}"
        if subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", rel], capture_output=True, text=True).stdout.strip():
            continue  # edited here: git's tree is the committed one
        want = subprocess.run(["git", "-C", str(REPO), "rev-parse", f"HEAD:{rel}"], capture_output=True, text=True).stdout.strip()
        if want:
            assert catalog.tree_id(folder) == want, folder.name


def test_the_index_lists_the_newest_commit_per_core_and_protocol_version(catrepo):
    idx = json.loads((catrepo["repo"] / "catalog" / "index.json").read_text())
    assert idx["catalog_version"] == catalog.CATALOG_VERSION and idx["commit"] == catrepo["c3"]
    e = idx["benchmarks"]["catx"]
    assert [(r["commit"], r["core"], r["protocol_version"]) for r in e["revisions"]] == [
        (catrepo["c3"], ">=9,<10", 2), (catrepo["c2b"], ">=1.1,<2", 2), (catrepo["c1"], ">=1.1,<2", 1)]
    assert e["revisions"][0]["tree"] == _git(catrepo["repo"], "rev-parse", f"{catrepo['c3']}:benchmarks/catx")
    assert e["path"] == "benchmarks/catx" and e["core"] == ">=9,<10" and e["scorer"] == "clf" and e["aliases"] == ["catx_alias"]
    assert catalog.main(["--repo", str(catrepo["repo"]), "--check"]) == 0  # reproducible from the commit it names
    p = catrepo["repo"] / "catalog" / "index.json"
    p.write_text(p.read_text().replace('"clf"', '"seq"'))
    assert catalog.main(["--repo", str(catrepo["repo"]), "--check"]) == 1


@pytest.mark.parametrize(("version", "want"), [("1.2.0", "c2b"), ("9.1.0", "c3"), ("1.0.1.dev3+gabc", "c3")])
def test_the_newest_revision_the_core_satisfies_is_chosen(catrepo, version, want):
    idx = catalog.build_index(catrepo["repo"])
    rev, st = catalog.select_revision(idx["benchmarks"]["catx"], version=version)
    assert rev["commit"] == catrepo[want]
    assert st["dev"] == ("dev" in version)


def test_no_fitting_revision_prints_the_pip_line_and_never_upgrades(catrepo):
    idx = catalog.build_index(catrepo["repo"])
    with pytest.raises(CatalogError) as ei:
        catalog.select_revision(idx["benchmarks"]["catx"], version="0.9.0")
    assert ei.value.code == "E_CORE_VERSION" and ei.value.exit_code == 3
    assert 'pip install --no-deps "yavih>=9,<10"' in str(ei.value)


def test_fetch_from_a_checkout_verifies_locks_and_ignores(catrepo, tmp_path, yavih_home):
    dest = tmp_path / "proj" / "yavih-bench" / "benchmarks"
    rec = catalog.fetch("catx_alias", source=catrepo["repo"], dest_dir=dest, version="1.2.0")
    folder = dest / "catx"
    assert rec["status"] == "fetched" and rec["commit"] == catrepo["c2b"] and rec["name"] == "catx"
    assert catalog.tree_id(folder) == rec["tree"] == _git(catrepo["repo"], "rev-parse", f"{catrepo['c2b']}:benchmarks/catx")
    lock = json.loads((dest.parent / "benchmarks.lock").read_text())
    assert lock["lock_version"] == catalog.LOCK_VERSION
    assert lock["benchmarks"]["catx"]["commit"] == catrepo["c2b"] and lock["benchmarks"]["catx"]["protocol_version"] == 2
    assert "/catx/" in (dest / ".gitignore").read_text()
    assert catalog.fetch("catx", source=catrepo["repo"], dest_dir=dest, version="1.2.0")["status"] == "unchanged"
    # an edited folder is never overwritten silently
    (folder / "README.md").write_text("edited\n")
    with pytest.raises(CatalogError, match="--refresh") as ei:
        catalog.fetch("catx", source=catrepo["repo"], dest_dir=dest, version="1.2.0")
    assert ei.value.code == "E_FETCH_MODIFIED"
    assert catalog.origin(folder)["tree"] != rec["tree"]
    out = catalog.fetch("catx", source=catrepo["repo"], dest_dir=dest, version="1.2.0", refresh=True)
    assert out["status"] == "refreshed" and catalog.tree_id(folder) == rec["tree"]
    assert catalog.origin(folder) == {"repo": catalog.DEFAULT_REPO, "commit": catrepo["c2b"], "tree": rec["tree"]}


def test_fetch_from_an_archive_and_from_file_urls_with_the_archive_cache(catrepo, tmp_path, yavih_home, monkeypatch):
    archives = tmp_path / "archives"
    archives.mkdir()
    for key in ("c2b", "c4"):
        c = catrepo[key]
        subprocess.run(["git", "-C", str(catrepo["repo"]), "archive", "--format=tar.gz", f"--prefix=yavih-{c}/", "-o",
                        str(archives / f"{c}.tar.gz"), c], check=True)
    # the archive of c4 holds the index and c3's folder: an archive serves the revision it holds
    with pytest.raises(CatalogError, match="9,<10"):
        catalog.fetch("catx", source=archives / f"{catrepo['c4']}.tar.gz", dest_dir=tmp_path / "a", version="1.2.0")
    rec = catalog.fetch("catx", source=archives / f"{catrepo['c4']}.tar.gz", dest_dir=tmp_path / "a", version="9.0.0")
    assert rec["commit"] == catrepo["c3"]
    # the default source through [catalog] url / archive (file:// here)
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[catalog]\nurl = "file://{catrepo["repo"]}/catalog/index.json"\narchive = "file://{archives}/{{commit}}.tar.gz"\n')
    monkeypatch.setenv("YAVIH_CONFIG", str(cfg))
    rec = catalog.fetch("catx", dest_dir=tmp_path / "b", version="1.2.0")
    assert rec["commit"] == catrepo["c2b"] and (yavih_home / "catalog" / "archives" / f"{catrepo['c2b']}.tar.gz").is_file()
    assert catalog.cached_index()["commit"] == catrepo["c3"]


def test_a_corrupt_archive_is_refused_and_installs_nothing(catrepo, tmp_path, yavih_home, monkeypatch):
    c = catrepo["c2b"]
    bad = tmp_path / "arch" / f"{c}.tar.gz"
    bad.parent.mkdir()
    (tmp_path / "src").mkdir()
    plain = tmp_path / "plain.tar"
    subprocess.run(["git", "-C", str(catrepo["repo"]), "archive", "--format=tar", f"--prefix=yavih-{c}/", "-o", str(plain), c], check=True)
    with tarfile.open(plain) as tf:
        tf.extractall(tmp_path / "src")  # noqa: S202 - our own archive
    src = tmp_path / "src" / f"yavih-{c}" / "benchmarks" / "catx"
    (src / "README.md").write_text("tampered\n")
    with tarfile.open(bad, "w:gz") as tf:
        tf.add(tmp_path / "src" / f"yavih-{c}", arcname=f"yavih-{c}")
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[catalog]\nurl = "file://{catrepo["repo"]}/catalog/index.json"\narchive = "file://{bad.parent}/{{commit}}.tar.gz"\n')
    monkeypatch.setenv("YAVIH_CONFIG", str(cfg))
    with pytest.raises(CatalogError) as ei:
        catalog.fetch("catx", dest_dir=tmp_path / "d", version="1.2.0")
    assert ei.value.code == "E_CATALOG_TREE" and not (tmp_path / "d" / "catx").exists()


def test_archive_members_that_leave_the_folder_are_refused(tmp_path):
    p = tmp_path / "evil.tar"
    with tarfile.open(p, "w") as tf:
        info = tarfile.TarInfo("top/benchmarks/catx/../../../escape.txt")
        info.size = 1
        import io

        tf.addfile(info, io.BytesIO(b"x"))
    with tarfile.open(p) as tf, pytest.raises(CatalogError, match="leaves the folder"):
        catalog._extract(tf, "benchmarks/catx", tmp_path / "out", strip=True)


def test_update_shows_the_change_and_fetches_only_with_yes(catrepo, tmp_path, yavih_home):
    dest = tmp_path / "proj" / "benchmarks"
    catalog.fetch(f"catx@{catrepo['c1'][:10]}", source=catrepo["repo"], dest_dir=dest, version="1.2.0")
    (rec,) = catalog.update(source=catrepo["repo"], dest_dir=dest, version="1.2.0")
    assert rec["status"] == "update_available" and rec["available"] == catrepo["c2b"]
    assert rec["changes"]["protocol_version"] == {"locked": 1, "available": 2}
    assert json.loads((dest.parent / "benchmarks.lock").read_text())["benchmarks"]["catx"]["commit"] == catrepo["c1"]
    (rec,) = catalog.update(["catx"], yes=True, source=catrepo["repo"], dest_dir=dest, version="1.2.0")
    assert rec["status"] == "updated"
    assert json.loads((dest.parent / "benchmarks.lock").read_text())["benchmarks"]["catx"]["commit"] == catrepo["c2b"]
    assert catalog.update(source=catrepo["repo"], dest_dir=dest, version="1.2.0")[0]["status"] == "up_to_date"


def test_cli_fetch_list_and_update_in_a_project(catrepo, tmp_path, yavih_home, monkeypatch, capsys):
    from yavih.cli import main

    proj = tmp_path / "work"
    proj.mkdir()
    monkeypatch.chdir(proj)
    monkeypatch.delenv("YAVIH_CONFIG", raising=False)  # project discovery on (the project yavih.toml init writes)
    assert main(["init", "--json"]) == 0
    capsys.readouterr()
    assert main(["bench", "fetch", "catx", "--from", str(catrepo["repo"]), "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["status"] == "ok" and Path(env["data"]["folder"]) == proj / "yavih-bench" / "benchmarks" / "catx"
    assert (proj / "yavih-bench" / "benchmarks.lock").is_file()
    assert main(["bench", "list", "--json"]) == 0
    rows = {r["name"]: r for r in json.loads(capsys.readouterr().out)["data"]}
    assert rows["catx"]["origin"]["kind"] == "fetched" and rows["catx"]["origin"]["commit"] == env["data"]["commit"]
    assert main(["bench", "update", "--from", str(catrepo["repo"])]) == 0
    assert "catx" in capsys.readouterr().out


def test_bench_list_shows_catalog_benchmarks_not_fetched_from_the_cached_index(catrepo, yavih_home, capsys):
    from yavih.cli import main

    (yavih_home / "catalog").mkdir()
    shutil.copy(catrepo["repo"] / "catalog" / "index.json", yavih_home / "catalog" / "index.json")
    assert main(["bench", "list", "--json"]) == 0
    rows = {r["name"]: r for r in json.loads(capsys.readouterr().out)["data"]}
    assert rows["catx"]["status"] == "not fetched" and rows["catx"]["origin"]["kind"] == "catalog"


def test_a_suite_entry_not_here_is_deferred_then_fetched(catrepo, tmp_path, yavih_home):
    from yavih.suites import fetch_missing, load_suite, without_unfetched

    (yavih_home / "catalog").mkdir()
    shutil.copy(catrepo["repo"] / "catalog" / "index.json", yavih_home / "catalog" / "index.json")
    sf = tmp_path / "s.yaml"
    sf.write_text("schema_version: 1\nname: s\ndescription: one catalog benchmark\ngroups:\n"
                  "  - capability: audio\n    entries:\n      - {benchmark: catx, readings: [v]}\n")
    s = load_suite(sf)
    (e,) = [e for _c, e in s.entries()]
    assert not e.fetched and e.status == "packaged"
    assert [x.status for _c, x in without_unfetched(s).entries()] == ["pending"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(sf.read_text().replace("catx", "nowhere"))
    from yavih.suites import SuiteError

    with pytest.raises(SuiteError, match="neither here nor in the catalog"):
        load_suite(bad)
    s2, done, failed = fetch_missing(s, source=catrepo["repo"])
    assert [d["name"] for d in done] == ["catx"] and not failed and all(x.fetched for _c, x in s2.entries())


def test_load_with_fetch_brings_a_name_that_is_not_here_and_an_edited_copy_warns(catrepo, tmp_path, yavih_home, monkeypatch):
    from yavih import config
    from yavih.benchmark import Benchmark, UnknownBenchmark

    with pytest.raises(UnknownBenchmark, match="yavih bench fetch catx"):
        Benchmark.load("catx")
    cfg = tmp_path / "config.toml"
    archives = tmp_path / "archives"
    archives.mkdir()
    c = catrepo["c3"]
    subprocess.run(["git", "-C", str(catrepo["repo"]), "archive", "--format=tar.gz", f"--prefix=yavih-{c}/", "-o",
                    str(archives / f"{c}.tar.gz"), c], check=True)
    cfg.write_text(f'[catalog]\nurl = "file://{catrepo["repo"]}/catalog/index.json"\narchive = "file://{archives}/{{commit}}.tar.gz"\n')
    monkeypatch.setenv("YAVIH_CONFIG", str(cfg))
    b = Benchmark.load("catx", fetch=True)  # a development build takes the newest revision
    # YAVIH_CONFIG names a project file: the folder goes to its benchmarks/ directory, the lock next to it
    assert b.folder == (tmp_path / "benchmarks" / "catx").resolve() and (tmp_path / "benchmarks.lock").is_file()
    assert b.folder_origin == {"repo": catalog.DEFAULT_REPO, "commit": c, "tree": catalog.tree_id(b.folder)}
    assert {"repo", "commit", "tree", "eval", "reader"} == set(b.folder_provenance())
    (b.folder / "README.md").write_text("edited\n")
    config.drain_warnings()
    assert Benchmark.load("catx").folder_origin["commit"] == c  # still the locked commit; the tree is the edited one
    assert any(w["code"] == "W_FETCHED_MODIFIED" for w in config.drain_warnings())


def test_an_exported_catalog_serves_fetch_without_the_checkout(catrepo, tmp_path, yavih_home):
    out = tmp_path / "mirror"
    assert catalog.main(["--repo", str(catrepo["repo"]), "--export", str(out), "catx"]) == 0
    idx = json.loads((out / "catalog" / "index.json").read_text())
    assert [r["commit"] for r in idx["benchmarks"]["catx"]["revisions"]] == [catrepo["c3"]]
    rec = catalog.fetch("catx", source=out, dest_dir=tmp_path / "w" / "benchmarks", version="9.0.0")
    assert rec["commit"] == catrepo["c3"] and rec["tree"] == catalog.tree_id(out / "benchmarks" / "catx")
