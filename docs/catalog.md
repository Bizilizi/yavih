# Catalog

The wheel ships the core only. The benchmark folders live in the repository's `benchmarks/` directory and reach a
user one at a time, fetched at a pinned commit: `yavih bench fetch <name>`. This page is the reference for the
index, the revision choice, the fetch, the lock and the offline paths; the code is `yavih.catalog`.

## The index

`catalog/index.json` on the repository's `main` branch lists every folder. It is generated from the repository's git
history alone, never from a working tree, so it is reproducible from the commit it names:

```bash
python -m yavih.catalog --write      # catalog/index.json for HEAD
python -m yavih.catalog --check      # exit 1 when it differs from a regeneration at the commit it names
```

The `catalog` workflow regenerates and commits it on every push to `main`; contributors never edit it.

```json
{
 "catalog_version": 1,
 "repo": "Bizilizi/yavih",
 "commit": "<the commit the index was generated from>",
 "benchmarks": {
  "esc50": {
   "name": "esc50", "title": "...", "header_title": "...", "aliases": [], "capability": ["audio"], "scorer": "clf",
   "obtain": "auto", "media_gb": 0.88, "clips": {"1": 400}, "protocol_version": 1, "protocol_hash": "...",
   "core": ">=1.1,<2", "path": "benchmarks/esc50",
   "revisions": [
    {"commit": "...", "path": "benchmarks/esc50", "tree": "<git tree id>", "core": ">=1.1,<2", "protocol_version": 1, "protocol_hash": "..."}
   ]
  }
 }
}
```

The metadata are read from the raw `benchmark.yaml` of the newest revision (no load, so a folder written for another
core is described too). `revisions` holds, for each distinct `(core, protocol_version)` in the folder's history, the
newest commit carrying it, newest first; a commit whose `benchmark.yaml` has no `core:` (a folder older than the
catalog) is not listed. `tree` is the git tree id of the folder at that commit (`git rev-parse <commit>:<path>`).

## Choosing a revision

Every folder declares `core:`, the PEP 440 specifier of the yavih core releases that can load and score it
([benchmark folder](benchmark-folder.md#benchmarkyaml)). `yavih bench fetch` takes the newest revision whose `core`
the installed yavih satisfies, so an older core gets an older revision of the folder instead of being told to
upgrade. A development build (`.devN`, a local version, a checkout without tags) is never refused: it takes the
newest revision and says so. When no revision fits, fetch stops with `E_CORE_VERSION` (exit 3) and the pip line of the
newest revision's `core`; yavih never upgrades itself. Loading a folder outside its `core` is refused the same way.

## Fetching

```bash
yavih bench fetch esc50                  # the newest revision this core can load
yavih bench fetch esc50@1a2b3c4d         # a listed revision
yavih bench fetch esc50 --refresh        # replace a folder edited here
yavih bench list [--catalog]             # every folder here, its origin, newer revisions, catalog folders not fetched
yavih bench update [<name>...] [--yes]   # what a newer compatible revision changes; --yes fetches it
```

Fetch reads the index (`[catalog] url`; the copy it downloads is cached at `$YAVIH_HOME/catalog/index.json` and used
when the network is down), downloads the repository archive at the chosen commit (`[catalog] archive`, cached per
commit under `$YAVIH_HOME/catalog/archives/`), extracts only the folder, checks that its tree id equals the index's
(`E_CATALOG_TREE` otherwise: nothing is installed), loads it (the manifest's sha256 is checked) and only then moves it
into place. It goes to the project's benchmarks directory (`yavih-bench/benchmarks/<name>`, or `[project]
benchmarks`), or to `$YAVIH_HOME/benchmarks/<name>` outside a project.

Next to that directory, `benchmarks.lock` records every fetched folder: `repo`, `commit`, `path`, `tree`, `core`,
`protocol_version`, `protocol_hash`, `source`, `fetched_at`. Commit the lock with the project; the fetched folders
themselves are listed in the directory's `.gitignore` (a managed block), so a project's own folders next to them stay
tracked.

Nothing changes silently:

- a folder already at the chosen revision is left alone (`unchanged`);
- a folder whose content is not what the lock records (edited here, or never fetched) is not overwritten without
  `--refresh` (`E_FETCH_MODIFIED`); loading an edited fetched folder warns `W_FETCHED_MODIFIED`;
- `yavih bench update` lists the changes of the newest compatible revision (commit, `core`, `protocol_version`,
  `protocol_hash`) and fetches only with `--yes`; an edited folder is reported, never overwritten. A new
  `protocol_version` or `protocol_hash` means new numbers are not comparable with the old ones.

`yavih plan init` fetches the suite's benchmarks that are not here before it writes bridge stubs (`--no-fetch` lists
them instead); `yavih suite show --fetch` does the same. A benchmark that cannot be fetched (offline, no fitting
revision) is listed and not planned (`W_NOT_FETCHED`). `Benchmark.load(name, fetch=True)` fetches a name that resolves
nowhere.

## Offline

`--from SRC` (on `bench fetch`, `bench update`, `suite show --fetch` and `plan init`) replaces the network:

- a yavih git checkout: the index is generated from its own history (current even on a branch the published index
  lags) and the folder is read with `git archive` at the chosen commit;
- a directory holding `catalog/index.json` and the folders at their `path`;
- a repository archive (`.tar.gz`, as `https://codeload.github.com/Bizilizi/yavih/tar.gz/<commit>` serves it): the
  revision is the one whose tree id the archive's folder has.

`[catalog] url` and `archive` also accept `file://` URLs, e.g. a mirror on a cluster without internet:

```toml
[catalog]
url = "file:///mirror/yavih/catalog/index.json"
archive = "file:///mirror/yavih/archives/{commit}.tar.gz"   # {repo} and {commit} are substituted
```

## Provenance

Every result records where its folder came from: `benchmark.folder_source` holds the `eval.py` / `reader.py` digests
and `repo`, `commit` (from the lock of a fetched folder; `null` otherwise) and `tree` (the tree id of the folder's
content when it was scored), and so do the results store and registry entries ([results](results.md),
[registry](registry.md)). `yavih bench info --json` shows it as `folder_origin`. The report compares results only on
the code digests: a README edit moves the tree without changing what scored the result.

## Codes

| code | exit | when |
|---|---|---|
| `E_CORE_VERSION` | 3 | no revision fits the installed core, or a folder is loaded outside its `core` (the pip line in the message) |
| `E_CATALOG_UNREACHABLE` | 3 | the index or an archive cannot be downloaded and nothing is cached |
| `E_CATALOG_INDEX` | 2 / 3 | not an index, an index without revisions, or a newer `catalog_version` (3: upgrade yavih) |
| `E_CATALOG_UNKNOWN` | 1 | a name or `@commit` the index does not list |
| `E_CATALOG_SOURCE` | 1 | a `--from` source that is not a checkout, an index directory or an archive, or lacks the folder |
| `E_CATALOG_TREE` | 2 | the fetched folder's tree id differs from the index's (nothing installed) |
| `E_CATALOG_ARCHIVE` | 2 | an archive member or link would leave the folder |
| `E_CATALOG_GIT` | 1 | a git command failed (`--from` a checkout, `python -m yavih.catalog`) |
| `E_FETCH_MODIFIED` | 1 | the folder here is not the one the lock records; `--refresh` replaces it |
| `E_LOCK` | 2 | an unreadable `benchmarks.lock` |
| `W_FETCHED_MODIFIED` | | a fetched folder no longer matches its lock (results record the edited tree) |
| `W_CATALOG_CACHED` | | the index is unreachable: the cached copy is used |
| `W_CATALOG` | | `bench list --catalog` could not read the index |
| `W_NOT_FETCHED` | | `plan init` could not fetch a suite benchmark: listed, not planned |
