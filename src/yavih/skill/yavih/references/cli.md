# CLI reference — commands, flags, `--json`, exit codes, codes

```
yavih [--json] [--version] [--config FILE] [--runs-root DIR] [--results DIR] [--data B.ROOT=PATH]... [--lenient] <command> ...          (also: python -m yavih ...)
  doctor    environment, data roots, the benchmark's authors' packages, GPU, configuration; --selftest runs the reference bridge
  config    show the configuration with the source of every value; get one value (for shell scripts)
  skill     install or show the agent skill (show --ref <name>: one reference)
  bench     list / info / files / show / path / prepare / fetch / verify benchmarks (files lists the shipped files, show prints one,
            prepare runs a numbered prepare/ step; verify --census writes media facts)
  bridge    new: a filled copy of a benchmark's bridge template; check: smoke-test a bridge (hard invariance checks) and write its
            profile (--synthetic needs no data)
  extract   extract one shard (python -m yavih.extract)
  features  validate / inspect / diff / stage feature sets
  probe     train and score the standardized readout; writes result.json and the results store
  report    lm-eval table + comparability-checked columns with floors; --merge-results, --join
  suite     list / show / new / missing: named benchmark sets (default, minimal, project suites) and what a model lacks
  plan      init (matrix.yaml + bridge stubs for a suite) / build (--sample N: a capped run) / status / calibrate;
            run --local runs a plan on this machine
  exec      run one plan unit (or a --range, or --slots K at once)
```

## Command table

| command | arguments | `data` under `--json` |
|---|---|---|
| `yavih init` | `[--dir D] [--site ID [--hosts GLOB]... [--launcher CMD]] [--force]` | `{dir, file, written, kept, existing, runs_root, results_root, user_config, next}`: writes the project folder `D/yavih-bench/` (`yavih.toml` with `[project] yavih = ">=M,<M+1"`, a commented data root and launcher and one `[site.ID]` table; `README.md`; `.gitignore` = `runs/`, `logs/`, `__pycache__/`; `bridges/`, `matrices/`, `suites/`, `benchmarks/`, `tools/`); never the user file `~/.config/yavih/config.toml` (`user_config`: where it would be; personal overrides only); `runs_root` / `results_root` = the roots in effect, by default `yavih-bench/runs` (ignored by git) and `yavih-bench/results` (committed); one project per repository: a directory another project file already governs is `E_PROJECT_EXISTS` (unless that project lists it in `[project] subprojects`), an existing `yavih.toml` is `E_EXISTS` unless `--force` refreshes the other files (never `yavih.toml`) |
| `yavih doctor` | `[--bench B] [--selftest]` | `{core, media, probe, external: {name: {ok, version, version_pin, pip}}, tools: {name: {path, version, version_pin, ok, problem, x264?, x264_pin?}}, torch: {version, cuda, untouched}, media_read: {clip_id, ok, split, error?}, media_info: {present, sha256, header_backends: {kind: [backend]}}, data_roots: {name: {path, exists, source, env_var, config_key, prepare_only, error}}, config, notes, gpu, launcher, decode_workers_default, yavih_home, runs_root, results_root, installed_from, package_dir, skill: {packaged, installed}}`; `installed_from` = `wheel`, `editable` (`pip install -e`) or `checkout` (imported from a source tree), `package_dir` = where the shipped files live; an installed skill copy (`.claude/skills/yavih`, `.codex/skills/yavih`, `.cursor/rules/yavih.mdc`, `AGENTS.md` under the current directory) whose stamp differs from the packaged skill, or has none, is `W_SKILL_STALE` (re-run `yavih skill install`); `config` = `config show --bench B`; `project` = `{dir, file, root, nested, stray, skipped, requirement: {spec, version, ok, dev, reason}}` (`W_PROJECT_NESTED`: a second project file in the repository not listed in `[project] subprojects`; `W_PROJECT_STRAY`: a `matrix*.yaml` outside the project folder; `W_PROJECT_VERSION`: the installed yavih is outside `[project] yavih`; `W_RUNS_SIZE`: the project-default runs root `yavih-bench/runs` holds more than 50 GB, the message says how to move it to scratch; the note `N_PROJECT` for no folder, a dev build or a capped scan); a root that does not resolve or does not exist is the warning `W_DATA_ROOT` naming the config key and env var to set (a missing `prepare_only` root is the note `N_DATA_ROOT_PREPARE`), an invalid configuration is `W_CONFIG`: doctor reports, it never aborts on configuration; `external` lists the benchmark's authors' packages (needed in the probe env only); `tools` checks a materializing benchmark's pinned tool (avsync: ffmpeg and its libx264 build, `W_TOOL_PIN` on a mismatch; run doctor on the node that will run the prepare step); `media_read` reads one row of the first split with rows among eval, train, folds, floor (`W_MEDIA_READ` when none can be read); without a census, a stream kind with no header backend (PyAV / soundfile / pillow, or `ffprobe`) is `W_HEADER_BACKEND` (every row would fail; `extract` refuses at start, exit 3) |
| `yavih config show` | `[--bench B]` | `{project: {file, dir, yavih, bridges, matrices, suites, benchmarks, subprojects} \| null, files: [{path, layer, how, exists}], site, site_source, lenient, values: {key: {value, source}}, warnings: [{code, message}], bench?: {name, folder, roots: {root: {value, source, exists, prepare_only, default, env_var, config_key, error}}}}`; `project` is null without a project file (a `[project]` path that does not resolve shows as `<message>`); `source` is `cli`, `env:NAME`, `file:PATH:LINE` (`[site.<id>]` appended), `yaml default <expr>` or `default`; `values` keys: `home.root`, `runs.root`, `results.root`, `scratch.root`, `data.root`, `datasets.<name>`, `data.<bench>.<root>`, the effective `resources.*` (cpus, memory limit, RAM budget, decode workers) with what decided them, `launcher` and `launchers.<name>` |
| `yavih config get` | `<bench>.<root> \| runs.root \| results.root \| home.root \| scratch.root \| data.root \| datasets.<name> \| resources.<key>` | `{key, value, source}`; text mode prints the bare value (what `prepare/*.sh` use: `ROOT="${ROOT:-$(yavih config get esc50.media_root)}"`); exit 1 when it does not resolve |
| `yavih skill install` | `[--target claude\|codex\|cursor\|agents-md] [--path P]` | `{path, target, files, stamp}`; claude / codex copy the skill directory to `.claude/skills/yavih` / `.codex/skills/yavih`, cursor writes `.cursor/rules/yavih.mdc`, agents-md writes a marked block into `AGENTS.md` (replaced in place on re-install); every form carries a stamp (yavih version + sha256 of the skill files: `.yavih_skill.json` in a directory, an HTML comment in a single file) that `doctor` compares (`W_SKILL_STALE`) |
| `yavih skill show` | `[--all \| --ref NAME]` | prints SKILL.md; `--all` appends every `references/*.md`; `--ref bridge` prints that one reference (bridge, cli, data, readouts, slurm) |
| `yavih bench list` | `[--fmt text\|md] [--suite S]` (`md`: the generated markdown table the docs embed, no data scan; `--suite`: only the suite's benchmarks, adding the suite group `suite_capability` and `readings`) | rows `{name, aliases, readout, capability: [ids of yavih.suites.CAPABILITIES], clips: {split: n}, media_gb, source_kind, obtain: {tier, how}, status: "missing" \| "partial 61%" \| "verified"}`; `obtain.tier`: `auto` (fetchable unattended), `manual-download` (a person downloads it from an official portal or the authors' links), `needs-licence` (a human signs a licence for their own account), `needs-data-owner` (ask the person who holds it), `site-only` |
| `yavih bench info` | `<name> [--readme-only \| --setup]` | README text (or setup.md with `--setup`, the same file as `bench show <name> setup`) + `{name, title, capability, readout, protocol_version, protocol_hash, input_hash, scoring_hash, manifest_hash, roles: {role: {splits, nominal, present}}, floors, metrics, source, obtain, budget, external, history}` (`input_hash`: what features are checked against: the manifest, splits, streams, contracts, readout and its inputs; `scoring_hash`: `protocol_version`, protocol, readout params, metrics, external pins, targets; no source file enters either); `paths.folder` = the benchmark folder (holds `bridge_template.py`) |
| `yavih bench files` | `<name>` | `{name, folder, files: [{rel, path, kind, size}]}`: every shipped file of the folder with its absolute path; `kind` ∈ readme, setup, yaml, template, probe, manifest, classes, conformance, prepare, test, other (no maintainer notes, bytecode or hidden files: the listing is the same from a wheel and a checkout) |
| `yavih bench show` | `<name> <what>` | prints one shipped file; `what` = `readme`, `setup`, `yaml`, `template`, `probe`, `classes`, `conformance` (`conformance/expected.json`), `prepare/<file>` or any path inside the folder; `--json`: `{path, text}`. A path leaving the folder or naming a maintainer note is `E_PATH`, a binary file (`manifest.csv.gz`) `E_BINARY`, a missing one `E_NOT_FOUND` (exit 1) |
| `yavih bench path` | `<name> [<what>]` | `{name, folder}`; prints the folder (`FOLDER=$(yavih bench path <b>)`), or with `<what>` (as `bench show`) that file's absolute path, e.g. for `sbatch "$(yavih bench path k400 prepare/01_download_torrent.sh)"` |
| `yavih bench prepare` | `<name> [<step> [-- args…]]` | without a step: `{name, folder, steps: [{step, name, file, path, interpreter, doc}]}`, the numbered `prepare/NN_*.{sh,py}` scripts; with one (`01`, `1`, `download`, `01_download.sh`, or another file under `prepare/`): runs it with this python (`.py`) or `bash` (`.sh`) in the current directory, environment + `FOLDER=<folder>` + `YAVIH_BENCH=<name>`; exits with the step's own code (`E_PREPARE_FAILED` when non-zero) |
| `yavih bridge new` | `<bench> --model M [--reading R] [--out DIR\|FILE] [--force]` | `{bench, model, reading, path, template, filled, next}`: the folder's `bridge_template.py` copied to the project's `bridges/<M>_<bench>.py` (`./bridges/` without a project) with `describe()["model"]` (and `Bridge.name`, and the reading when given) filled and a header naming `yavih bench show <bench> setup`; refuses an existing file (`E_EXISTS`) unless `--force`. Everything else stays `TODO`, so `bridge check` refuses the copy (`E_TEMPLATE_UNFILLED`) until it is filled |
| `yavih bench lint` | `<name\|path>... [--all] [--skip CHECK]` | `[{name, folder, ok, errors, warnings}]`: the folder contract (`yavih.lint`): required files, strict `benchmark.yaml`, README.md / setup.md headings and pins, contract yaml copies, budget vs frozen rows, `obtain`, bridge template, citations a pip install can open (no shipped folder file may point into the folder's maintainer notes, which stay in the source repository: `E_NOTES_LINK`), `probe.py` (and an optional `reader.py`) importable without torch, `capability`: `capability: [...]` from the vocabulary (`E_CAPABILITY`) and listed so by the shipped suites (`W_SUITE_CAPABILITY`), `conformance`: the folder's `conformance/expected.json` is current (the `bench conformance` codes), `hashed`: no prose inside a hashed yaml view (`E_HASHED_PROSE`), and `hashes`: the newest `changelog` entry records the `protocol_hash` and `input_hash` the folder computes (`E_CHANGELOG_HASH` prints the exact lines to write, and when to list the old hash under `supersedes` / `supersedes_input`); errors exit 2, `TODO` markers are warnings. Run it on every folder you write or edit |
| `yavih bench new` | `<name> [--readout clf\|seq\|grid\|box] [--like B] [--dir D]` | `{name, folder, lint, todo}`: a scaffold copied from `--like` (default per readout: esc50, ave, nyuv2_depth, vggss_box) with a fresh README.md / setup.md / `make_manifest.py` / `prepare/` / `tests/` and `TODO` wherever a fact is needed; it passes `bench lint`. Written to `D/<name>` (default: the project's `benchmarks/`, else the first `$YAVIH_BENCH_PATH` directory, else `.`); folders in the project's `benchmarks/`, on `$YAVIH_BENCH_PATH` and in pip-installed packs (entry-point group `yavih.benchmarks`) resolve by name after `$YAVIH_HOME/benchmarks` and the packaged ones, never shadowing a packaged name. Contributing one: https://github.com/Bizilizi/yavih/blob/main/docs/contributing-a-benchmark.md |
| `yavih bench trial` | `<name\|folder> --synthetic [--readout-fast] [--keep DIR]` | `{benchmark, readout, ok, exit_code, stages: [{stage, status, seconds, detail, error}], headline: {metric, value, ci95, floor_name, floor, clears_floor}, failed, paths, limits}`: the folder's whole chain in process, stage by stage: `conformance` (the folder's fixture), `stand_in`, `bridge` (the oracle bridge), `bridge_check`, `extract`, `validate`, `probe` (without a results file), `report`, `headline` on a generated stand-in; never touches the runs root, results store, `$YAVIH_HOME` or configuration. `--synthetic` is required (without it: exit 1, pointing at https://github.com/Bizilizi/yavih/blob/main/docs/trial-run.md); `--readout-fast` uses the test-only fast `ProbeConfig` and the headline must clear its floor. `E_TRIAL` (exit 2) a stage failed, `E_TRIAL_NOT_READY` (exit 3) no stand-in can be built, `W_TRIAL_HEADLINE` the headline missed its floor without `--readout-fast`. Single-label stand-in: the folder's `probe.py` hooks and `external` packages are not exercised |
| `yavih bench conformance` | `<name\|path>... [--all] [--record [--same-version --reason TEXT] [--force]]` | check (or record) the folder's `conformance/expected.json`: the numbers its scoring path (readout, metrics, `probe.py`, pinned external package) produces on a generated population, at its `protocol_version`, within the per-metric tolerance (`metrics.tolerance`, default 0.001). `E_CONFORMANCE_DRIFT` = the numbers moved at the same version (bump `protocol_version`, re-record), `E_CONFORMANCE_STALE` = recorded at another version, `E_CONFORMANCE_MISSING` = no fixture; `W_CONFORMANCE_SKIPPED` = the external package is absent here. Code is never hashed; this fixture is what holds a declared version to its numbers |
| `yavih bench hash` | `<name\|path> [--explain]` | `{protocol_hash, input_hash, scoring_hash, manifest_hash, folder_source}`; `--explain` adds the canonical JSON and digest of every hashed component, enough to recompute the hashes with stdlib only (what a third party matches to claim the hash) |
| `yavih registry export` | `<results_root\|results_file> --org ORG --out DIR [--model NAME]... [--include-nonstandard]` | registry entries (`entries/<benchmark>/<eval_hash12>/<org>/<model>/<reading>.json`) + protocol payloads from a results store; `W_REGISTRY_UNVERIFIED` = the folder's conformance fixture did not pass here (the entry is written, `check` refuses it) |
| `yavih registry check` | `DIR` | every entry: schema, id = path, `eval_hash = H(protocol_hash, probe_hash)`, protocol payload recomputes the hash, conformance passed, tolerance recorded; `W_REGISTRY_NONSTANDARD` / `W_REGISTRY_COVERAGE` are shown-as-such, never ranked; errors exit 2 |
| `yavih bench fetch` | `<name> [--refresh]` | syncs the folder into `$YAVIH_HOME/benchmarks/<name>` (absent files copied, files the last fetch copied refreshed, locally edited files kept unless `--refresh`), stamps it (`.yavih_fetch.json`: yavih version, protocol hash, per-file sha256) and verifies the manifest sha; never downloads media. A home copy whose packaged folder changed since (a yavih upgrade) is refused at load (`E_HOME_BENCH_STALE`, exit 1; a warning with `--lenient`); an unstamped one warns (`W_HOME_BENCH_UNSTAMPED`) |
| `yavih bench verify` | `<name> [--census] [--decode-sample 32] [--fingerprint-sample 256]` | `{manifest_sha_ok, per_split: {present, nominal, missing_first10}, read: {ok, failed}, decode_sample: {ok, failed, lib}, labels_ok, targets_sha_ok, derived_sha_ok, curation_invariants_ok, media_info: {rows, sha256, path}, external: {name: {ok, version}}, media_fingerprint, data_roots, data_status_path}`; `read` reads the bytes of sampled rows; `decode_sample` is a media-health check with `yavih.media.decode` (skipped without `[media]`), never a protocol step; `--census` writes the media-info table `$YAVIH_HOME/media_info/<name>-<manifest_hash12>.jsonl.gz`; `derived_sha_ok` is `null` where the folder lists no derived files |
| `yavih bridge check` | `<bridge.py> --bench B [--n 32] [--arg k=v]... [--decode-workers N] [--out profile.json \| --synthetic] [--allow-loopback]` | `{describe, describe_sha256, model, reading, used, media, N, D, pos_mode, streams, invariance: {rerun_bit_identical, rerun_rel_l2, batch_rel_l2, order_rel_l2, tolerance, tolerance_reason}, rules_checked, clips_per_s, decode_workers, peak_gpu_gb, warnings, notes}` = `profile.json` (`notes`: `N_RULE_CUSTOM` per opted-out rule; with `--allow-loopback` also `allow_loopback`, `loopback_connects`); exit 2 on a failed hard check (no profile written); `--synthetic` runs the same checks on tiny generated media standing in for the benchmark (its name, readout, stream names/kinds, guaranteed facts, label facts; needs PyAV, soundfile, pillow), adds `synthetic: true` and writes no profile |
| `yavih extract` / `python -m yavih.extract` | `--bench B --bridge P --split S (--shard i/n \| --rows lo:hi) --out DIR [--arg k=v]... [--canaries 8] [--decode-workers N] [--read-ahead 2] [--deadline none\|+S\|<dur>\|<epoch>\|slurm] [--expect-profile P] [--allow-loopback]` | sidecar summary; `--out` is the run dir `<runs root>/<b>/<model>/<reading>`; `--allow-loopback` lets `encode` reach a local server on 127.0.0.0/8 / ::1 (recorded in the sidecar's `side_effects`; any other connect is still `E_NETWORK`); `--rows lo:hi` extracts manifest rows `[lo, hi)` as the one shard `0/1` of a sampled layout (what `plan build --sample` plans); `--expect-profile` (what `plan` passes) skips a matching shard without loading the model and refuses a drifted configuration (`E_CONFIG_DRIFT`) |
| `yavih features validate` | `<features_dir or features/> [--verify-sha] [--no-write-meta] [--waive CODE=reason]...` | per split `ValidationReport.to_json()`: `{ok, errors, warnings, notes, split, coverage, N, D, dtype, pos_mode, bytes, backend_hint, eligibility, missing_shards, waived}`; `--waive` (repeatable; `all=reason` = every waivable check) waives a waivable check for features already extracted, recorded in `_meta.json` `waived` with `source: cli` (no hash changes; carried over by later validations of the same configuration) |
| `yavih features inspect` | `<features_dir>` | per-dim std histogram, near-constant dims, mean abs, pos summary, dead rows, `used` summary per stream |
| `yavih features diff` | `<a> <b>` | `{rows_compared, max_abs, max_rel_l2, verdict}` |
| `yavih features stage` | `<features/> <scratch>` | copies shards + sidecars + `_meta.json` to `<scratch>/<unit_id\|run>.<host>.<pid>/`; prints the path for `probe run --scratch` |
| `yavih probe run` | `<run_dir> [--seeds 0,1,2] [--deadline none\|+S\|<dur>\|<epoch>\|slurm] [--scratch DIR] [--allow-low-coverage] [--loader-workers N] [--log-samples] [--out DIR] [--device cpu\|cuda\|cuda:N] [--results DIR \| --no-results] [--waive CODE=reason]...` | list of result summaries (`--waive`: an operator's waivers for the validate prologue, as `features validate --waive`); `paths.results_file`, `paths.samples_files`; `--out DIR` probes read-only features (shards mirrored into `DIR/features/` as symlinks; `_meta.json` and `probe/` written under DIR, the source never); `--device` overrides cuda/cpu auto-detection |
| `yavih report build` | `<glob>... [--fmt md\|csv\|json] [--out PATH] [--seeds aggregate\|list] [--allow-coverage-diff] [--detail] [--join vggsounder a=RUN v=RUN av=RUN] [--results DIR] [--merge-results] [--suite S] [--registry DIR]` | table rows of both views: the per-model `benchmark \| reading \| version \| eval_hash \| readout \| metric \| value \| ci95 \| floor` table and the comparability columns (`(benchmark, eval_hash)` columns, `(model, reading)` rows); csv/json rows carry `eval_hash`, `model`, `reading`, `describe_sha256`, `features_fingerprint`; rows from a capped run are marked `partial`; values within a metric's conformance tolerance of the column's best are ties (`≈` in md, `tie` in json/csv), never ordered by value; `--registry DIR` adds a model registry's accepted entries as rows `<org>/<model>` (flag `registry`); `--suite S` appends `## Suite completeness: <S>` per model (json: `suite`) |
| `yavih plan init` | `--suite S --model M [--launcher PYTHON\|NAME] [--out FILE] [--bridges-dir DIR] [--force]` | `{path, suite, model, launcher, launcher_configured, project, entries, pending, joins, bridges, plan, next}`: writes the matrix (in a project `yavih-bench/matrices/<M>.yaml`, else `./matrix.yaml`; one entry `<M>-<benchmark>-<reading>` per suite pair, `args: {reading: R}` for a multi-reading template, `profile: <matrix stem>.profiles/<M>-<benchmark>-<reading>.profile.json`; `defaults.launcher` = `--launcher`, or `null` = the configured launcher of the site at execution, `W_LAUNCHER_UNSET` when none is configured either; pending entries and join commands as comments) and one bridge stub per benchmark into the project's `bridges/` (`bridge new`; an existing bridge is kept); prints one `yavih bridge check … --out <profile>` per entry, then `plan build <matrix> --out <plan>` with `plan` = `<runs root>/_plans/<M>.jsonl` (plan state lives with the runs, never in `yavih-bench/`). An existing matrix is `E_EXISTS` (exit 1) unless `--force` |
| `yavih plan build` | `<matrix.yaml> [--out plan.jsonl] [--pending] [--kind extract\|probe\|report] [--shard-minutes M] [--format jsonl\|count] [--sample N]` | unit records (`count` with `--out`: `{count, sha256}`, the file's sha256 for `exec --plan-sha`); a split whose layout is frozen (`features/<split>/_layout.json`) is planned with exactly that layout whatever `--shard-minutes` or the profile's `clips_per_s` say; the plan file stores `cwd` relative to itself; `--sample N` plans a capped run: the first `N` rows of every split in manifest order (not class-balanced), one `extract --rows 0:k` unit per split, probe units with `--no-results`, all under the runs root `<out>-sample<N>`; its numbers only say the chain runs |
| `yavih suite list` | | `[{name, origin, path, description, shadows, error}]`; `origin` `shipped` or `project`, `shadows` when a project suite replaces the shipped one of its name; an invalid project file is listed with its error (`W_SUITE_INVALID`) |
| `yavih suite show` | `<name\|file.yaml>` | the suite's capabilities, benchmarks, readings and joins. Resolution: a path, then `<project suites dir>/<name>.yaml` (`$YAVIH_SUITES`, else the project's `suites/`, else `./suites`), then the shipped `default` / `minimal` |
| `yavih suite new` | `<name> [--from default] [--out DIR\|FILE] [--force]` | copies a suite (comments included) into the project suites directory with its `name` replaced, to edit; an existing file is `E_EXISTS` unless `--force` |
| `yavih suite missing` | `<name> [<glob>...] --model M [--results DIR]` (globs before the options; default: the runs root) | per suite `(benchmark, reading)` of model `M`: `done`, `missing` or `pending`; per join `done`, `join-missing` (with its `report build --join` command) or `join-unknown`; exit 0 either way. An unknown or invalid suite is `E_SUITE` (exit 1) |
| `yavih plan status` | `<plan.jsonl> [--root DIR]` | `{done, pending, running, not_ready, preempted, misconfigured, failed, not_applicable, total, by_kind, stale_running, repeated, units: [{id, kind, benchmark, bridge, split, shard, seed, status, exit_code, message, log, host, attempts}]}` (`preempted` = exit 4 units, resubmittable, never `failed`; `misconfigured` = exit 1, fix the configuration; `running` = a live `running` record; `stale_running` = units whose process ended without a final record, counted as `pending`; `repeated` = units whose last 3+ attempts ended with the same exit 3 or 4: resubmitting again will not help). Text mode prints the counts per kind and the misconfigured / failed / repeated / not_ready / preempted / running units with the last line of their message; `--json` has every unit |
| `yavih plan calibrate` | `<runs_root>` | multipliers written to `<runs_root>/_calibration.json`; a unit's `needs.minutes` is raised ×1.5 only when its own deadline stopped it while running (`status: deadline`, not a start refusal or a preemption) |
| `yavih plan run` | `<plan.jsonl> --local [--slots K] [--deadline D] [--deadline-margin 900] [--root DIR] [--cpus N] [--deps plan.jsonl]` | `{units: [last…], counts, blocked, not_started, already_done, total, status, exit_code}`: every unit not done yet, dependencies first, on this machine (`--slots K`: K units at a time, each a separate `yavih exec --id` process on the g-th visible GPU); a failed / misconfigured / not-ready unit is not retried and its dependents are `blocked`; a deadline stops new starts. `--local` is required: yavih never writes or submits sbatch |
| `yavih exec` | `<plan.jsonl> (--index N \| --id X \| --range a:b) [--deps plan.jsonl] [--deadline none\|+S\|<dur>\|<epoch>\|slurm] [--deadline-margin 900] [--slots K] [--root DIR] [--cpus N] [--plan-sha HEX]` | `logs/last.json` content (several units: `{units, status, exit_code, not_started}`, worst code); `--deps` = the full plan, used only to look up dependencies (required when `<plan.jsonl>` is a `--pending --kind probe\|report` file); `--range a:b` runs units a … b-1 in order and stops at the deadline; `--slots K` runs K units at once (with `--index i`: units K·i … K·i+K-1), each a separate `yavih exec --id` process with `CUDA_VISIBLE_DEVICES` = the g-th entry of the inherited list; `--root` = the directory the units' `cwd` (the only relative path a plan stores; the project) means on this machine (a container mount); `out` / `done` stay the absolute runs-root paths, so mount the runs root at its own path; `--cpus` = CPUs per unit (decode workers `cpus − 1`; default `needs.cpus`, clamped to the CPUs the affinity mask / cgroup quota allow); `--plan-sha` refuses a plan file whose sha256 does not start with HEX; the margin stays above the sbatch `--signal=B:TERM@600` lead (references/slurm.md) |

Text output worth knowing: `features validate` prints the shard table (`shards 31/32 done, 0 in
progress, 1 missing: --shard 7/32`), the rules line (`rules checked: video uniform + spatial;
audio span`) and the eligibility line (`eligible for box: no (grid coverage median 0.18 < 0.25
over the full frame) → probe will record not_applicable`); an ineligible but well-formed set exits
0. `exec` prints one header and one footer line `[yavih] unit <id> <kind> <status> wall_s=…`;
every unit that ran leaves `<out>/logs/<id>.<job|pid>.<utc>.log` (an extract child's output; for an
in-process probe/report a header and, on failure, the full traceback), named in `last.<id>.json`
`log`. While a unit runs its `last.<id>.json` holds `status: running` (`host`, `pid`, `started_epoch`).
`--deadline` has one parser for `extract`, `probe run`, `exec` and `plan run`: `none`; `+3600`, `+90m`,
`2h`, `1h30m` (from now); an epoch in seconds (a bare number before 2001 is refused as a duration
typo); an ISO time (`2026-10-10T18:00:00Z`); `slurm` = `SLURM_JOB_END_TIME − 900 s` (above the
templates' 600 s signal lead, so a unit stops on its own clock first). `slurm` inside a job without
`SLURM_JOB_END_TIME` (older Slurm releases) is `E_DEADLINE_SPEC`, exit 1 — never "now"; outside
SLURM it means no deadline. A unit whose child exits 1 is recorded `misconfigured` (exit 1).

## The `--json` envelope

Exactly one object on stdout (logs and `[yavih] warning …` lines on stderr). While a command runs
under `--json`, anything it prints to stdout — a bridge's model constructor, a library — goes to
stderr: `sys.stdout` is redirected and, when stdout is a real descriptor, fd 1 too, so C-level and
child-process writes are covered as well (a descriptor duplicated from stdout before the command
started is not).

```json
{"status": "ok|error|skipped|not_applicable", "data": {...}, "errors": [Finding], "warnings": [Finding], "paths": {"name": "path"}}
```

`Finding = {"code": "E_…|W_…|N_…", "message": str, "where": str|null}`. Parse `status` first;
`errors[0].code` names the rule; `paths` has `profile`, `plan`, `folder`, `data_status`, `skill`,
`out`, `staged`, `results_file`, `samples_files` where applicable. Under `--json` a usage error is
still one envelope (`E_USAGE`, exit 1).

## Exit codes

| code | meaning | scheduler action |
|---|---|---|
| 0 | done, skipped, or `not_applicable` recorded | none |
| 1 | usage / configuration error (bad args, unknown `--arg`, bad yaml, bad profile, bad `--deadline`, a plan older than the split's frozen layout `E_LAYOUT`) | fix, resubmit |
| 2 | contract violation the agent must fix (validation errors, hash mismatch, describe schema, failed invariance) | fix bridge/data |
| 3 | not ready: data missing, dependency DONE missing, decoder missing, authors' package missing or mis-versioned, scratch short, the shard is being written by another live task (`E_SHARD_CLAIMED`) | resubmit later |
| 4 | deadline reached or SIGTERM/SIGUSR1, or the shard's claim was taken over meanwhile (`E_CLAIM_LOST`); nothing partial left; resumable | resubmit |
| 5 | internal/unexpected failure (also: an exception from `encode` other than `NotApplicable`/`MediaUndecodable`) | inspect `logs/` |

`plan status` maps `logs/last.json`: exit 4 → `preempted`, exit 3 → `not_ready`, exit 1 →
`misconfigured`, 2/5 → `failed`, no record → `pending`, matching DONE → `done`, a `running` record → `running` while its
process can be alive (same host: pid alive; other host: within 2 × `needs.minutes` + 30 min and
the recorded deadline), else `pending` (listed under `stale_running`).

## Codes

### Bridge / extract / bridge check (exit 2 unless noted)

`E_DESCRIBE` schema violation (missing MUST key, `model` or `reading` outside its charset,
`views != 1`, an `input` rule outside the vocabulary, `batch_tolerance` above 1e-3 or without
reason, an unknown key) · `E_TEMPLATE_UNFILLED`
(bridge check) `TODO` left in `model`, `weights_ref` or a `norm.source`, or a null `norm` constant ·
`E_CONSTANT_FEATURES` (bridge check) every smoke clip encodes to the same features · `E_SMOKE` (exit 3)
no present row in the smoke set (names the data roots searched; `--synthetic` needs none) ·
`E_SYNTHETIC` (exit 1; 3 without the media writers) `--synthetic` cannot build its stand-in · `E_INPUT` a rule the benchmark's facts cannot evaluate (the message offers a supported rule
or the `custom` opt-out) · `E_PRECISION_UNDECLARED` · `E_ARGS_UNCONSUMED` · `E_ARG` unknown/malformed
`--arg` (exit 1; a bridge file that fails to import shows its traceback) · `E_TRAIN_MODE` · `E_CONFIG_DRIFT` the live `describe_sha256`, `model` or `reading`
differs from the `--expect-profile` profile (names the differing keys) ·
`E_USED` missing or ill-formed `used` report · `E_NETWORK` a socket connection
inside `encode` (loopback only with `--allow-loopback`) · `E_RERUN_DRIFT` rerun not bit-identical (with `precision.deterministic: false`: above the tolerance) · `E_BATCH_INVARIANCE` alone or shuffled
pass above the tolerance (names the pass and clip) · `E_TIMES_OUT_OF_MEDIA`, `E_RULE_MISMATCH`,
`E_POS_SHAPE`, `E_POS_RANGE` on the smoke set · `E_WAIVE_INVALID` `describe()["waive"]` names a non-waivable code
or gives no reason (Waivers below; a waived check is `W_WAIVED`, the profile is written and lists `waived`). Warnings: `W_SIDE_EFFECT` (writes outside the run
directory during `encode`), `W_NO_PREPROCESSING_SOURCE`, `W_LOOSE_TOLERANCE`, `W_RULE_UNCHECKED`
(a checkable rule a null media fact kept from being evaluated), `W_FP16_RANGE`,
`W_WEIGHTS_UNFINGERPRINTED` (null `weights_sha256`: recorded, nothing else), `W_RERUN_NONDETERMINISTIC` (rerun within the tolerance but not bit-identical; only with `deterministic: false`). `doctor --bench`: `W_HEADER_BACKEND` (no census and no header backend for a stream kind); `doctor`: `W_SKILL_STALE` (an installed skill copy older than the packaged one). `bench show` / `bench path` / `bench prepare` / `bridge new` (exit 1): `E_PATH`, `E_BINARY`, `E_NOT_FOUND`, `E_PREPARE_STEP`, `E_EXISTS`; `bench prepare` exits with the step's own code (`E_PREPARE_FAILED`). Notes: `N_RULE_CUSTOM`
(a `custom` rule, the legitimate opt-out; never a warning). `MediaUndecodable` rows become missing
rows (coverage), not errors.

### `features validate`

Errors: `E_FORMAT` (a feature format this yavih does not read: re-extract), `E_NA_AND_SHARDS`, `E_SHARD_SET` (lists
`--shard i/n` to rerun), `E_RANGE_TILING`, `E_HASH_MANIFEST`, `E_HASH_INPUT` (the benchmark's
input side changed since extraction: the message names the differing components — name, manifest, splits,
streams, contracts, readout and its inputs; no source file — and the yavih versions/commits that wrote and read
the shards; re-extract. A scoring-side change — a `protocol_version` bump, a metric or pin, a readout version — is
never a validate error, and a `probe.py` / `reader.py` edit moves no hash at all),
`E_DESCRIBE_MISMATCH` (sidecars of one split with two `describe_sha256`, `model` or `reading`; another
decoder build or batch size is not a mismatch), `E_ARGS_UNCONSUMED`, `E_TAR_SHA` (only `--verify-sha`), `E_MEMBERS`,
`E_ROWS`, `E_TIMES_OUT_OF_MEDIA`, `E_RULE_MISMATCH`, `E_FINITE`, `E_POS_SHAPE`, `E_POS_RANGE`,
`E_BATCH_INVARIANCE` (canaries), `E_DEAD_ROWS`, `E_DUPLICATE_CLIPS`, `E_WAIVE_INVALID` (a waiver of a non-waivable code
or without a reason). `E_POS_SHAPE` is structural (missing pos, shape, non-finite values, a stream column outside `-1` /
`[0, len(streams))`, constant pos differing across shards); `E_POS_RANGE` is a coordinate `t`/`y`/`x` outside `[0,1] ∪ {-1}`.
Notes: `N_ELIGIBILITY` (the
verdict → `_meta.json`), `N_CANARY_DRIFT` (canary drift within tolerance, recorded in `_meta.json`
`canary.max_rel_l2`; nothing to do), `N_RULE_CUSTOM` (a `custom` temporal or spatial rule: the
opt-out, recorded in `rules_checked`; nothing to do). Warnings: `W_LOOSE_TOLERANCE`, `W_RULE_UNCHECKED`
(only a checkable rule on a row whose needed fact is null),
`W_COVERAGE`, `W_MEDIA_FP`, `W_FP16_RANGE`, `W_CONSTANT_CLIP`, `W_SIDE_EFFECT`, `W_SIZE` ("probe
will use memmap backend; request scratch"), `W_WEIGHTS_UNFINGERPRINTED`, `W_SOURCE_SHA`, `W_WAIVED` (a waived error: `"<CODE>: <message> — waived: <reason>"`; replaces `W_DEAD_ROWS` / `W_DUPLICATE_CLIPS`),
`W_STALE_TMP`.

### Waivers

A check that is wrong for a model is waived with a reason, never edited out: the bridge's `describe()["waive"] = {CODE:
reason}` (`{"*": reason}` = all waivable; part of `describe_sha256`), or an operator's `features validate --waive
CODE=reason` / `probe run --waive` / matrix entry `waive: {CODE: reason}` (`all=reason`; no hash changes). A waived error
is `W_WAIVED`; `bridge check` records `waived` in the profile (and `plan` accepts a profile whose invariance numbers a
waiver covers), validate in `_meta.json` `waived: [{code, reason, source: describe|cli, count}]`, the probe in
`result.json` `waived`, the results store in `configs[key].waived`; `report build` marks the row `†` with a footnote —
never greyed, excluded or flagged.

| waivable | fires in |
|---|---|
| `E_TIMES_OUT_OF_MEDIA`, `E_RULE_MISMATCH`, `E_POS_RANGE` | validate, bridge check (smoke set) |
| `E_BATCH_INVARIANCE` | validate (canaries), bridge check (alone / order passes) |
| `E_RERUN_DRIFT`, `E_CONSTANT_FEATURES` | bridge check |
| `E_DEAD_ROWS`, `E_DUPLICATE_CLIPS` | validate |
| `E_ARGS_UNCONSUMED` | bridge check / extract start, validate |

Never waivable (`E_WAIVE_INVALID`: exit 2 in a `describe()`, exit 1 on the command line or in a matrix): structural
`E_FORMAT`, `E_NA_AND_SHARDS`, `E_SHARD_SET`, `E_RANGE_TILING`, `E_MEMBERS`, `E_ROWS`, `E_TAR_SHA`, `E_POS_SHAPE`,
`E_FINITE`; identity `E_HASH_MANIFEST`, `E_HASH_INPUT`, `E_DESCRIBE_MISMATCH`, `E_CONFIG_DRIFT`; `E_TEMPLATE_UNFILLED`;
every other error code; warnings and notes need no waiver.

### `probe`

`E_NO_FEATURES` (exit 3) · `E_SCRATCH_SPACE` (exit 3) · `E_EXTERNAL_MISSING` /
`E_EXTERNAL_VERSION` (exit 3, message = the pip line from `benchmark.yaml[external]`, checked
before training) · `E_META_MISSING` · `E_VALIDATION` · `E_FORMAT` (a feature format this yavih does not read) ·
`E_HASH_MISMATCH` (role splits extracted under different benchmark hashes) · `E_HASH_MANIFEST` /
`E_HASH_INPUT` (features extracted for another benchmark input side; re-extract) · `E_HASH_PROTOCOL`
(a plan-time `not_applicable` unit planned under another benchmark protocol: re-run `plan build`) ·
`E_LEAKAGE` · `E_COVERAGE` (`--allow-low-coverage` → `W_COVERAGE` in the result) · `E_FOLDS` ·
`E_METRICS_INCOMPLETE` · `E_NONSTANDARD_CONFIG` (exit 1) · `E_RESULTS_ROOT` (exit 1, before training: the run dir is not `<runs root>/<benchmark>/<model>/<reading>` and no `--results` / `YAVIH_RESULTS` / `[results] root`) · warning `W_NONDETERMINISTIC` in `result.json` (ops without a deterministic kernel ran; `probe_fit.json` `determinism` names them) · deadline / preemption (exit 4,
`probe_ckpt.pt` written, resume on rerun).

### `plan` / `exec` / `report`

`E_MATRIX` malformed `matrix.yaml` (exit 1) · `E_PROFILE` profile missing, malformed, disagreeing
with its entry or without passed hard checks (exit 1) · `E_UNIT_<STATUS>` the exec envelope's error
for a non-zero unit · `E_EXEC` bad selection (`--index` out of range, `--range`, `--slots` with
more slots than inherited GPUs, a `--plan-sha` mismatch; exit 1) · `E_DEADLINE_SPEC` an unusable
`--deadline` (exit 1) · `E_LAYOUT` the unit's `--shard i/n` disagrees with the split's frozen layout:
re-run `plan build` (exit 1; `plan build` itself refuses a matrix `shards` pin that disagrees with a
frozen layout) · `E_SHARD_CLAIMED` another live task holds the shard's claim (exit 3) ·
`E_CLAIM_LOST` the claim was taken over while this writer ran; nothing published (exit 4) · `E_POPULATION_DIFFERS` report refuses a comparison whose eval `ids_sha256`
differ below `min_eval_coverage` unless `--allow-coverage-diff` · `plan build` refuses two
entries resolving to one `(benchmark, model, reading)` and a profile without `model`/`reading`
(exit 1) · report warnings `W_BRIDGE_CHANGED` (one `(benchmark, eval_hash, model, reading)` row holds
results of several `describe_sha256`: the newest configuration is shown, the others under
`--detail`; a configuration whose probe output a re-run in the same run directory moved to
`probe/_stale/` counts, marked `moved_aside`), `W_FEATURES_DIFFER` (same configuration, several feature sets: the newest is shown),
`W_WEIGHTS_UNFINGERPRINTED` (listed once) · `--join vggsounder` refuses readings of different
`eval_hash`, `protocol_hash` or eval ids (`E_JOIN_MIXED`, `E_JOIN_POPULATION`), routings that are not
audio / video / both (`E_JOIN_ROUTING`), readings of different `model` names (`E_JOIN_MODEL`) or
probes (`E_JOIN_PROBE`) (exit 2) and a missing package (exit 3); different weights fingerprints are
the warning `W_JOIN_WEIGHTS`.


### Every command (the envelope's fallback codes)

An exception without its own code is reported by class: `E_USAGE` (exit 1: configuration, schema, an unknown
benchmark or argument) · `E_CONTRACT` (exit 2: validation, manifest, describe or format errors) · `E_NOT_READY`
(exit 3: data, media or a decoder missing) · `E_DEADLINE` (exit 4: a timeout or `--deadline`) · `E_INTERRUPTED` (exit
4: an interrupt; the probe checkpoints first) · `E_NOT_IMPLEMENTED` (exit 5) · `E_INTERNAL` (exit 5, traceback on
stderr in text mode). `W_INTERNAL`: a non-essential `doctor` probe crashed (reported, not fatal).

### Extract and the probe (beyond the lists above)

Extract: `E_BRIDGE` (exit 5) the bridge raised inside `encode` / `prepare` (a bridge bug is not coverage) ·
`E_FEATURES` `encode()` returned a wrong shape or `pos` for the declared `pos_mode` · `E_PRECISION_DRIFT` torch
precision flags differ after `encode` from the declaration · `E_PREPARE_NONDETERMINISTIC` (bridge check) `prepare()`
twice on one sample differs · `E_ORACLE` (bridge check) the test oracle bridge outside `YAVIH_TESTING=1` ·
`E_READER` (exit 5) the folder's `reader.py` yielded rows out of order or outside the shard range · `E_READ_INTERNAL`
(exit 5) reading the stored bytes raised · `E_WRITE` (exit 5) a row's features or provenance cannot be serialized ·
`E_PREEMPTED` (exit 4) SIGTERM / SIGUSR1: shard abandoned, nothing partial · `W_MISSING_ROWS` (bridge check) sampled
rows with missing or undecodable media. Probe (exit 2 unless noted): `E_READOUT` (exit 1) a `--readout` other than
the benchmark's · `E_PROBE_PY` (exit 1) the folder's `probe.py` does not import · `E_SPLIT_MISSING` a role split
without features · `E_META_STALE` a shard written by another configuration than `_meta.json` describes: re-run
validate · `E_ELIGIBILITY_MISMATCH` splits disagree on eligibility · `E_HASH_PROBE` a planned unit's `probe_hash` /
`eval_hash` differ from the config's · `E_PLAN_PROFILE` (exit 1) a plan-time `not_applicable` unit without a usable
profile · `E_LABELS` a row with not exactly one label on a single-label readout, or a label outside `classes.csv` ·
`E_STREAM` a stream outside `describe()["streams"]` · `E_POS_MISSING` a readout that needs `pos` got none ·
`E_POS_SHUFFLE` (box) the shuffled control's partners carry different token counts · `E_DURATIONS` (seq) tokens with
`pos.t` but no `media.duration_s` · `E_EXTERNAL` (exit 3) the base of `E_EXTERNAL_MISSING` / `E_EXTERNAL_VERSION`.
`plan build`: `E_PLAN` (exit 1) a plan that cannot be built or read (two entries on one run directory, a frozen layout
against a `shards` pin, another plan schema, duplicate ids) · `W_NO_CENSUS` frame streams without a census: shard
sizes are estimated.

### `report`, joins and the registry

Report warnings: `W_NONSTANDARD` a column of non-pinned probe configs (comparable with nothing) ·
`W_POPULATION_DIFFERS` eval `ids_sha256` differ within a column · `W_MISSING_READING` a model lacks an expected
reading in a column · `W_FOLDER_SOURCE_DIFFERS` results in one column scored by different versions of the folder's
`probe.py` / `reader.py` (`benchmark.folder_source`; a result without one is unknown, not different): comparable,
listed for diagnosis. `--join`: `E_JOIN_USAGE` (exit 1) no `<reading>=RUN` pairs · `E_JOIN_UNKNOWN` (exit 1) no such
benchmark · `E_JOIN_UNSUPPORTED` (exit 1) the folder defines no join · `E_JOIN_MISSING` a run, seed or DONE result
not found · `E_JOIN_AMBIGUOUS` a run holding several probe columns (name one) · `E_JOIN_STATUS` a reading whose
status is not `ok` · `E_JOIN_SHAPE` saved eval outputs disagree with the eval rows. Registry (`registry export` /
`check`, errors exit 2): `E_REGISTRY_SCHEMA` an entry or protocol payload that does not parse, is another schema,
or records no tolerance (also a bad `--org`, a non-ok or join entry at export) · `E_REGISTRY_ID` a file not filed
under its id · `E_REGISTRY_HASH` `eval_hash` ≠ `H(protocol_hash, probe_hash)` · `E_REGISTRY_PROTOCOL` a protocol
payload that does not recompute `protocol_hash` · `E_REGISTRY_CONFORMANCE` conformance not passed (skipped or
unverified is refused) · `W_REGISTRY_NONSTANDARD` a non-standard entry (stored, never ranked) ·
`W_REGISTRY_COVERAGE` eval coverage below `min_eval_coverage` (shown partial, never ranked) ·
`W_REGISTRY_UNVERIFIED` exported although conformance did not pass here (`check` refuses it) ·
`W_REGISTRY_SKIPPED` a results-store entry export skipped (a join, a non-ok or non-standard entry).

### Benchmark folders: `bench lint`, `bench conformance`, `bench trial`, `bench verify`, fetched copies

`bench lint` (errors exit 2), by check: `yaml` `E_YAML` unreadable, `E_SCHEMA` schema violation, `E_NAME` name ≠
folder · `hashes` `E_CHANGELOG_HASH` · `hashed` `E_HASHED_PROSE` · `budget` `E_MANIFEST` unreadable manifest,
`E_BUDGET` `budget.clips` ≠ frozen rows, `E_CURATION` `manifest.upstream.curation` missing or its sums wrong ·
`header` `E_HEADER` no `header` block, `W_HEADER_STALE` header image not rendered from the current header ·
`readme` / `setup` `E_README` / `E_SETUP` headings, header line, links, pins; `E_README_HASH` a quoted
`protocol_hash` that is not current · `external` `E_EXTERNAL` a pin without a full `pip install` line · `obtain`
`E_OBTAIN` · `capability` `E_CAPABILITY` missing or outside the vocabulary, `W_SUITE_CAPABILITY` a shipped suite
lists the folder under a capability it does not carry · `template` `E_TEMPLATE` no literal `model`, or a `reading`
outside a | v | av · `citations` `E_CITATION` a citation a pip install cannot open, `E_NOTES_LINK` · `imports`
`E_IMPORT` `reader.py` / `probe.py` need torch or an external package at import · `todo` `W_TODO` · `E_LINT_CRASH` a
check crashed. `bench conformance` and the lint check `conformance`: `E_CONFORMANCE_MISSING` no
`conformance/expected.json` (record one) · `E_CONFORMANCE_STALE` recorded at another `protocol_version` or
generator · `E_CONFORMANCE_DRIFT` a value outside its tolerance at the same version (names the first six) ·
`E_CONFORMANCE_SCHEMA` the file does not parse, is another schema or records another benchmark ·
`E_CONFORMANCE_UNSUPPORTED` a readout without a population, or a scoring refusal unrelated to an external
package · `W_CONFORMANCE_SKIPPED` a pinned external package is absent or fails the folder's guard: nothing
compared. `bench trial`: `E_TRIAL` (exit 2) a stage failed · `E_TRIAL_NOT_READY` (exit 3) no stand-in ·
`W_TRIAL_HEADLINE`. `bench verify` warnings: `W_DECODE_SAMPLE` sampled rows that do not decode · `W_CURATION`
curation sums do not hold · `W_TARGETS` / `W_DERIVED` derived targets or files missing or of another sha256 ·
`W_EXTERNAL` an external package missing or off its pin (probe env). A copy under `$YAVIH_HOME/benchmarks`:
`E_HOME_BENCH_STALE` (exit 1; `W_HOME_BENCH_STALE` with `--lenient`) the packaged folder changed since it was
fetched · `W_HOME_BENCH_UNSTAMPED` no fetch stamp · `W_HOME_READONLY` `$YAVIH_HOME` not writable (caches not kept).
`suite`: `E_SUITE` (exit 1) an unknown or invalid suite, `W_SUITE_INVALID` (`suite list`) a suite file that does
not load. `bench list`: `W_BENCH_LOAD` a folder that does not load (listed, the others still shown).

### `init`, `doctor`, configuration, `skill`

`init`: `E_PROJECT_EXISTS` (exit 1) another project file already governs the directory. `doctor`:
`W_PROJECT_NESTED` a second project file in the repository · `W_PROJECT_STRAY` a matrix outside the project folder ·
`W_PROJECT_VERSION` the installed yavih outside `[project] yavih` · `W_RUNS_SIZE` the project-default runs root over
50 GB (move it to scratch with `[runs] root`) · `N_PROJECT` no project folder, a development
build or a capped scan · `W_DATA_ROOT` a data root that does not resolve or exist (`N_DATA_ROOT_PREPARE` for a
`prepare_only` root) · `W_MEDIA_READ` / `W_MEDIA_INFO` the media read sample or the media-info table failed ·
`W_TOOL_PIN` a prepare tool off its pinned build · `W_EXTRA_MISSING` an extra incomplete · `W_TORCH_TOUCHED` pip may
have replaced the model env's torch · `W_DECODER` a decoder probe failed · `W_SELFTEST` `--selftest` failed ·
`W_CONFIG` / `W_CONFIG_MISSING` the configuration does not load, or names a file that does not exist ·
`W_LAUNCHER_UNSET` (`plan init`) no launcher given or configured. Configuration (exit 1 strict; warnings with
`--lenient`): `W_CONFIG_UNKNOWN` an unknown table or key · `W_CONFIG_TYPE` a value of the wrong type ·
`W_CONFIG_UNKNOWN_BENCH` / `W_CONFIG_UNKNOWN_ROOT` a benchmark or data root nothing declares ·
`W_CONFIG_UNSET_VAR` an unset `$VAR` · `W_CONFIG_ENV` an unknown `YAVIH_*` variable · `W_CONFIG_SITE` `YAVIH_SITE`
names no table (or several sites match: a warning). `skill`: `E_SKILL_MISSING` (exit 5) the packaged skill is
missing · `W_SKILL_INCOMPLETE` it has no SKILL.md.

## Results store

`<results_root>/<model>/results_<timestamp>.json` per probe invocation (written after DONE;
never read to decide anything; one directory per model NAME, every reading inside, entries keyed
`<benchmark>:<reading>`) and, with `--log-samples`, `samples_<benchmark>_<reading>_<timestamp>.jsonl`.
Root, decided before training: `--results DIR` → `YAVIH_RESULTS` → config `[results] root` → `yavih-bench/results`
when a project is discovered → `results/` next to the runs root of a standard run dir
`<runs root>/<benchmark>/<model>/<reading>`; any other layout is refused (`E_RESULTS_ROOT`, exit 1), never guessed. `--no-results` skips writing. `report build --merge-results` writes one merged file per
model (newest per `(benchmark, reading, eval_hash)`); `--join vggsounder` writes the key
`vggsounder:a+v+av` into the same `<results_root>/<model>/`. Keys: references/readouts.md.

## Configuration (`yavih.config`)

One `Settings` object per process, built from these layers (later wins, key by key):

1. built-in defaults and the `benchmark.yaml` root defaults (`${data.root}`, `${datasets.<name>}`);
2. the user file `$XDG_CONFIG_HOME/yavih/config.toml` (else `~/.config/yavih/config.toml`), then its
   active `[site.<id>]` table;
3. the project file: `--config FILE`, else `$YAVIH_CONFIG`, else the nearest `yavih-bench/yavih.toml` (else
   `yavih.toml`) walking up from the current directory, stopping at the git root (one project per repository);
   then its active `[site.<id>]` table;
4. environment variables (below);
5. CLI flags `--config`, `--runs-root`, `--results`, `--data <bench>.<root>=PATH` (also `root=PATH`,
   `datasets.<name>=PATH`), `--lenient` — before or after the subcommand. They are exported to the
   environment of the children a command starts (`exec` → `extract` / `probe run`).

```toml
# ~/.config/yavih/config.toml (user: personal overrides) or yavih-bench/yavih.toml (project); every key is optional
launcher = "/envs/mymodel/bin/python"   # default launcher of extract units (a python path or a prefix command); top-level keys come before any table
[project]                               # project file only
yavih = ">=1,<2"                        # the yavih the project expects; doctor warns W_PROJECT_VERSION, never fails
# bridges = "bridges"                   # relocate a project directory; also matrices, suites, benchmarks; subprojects = [...]
[launchers]
vjepa = "/envs/vjepa/bin/python"        # a matrix entry's `launcher: vjepa`; per site under [site.<id>.launchers]
[data]
root = "/data/datasets"                 # ${data.root}
[data.in1k]                             # one table per benchmark (name or alias); keys = its declared roots
media_root = "$GROUP/imagenet2012"      # ~ and $VAR expand; a relative path is relative to THIS file
[datasets]
vggsound = "/data/vggsound"             # ${datasets.vggsound}: vggsounder, vggss_box and avsync share it
[runs]
root = "/scratch/me/yavih-runs"         # default yavih-bench/runs (ignored by git); a relative path is relative to THIS file
[results]
root = "/data/yavih-results"            # default yavih-bench/results (committed), else <runs root>/../results
[scratch]
root = "/tmp/yavih"                     # staged shards; default SLURM_TMPDIR; never TMPDIR
[home]
root = "~/.yavih"
[resources]                             # site knobs only; nothing here changes a number
ram_budget_gb = 24                      # else min(16, 0.45 x the SLURM / cgroup memory limit)
cpus = 8                                # else SLURM / PBS / affinity / cgroup cpu.max / os.cpu_count
decode_workers = 7                      # else cpus - 1 for an allocation, cpus // 2 on a whole machine
[site.cluster]                          # chosen by YAVIH_SITE=cluster or a hosts glob
hosts = ["gpu*", "*.cluster.example.org"]
[site.cluster.data]
root = "/scratch/<user>/datasets"
```

`${datasets.X}` is `[datasets] X`, else `${data.root}/X`. A data root resolves as: `Benchmark.load(data=…)`
→ `--data` → `YAVIH_DATA__<BENCH>__<ROOT>` → project file → user file (site tables first) → the yaml
default. The schema is strict: an unknown table or key, a near miss of a benchmark name (`[data.ESC50]`,
`YAVIH_DATA__ESC_50__…`), a root the benchmark does not declare, an unset `$VAR`, or an unknown `YAVIH_*`
variable is exit 1 with the list of problems. `--lenient` / `YAVIH_LENIENT=1` turns them into warnings
(for a HOME shared by installs of different yavih versions). A `[data.<name>]` table naming no known
benchmark only warns (a folder loaded by path may carry any name; its keys are checked when it is loaded).
`yavih config show [--bench B]` prints every value with its source; `doctor` embeds it.
`$YAVIH_HOME/data_status/<b>-<roots digest>.json` is keyed by the resolved roots, so sites sharing one HOME
keep their own presence and `verified` stamps; a read-only HOME warns once (`W_HOME_READONLY`) and
caches are not kept.

## Environment

`YAVIH_HOME` (`[home] root`; default `~/.yavih`: fetched benchmarks, `data_status/`, `media_info/`) ·
`YAVIH_RUNS` (`[runs] root`; default `yavih-bench/runs`, else `./runs`, absolute) · `YAVIH_RESULTS` (`[results] root`;
default `yavih-bench/results`, else `<runs root>/../results`) · `YAVIH_SCRATCH`
(staged shards for the probe; `[scratch] root`; else `SLURM_TMPDIR`; never an implicit `TMPDIR`; with none
the probe reads the features in place, `W_SIZE`) · `YAVIH_CONFIG` (the project file,
like `--config`) · `YAVIH_SITE` (the active `[site.<id>]`) · `YAVIH_LENIENT=1` · `YAVIH_LAUNCHER` (the default
launcher, `launcher`) · `YAVIH_BENCH_PATH` (colon-separated directories of out-of-tree benchmark folders) ·
`YAVIH_SUITES` (the project suites directory) ·
`YAVIH_DATA__<BENCH>__<ROOT>=<path>` (one root; upper-case names, double underscores) · `YAVIH_DATA__ROOT`
(`[data] root`) · `YAVIH_DATASETS__<NAME>` (`[datasets] <name>`) · `YAVIH_RAM_BUDGET_GB`
(`[resources] ram_budget_gb`) · `YAVIH_CPUS` (`[resources] cpus`: CPUs this process may use; set by
`exec --cpus` / `--slots` for its children; decode workers `YAVIH_CPUS − 1`; else SLURM / PBS / the affinity
mask / cgroup quota) · `YAVIH_DECODE_WORKERS` (`[resources] decode_workers`) · `YAVIH_VALIDATE_WORKERS` (threads scanning shards in
`features validate`) · `YAVIH_SLOT_GPU` (set by
`--slots`: the slot's device mask, applied after the unit's `env`) · `YAVIH_CLAIM_LEASE_S` (shard-claim
heartbeat lease, default 900 s) · `YAVIH_PROBE_CKPT_EVERY_S` (seconds between probe checkpoints, default
600) · `YAVIH_JSON=1` · `YAVIH_DEADLINE` and `YAVIH_UNIT_ID` (set by `exec` for the child) · `YAVIH_BENCH` (set by `bench prepare` for
the step it runs, with `FOLDER`) · `YAVIH_TESTING=1` (non-pinned `ProbeConfig`; results get `standard: false`). Any other `YAVIH_*`
name is an error (a warning with `--lenient`).
