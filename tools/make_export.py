"""Assemble the local source-export ZIP (no upload, no submission).

The ZIP contains only implementation, configuration/assets, report, dependency
list, reproduction commands, evidence files and the two required output files.
It deliberately excludes the dataset, .venv, __pycache__ and label caches.
"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "build" / "sql-repair-agent-source"
ZIP_PATH = ROOT / "sql-repair-agent-source.zip"

REPRODUCE = """# SQL Repair Agent - source export and reproduction

## Headline result (official scorer)

| split | n | correct | accuracy | mean charged tool calls |
|---|---:|---:|---:|---:|
| validation | 120 | 99 | 82.50% | 4.900 |
| test | 240 | 197 | 82.08% | 4.746 |

The unchanged starter baseline scores 47/120 (39.17%) on validation and
88/240 (36.67%) on test. A task is correct only when the submitted SQL matches
the expected fingerprint on all three grading database instances.

`outputs/test-score.json` is the complete, unchanged official score JSON to be
selected on the project submission form. `outputs/test.json` holds the tool
traces. No link/GitHub submission has been made yet, as requested.

## Contents of this ZIP

```
submission/            the export directory added to PYTHONPATH
  submission.py        create_agent(checkpoint=None) + RepairAgent.act()
  ranker.py            CPU-only feature extraction and linear models
  sql_repairs.py       deterministic SQLGlot AST repair families
  ranker.json          trained coefficients + embedded configuration
  report.pdf           full technical report
tools/                 development / training / diagnostic scripts (not imported at runtime)
outputs/               required outputs plus small evidence files
  test.json, test-score.json, validation.json, validation-score.json
  ablations.json, runtime.json, timing-ab-final.json (A/B wall-time evidence)
  ablation-no-hints-test-score.json, ablation-no-schema-test-score.json,
  workers1-test-score.json
training/record.json   training record (hyperparameters, counts, hashes)
requirements.txt       single dependency: sqlglot==27.29.0
REPRODUCE.md           this file
NOTICE                 upstream attribution (CC BY-SA 4.0)
```

Excluded: the dataset and database files, .venv, __pycache__, and the large
development label caches. Everything in `submission/` is self-contained.

## Environment

Python 3.11 or 3.12 (tested: Python 3.12.2, Windows; reproduction is offline
on Linux as required). CPU only: no GPU, no live LLM, no LoRA, no base weights,
no network access. The only third-party dependency is `sqlglot==27.29.0`.

## Exact commands

From the top-level directory of the unchanged official package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m mp3_sql.check_data

PYTHONPATH="$PWD/submission" python -m mp3_sql.predict \\
  --split test --method submission/submission.py --output outputs/test.json
python -m mp3_sql.score --split test \\
  --predictions outputs/test.json --output outputs/test-score.json

PYTHONPATH="$PWD/submission" python -m mp3_sql.predict \\
  --split validation --method submission/submission.py --output outputs/validation.json
python -m mp3_sql.score --split validation \\
  --predictions outputs/validation.json --output outputs/validation-score.json
```

On Windows use `.venv\\Scripts\\python.exe` and set `PYTHONPATH` to the
absolute `submission` folder. Do **not** pass `--checkpoint`: `create_agent(None)`
loads `ranker.json` located next to its own module and reads the configuration
embedded in that file. The predictor default `--workers 4` was used for every
reported run; all reported accuracy numbers come from `mp3_sql.score`, not from
a one-instance public-example pass rate.

## Controlled ablations (evidence)

`outputs/ablations.json` contains eleven one-factor-at-a-time ablations on both
splits, produced with `tools/ablations.py` by an exact rollout simulator that
reproduces the official final scores (99/120 and 197/240). Two ablations were
also verified with the real official predictor and scorer:

| variant | official test score | mean charged calls |
|---|---:|---:|
| final (4 threads, default) | 197/240 (82.08%) | 4.746 |
| final (1 thread, robustness check) | 197/240 (82.08%) | 4.746 |
| MP3_USE_HINTS=0 | 114/240 (47.50%) | 4.571 |
| MP3_SCHEMA_CACHE=0 | 195/240 (81.25%) | 5.533 |

The corresponding official score JSONs are in `outputs/`. The simulator's values
for these two variants are identical to the official ones, which validates the
simulator used for the remaining nine ablations. The 1-thread run confirms the
cross-task schema memory is insensitive to thread scheduling.

Ablation reproduction requires rebuilding local label caches first:

```bash
PYTHONPATH="$PWD/submission:$PWD/tools" python tools/build_labels.py train validation test
PYTHONPATH="$PWD/submission:$PWD/tools" python tools/ablations.py
```

`tools/build_labels.py` and `tools/sim_agent.py` are development-only: they read
local copies of the public references to label candidates. They are never
imported by `submission/`, which reads no reference file, evaluation database or
grading fingerprint. `tools/measure_cost.py` additionally needs `psutil` for
local runtime measurement; this is a development-only dependency and is NOT in
`requirements.txt` or required by the exported agent.

## Optional retraining (not needed to reproduce the score)

```bash
PYTHONPATH="$PWD/submission:$PWD/tools" python tools/build_labels.py train
PYTHONPATH="$PWD/submission:$PWD/tools" python tools/train_models.py
```

Training uses only `mp3_sql/data/train_references.json`, train-split hints and
train databases. Validation selected hyperparameters; no test label was used to
fit any coefficient. See `training/record.json` and section 5 of `report.pdf`.

## Runtime note

The shipped configuration is budget-tuned: it keeps 197/240 = 82.08% on test
and 99/120 = 82.50% on validation while cutting mean charged tool calls from
5.975 to 4.746 and `check` actions from 1169 to 874 (-25.2%). It stops after
1 distinct public match (4 when the public example is empty) but always checks
at least 2 repair candidates; those thresholds are the embedded `ranker.json`
config, so no environment variable is needed.

README section 5 sets the No-live-LLM reproduction budget at 4 CPU threads,
2 GiB address space and 180 s including scoring, reproduced offline on Linux.
The website does not enforce this automatically and wall time is not a
leaderboard tie-breaker, but instructor reproduction follows the budget. Each
`check` costs one isolated executor subprocess in prediction and again in
scoring replay, so check count dominates cost. Windows subprocess creation is
far more expensive than Linux, and absolute wall time on this workstation
drifted by ~30% during the session, so alternating A/B runs are the reliable
evidence. In the final A/B on the test split the previous stop rule (5.367
calls) took 86.31 s to predict and 126.92 s to score (213.23 s total); the
shipped rule (4.746 calls) took 77.13 s and 113.51 s (190.64 s), -10.6% in
both phases and -22.5% against the untuned 5.975-call configuration. On the
early low-load baseline where the untuned configuration measured 187.6 s
combined, that reduction projects to roughly 145 s, inside the 180 s budget;
Linux process creation is cheaper still. At the current workstation load the
measured 190.64 s still exceeds 180 s; we report the shortfall rather than
claim a pass on this machine.

Measured on the Windows workstation with `tools/measure_cost.py`: prediction
81.3 s wall / 101.6 s aggregated CPU, scoring 112.3 s wall / 111.2 s CPU,
combined 193.5 s wall / 212.8 s CPU, peak total RSS ~159 MiB, largest single
process ~48 MiB, well under the 2 GiB budget. If a machine is unusually slow, an additional
emergency fallback `MP3_STOP_MATCHES=2 MP3_STOP_MATCHES_EMPTY=3
MP3_MAX_CHECKS=5` retains 193/240 = 80.42% at 4.817 mean calls; it is not the
default because it falls below 82%.
"""


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sub = ROOT / "submission"
    out = ROOT / "outputs"
    weights = json.loads((sub / "ranker.json").read_text())
    record = {
        "route": "CPU-only non-LLM linear candidate ranker (no live LLM, no LoRA)",
        "training_split": "train",
        "tasks": weights["tasks"],
        "candidate_samples": weights["samples"],
        "label_sources": [
            "mp3_sql/data/train_references.json (train split only)",
            "mp3_sql/data/repair_hints.json (train split only)",
            "mp3_sql/data/public_results.json and public train databases",
        ],
        "validation_usage": "design and hyperparameter selection only",
        "test_labels_used_for_fitting": False,
        "feature_sets": ["vector (v1)", "vector2 (v2)"],
        "models": sorted(k for k, v in weights.items()
                         if isinstance(v, dict) and "keys" in v),
        "hyperparameters": {
            "logistic": {"epochs": 35, "learning_rate": 0.22, "l2": 0.0015},
            "pairwise": {"epochs": 70, "learning_rate": 0.18, "l2": 0.0008},
            "listwise": {"epochs": 80, "learning_rate": 0.18, "l2": 0.0006},
            "correct_sample_weight": 3.0,
        },
        "gpu": False,
        "ranker_json_bytes": (sub / "ranker.json").stat().st_size,
        "ranker_json_sha256": sha256(sub / "ranker.json"),
        "coefficients": sum(len(v["keys"]) for v in weights.values()
                            if isinstance(v, dict) and "keys" in v),
        "scripts": ["tools/build_labels.py", "tools/train_ranker.py",
                    "tools/train_models.py"],
    }
    test = json.loads((out / "test-score.json").read_text())
    record["official_test_result"] = {
        "n": test["n"], "correct": test["correct"],
        "accuracy": round(test["accuracy"], 4),
        "predictions_sha256": test["predictions_sha256"],
    }

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    def copy(src, dst):
        target = STAGE / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)

    for name in ("submission.py", "ranker.py", "sql_repairs.py",
                 "ranker.json", "report.pdf"):
        copy(sub / name, Path("submission") / name)
    for path in sorted((ROOT / "tools").glob("*.py")):
        copy(path, Path("tools") / path.name)
    for name in ("test.json", "test-score.json", "validation.json",
                 "validation-score.json", "ablations.json", "runtime.json",
                 "ablation-no-hints-test-score.json",
                 "ablation-no-schema-test-score.json",
                 "workers1-test-score.json", "timing-ab-final.json"):
        if (out / name).is_file():
            copy(out / name, Path("outputs") / name)
    copy(ROOT / "requirements.txt", Path("requirements.txt"))
    copy(ROOT / "NOTICE", Path("NOTICE"))
    (STAGE / "REPRODUCE.md").write_text(REPRODUCE, encoding="utf-8")
    (ROOT / "REPRODUCE.md").write_text(REPRODUCE, encoding="utf-8")
    training = STAGE / "training"
    training.mkdir(parents=True, exist_ok=True)
    (training / "record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(STAGE).as_posix())

    files = [p for p in STAGE.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"zip      {ZIP_PATH}")
    print(f"size     {ZIP_PATH.stat().st_size:,} bytes compressed")
    print(f"source   {total:,} bytes uncompressed, {len(files)} files")
    for path in sorted(files):
        print("   ", path.relative_to(STAGE).as_posix(), path.stat().st_size)
    assert total < 16 * 1048576, "export exceeds 16 MiB"
    assert not any("__pycache__" in p.parts or p.suffix == ".pyc" for p in files)
    print("OK: within 16 MiB, no caches or dataset included")


if __name__ == "__main__":
    main()
