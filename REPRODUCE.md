# SQL Repair Agent - source export and reproduction

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

PYTHONPATH="$PWD/submission" python -m mp3_sql.predict \
  --split test --method submission/submission.py --output outputs/test.json
python -m mp3_sql.score --split test \
  --predictions outputs/test.json --output outputs/test-score.json

PYTHONPATH="$PWD/submission" python -m mp3_sql.predict \
  --split validation --method submission/submission.py --output outputs/validation.json
python -m mp3_sql.score --split validation \
  --predictions outputs/validation.json --output outputs/validation-score.json
```

On Windows use `.venv\Scripts\python.exe` and set `PYTHONPATH` to the
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
