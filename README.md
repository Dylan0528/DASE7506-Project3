# DASE7506 MP3 - SQL Repair Agent (source export)

Official `mp3_sql.score` result: test **197/240 = 82.08%**, validation **99/120 = 82.50%**.

## Artifacts

- `sql-repair-agent-source.zip` - immutable source export (232346 bytes, SHA-256 `f26288689c920b19dc2ce81e7d0479286d3199c84aece99bb8e8ea11b6b7c5a5`)
- `submission/report.pdf` - 7-page technical report
- `REPRODUCE.md` - exact reproduction commands
- `outputs/test.json`, `outputs/test-score.json` - official traces and score (the score JSON is the file to select on the project submission form)

## Reproduce

See `REPRODUCE.md`. CPU-only, no live LLM / LoRA / GPU, single dependency `sqlglot==27.29.0`. The exported agent lives in `submission/`; `tools/` holds development-only scripts that are never imported at runtime.
