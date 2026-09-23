# Verified-party matching POC

This local POC derives an ADM-shaped dataset from the `related-parties` workbook, builds a persistent verified-party graph, retrieves ADM records with exact, word TF-IDF, character TF-IDF and optional embedding indexes, resolves competition at the global root, and writes events plus a complete prediction workbook.

## Setup

Use Python 3.9 through 3.12. Start with the core lexical installation; add the optional integrations only after the baseline works. Public Hugging Face models do not require an account. The first model-enabled run downloads weights unless `embedding_model_path` points to a local copy.

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install .
```

`python -m pip install .` deliberately avoids editable-install limitations in older pip versions such as 21.2.4. If you later replace source files, reinstall or run scripts with `PYTHONPATH=src` to use the new code. Do not copy `.venv`, `*.egg-info`, `artifacts`, `data/prepared`, `outputs`, or `state` between computers.

Optional integrations:

```bash
python -m pip install ".[llm]"  # Azure OpenAI expansion
python -m pip install ".[ml]"   # embeddings/ANN; large download
```

After installing the ML extras, set `embedding_enabled = true` in `config.toml`. It is off by default so the first run is fast and does not unexpectedly download a model.

For editable installation, first run `python -m pip install --upgrade pip setuptools wheel`, then use `python -m pip install -e .`.

## Commands

Synthetic smoke run:

```bash
python scripts/prepare.py --sample
python scripts/train.py
python scripts/run.py --mode serial --fresh-state --output outputs/smoke_serial
python scripts/run.py --mode parallel --fresh-state --output outputs/smoke_parallel
```

Company workbook run:

```bash
python scripts/prepare.py --workbook "/path/to/related-parties.xlsx"
python scripts/train.py
python scripts/run.py --mode parallel --fresh-state
```

For this decision-rule and report update, the existing trained artifact can be reused; the feature schema has not changed. After copying the changed source files, run from the repo root with `PYTHONPATH=src` so Python uses those files without reinstalling. `prepare.py` is only needed again if the workbook or preparation settings changed.

```bash
PYTHONPATH=src python scripts/run.py --mode parallel --fresh-state --output outputs/exp4
```

Use `--limit 1000` for a quick prefix sample or `--ids-file adm_ids.txt` for a targeted set of ADM IDs.

Add `--cross-encoder` to training only after the feature baseline works. Set `[expansion].mode = "azure"` and populate `.env` to call Azure OpenAI. `cache_only` reuses prior expansions without network calls. The LLM receives official names plus temporary request references; authoritative party IDs are attached locally and never supplied by the model.

## Outputs

Each run writes:

- `events.jsonl`: API-shaped accepted mapping events.
- `decisions.jsonl`: every decision and its evidence.
- `predictions.xlsx`: `Summary` groups every accepted raw-name match under its predicted verified party; `Stats` has run metrics; `Detail` has every source row.
- `run_metrics.json`: full held-out, split, case and confidence metrics.
- `diagnostics.json`: compact retrieval funnel, connector cohorts and error buckets.
- `run_stats.json` and `analysis.md`.

Evaluation should use `--fresh-state`. Incremental runs omit that flag and reuse `state/<account>/graph.json` plus `mappings.jsonl`.

## Important semantics

- Only verified parties are graph nodes.
- Expansion candidates retrieve ADM names for their owner. `suggested_parent` never does.
- Low-confidence LLM candidates and parents are discarded using the configurable expansion thresholds.
- OBO and VIA segments are matched independently. A unique, long official-name prefix followed only by generic company descriptors (for example, `Carahsoft` for `Carahsoft Technology Corp.`) gets a narrow deterministic match rule. It does not lower the general cutoff; it is disabled when another verified root shares that prefix or owns the exact short name.
- Distinct roots use the leftmost valid OBO segment or rightmost valid VIA segment. If that preferred segment is just below cutoff, the run abstains instead of automatically emitting the other segment. A uniquely identified short-name match on the preferred side can resolve a conflict directly; other conflicting-root choices still require the calibrated connector gate. Mixed or malformed connectors abstain. Set `connector_policy = "legacy"` to compare with the earlier pooled behavior.
- Proposals are grouped by global root before distinct roots compete.
- Retrieval is bounded by `max_roots_per_mention` and `max_variants_per_root`; cap-hit statistics are written to diagnostics.
- Headline precision and recall use only untouched `TEST_KNOWN` and `TEST_UNSEEN` rows. Training and calibration results remain visible but are not mixed into the headline.
- Evaluation resolves each workbook canonical label through the verified graph before comparing it with the emitted global parent.
- OBO/VIA, no-OBO, no-VIA and plain-name cohorts are reported separately.
- `UNKNOWN` rows appear in predictions but not supervised metrics.
- Detail keeps one `Correct Answer` (the expected global parent), accepted prediction or `NO_MATCH`, a readable result and a small set of review clues. IDs and full scoring evidence remain in `decisions.jsonl`, not Excel. Stats shows unknown prediction count and connector cohorts. Summary lists all accepted matches by predicted verified party, including matches on unknown-label rows; large alias lists continue on additional rows. `run_stats.json` and `analysis.md` include stage timings and process-memory snapshots; `end_to_end_seconds` includes workbook generation.
- Rows that already have a verified parent remain in predictions, but are excluded from matching, training, and supervised metrics.
- The matcher never reads `labels.jsonl`; only reporting and training do.
- A matcher artifact is rejected when it was trained against a different prepared workbook.
