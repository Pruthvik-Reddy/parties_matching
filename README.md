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

After copying changes to preparation or reporting, run all three commands again. Use `PYTHONPATH=src` so Python uses the copied source files without reinstalling, and `--fresh-state` so the old graph and mappings do not contaminate the cleaned dataset.

```bash
PYTHONPATH=src python scripts/prepare.py --workbook "/path/to/related-parties.xlsx"
PYTHONPATH=src python scripts/train.py
PYTHONPATH=src python scripts/run.py --mode parallel --fresh-state --output outputs/cleaned_exp
```

Use `--limit 1000` for a quick prefix sample or `--ids-file adm_ids.txt` for a targeted set of ADM IDs.

Add `--cross-encoder` to training only after the feature baseline works. Set `[expansion].mode = "azure"` and populate `.env` to call Azure OpenAI. `cache_only` reuses prior expansions without network calls. The LLM receives official names plus temporary request references; authoritative party IDs are attached locally and never supplied by the model.

## Outputs

Each run writes:

- `events.jsonl`: API-shaped accepted mapping events.
- `decisions.jsonl`: every ML-path decision and its evidence; `rules_decisions.jsonl` holds enhanced rules comparison decisions, and `rules_baseline_decisions.jsonl` preserves the previous rules decisions. Neither rules file emits events.
- `predictions.xlsx`: `Summary` compares ML predictions and ground truth by verified root; `Stats` compares held-out ML and rules-only precision/recall/F1; `Detail` has every cleaned unverified row in five columns, including a rules-only prediction.
- `data/prepared/cleaning_report.json` and `excluded_rows.jsonl`: counts and source-row references for exact duplicates, verified self-rows, and conflicting labels set aside during preparation.
- `run_metrics.json`: full held-out, split, case and confidence metrics.
- `rules_changes.csv`: only rows where the enhanced rules prediction differs from the previous rules prediction, with the label and correctness for review.
- `diagnostics.json`: compact retrieval funnel, connector cohorts and error buckets.
- `run_stats.json` and `analysis.md`.

Evaluation should use `--fresh-state`. Incremental runs omit that flag and reuse `state/<account>/graph.json` plus `mappings.jsonl`.

## Temporary rules diagnosis

For the rules-only recall investigation, opt in to a sidecar candidate trace, then summarize it separately:

```bash
PYTHONPATH=src python scripts/run.py --mode parallel --fresh-state --diagnose-rules --output outputs/rules_diagnostic
PYTHONPATH=src python scripts/analyze_candidates.py --run outputs/rules_diagnostic
```

`rules_candidate_trace.jsonl` captures missed labeled **plain-name** rows, including the expected root's final rules rank/score/guard and the top five candidates. `rules_diagnostics.md` gives calibration/test failure counts and examples; `rules_diagnostics.csv` supports row-level review. OBO/VIA and unknown-label rows are intentionally outside this first audit. These files contain party names and stay local with the other confidential outputs. The flag is off by default and does not alter match decisions or events. The hook and module are marked `TEMPORARY DIAGNOSTIC` for removal after the rules investigation.

## Enhanced rules-only comparison

The rules path first calculates its previous decision. With `decision.rules_enhanced_enabled = true`, only rejected plain-name rows are reconsidered using guarded name views, a two-way soft-token comparison, and a verified-root-unique short-name rule. A supplementary core-name anchor checks remaining rejected rows where a complete official name begins the raw name and is followed by extra context; it requires an unambiguous existing winner and checks the remainder for competing parties. It reuses the roots retrieved for the original full name; no new root or company-specific alias is inserted. Previously accepted rules decisions stay intact. This rules enhancement does not change connector or ML decisions; the separate left-preferred OBO policy described below affects both ML and rules. The Stats sheet compares enhanced, enhanced-before-core-anchor, and previous rules results from the same run; `rules_changes.csv` lists changed rows. The before-core-anchor row excludes the new anchor promotions, so its difference from the final row measures their incremental effect under the new connector policy. `rules_baseline_decisions.jsonl` preserves the previous rules decisions, while `rules_decisions.jsonl` contains the enhanced results. Set `rules_enhanced_enabled = false` to restore the previous rules output. This code-only change does not require retraining; rerun `scripts/run.py` against the existing prepared data and model artifact.

## Important semantics

- Only verified parties are graph nodes.
- Expansion candidates retrieve ADM names for their owner. `suggested_parent` never does.
- Low-confidence LLM candidates and parents are discarded using the configurable expansion thresholds.
- OBO and VIA segments are retrieved and scored independently. There is no fixed-confidence short-name override; short forms must pass the same score and guard checks as other non-exact matches. Unique exact-name hits retain their existing deterministic score floor. `run_stats.json` records whether trained or fallback scoring was used.
- Distinct roots use the leftmost valid OBO or VIA segment, preserving that segment's confidence. If the preferred leftmost segment is just below cutoff, the run abstains instead of automatically emitting the other segment. Mixed or malformed connectors abstain. Set `connector_policy = "legacy"` to compare with the earlier pooled behavior.
- Remaining generic heuristics include the unique exact-name score floor, configured ambiguity/near-cutoff margins, and fallback scoring when a trained model is unavailable. These are not learned probabilities; check the scorer modes in `run_stats.json` and validate changes on held-out data.
- Proposals are grouped by global root before distinct roots compete.
- Retrieval is bounded by `max_roots_per_mention` and `max_variants_per_root`; cap-hit statistics are written to diagnostics.
- Headline precision and recall use only untouched `TEST_KNOWN` and `TEST_UNSEEN` rows. Training and calibration results remain visible but are not mixed into the headline.
- Evaluation resolves each workbook canonical label through the verified graph before comparing it with the emitted global parent.
- OBO/VIA, no-OBO, no-VIA and plain-name cohorts are reported separately.
- `UNKNOWN` rows appear in predictions but not supervised metrics.
- Preparation uses exact raw-name and canonical-label text for duplicate and conflict checks; it does not normalize these labels or infer a merge from name similarity. An exact raw-name = canonical-name row is treated as a verified self-row, retained in the catalog but excluded from unverified matching and evaluation. Duplicate raw+label pairs collapse to their first row. Raw names carrying two distinct known labels are set aside for review. The source workbook stays unchanged.
- Detail shows unverified name, ML match (or `NO_MATCH`), graph-root label, ML result, and rules-only prediction. Whole-row shading marks correct matches green, no match yellow, wrong matches red, and unscored rows gray. Rules-only has no trained identity/target model, calibration, or cross-encoder; its enhanced path is isolated from ML and keeps a separate cutoff. Summary and events remain ML-based. IDs and scoring evidence remain in the JSONL files, not Excel. Summary includes unknown-label predictions but does not count them as correct; long alias lists spill into continuation rows without repeating counts. `run_stats.json` and `analysis.md` include stage timings and process-memory snapshots; `end_to_end_seconds` includes workbook generation.
- The rules-only short-name containment experiment and the preserved blanket-cutoff option are documented in [EXPERIMENTS.md](EXPERIMENTS.md). Stats shows the rules-only result and its no-containment baseline from the same run. Disable the rule with `rules_containment_enabled = false`; `rules_plain_threshold` remains independently adjustable.
- Rows that already have a verified parent remain in predictions, but are excluded from matching, training, and supervised metrics.
- Normal matching never reads `labels.jsonl`; only reporting and training do. The opt-in temporary diagnostic trace reads held-out labels solely to record missed candidates, never to choose matches.
- A matcher artifact is rejected when it was trained against a different prepared workbook.
