# Verified-party matching POC

This local POC derives an ADM-shaped dataset from the `related-parties` workbook, builds a persistent verified-party graph, retrieves ADM records with exact, word TF-IDF, character TF-IDF and optional embedding indexes, resolves competition at the global root, and writes events plus a complete prediction workbook.

For a separate inference-only CSV handoff of the enhanced rules matcher, see [rules_only_handoff/README.md](rules_only_handoff/README.md). It does not require the POC training or reporting workflow.

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
- `predictions_rules_only.xlsx` (opt-in): demo workbook with the same Summary/Stats/Detail structure, but its matches, row colors, and all reported evaluation metrics use the enhanced rules decisions only. It has no ML comparison column or ML results.
- `data/prepared/cleaning_report.json` and `excluded_rows.jsonl`: counts and source-row references for exact duplicates, verified self-rows, and conflicting labels set aside during preparation.
- `run_metrics.json`: full held-out, split, case and confidence metrics.
- `rules_changes.csv`: only rows where the enhanced rules prediction differs from the previous rules prediction, with the label and correctness for review.
- `diagnostics.json`: compact retrieval funnel, connector cohorts and error buckets.
- `run_stats.json` and `analysis.md`.

Evaluation should use `--fresh-state`. Incremental runs omit that flag and reuse `state/<account>/graph.json` plus `mappings.jsonl`.

To make the rules-only demo workbook from a **completed** run, without training or matching again:

```bash
PYTHONPATH=src python scripts/report_rules.py --run outputs/YOUR_RUN_FOLDER
```

This writes only `predictions_rules_only.xlsx` inside that run folder and leaves `predictions.xlsx`, decisions, events, and other reports unchanged. Its held-out recall is correct enhanced-rules matches divided by all scorable held-out rows, so `NO_MATCH` rows count against recall. Use the same `data/prepared` dataset that produced the run.

## Temporary rules diagnosis

For the rules-only recall investigation, opt in to a sidecar candidate trace, then summarize it separately:

```bash
PYTHONPATH=src python scripts/run.py --mode parallel --fresh-state --diagnose-rules --output outputs/rules_diagnostic
PYTHONPATH=src python scripts/analyze_candidates.py --run outputs/rules_diagnostic
```

`rules_candidate_trace.jsonl` captures missed labeled **plain-name** rows, including the expected root's final rules rank/score/guard and the top five candidates. `rules_diagnostics.md` gives calibration/test failure counts and examples; `rules_diagnostics.csv` supports row-level review. OBO/VIA and unknown-label rows are intentionally outside this first audit. These files contain party names and stay local with the other confidential outputs. The flag is off by default and does not alter match decisions or events. The hook and module are marked `TEMPORARY DIAGNOSTIC` for removal after the rules investigation.

## Enhanced rules-only comparison

The rules path first calculates its baseline decision. With `decision.rules_enhanced_enabled = true`, rejected plain-name rows are reconsidered using guarded name views, a two-way soft-token comparison, and a verified-root-unique short-name rule. A supplementary core-name anchor checks remaining rejected rows where a complete official name begins the raw name and is followed by extra context; it requires an unambiguous existing winner and checks the remainder for competing parties. It reuses the roots retrieved for the original full name; no new root or company-specific alias is inserted. Previously accepted plain-name rules decisions stay intact. Separately, `rules_connector_short_name_enabled` lets an independently scored OBO/VIA segment below the ordinary connector cutoff count as valid only when its name is a unique prefix of an official verified name, passes the existing score, lexical, margin, and conflict checks, and identifies one graph root. Single-word mentions must also be a distinctive token across verified roots. A multiword prefix shared by separate verified roots is not automatically assigned to either root; if the preferred segment is below cutoff, the rules path abstains rather than choosing the other segment. `rules_multiword_prefix_enabled = false` reverts only this new multiword extension and preserves the previous single-token, view, and core-anchor behavior. This never changes ML decisions and does not add retrieval candidates. The Stats sheet compares enhanced, enhanced-before-core-anchor, and baseline rules results from the same run; `rules_changes.csv` lists rows changed by the post-baseline plain-name enhancement. `rules_baseline_decisions.jsonl` includes the connector short-name recovery but excludes the post-baseline plain-name enhancement; `rules_decisions.jsonl` contains both. This code-only change does not require retraining; rerun `scripts/run.py` against the existing prepared data and model artifact. It cannot infer that separate verified roots are the same company; those relationships must be established in the verified graph before a shared short name can resolve unambiguously.

The optional `decision.rules_second_pass_enabled = true` runs after the enhanced rules, only for still-unmatched plain names. It scans already-retrieved official candidates across roots for a complete official name with extra context, a root-unique short name, a legal-suffix variation, or one small spelling change. It abstains on competing evidence and honors each job's request cutoff. It does not change OBO/VIA, ML decisions, events, or the separate `rules_only_handoff/` package. Set the flag to `false` to restore the previous enhanced-rules result. One run writes `rules_pre_second_pass_decisions.jsonl` and `rules_second_pass_comparison.json`, which compare before/after held-out precision, recall, F1, newly correct matches, and newly wrong matches; the demo workbook reflects the new final rules decisions. The numerical confidence remains the original rules score, not a newly calibrated probability. If the comparison shows poorer precision or F1, turn off the flag and rerun without retraining.

`decision.rules_regional_enabled = true` adds a separate final route for plain names that still have `NO_MATCH`. It requires every meaningful official-name token in the raw name, a lexical TF-IDF signal of at least 0.55, a location term, and no unexplained extra tokens. It refuses a core shared by separate verified roots, a conflicting digit or distinctive token, an exact competing candidate, and any job cutoff above the original rules score. Geography and branch/office words are descriptive evidence, **not proof of common ownership**; the match method says `regional_heuristic_not_proven_ownership`. The vocabulary in `src/party_matching/regional_terms.py` is intentionally explicit and incomplete; unknown terms abstain. Set only `rules_regional_enabled = false` to restore the pre-regional enhanced result, leaving the existing second pass enabled. Each run writes `rules_pre_regional_decisions.jsonl` and `rules_regional_comparison.json` so the new route's held-out additions and precision/recall/F1 effect can be reviewed independently. No retraining is needed. Existing matches from earlier rules stages are never changed by this route.

## Important semantics

- Only verified parties are graph nodes.
- Expansion candidates retrieve ADM names for their owner. `suggested_parent` never does.
- Low-confidence LLM candidates and parents are discarded using the configurable expansion thresholds.
- OBO and VIA segments are retrieved and scored independently. The rules-only unique-short-name exception uses the calculated score and conservative guards; it does not assign a fixed confidence. Unique exact-name hits retain their existing deterministic score floor. `run_stats.json` records whether trained or fallback scoring was used and how many connector short-name matches were made.
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
