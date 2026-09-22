# Verified-party matching POC

This local POC derives an ADM-shaped dataset from the `related-parties` workbook, builds a persistent verified-party graph, searches ADM records from every verified name and expansion, resolves competition at the global root, and writes events plus a complete prediction workbook.

## Setup

Use Python 3.9 through 3.12. Start with the core lexical installation; add the optional integrations only after the baseline works. Public Hugging Face models do not require an account. The first model-enabled run downloads weights unless `embedding_model_path` points to a local copy.

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install .
```

`python -m pip install .` deliberately avoids editable-install limitations in older pip versions such as 21.2.4. If you later replace any source files, run that command again. Do not copy `.venv`, `*.egg-info`, `artifacts`, `data/prepared`, `outputs`, or `state` between computers.

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

Use `--limit 1000` for a quick prefix sample or `--ids-file adm_ids.txt` for a targeted set of ADM IDs.

Add `--cross-encoder` to training only after the feature baseline works. Set `[expansion].mode = "azure"` and populate `.env` to call Azure OpenAI. `cache_only` reuses prior expansions without network calls. The LLM receives official names plus temporary request references; authoritative party IDs are attached locally and never supplied by the model.

## Outputs

Each run writes:

- `events.jsonl`: API-shaped accepted mapping events.
- `decisions.jsonl`: every decision and its evidence.
- `predictions.xlsx`: complete `Summary` and `Detail` sheets.
- `run_metrics.json`, `run_stats.json`, and `analysis.md`.

Evaluation should use `--fresh-state`. Incremental runs omit that flag and reuse `state/<account>/graph.json` plus `mappings.jsonl`.

## Important semantics

- Only verified parties are graph nodes.
- Expansion candidates retrieve ADM names for their owner. `suggested_parent` never does.
- Low-confidence LLM candidates and parents are discarded using the configurable expansion thresholds.
- OBO and via mentions are matched independently; no connector direction is assumed.
- Proposals are grouped by global root before distinct roots compete.
- `UNKNOWN` rows appear in predictions but not supervised metrics.
- Rows that already have a verified parent remain in predictions, but are excluded from matching, training, and supervised metrics.
- The matcher never reads `labels.jsonl`; only reporting and training do.
- A matcher artifact is rejected when it was trained against a different prepared workbook.
