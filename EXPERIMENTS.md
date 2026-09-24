# Rules-only recall experiment

## Baseline and reason for the change

On the existing `ml_vs_rules` calibration run, 3,384 plain-name rows were rejected for `INSUFFICIENT_SUPPORT` even though the labeled verified root ranked first. In 3,295 of those, that root was also the provisional winner. This points to the rules confidence gate, but does **not** establish that accepting every low-score winner would be safe. The earlier blanket 0.60-cutoff audit added 2,797 calibration matches, only 2,546 correct (91.0% incremental precision).

## Experiment 1: distinctive short-name containment

The new path applies **only to the rules-only comparison for plain names**. It does not change ML decisions, events, OBO/VIA, training, or the graph. Below the ordinary plain-name cutoff, it may accept an official verified name when the raw name's meaningful tokens are a strict subset of it, at least one shared token is rare in the verified catalog, there are at most two extra meaningful tokens, lexical retrieval agrees, and the normal conflict and competing-root checks pass. The reported confidence remains the original rules score; the rule does not assign a fixed high confidence.

Settings in `config.toml` under `[decision]`:

| Setting | Initial value | Meaning |
| --- | ---: | --- |
| `rules_plain_threshold` | `0.80` | Ordinary rules-only plain-name cutoff; **lower this to run the blanket-cutoff experiment**. |
| `rules_containment_enabled` | `true` | Set `false` for the same rules-only baseline without this experiment. |
| `rules_containment_min_confidence` | `0.40` | Minimum original rules confidence for this narrow path. |
| `rules_containment_min_margin` | `0.08` | Minimum lead over the next eligible verified root. |
| `rules_containment_min_lexical` | `0.50` | Minimum of word or character TF-IDF similarity. |
| `rules_containment_max_token_df` | `3` | A distinctive raw token may appear in at most this many verified names. |

Accepted rows have decision tier `RULES_UNIQUE_CONTAINMENT` in `rules_decisions.jsonl`. The Stats sheet shows rules-only results with the rule and the counterfactual baseline without it **from the same run**; `run_metrics.json` also reports the rule's calibration and held-out match counts and precision. No second full run is needed to measure this rule's incremental effect. These are experimental, uncalibrated rules scores, not probabilities.

To compare fairly, use the same prepared data and `--fresh-state` in separate output folders. No retraining is needed for this change because the existing matcher artifact already stores verified-name token IDF values. Use `PYTHONPATH=src` after copying source files so the updated package is loaded without reinstalling. To revert this experiment, set `rules_containment_enabled = false`; the plain cutoff remains available independently.

Do not claim an 80% recall gain until the held-out results establish it. Next investigate rank-1 ambiguity, lower-ranked correct roots, and retrieval misses separately.
