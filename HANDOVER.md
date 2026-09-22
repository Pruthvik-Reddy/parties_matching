# POC handover

The active implementation is documented in `CODE_MAP.md`; `README.md` contains the only supported commands. The final design agreed in this project overrides older notes that call for a mandatory joint cross-encoder head. The current primary experiment is feature scoring for all proposals with an optional selective cross-encoder for ambiguous shortlisted pairs.

Production quality cannot be claimed until the company workbook is prepared and the untouched `TEST_KNOWN` and `TEST_UNSEEN` results are inspected. The synthetic fixture validates behavior and contracts only.

Before a full company run, confirm the workbook headers in `config.toml`, run the feature baseline, inspect retrieval recall, then enable embeddings, LLM expansion, and cross-encoder training one at a time.

The current baseline uses exact lookup plus word and character TF-IDF retrieval, bounded root/variant shortlists, multi-signal identity scoring, digit/distinctive-token guards, and optional isotonic calibration when the calibration split is large enough. Headline metrics are held-out only; development rows and UNKNOWN predictions are reported separately.
The calibration split is partitioned into target-model, isotonic, and threshold-selection groups when it is large enough. The small synthetic fixture falls back to the fixed threshold and does not establish confidence calibration.

For the verified transfer between computers, copy the complete source bundle and install it with `python -m pip install .`. Do not merge individual changed files into an older copy and do not transfer generated state or model artifacts.
