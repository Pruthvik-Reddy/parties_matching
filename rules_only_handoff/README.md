# Enhanced rules-only party matcher

This folder is a standalone, inference-only extraction of the POC's enhanced
rules path. Give it a CSV of verified parties and a CSV of names to match; it
writes one prediction per unverified row. It does **not** train a model, read
labels, calibrate thresholds, call an LLM, build a parent hierarchy, or emit
POC events.
Python 3.9–3.12 and the packages in `requirements.txt` are required.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python match_csv.py --verified examples/verified.csv --unverified examples/unverified.csv --output predictions.csv
```

For your own files, replace both input paths. The output path must differ from
both inputs. The command recreates the retrieval index from the supplied
unverified CSV each time. The Python API also rebuilds it on each call; a
persistent-index service would be a separate later optimization. There is no
training or calibration command to run.

### Input and output columns

`verified.csv` requires `verified_name`. `verified_id` is recommended; if
omitted, IDs `v1`, `v2`, … are assigned in CSV order. Optional `aliases`
contains `|`-separated, explicitly supplied names. No aliases are inferred.
Each name or alias maps directly to its own verified ID. Duplicate IDs are
errors, and a nonblank `parent_id` is rejected rather than silently ignored.
Review supplied aliases before handoff.

`unverified.csv` requires `unverified_name`. `unverified_id` is recommended;
if omitted, IDs `u1`, `u2`, … are assigned in CSV order. A blank name or
duplicate ID is an error.

`predictions.csv` contains `unverified_id`, `unverified_name`, `decision`,
`verified_id`, `verified_name`, `score`, `reason`, `decision_tier`, and
`matched_mention`. Verified ID/name are blank on `NO_MATCH`. `score` is a
rules score for diagnostics, **not** a calibrated probability. A `NO_MATCH`
can still have a high score because a guard, ambiguity check, or cutoff
rejected the candidate. Input order is preserved.

## Code walkthrough

Start at `match_csv.py`. It validates the CSV headers and rows, then calls
`party_matching_rules.api.match_parties`. The API is also callable directly
with lists of dictionaries; it returns the same prediction dictionaries.

1. **Verified catalog — `catalog.py`.** Each verified row is an independent
   party with an official name and any supplied aliases. There are no edges,
   parent IDs, or global-parent lookups. The catalog tracks when a name or
   alias belongs to multiple verified IDs so matching can detect collisions.
   Nothing is persisted or updated, and no aliases are inferred.
2. **Mentions and retrieval — `domain.py`, `matching.py`.** An unverified name
   is normally one mention. OBO/VIA splits it into separate mentions, each
   matched independently. The code builds an exact-name lookup plus character
   3–5-gram and word 1–2-gram TF-IDF indexes over the unverified mentions.
   It queries these with verified names/aliases, combines ranked hits using
   reciprocal-rank fusion, and caps retained verified IDs and name variants.
   This reverse index mirrors the POC batch algorithm; it is rebuilt for
   each invocation.
3. **Pair scoring — `matching.py`.** Every retrieved mention/name pair gets
   normalized-name, base-name (legal suffix removed), compact-name, character
   similarity, Jaro-Winkler, Levenshtein, token overlap/coverage, TF-IDF,
   acronym and collision features. `rule_identity_score` combines them with
   the POC's fixed weights. Exact official-name hits get a deterministic
   score floor. The scorer contains no learned model.
4. **Verified-party decision — `matching.py`.** Multiple names or aliases
   for one verified ID are grouped; only its strongest variant competes.
   The fixed rules target score factors in the gap from the runner-up.
   Digit and distinctive-token conflicts can veto a non-exact match. A
   close second verified ID triggers ambiguity abstention. Plain names use
   a 0.800 cutoff; connector mentions ordinarily use a 0.933 cutoff. A guarded
   distinctive-containment rule can recover a clear, short official-name
   subset below the plain cutoff. A separate connector exception accepts a
   calculated score below 0.933 only when a single token of at least five
   characters uniquely identifies one verified ID, is present in its
   official name, and passes the score, lexical, margin, and conflict checks.
   It never assigns a fixed score. These fixed defaults came from the POC
   configuration/results and are not recalibrated from the new CSVs.
5. **OBO/VIA — `matching.py`.** Each part gets its own candidates and score.
   The current POC policy prefers the *leftmost valid* part for both OBO and
   VIA when they point to different verified IDs. If the preferred part is just
   below cutoff, the matcher abstains instead of silently taking the other
   part. If only one part has a valid match, it can be selected. Mixed or
   malformed connectors abstain.
6. **Enhanced recovery — `rules_enhancement.py`.** This runs only on rejected
   plain names; it never replaces an accepted baseline match. The connector
   exception above runs during the independent segment decision. Plain-name
   recovery may accept a verified-party-unique short name, safely score a
   trimmed/spacing/plural name view, or recognize a complete
   official name followed by contextual text. Removed text is checked for
   competing-party and brand-like signals. A view can strengthen only a root
   already retrieved for the full name; it cannot invent a new candidate.
   Some internal fields retain the POC term `root`; in this flat package,
   `root_party_id` is always the same as the verified party ID.

## What is fixed, and what can differ from the POC

`MatcherSettings` in `api.py` exposes the fixed cutoffs and retrieval caps to
Python callers. The CLI deliberately uses those defaults; no labels or
calibration data are needed. The verified CSV supplies a flat catalog with
only explicitly supplied aliases. With different verified/unverified
populations, TF-IDF and token-frequency statistics change naturally, so scores and predictions may
differ from a historical POC run. The score should not be presented as a
probability of correctness.

The Python dependencies are NumPy, scikit-learn, sparse-dot-topn and
RapidFuzz. There is no joblib model, sentence-transformers, cross-encoder,
LLM, Excel generator, or POC training dependency in this handoff. A C# team
can invoke the Python CLI or host the Python API as a service; this folder is
not a native .NET/NuGet assembly.

## Test

From this folder:

```bash
python -m unittest discover -s tests -v
```

The source under `party_matching_rules/` is checked in and is sufficient to
run. This handoff is frozen independently of the POC; recipients do not need
the parent repository or its extraction script. Do not rerun the old
`../scripts/build_rules_handoff.py` generator: it predates this flat-catalog
simplification and would overwrite it.
