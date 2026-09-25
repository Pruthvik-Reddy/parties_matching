# Verified-party-first suggestions

## Two-snapshot demo: actual verified-name groups, then expanded verified list

Use this when you want to show how suggestions change as more verified names
become available. It reuses the enhanced POC rules, including OBO/VIA, and
does **not** change the main matcher, saved graph, events, or mappings.

```powershell
python -m suggestions.snapshot_demo --config config.toml --max-families 3 --output-dir outputs/verified_snapshots_1 --xlsx
```

For the nine examples supplied for the demo, use the checked-in list instead:

```powershell
python -m suggestions.snapshot_demo --config config.toml --representatives-file suggestions/demo_representatives.json --output-dir outputs/listed_snapshots_1 --xlsx
```

Every listed name must exactly and uniquely occur in the prepared verified
catalog. The command reports **all** missing names together so you can correct
the list or prepare the full work catalog; it does not silently replace a
missing name with a similar one. If a listed group has no additional verified
names, its initial and expanded states may be identical, which is reported
rather than fabricated. Edit the JSON list to use a different set of real
verified parties.

Without `--representative` or `--representatives-file`, the command picks up
to three actual verified-name
groups in the prepared catalog. Each group must have at least two verified
names sharing a non-generic first meaningful token and at least three eligible
unverified names containing that token; the groups with the most such rows are
selected. Use `--max-families` and `--min-family-unverified` to adjust these
selection limits. For each selected group, the shortest meaningful verified
name is kept as the **initial simulated representative**. Other verified names
in that group are withheld initially and return in the expanded catalog. All
verified names outside the selected groups remain in both snapshots and can
compete for suggestions. The `Scenario` sheet and `scenario.json` list the
groups, representatives, and withheld names. A shared first token is only a
transparent demo-selection heuristic, not proof of ownership or an actual
onboarding order. If no suitable group exists, the command stops and asks for
a fuller prepared catalog or different limits; it never makes up company names.

You can still focus on one known party with `--representative "Exact Current
Verified Name"`. In this explicit mode only, `--withhold-verified "Exact
Current Verified Name"` can add a party with a different leading token to the
initially withheld set, and `--add-verified "New Name"` can simulate a truly
new verified party. Both flags are repeatable.

By default, both snapshots score the **same eligible unverified names** that
contain any selected group token in a parsed segment. This is the quick demo
mode, not a full-dataset recall test; it includes matching OBO/VIA segments. Add
`--all-unverified` to score the entire eligible dataset in both snapshots,
which takes longer. Use a fresh `--output-dir` for each run.

Each of `initial/` and `expanded/` contains `suggestions_current.xlsx`,
`strong_suggestions.csv`, `review_candidates.csv`, and `summary.json`. The
expanded workbook also has a `Changes` tab, and the parent folder has
`changes.csv`. `MOVED` means a strong suggestion switched verified party;
`NEW_STRONG` and `LOST_STRONG` show threshold/ambiguity changes. Unchanged
names remain on the snapshot detail tabs, but are omitted from `Changes`.
Neither run transfers a confirmed mapping. The demo deliberately uses a wider
**review-only** filter than the normal suggestions command (minimum identity
and lexical scores 0.45, up to three close candidates); this may include
false positives and is not a promise of 90% coverage. Strong suggestions
still use the POC's existing acceptance rules and cutoffs. The `Scenario` sheet
reports the share of selected names appearing in either strong or review;
that suggestion coverage is **not** retrieval recall or match accuracy.
The snapshot demo's `--xlsx` uses the POC's existing Python `XlsxWriter`
dependency. It does not require Node.js, `@oai/artifact-tool`, or
`build_workbook.mjs`; that JavaScript file remains for the other suggestions
commands.

## Recommended: enhanced POC rules, scoped for a quick run

`python -m suggestions.poc` is a read-only inference path using the **current
POC rules-only** retrieval, scoring, connector decisions and enabled recovery stages.
It does not run ML decisions, emit events, write mappings, or save the graph.
It reuses the existing `artifacts/matcher.joblib` for the POC's IDF weights and
rules cutoff when available, without running the trained model or requiring
retraining for a newly prepared verified list. If the artifact is missing,
`summary.json` warns that fallback scores may differ from the evaluated POC.
It uses all current verified parties as competing candidates, but can score a
smaller set of unverified names for a quicker demo. The output is one final
current-state snapshot, not a history of previous assignments.

```powershell
python -m suggestions.poc --config config.toml --min-anchor-group-size 3 --max-groups 20 --output-dir outputs/suggestions_poc_demo
```

`--min-anchor-group-size 3` means **more than two raw unverified names sharing
a leading meaningful token** with at least one verified party. It is a cheap
*input selection* proxy, not the number of known or accepted matches to a
verified party. It can omit typos, aliases, reordered names and rows without
the shared first word. `--max-groups 20` then selects the 20 largest such groups.
Use `--focus-verified "Abbott Laboratories"` to focus on one brand's leading
name while retaining all verified parties as competitors. Omit all three
scope options to score every eligible plain unverified name. Narrowing the
input changes the TF-IDF corpus, so scoped scores and results are not a
direct replacement for the full POC's 68% held-out recall measurement.

The command always writes `strong_suggestions.csv`, `review_candidates.csv`,
and `summary.json`. If held-out labels exist, the JSON also reports precision
and recall for **only the selected held-out rows**; these are not the
full-dataset metrics. For an Excel workbook, add `--xlsx`. It automatically
uses the bundled Codex spreadsheet runtime when present; otherwise provide
`--artifact-modules PATH_TO_NODE_MODULES` and, if needed, `--node PATH_TO_NODE`.
The workbook has `Verified parties` with strong and review counts, followed by
`Strong suggestions` and `Review candidates`. All qualifying rows in the
selected scope are listed; neither sheet has a 10–15-row sample cap. The
detail tabs identify `PLAIN`, `OBO`, or `VIA` and show the matched segment and
connector resolution. OBO/VIA use the POC's existing guarded connector policy;
the workbook does not simply choose the highest-scoring segment.

**Strong** means the enhanced POC rules returned `MATCH` for this scoped run.
It is not a confirmed mapping or calibrated probability. **Review** means
the rules rejected the row, but an unguarded retrieved candidate has identity
score at least 0.60 and lexical evidence at least 0.55. At most two candidates
within 0.10 of the leading score appear per unverified name; these limits are
configurable with `--review-score-floor`, `--review-lexical-floor`, and
`--review-max-candidates`. This review filter is an unvalidated starting point,
not a measured precision tier. Rows with no plausible unguarded candidate
remain without a suggestion. For rejected OBO/VIA names, review candidates
come only from the POC's preferred connector segment. Malformed or mixed
connector chains remain abstentions, with no review candidate. Connector
precision was weaker than plain-name precision in the earlier POC analysis,
so inspect these suggestions separately before treating them as matches.

To see reassignment without editing the prepared catalog, run once, then run
again to a *new output directory* with `--add-verified "Microsoft Azure
Marketplace"`. The second workbook contains only its current grouping. For a
real new batch, update and prepare the verified list first, then rerun. The
POC's multipart rule derives a delimiter-position prior from available
TRAIN/CALIBRATION labels; if `labels.jsonl` is absent, that stage is skipped
and `summary.json` says so. In a scoped run its prior may differ from a full
run. No TEST label is used to select a candidate or make a decision.

The older prototype below remains available for comparison, but it is **not**
the recommended matching-quality demonstration.

## Earlier name-only prototype

This is a separate, read-only experiment. It does not change the ML matcher,
enhanced rules-only POC, graph, emitted events, accepted mappings, or frozen
`rules_only_handoff/` package. One-to-one matching is not part of this command.

## What it reads and writes

Input: the existing `data/prepared/verified_parties.json` catalog and
`data/prepared/adm_records.jsonl` unverified names. It never reads labels. An
initial verified name must exist in the catalog; an added name may be new.
For the current list, it re-scores **every eligible plain unverified row** and
groups all provisional suggestions under the verified party. There is no
sampling or 10–15-row cap. OBO/VIA rows are excluded because this experiment
does not decide which connector segment represents the party to match. Blank,
ineligible and already verified rows are excluded too, with counts in JSON.

Output:

- `suggestions_current.csv` — every current provisional suggestion, sorted by
  verified party, with that party's total suggestion count beside each row.
- `ambiguous_review.csv` — rows for which equally strong verified names compete;
  these are not assigned to either party or included in suggestion counts.
- `summary.json` — all verified parties including zero-count parties, totals,
  and backend-only transition counts when new verified parties are added.
- `suggestions_current.xlsx` — optional formatted workbook with `Verified
  parties`, `Suggestions`, and `Ambiguous review` sheets. It contains the
  *current* assignment only, with no before/after or change-history display.

Example:

```powershell
python -m suggestions.current --prepared-dir data/prepared --initial-verified "Abbott Laboratories" --add-verified "Abbott Molecular" --output-dir outputs/suggestions_abbott
```

To use every catalog party as the initial verified list, use `--initial-all`
instead of repeating `--initial-verified`. You can also omit `--add-verified`
to generate a first snapshot. Use a new output directory per run; results are
never silently overwritten. CSV files open in Excel and require no Node.js.
For the formatted workbook, append `--xlsx --artifact-modules
PATH_TO_NODE_MODULES` and optionally `--node PATH_TO_NODE`, where the modules
directory contains `@oai/artifact-tool`.

The old `python -m suggestions.demo` entry point remains an alias for this
complete current-snapshot command, but its old sample-size options are gone.

## How reassignment works

Each run rebuilds a small search index over the currently available verified
names and evaluates each eligible unverified name against it. If `Abbott
Molecular` is initially only weakly suggested under `Abbott Laboratories` and
then `Abbott Molecular` is added as a verified party, the next run places that
unverified name under the new, more specific party. It does *not* edit prior
workbooks or confirmed mappings. The backend counts such transitions in
`summary.json`, but the workbook deliberately shows only the latest grouping.
Equal-strength competing names remain in the review sheet.

## Name evidence and limitations

This code carries forward conservative ideas from enhanced rules matching:
normalizing case/punctuation and leading `The`, recognizing common legal-form
and `Labs`/`Laboratories` variations, requiring a meaningful name anchor,
preferring a full-name match over a shared-brand suggestion, and abstaining on
equal evidence. It is **not** a copy of the POC's full retrieval/scoring/graph
pipeline. No ownership, parent/subsidiary, rename, or business relationship is
inferred from a shared brand. Thus every exported result is a *suggestion for
review*, never a confirmed match; no precision or recall claim is made.
