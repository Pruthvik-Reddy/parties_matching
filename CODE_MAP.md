# Active code map

| Stage | Active code | Input | Output |
|---|---|---|---|
| Workbook preparation | `party_matching.prepare.prepare_workbook` | Related-parties workbook | Runtime ADM JSONL, isolated labels/source rows, verified jobs |
| Name parsing | `party_matching.domain.parse_mentions` | ADM raw name | Stable organization mentions with OBO/via context |
| Verified graph | `party_matching.graph.VerifiedGraph` | Verified imports and cached expansions | Verified-only forest, candidate index, root paths |
| Expansion | `party_matching.expansion.ExpansionService` | Verified parties | Cached candidates and separate suggested parent |
| Retrieval | `party_matching.matching.MentionRetriever` | Verified official/candidate names and ADM mentions | Exact, TF-IDF, and optional embedding hits |
| Proposal collection | `party_matching.matching.collect_proposals` | All graph variants | Account-wide deduplicated proposals |
| Identity and assignment | `FeatureScorer`, `CrossEncoderReranker`, `decide_records` | Proposals and graph roots | One `FinalDecision` per ADM record |
| Persistence and events | `party_matching.matching.run_matching` | Final decisions | Graph, mapping ledger, events JSONL |
| Evaluation and Excel | `party_matching.reporting.build_reports` | Source rows, labels, decisions | Metrics, analysis, Summary/Detail workbook |

Record trace: raw name → `OrganizationMention` → retrieval evidence against a verified official/candidate name → identity score → candidate owner → verified graph root → distinct-root competition → `FinalDecision` → event and Excel row.

