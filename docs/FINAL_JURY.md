# Final Jury

`FinalJuryAgent` receives the exact profile, accepted Paper IR, frozen candidate
identity, requirement coverage, and deterministic rule results. It returns an
11-dimension structured scorecard and typed findings. It is review input, not
authority.

`FinalJuryGate` recomputes dimension names, maxima, total score, deterministic
findings, and the final decision. A failed Phase 5 pointer, non-ready Phase 6
paper, missing/partial requirement, blocking failed rule, unresolved Critical,
or score below the profile threshold yields FAIL. Unknown blocking rules and an
unmet configured independent-review requirement yield HUMAN_REVIEW. Low
innovation alone is not a hard failure.

Persisted `claimed_score` and decision are rechecked. The immutable report digest
binds candidate content, Paper IR/manifest/version, model identity/digest,
official result, profile digest/version, requirement/rule snapshots, and the
artifact set. Finding rows and report JSON must agree, so deletion or severity/
count tampering invalidates the report. A Mock jury may prove routing/schema/
workflow behavior and may coexist with real deterministic gates, but
`reviewer_is_mock=true` remains bound into the frozen snapshot and is never
reported as a live jury review. A profile requiring an independent reviewer
cannot freeze a Mock report.

Jury findings cannot edit a paper, model, result, or package. Corrections use a
typed scope and deterministic invalidation map. Model/data/result changes return
to upstream workflows; only low-risk format-only corrections qualify for the
automatic-fix contract.
`FORMAT_ONLY` is fail-closed to a small deterministic component vocabulary;
equation/model/parameter/constraint changes cannot be relabeled as formatting.
