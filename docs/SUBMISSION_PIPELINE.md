# Submission pipeline

The Phase 7 path is:

```text
explicit verified_result_id + explicit ready paper version
-> requirement coverage -> competition rules -> Final Jury
-> SubmissionCheck -> SubmissionFreeze -> manifest -> ZIP -> re-open verify
```

`SubmissionCheck` recomputes failed-rule and missing-requirement counts from
underlying records. It rechecks jury arithmetic, artifact hashes, profile
allow-lists, PDF identity, anonymity, hidden files, secrets, absolute paths, and
required outputs. Blocking FAIL prevents freeze; unknown blocking rules remain
HUMAN_REVIEW.

`SubmissionFreeze` accepts only one mutually consistent PASS jury/check/candidate
chain. It reruns competition rules and rebuilds requirement coverage from the
exact Paper IR and current structured problem requirements. A process-local lock
makes a repeated/concurrent freeze of the same candidate idempotent; the database
current-state lock and continuous revision check prevent two concurrent runs from
both becoming the official current submission. Package generation remains
outside the database transaction, so build/write/roundtrip failure cannot commit
`FROZEN` state.

`SubmissionPackageBuilder` independently rechecks the Jury and SubmissionCheck,
re-reads immutable source bytes, verifies MIME and hashes, rejects symlinks,
Windows reparse points, and disallowed paths, writes canonical JSON, and creates
a deterministic ZIP. The manifest pins the exact profile digest, requirement,
Jury, check, candidate and artifact-set digests, model id/version/digest,
verified result, Paper IR/manifest/version, every source file hash, and semantic
package hash.

The integrity verifier reopens stored manifest and ZIP bytes, reruns supported
security/anonymity and final PDF/profile checks, and rejects missing/ghost,
duplicate, case/Unicode-colliding, absolute, traversal, symlink, nested-archive,
over-count, over-size, and excessive-compression entries. It recomputes every
binding and rejects a manually promoted status. The read API also re-proves that
the snapshot still names the current official result, accepted paper, profile,
requirements, Jury, and check. Any protected-byte, persisted approval, manifest,
or package mutation returns `DIRTY` (or an integrity error when no trustworthy
snapshot can be parsed). Restoring exact original bytes/records allows a later
content-based verification to report `FROZEN`; timestamps have no authority.
Historical snapshots remain immutable. `FROZEN` means locally ready to submit;
this project does not claim `SUBMITTED` without an external platform
acknowledgement.

The default one-shot endpoint does not freeze unless `freeze_on_pass=true`.
Split endpoints rerun all gates from the exact paper/profile version; rerunning
invalidates current readiness before producing a new check or snapshot.

The final package contains only profile-allowed formal PDF/TeX/BibTeX,
figure/table, optional code/data/README artifacts, plus
`submission_manifest.json`. Runtime caches, tests, credentials, local paths, and
internal audit/fixture artifacts are forbidden. Text, PDF text/metadata/
attachments, and supported image metadata are Unicode-normalized before common
identity, local-path, credential, database URL, private-key, and Mock-marker
checks. A required code reproduction rule has no automatic executor in Phase 7
and therefore fails closed at `HUMAN_REVIEW`; it cannot be silently promoted.
