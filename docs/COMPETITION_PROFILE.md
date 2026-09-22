# Competition profiles

`CompetitionProfile` is an immutable, versioned rule contract. A submission
stores both `profile_id` and `version`; names such as “MCM” are never branching
logic. Each `CompetitionRule` records type, severity, parameters, source,
source location, and verification status.

The canonical profile digest includes all nested fields and rules. Registry
entries are defensive snapshots, and PostgreSQL stores the digest separately
from JSON; reads compare scalar identity, digest, parent JSON, and child rule
rows. Changing a field without creating a new profile version is an integrity
error. A fixture source cannot become `VERIFIED` merely by changing its status.
Duplicate rule IDs, conflicting page limits, non-finite/unbounded numeric rule
values, unsafe file patterns, and custom filename regexes are schema errors.

Blocking rules with `UNVERIFIED`, `UNKNOWN`, or `CONFLICT` provenance produce
`HUMAN_REVIEW`, never PASS. `PARTIALLY_VERIFIED` profiles rely on each rule's
status. A profile extracted from a future rules PDF must begin as proposed or
unverified until deterministic/manual verification confirms its sources.

The built-in `GENERIC_MODELING_TEST_PROFILE` is version 1 with
`TEST_FIXTURE` provenance. It exercises total page count, exact PDF filename,
real PDF signature, anonymity, required results section, code/data inclusion,
file count, and package allow-lists. It is not an official MCM, ICM, or CUMCM
profile. It remains the only profile exposed by the ordinary Phase 7 test API.

Phase 8 adds an isolated `COMAP MCM 2024 official benchmark profile` version 1.
Its blocking page, PDF, file-count/size/name, anonymity, required-section,
code/data, AI-disclosure, and control-number rules cite hash-pinned official
2024 problem/tips PDFs and page locations. The profile is `VERIFIED` for the
historical technical benchmark after duplicate/conflict and provenance checks.
It is not silently substituted into ordinary submissions and is not evidence
that current or future COMAP rules are unchanged. The control-number handler and
AI-policy compatibility can still cause fail-closed rule/human-review outcomes.

`RuleEngine` reads actual registered bytes and final PDF metadata/text. It does
not trust extensions or agent prose. Non-deterministically evaluable blocking
rules require human review. Deadline modes may reduce optional polish only;
they never disable requirement, anonymity, secret, artifact, or manifest gates.
Custom rules are data only and cannot execute Python or shell. A required but
unsupported reproduction handler must be represented by a blocking custom rule,
which remains `HUMAN_REVIEW` until a real deterministic handler exists.

The historical profile verification is source-specific. A real submission claim
still requires a successful live Phase 8 case, current rule review, independent
human acceptance, and an exact registered control number; the current formal
benchmark is `NOT_READY` before those gates.
