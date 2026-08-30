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
profile.

`RuleEngine` reads actual registered bytes and final PDF metadata/text. It does
not trust extensions or agent prose. Non-deterministically evaluable blocking
rules require human review. Deadline modes may reduce optional polish only;
they never disable requirement, anonymity, secret, artifact, or manifest gates.
Custom rules are data only and cannot execute Python or shell. A required but
unsupported reproduction handler must be represented by a blocking custom rule,
which remains `HUMAN_REVIEW` until a real deterministic handler exists.

**NOT YET VERIFIED AGAINST REAL COMPETITION RULESET.**
