# CI portability and clean-runner repair

## Failure and correction

GitHub runs 35683442439 and 35684229277 failed Linux mypy because two
benchmark file checks accessed Windows-only `stat_result.st_file_attributes`.
The checks now use optional attribute lookup while retaining reparse-point
rejection. Strict mypy also passes with `--platform linux` on the local host.

Full reproduction exposed a stale model-explorer prompt-version assertion and
Case A policy review digests left behind after validation-requirement bindings
changed. The prompt assertion now matches version 2.5.0. The existing contract
review script was rerun successfully and the scenario review script executed
all 19 scenarios in the real solver sandbox (1/1 standalone metrics and 19/19
scenarios passed). Review sidecars now bind policy digest
`c61422eded3f1e1c05347090ce4737f94a0c967762d1dade907f447b73eb09bd`.
The mathematical model, policy requirements, acceptance thresholds, and old v1
draft are unchanged; previous sidecars remain in Git history. This is fixture
review maintenance, not a new live benchmark attempt or project acceptance.

The workflow also builds the solver and PDF images before running pytest,
so a fresh runner can execute the PostgreSQL replay and real computation/PDF
tests. The paper Dockerfile requires the repository root as its build context.
No test or coverage gate is disabled.

## Local validation

- Clean temporary PostgreSQL database migrated from zero to head; Alembic check passed.
- Full suite: 636 passed, 8 environment-gated skips, coverage 86.77% (minimum 85%).
- Ruff formatting/lint and strict mypy passed; Linux-targeted mypy passed.
- The first local PostgreSQL failure was reproduced against the developer
  database containing saved provider routes; using a fresh CI database passed.
  The developer database was not cleared.

GitHub Actions on the pushed commit is the final Linux execution check.

## Linux execution follow-up

Run 35696103005 passed static checks, migrations, and all image builds, then
exposed a POSIX cleanup bug: the sandbox workspace is intentionally mode 0555,
but the old Windows-oriented retry only made the child writable and could strip
directory search permission. All 29 failing tests reported this cleanup error.
Cleanup now restores owner read/write/search permissions on directories inside
the validated per-run tree before deletion, preserves existing file mode bits,
and does not traverse symlink targets. Runtime read-only container boundaries
remain unchanged. Regression tests cover nested read-only directories, external
symlink preservation, and rejection of an out-of-root cleanup target.

The actual cleanup methods also passed a standalone Linux reproduction inside
the sandbox image as UID 65532 (nested mode-0555 directories, external symlink
targets retaining their contents/modes, and out-of-root rejection). Local Ruff,
strict mypy, and the executor unit tests passed; the POSIX-only symlink test is
intentionally skipped on Windows and runs in Linux CI.
The sandbox, solver, and scenario-replay subset passed: 49 passed, 2 skipped
(POSIX-only regression on Windows and optional licensed Gurobi integration).
