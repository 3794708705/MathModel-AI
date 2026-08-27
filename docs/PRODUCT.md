# Product scope

MathModel AI accepts competition problems and attachments and ultimately builds
a reproducible submission package. Correctness and provenance outrank novelty.

The current release completes Phase 4. It preserves the Phase 1–3 foundation,
reasoning, attachment, profiling, and sandbox contracts, then converts a selected
candidate into a versioned solver-independent `MathematicalModel`, checks symbols,
equations, parameters, and dimensions, chooses an algorithm and available solver,
executes real numerical code, and persists a canonical result with exact evidence
links.

Supported acceptance paths are continuous LP through SciPy/HiGHS, small MILP
through SciPy, integral CP-SAT through OR-Tools, and basic continuous NLP through
SciPy. Gurobi is an optional runtime with explicit availability/license handling.
Mock model outputs still prove only schemas and orchestration; they never prove
mathematical quality or numerical success. Full Phase 5 verification and all
literature/paper/submission work remain outside the current release.
