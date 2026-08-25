# Data and execution foundation

Phase 3 implements an independently gated sub-workflow:

```text
FILES -> DATA -> EXECUTION
```

It does not generate a mathematical model, solver program, or validated result;
those are Phase 4 and later responsibilities.

## File pipeline

`POST /api/v1/projects/{project_id}/files` accepts one CSV, XLSX, PDF, PNG,
JPEG, WebP, or TXT attachment. The transport body, exact file size, filename,
extension, declared MIME, byte signature/package structure, archive expansion,
image pixels, and PDF structure are bounded before acceptance. Storage keys are
UUID-generated and never derived from user paths. The raw object is moved once,
hashed with SHA-256, and never overwritten.

CSV is parsed by Polars; XLSX is read in non-formula-evaluating/read-only mode;
PDF text is extracted with pypdf and a bounded number of page previews are
rendered with PyMuPDF; Pillow verifies images. Parsed previews, extracted text,
page images, profiles, original files, code, and sandbox outputs all receive
typed `ArtifactRecord` entries.

## Deterministic data layer

Python computes row/column counts, physical and semantic schema candidates,
units inferred from headers, missingness, duplicate rows, descriptive numeric
statistics, IQR outliers, correlations, top values, feature/target/time/spatial
candidates, quality issues/scores, and shared-key candidates across workbook
sheets. These values are `deterministic=true` and are the only profile values
the DataAgent may interpret.

The DataAgent uses its versioned structured prompt and `ModelRouter`. Ordinary
structured profiles start at `FLAGSHIP_HIGH`; attached media or large context
selects the configured Gemini target at `FLAGSHIP_XHIGH`. Google requests use
inline media with an explicit MIME type and bounded base64 payload. If Google is
unavailable, fallback remains explicit in the route; Mock output always carries
`is_mock=true` and proves only schema/workflow behavior.

## Sandbox

`POST /api/v1/projects/{project_id}/executions` is available only after DATA
passes. `SandboxExecutor` invokes Docker without a shell and uses:

- an inspected image digest for the actual run;
- `--network none`, UID/GID `65532`, read-only root, and dropped capabilities;
- `no-new-privileges`, bounded CPU, RAM/swap, PIDs, and wall-clock time;
- read-only source/input bind mount plus bounded no-exec tmpfs filesystems for
  temporary and output data; output is copied to the host only after exit;
- bounded stdout/stderr and artifact count/size.

The image is built with:

```text
docker build -t mathmodel-ai-sandbox:phase3 sandbox
```

The Dockerfile defaults to `python:3.12-slim`. `PYTHON_IMAGE` is a build-time
override for offline verification only; the actual digest is always recorded.
An unavailable image produces `UNAVAILABLE`, not a fabricated success.

## API registry endpoints

- `GET /api/v1/projects/{project_id}/files`
- `GET /api/v1/projects/{project_id}/datasets`
- `GET /api/v1/projects/{project_id}/data-profiles`
- `GET /api/v1/projects/{project_id}/artifacts`
- `POST /api/v1/projects/{project_id}/data/analyze`
- `GET /api/v1/projects/{project_id}/data-understanding`
- `POST/GET /api/v1/projects/{project_id}/executions`

Migration `20260826_0003` owns the five Phase 3 registries. Bytes stay in the
file store; PostgreSQL and versioned `ProblemState` retain metadata and links.
