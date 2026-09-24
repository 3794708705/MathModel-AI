# MathModel AI

> 一个以证据、可追溯性和可复现性为优先级的数学建模竞赛辅助系统。
>
> **当前结论：Provider/模型注册与路由模块已经 `READY`，Phase 1–7 的工程主链已经实现并通过其约束范围内的测试；真实竞赛 Benchmark 和整个项目仍为 `NOT_READY`。本项目目前是研究与工程原型，不是“自动获奖器”，也不能替代参赛者、指导教师、评审人员或领域专家。**

> **WIP 分支说明：**本分支公开尚未完成的 Case A/C 科学验证工作与一份 C 题中文独立复算稿，不表示自动 Benchmark 或正式竞赛提交通过；失败运行记录保留在本地 `var/`，不随仓库发布。

## 中文项目简介

MathModel AI 想解决的不是“让大模型一次生成一篇看起来像论文的答案”，而是把数学建模过程拆成可检查、可复算、可追责的工程链路：从题目理解、数据处理、模型选择、数学表达、求解和验证，到敏感性/鲁棒性分析、红队审查、论文证据绑定、PDF 生成和最终提交包冻结。系统要求重要结论能够回溯到确切的题目版本、数据、模型、方程、程序、求解记录、验证记录和文件哈希。

项目坚持两个原则：

1. **大模型负责提出、解释和评审候选方案，确定性程序负责验证事实。** 工作流成功不等于数学正确，模型自述也不能代替独立计算。
2. **失败必须被保留和如实报告。** 缺少密钥、证据、独立验证或人工审查时，系统会停在 `BLOCKED`、`NOT_EVALUABLE` 或 `NOT_READY`，而不是用 Mock、猜测或漂亮文字伪造成功。

## 项目目的与希望达到的目标

### 为什么做这个项目

- 降低数学建模过程中多 Agent、多工具和多轮修改难以追踪的问题。
- 让每个数值、图表、引用和结论都有来源，而不是只保留最终文本。
- 把 LLM 的创造能力与传统求解器、独立校验、版本控制和安全沙箱结合起来。
- 建立一套可以对真实竞赛题进行重复实验、比较失败原因和持续改进的 Benchmark，而不是依赖单次演示。

### 最终希望达到

- 对不同类型的真实数学建模题形成可靠的“理解 → 建模 → 求解 → 验证 → 写作 → 提交”闭环。
- 至少两个真实案例产生独立验证通过的结果、论文、PDF 和可复现提交包，并对一个主案例完成重复运行。
- 引入与求解 Agent 隔离的独立模型/人工评估，避免系统自己给自己打分。
- 支持更多动态系统、统计模型、非线性/全局优化和数据驱动任务，同时保持证据链与失败闭合。
- 最终达到可供真实团队使用的辅助工具标准，但始终保留人工决策、竞赛规则核对和结果复审。

## 当前做到什么程度

| 范围 | 当前状态 | 说明 |
|---|---|---|
| Phase 1–7 工程主链 | **已实现 / 约束范围内通过** | 状态持久化、数据处理、隔离执行、数学模型、求解、独立验证、修复、论文和提交包链路已经实现。|
| Phase 8 Benchmark 控制面 | **已实现，但验收 `NOT_READY`** | 支持官方资源哈希、盲解/评测隔离、不可变运行记录、预算门、失败记录和报告重建；尚无完整真实案例通过全部门。|
| Phase 8.1 Provider Registry | **`READY`** | Provider/Model 注册、只写密钥、能力 Probe、端点信任、Agent 独立路由和历史身份绑定已通过验收。|
| Phase 8.2 Web 控制台 | **可运行** | React 控制台可管理 Provider、模型、路由、项目、Benchmark 历史和系统状态；业务规则仍以后端为准。|
| 真实 DeepSeek 接入 | **协议与 Provider smoke 已通过** | 证明真实 Provider 调用和生成程序链路可工作，不代表整个竞赛流程已通过。|
| COMAP MCM 2024 Case A | **科学链局部通过，整体 `NOT_READY`** | 新的真实求解与独立复算证据已落盘；Paper v5 达到 `READY_FOR_FINAL_JURY`，但最终评审、完整 AI 使用报告、真实队伍控制号和正式提交仍未通过。历史失败尝试没有被改写。|
| COMAP MCM 2024 Case C | **真实运行，验证 `FAIL`** | 使用官方题面、官方逐分数据和真实 DeepSeek Flash 多轮诊断；最新自动运行仍缺数据派生结果和留出验证，不能称为 verified result。另有[中文独立复算稿](output/pdf/mcm2024c_chinese_independent_study.pdf)及[复现说明](analysis/mcm2024c/README.md)，明确不是自动流水线 PASS 或正式提交。|
| COMAP MCM 2024 Case B | **未运行** | 不以其他案例或结构检查结果代替真实运行证据。|
| 全项目竞赛就绪 | **`NOT_READY`** | 在真实多案例、独立评估、论文与提交包门全部通过前，不应宣称可无人值守参赛。|

本 WIP 分支的[最近一次 GitHub CI](https://github.com/3794708705/MathModel-AI/actions/runs/35962288336)已通过后端、前端、数据库迁移与三个沙箱镜像构建。当前本地后端全量为 695 通过、12 条件跳过，ruff 与 strict mypy 通过；新加入的逐分数组契约还需本次推送后的远端 CI 复核。测试通过只证明软件契约，不等于真实建模质量通过。C 题中文独立复算稿现为 23 页，仍不是自动流水线 PASS。详细边界见 [当前目标](GOAL.md)。

## 哪里可以运行

### 可以直接运行或配置后运行

- **本地 API 与 Web 控制台**：Python 3.12、PostgreSQL、Node.js 和依赖准备完成后，可启动 FastAPI 后端与 React 前端。
- **Mock 开发链路**：适合验证接口、状态机、持久化、错误处理和 UI；不能用于证明数学质量。
- **真实模型 Provider**：可使用 OpenAI、Google、Anthropic 原生适配器，以及 DeepSeek、Qwen 等 OpenAI-compatible 服务和受限 Custom JSON HTTP 端点。需要用户自己的 API Key、能力 Probe 和明确的付费测试授权。
- **确定性求解**：SciPy/HiGHS、SciPy MILP、OR-Tools CP-SAT 和基础 SciPy NLP 路径可用；生成代码会进入无网络、非 root、资源受限的 Docker 沙箱。
- **论文与提交链路**：当上游存在真实 verified result、引用证据、Docker PDF 环境和竞赛规则配置时，可以生成受约束 PDF 并冻结哈希绑定的提交包。
- **Benchmark 控制面**：可以登记官方题目、固定资源哈希、保存每次尝试、重算指标、生成 JSON/Markdown 报告并如实给出失败状态。

### 目前不能运行、没有完成或不能宣称的部分

- 不能声称已经自动完成 COMAP A/B/C 三题，也不能声称达到真实竞赛提交标准。
- Case A 的动态系统仍缺少与正式模型严格绑定的独立轨迹、平衡点、Jacobian 稳定性和场景重放验证；不能用虚构目标函数或普通 bounds check 代替。
- 尚无独立 Benchmark LLM/人工评委，问题理解、模型合理性和论文质量仍需要真正的外部复审。
- 尚无可靠的进程级 Benchmark 自动恢复、异步人工取消、通用提交代码复现执行器。
- 尚未实现本地扫描 PDF OCR、S3/MinIO 存储适配器、任意模型家族和通用非线性全局优化。
- Gurobi 路径需要单独的镜像、`gurobipy` 和合法许可证；当前环境不能把它列为已验证能力。
- 网络检索、Crossref、真实 Provider、Docker 和 LaTeX 任一环境缺失时，相应流程会 fail closed。

## 我的方法与项目中的核心设计

这里的“我的方法”主要指本项目组合、实现并持续收紧的一套工作流与证据方法，而不是声称发明了所有基础算法：

1. **Evidence-first 不可变状态链**：每个阶段生成新版本，不覆盖旧结果；模型、程序、运行、证据、论文和提交包通过 ID 与 digest 精确绑定。
2. **LLM + Deterministic Gate 混合架构**：LLM 负责候选生成和结构化评审，Pydantic Schema、求解器、独立计算器和规则引擎决定能否进入下一阶段。
3. **Typed Mathematical IR**：先形成与求解器无关的数学模型，再由算法选择器和 Solver Router 选择 SciPy、OR-Tools、Gurobi 或受控生成程序。
4. **独立验证 AND Gate**：不信任原求解结果中的“feasible”标记；重新计算变量、约束、指标、场景和物理产物哈希，所有必要义务共同通过才允许晋级。
5. **不可变修复循环**：Red Team 的关键发现进入有上限的模型修复，新版本必须重新求解和验证，不能直接修改历史结果。
6. **Provider/Model/Capability 分离**：稳定 Model ID、端点信任、凭据、模型能力和探测证据是不同事实；每个 Agent 可独立路由，后端硬过滤始终具有最终决定权。
7. **Claim–Evidence Graph**：论文中的结构化结论绑定已验证证据，图表可复现，最终包绑定文件清单和哈希。
8. **Blind Benchmark**：求解输入与评测资源隔离，保留所有尝试、失败、人工介入、成本和环境阻塞，报告从原子记录确定性重建。

## 借鉴和吸收的方法

本项目没有把“借鉴”包装成原创。它吸收的是公开领域中已经成熟的思想和工程模式，再按照可审计数学建模的目标重新组合：

- 借鉴数学建模竞赛常见流程：问题重述、假设、符号、模型构建、求解、敏感性/鲁棒性分析、优缺点、引用与论文交付。
- 借鉴多 Agent 工作流中的角色分工、候选生成、Jury、Red Team 和 Repair 思路，但用确定性 Gate 限制 Agent 自我确认。
- Provider/API 使用体验参考了 **DeepSeek Harness 风格**的 Provider Catalog、Custom Provider、只写凭据、模型发现和无需重启的路由更新；MathModel AI 继续保留 CapabilityProbe、EndpointTrust、证据来源和 Router 硬过滤。
- 借鉴 Ports and Adapters、依赖反转、不可变审计日志、内容寻址、状态机、零信任网络和可复现研究等软件工程思想。
- 借鉴传统数值优化与科学计算工具的成熟能力，包括 HiGHS、SciPy、OR-Tools CP-SAT、可选 Gurobi，以及扰动实验、场景分析和独立复算。
- 借鉴 JSON Schema、Pydantic、OpenAPI、关系数据库迁移、容器沙箱和测试金字塔来约束 AI 输出与运行环境。

这些借鉴指设计思想和公开工具的使用，不表示第三方项目为本项目的结果背书，也不表示可以忽略对应依赖、数据和竞赛规则的许可证或使用条款。

## 使用的主要工具

| 类别 | 工具 |
|---|---|
| 后端 | Python 3.12、FastAPI、Pydantic v2、SQLAlchemy 2、Alembic、PostgreSQL、HTTPX |
| 前端 | React 19、TypeScript、Vite、TanStack Query、OpenAPI TypeScript |
| 数据与文件 | Polars、OpenPyXL、PyMuPDF、pypdf、Pillow |
| 数学求解 | SciPy/HiGHS、SciPy MILP、OR-Tools CP-SAT、SciPy Optimize、可选 Gurobi |
| 隔离与交付 | Docker、非 root/无网络计算沙箱、LaTeX/BibTeX PDF 沙箱、确定性 ZIP 与 SHA-256 |
| 模型接入 | OpenAI、Google、Anthropic、OpenAI-compatible（含 DeepSeek/Qwen）、受限 Custom JSON HTTP |
| 工程质量 | uv、pytest、coverage、Ruff、strict mypy、Vitest、Testing Library、ESLint |
| 协作与版本 | Git、GitHub、不可变数据库记录、版本化配置与迁移 |

## 中文快速启动

推荐环境为 Windows PowerShell、Python 3.12、[uv](https://docs.astral.sh/uv/)、Node.js 和 Docker Desktop。首先在项目根目录准备依赖和数据库：

```powershell
Copy-Item .env.example .env
docker compose up -d postgres
uv sync --dev
uv run alembic upgrade head
npm --prefix frontend ci
```

在第一个 PowerShell 窗口启动后端：

```powershell
.\scripts\dev-backend.ps1
```

在第二个 PowerShell 窗口启动前端：

```powershell
.\scripts\dev-frontend.ps1
```

然后访问：

- Web 控制台：`http://127.0.0.1:5173`
- FastAPI：`http://127.0.0.1:8000`
- API 文档：`http://127.0.0.1:8000/docs`

如需运行隔离 Python、求解器和论文 PDF 流程，还要构建对应沙箱：

```powershell
docker build -t mathmodel-ai-sandbox:phase3 sandbox
docker build -f sandbox/solver.Dockerfile -t mathmodel-ai-solver:phase4 sandbox
docker build -f sandbox/paper.Dockerfile -t mathmodel-ai-paper:phase6 .
```

默认 Mock 仅用于工程验证。调用真实模型时，请在本地未跟踪的 `.env` 中配置密钥，或为后端设置随机 `MM_SECRET_MASTER_KEY` 后通过 Web UI 写入；任何真实凭据都不要提交到 Git。完整配置与启动说明见后文和 [自定义 Provider 文档](docs/CUSTOM_PROVIDERS.md)。

## 希望后来者帮助完善

这是一个仍在成长的项目。欢迎后来者通过 Issue、设计讨论和 Pull Request 帮助完善，当前最有价值的方向包括：

1. 完成 Case A 动态系统的独立指标计算、场景重放和科学审查，并以新运行保留结果。
2. 恢复 Case B/C，建立至少两个真实成功案例和一个可重复主案例。
3. 增加独立 LLM/人工评估协议、盲评数据和引用/竞赛规则复核流程。
4. 扩展动态模型、统计/机器学习模型、随机优化、全局优化和大型模型的 typed IR 与求解适配器。
5. 实现 OCR、S3/MinIO、孤儿产物清理、Benchmark crash/resume、异步取消和通用代码复现。
6. 改进跨平台部署、CI、可观测性、成本流式中止、中文界面、教程和示例数据。
7. 审查第三方依赖与竞赛资料的许可边界，并帮助项目选择合适的正式开源许可证。

贡献时请不要提交 API Key、`.env`、比赛账号、受版权限制的题目文件或无法公开的数据。新功能应尽量附带测试、迁移、证据边界和失败行为说明；请保留历史失败，不要为了让状态变绿而降低 Gate。

## English technical summary

MathModel AI is an evidence-first automation system for mathematical modeling
competitions. Phases 1–7 implement persisted reasoning, guarded data ingestion,
isolated execution, typed mathematical models, deterministic solver routing,
independent validation, bounded repair, evidence-linked paper production, and
hash-frozen submission packages.

Phase 8 implements immutable real-case benchmark runs, blind solve/evaluation
isolation, hash-pinned official resources, deterministic acceptance
recomputation, and auditable reports. The control plane is implemented, but the
formal benchmark remains `NOT_READY`. A retained live DeepSeek Case A run
reached a feasible generated solve; independent dynamic/scenario verification
remained `NOT_EVALUABLE`, so no verified result, paper, PDF, or package was
accepted. Cases B and C have not been resumed.

Phase 8.1 provides the persisted provider/model registry, secure credential
handling, capability probing, endpoint controls, and Agent-specific routing. It
is `READY` within its documented acceptance boundary. See
[docs/CUSTOM_PROVIDERS.md](docs/CUSTOM_PROVIDERS.md) and
[docs/MODEL_REGISTRY.md](docs/MODEL_REGISTRY.md).

Phase 8.2 provides the React control center for provider/model setup, Agent
routing, projects, benchmark history, and system status. It consumes the
FastAPI OpenAPI contract and does not reimplement workflow or readiness rules
in the browser. See [docs/WEB_UI.md](docs/WEB_UI.md).

## Local Development

The tested local baseline is **Python 3.12**. On Windows, use the repository
`.venv` directly; do not rely on an activated Anaconda base Python 3.13 or
another `python` found earlier on `PATH`.

Synchronize dependencies once, then migrate PostgreSQL:

```powershell
Copy-Item .env.example .env
docker compose up -d postgres
uv sync --dev
uv run alembic upgrade head
docker build -t mathmodel-ai-sandbox:phase3 sandbox
docker build -f sandbox/solver.Dockerfile -t mathmodel-ai-solver:phase4 sandbox
docker build -f sandbox/paper.Dockerfile -t mathmodel-ai-paper:phase6 .
```

`uv sync` is intentionally not run on every startup. The single recommended
local backend launcher is:

```powershell
Set-Location C:\Users\zeon\Projects\MathModel-AI
.\scripts\dev-backend.ps1
```

It resolves the repository, calls `.\.venv\Scripts\python.exe` directly, runs a
non-secret startup preflight, and starts the `create_app` Uvicorn factory with
reload enabled. The command intentionally stays in the foreground; keep that
PowerShell window open and press `Ctrl+C` to stop the backend. The preflight
bounds a failed PostgreSQL connection attempt and reports database unavailability
instead of waiting silently. Its underlying stable CLI supports custom
development options:

```powershell
.\.venv\Scripts\python.exe -m mathmodel_ai serve --host 127.0.0.1 --port 8000 --reload
```

`mathmodel_ai.main:application` is not an ASGI target: `application` is local to
`create_app`. Do not use that obsolete command.

## Web UI

With the backend running on port 8000, use the frontend launcher:

```powershell
.\scripts\dev-frontend.ps1
```

Run it in a second PowerShell window; it also remains in the foreground until
`Ctrl+C`. The script does not install packages. If `frontend/node_modules` is
absent, run `npm ci` in `frontend` first.

Open `http://127.0.0.1:5173`. Backend API and FastAPI documentation are at
`http://127.0.0.1:8000` and `http://127.0.0.1:8000/docs`. Use
`VITE_API_BASE_URL` when the backend is not at
`http://127.0.0.1:8000`. Browser-entered provider credentials require a random
URL-safe base64 32-byte `MM_SECRET_MASTER_KEY` on the backend; keys are encrypted
server-side, are never returned, and are cleared from component state after a
successful write. Environment-backed `credential_ref` values remain supported.
Without a master key, a new local environment still starts and displays the
encrypted credential store as not configured. If encrypted rows already exist,
they remain unavailable and providers fail closed until the original key is
restored. A missing live Provider credential never blocks the control center.
For an integrated local image, run `docker compose up --build` after creating
the untracked `.env`.

The API exposes liveness at `/health/live`, database readiness at
`/health/ready`, and non-secret runtime metadata at `/api/v1/system`. Phase 2
reasoning is available under `/api/v1/projects/{project_id}`; create a fixture
project with `POST /api/v1/projects`, then call `POST
/api/v1/projects/{project_id}/reasoning/run`.

Phase 3 endpoints under the same project prefix accept files, expose file/
dataset/profile/artifact registries, run structured data understanding, and
execute explicitly submitted Python only after the FILES and DATA gates pass.
See [docs/DATA_EXECUTION.md](docs/DATA_EXECUTION.md) for contracts and limits.

Phase 4 extends the main workflow from `SELECT` through `MODEL` and `SOLVE`.
`POST /api/v1/projects/{project_id}/mathematical/run` constructs one typed,
solver-independent model, applies the MODEL gate, and selects `AUTO`,
`DETERMINISTIC`, or `GENERATED` execution. `AUTO` remains deterministic-first;
custom generated programs pass through CodeAgent, approved dependency checks,
the same real Docker sandbox, and a strict `result.json` contract. The SOLVE gate
requires recomputed feasibility plus content-consistent model/run/program/code/
execution/result evidence before a result is verified. See
[docs/MATHEMATICAL_CORE.md](docs/MATHEMATICAL_CORE.md) and
[docs/SOLVER_ARCHITECTURE.md](docs/SOLVER_ARCHITECTURE.md).

Phase 5 extends an accepted `SOLVE` result through `VALIDATE`, `SENSITIVITY`,
`ROBUSTNESS`, and `RED_TEAM`. Validation recomputes variables, constraints,
objective, expected outputs, and declared checks without trusting the Phase 4
feasibility record. Sensitivity and robustness perturb immutable model copies
and require a real `ExecutionRecord` for every accepted scenario. A non-Mock
Critical Red Team finding enters `MODEL_REPAIR`; an accepted repair creates the
next model version and must be solved and verified again. Automatic repair is
capped at three cycles. SOLVE leaves results `UNVERIFIED`; only the final
deterministic gate sets `verified_result_id` for the exact accepted formal result. See
[docs/VERIFICATION_REPAIR.md](docs/VERIFICATION_REPAIR.md).

Phase 6 consumes only the explicit Phase 5 `verified_result_id`. `POST
/api/v1/projects/{project_id}/paper/run` builds an immutable evidence snapshot,
retrieves and verifies reference metadata, links structured claims to evidence,
generates reproducible figures/tables, renders deterministic LaTeX/BibTeX, and
compiles a real PDF in a network-disabled non-root container. Paper records and
artifacts are available from the `/literature`, `/paper`, and
`/paper/artifacts` project routes. See
[docs/PAPER_PIPELINE.md](docs/PAPER_PIPELINE.md).

Phase 7 binds an explicit accepted paper and verified result to one immutable
`CompetitionProfile` version. `POST /api/v1/projects/{project_id}/final/run`
recomputes requirement coverage and competition rules, runs structured Final
Jury review behind deterministic gates, performs submission checks, and can
freeze a real manifest/ZIP when `freeze_on_pass=true`. Frozen artifacts are
rehash-verified on read; modification returns `DIRTY`, never silently ready.
See [docs/COMPETITION_PROFILE.md](docs/COMPETITION_PROFILE.md),
[docs/FINAL_JURY.md](docs/FINAL_JURY.md), and
[docs/SUBMISSION_PIPELINE.md](docs/SUBMISSION_PIPELINE.md).

Phase 8 endpoints are `POST /api/v1/benchmarks/runs`, `GET
/api/v1/benchmarks/runs/{id}`, `GET /api/v1/benchmarks/runs/{id}/cases`, and
`GET /api/v1/benchmarks/runs/{id}/report`. Formal runs bind the full Git commit
and source-tree digest, versioned configuration, exact case/profile digests,
every attempt, atomic metrics, failures, and interventions. See
[docs/FINAL_PROJECT_ACCEPTANCE.md](docs/FINAL_PROJECT_ACCEPTANCE.md).

Provider setup uses `/api/v1/providers`, `/api/v1/models`, and
`/api/v1/model-routing`. Configure secrets as environment variables referenced
by `credential_ref`, or submit them once through the encrypted Web UI bridge;
responses never contain raw keys or secret references. Run a model capability
probe and a route preview before a paid Agent smoke. A successful proxy call
proves protocol behavior, not official model identity.

## Capability status

### Implemented

- Phase 1-7 application workflows and Phase 8 benchmark persistence, gates,
  reports, API, official manifests, resource validation, and profile isolation.
- Phase 8.1 provider/model registries, capability probe, secure custom endpoint
  adapters, capability routing, migration, APIs, and audit trace fields.
- Deterministic rejection of Mock/non-live promotion, score/status/count tamper,
  hidden attempts, evaluation leakage, source tamper, and budget overruns.

### Validated

- Phase 1-7 unit, integration, adversarial, PostgreSQL, Docker solver, real PDF,
  and real package paths under their documented fixture/environment boundaries.
- Phase 8 manifest/profile integrity, blind loading, report recomputation,
  exception terminalization, retry-history retention, deadline/budget gates,
  source-tree identity, API, and live Crossref metadata resolution.

### Benchmark-tested

- Official COMAP MCM 2024 Problems A, B, and C were downloaded from hash-pinned
  official URLs and structurally inspected; Problem C includes its official CSV
  and data dictionary.
- The real DeepSeek provider smoke is accepted. A retained Case A run made eight
  real provider calls and two generated Docker executions, reached a feasible
  result, and passed the SOLVE gate.
- The full live competition solve is still not benchmark-validated. Case A's
  independent dynamic/scenario verification is `NOT_EVALUABLE`; it has no
  verified result, paper, PDF, or package. Cases B and C have not been resumed.

### Environment-dependent

- Live reasoning/paper/final-jury execution requires a user-supplied non-Mock
  provider credential. Gurobi needs its optional licensed runtime. Crossref and
  official benchmark retrieval need network access; Docker is required for
  isolated solving and PDF compilation.

### Not implemented

- A local scanned-PDF OCR engine, S3/MinIO storage adapter, automatic benchmark
  crash/resume coordinator, asynchronous human-cancel endpoint, independent
  benchmark LLM/human evaluator, and general submission-code reproduction
  executor.

## Verification

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy --strict src
uv run pytest --cov=mathmodel_ai --cov-report=term-missing
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for boundaries and
[docs/REASONING_CORE.md](docs/REASONING_CORE.md) for the Phase 2 contracts.

## Known limitations

- Mock reasoning verifies schemas, routing, workflow, and persistence only; it
  does not prove that a candidate is mathematically good.
- Live provider calls require user-supplied keys and an explicit paid-test opt-in.
- The local file store has no S3/MinIO adapter, scanned-PDF OCR, or orphan cleanup.
- Phase 4 adapters intentionally cover flattened scalar LP/MILP/integer models,
  basic continuous NLP, and integral CP-SAT input—not every model family in the
  extensible schema.
- Gurobi is optional and needs a separately configured image, `gurobipy`, and a
  runtime license file; SciPy/OR-Tools remain usable without it.
- Bootstrap robustness is explicitly blocked until a dataset resampling
  contract binds sampled rows to model parameters; it is never approximated by
  parameter noise.
- Live literature and PaperAgent checks remain explicit opt-ins; their default
  skip statuses are not evidence of success.
- The normal submission API keeps its deterministic test profile isolated from
  the verified COMAP 2024 benchmark profile. The latter is a historical
  technical-benchmark ruleset, not a claim that current COMAP rules are identical.
- Phase 8 remains `NOT_READY` until the independent verification obligations are
  satisfied and at least two cases produce verified papers/packages. A working
  live provider or a feasible solver result alone is not an acceptance pass.
- Benchmark cost enforcement is post-attempt for an in-flight provider chain;
  it prevents acceptance and blocks later cases but does not stream-cancel an
  already-issued provider request.
- Problem-understanding accuracy, model appropriateness, and Red Team usefulness
  remain zero/Human Review until an independent evaluator scores them; workflow
  stage success is not treated as rubric accuracy.
- FastAPI's current test client emits an upstream `httpx2` migration warning.
