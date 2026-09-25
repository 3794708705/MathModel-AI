# 当前目标

把 MathModel AI 的自动建模程序做成可交付的完整流程：在受支持的真实赛题上，从题目和官方数据进入系统，经真实模型、沙箱求解、独立科学验证，自动生成证据绑定的中文论文和可核验交付物；以 2024 MCM C 作当前端到端验收案例，保留 Case A 和所有失败尝试，并将通过检查的改动上传现有 GitHub 仓库。完整流程不等于保证任意模型、任意题目可解，也不冒充需要外部队伍身份的正式 COMAP 提交。

## 验收

1. 保留 Case A 及 C 的已有证据和失败记录；C 题仅使用官方题面与数据、真实非 Mock 模型和执行器，自动运行到绑定原始逐分数据的已验证结果，主要问题与模型自报验证要求均有独立复算证据，不降低质量门。
2. 从该已验证结果自动生成约 23 页中文论文、真实文献支持的声明和可核验的本地交付物；逐页检查 PDF。23 页是用户篇幅要求，不冒充 COMAP 官方最低页数；正式竞赛提交仍受真实队伍信息、AI 使用报告和外部规则约束。
3. 对通用缺陷作最小修复与回归，前后端/静态/CI 检查通过，敏感信息不入仓；上传 `3794708705/MathModel-AI` 并核对远端。任何未通过的科学或提交门如实标记，不能用局部软件测试替代端到端完成。

## 当前状态

- 本轮唯一真题诊断：benchmark run `244899b9-d59c-47bb-9cd1-19b505f0ac01`，Case A attempt `aa8c0e2b-dd95-4d9f-b73a-770ef00f3afc`，报告在 `var/benchmarks/runs/244899b9-d59c-47bb-9cd1-19b505f0ac01/benchmark-results.json`。真实 DeepSeek Flash、真实沙箱求解与独立复算均执行；新验证结果 `096db580-9321-454f-93d1-10d8e35e5e18` 已落盘，旧结果不变。完整 benchmark 状态仍是 `NOT_READY`，原尝试 `FAIL` 停在论文/最终提交，不可事后改写为 PASS。
- 原 Paper v1 因文献断言强于来源证据被拒；v3 因要求覆盖与 PDF 文本抽取问题失败；v4 仅余 Q3 概念性输出的类型分类误判。失败版本均保留。对无效证据字段增加作者预检，并修正把稳健性情景指标错误绑定到敏感性证据的反馈；未手工修改求解结果或论文声明。
- 同一新科学快照的 Paper v5（paper ID `3d769d74-a3a2-4188-870c-76f371291c62`）经确定性重验为 `READY_FOR_FINAL_JURY`，质量错误 0，15/15 题目输出由已支持的可见声明覆盖。六页 PDF 位于 `var/storage/projects/d0c0d290-826c-4a37-be5f-092b583efa16/artifacts/6869ddc9-5803-49a9-a8eb-f2225752ff0c/paper.pdf`，SHA-256 `14558d62970056078bdf7fb6c2e2c4feb6435d0134d264f457ca7a6b9963c128`，已逐页目检；存在重复摘要、超长符号表等排版限制。
- v5 不冻结最终评审已落盘：Jury `FAIL`，Submission Check `HUMAN_REVIEW`，提交包未生成。硬阻断为缺完整 AI 使用报告及注册队伍控制号无法核实；评审仍指出未解决的重大红队意见、目标函数解释和参考文献不足。不能将论文质量门通过等同正式竞赛提交成功。
- 旧尝试的验证结果 `f4f73f2d-a524-49b1-bac0-6d24fcef814f` 和 Paper v9 `18c474a3-58e6-4732-b74c-400a0bf11d77` 保留。原有工作树改动在用户授权后作为 WIP 独立分支上传，不回写任何历史运行结论。
- 第二题 C 已用官方题面、官方逐分数据和真实 DeepSeek Flash 连续诊断，失败尝试均保存在各自 benchmark run 下；依次修复 CSV 方言解析、数据列名反馈、潜在状态欠定检测、题目结构化重试反馈，以及只读输入文件未传入生成代码沙箱。
- GitHub 远端是公开仓库 `3794708705/MathModel-AI`。GitHub 连接器现已授权给账号 `3794708705`，仓库权限含 `push`；WIP 分支从 `main` 提交 `231d9944770886382568d606747397972954f05e` 建立，保持 `main` 不变。科学链未通过，不作为完成版提交。
- 输入挂载后的 C 单次重跑 `e5b27e18-f1ba-454b-9d26-412fba5ba43f` 保留为 `FAIL`：真实沙箱已读取 7284 点/31 场，但输出因 NumPy 布尔序列化失败一次，后两次被独立可行性/目标复算判 `MODEL_INVALID`。模型硬约束含未物化的逐分数据绑定符号，且标量结果违反初值约束；不能放宽验证门。新增通用建模门，提前指出不可供标量复算的符号，并以 C 原模型拒绝、A 已验证模型接受作现场核验。
- 随后的 C 单次运行 `439b7c54-6988-4e27-99f6-210a583b4e0d` 到达真实求解（solver_success=1），但独立验证为 `NOT_EVALUABLE`：三项自报验证要求均 `UNCHECKED`；模型还把无数据绑定的观测率写成 0.5046，其唯一证据只说明 CSV 存在，而官方 CSV 复算为 0.51043。已加通用门：DATA 数值常量必须被其证据及原题面明确陈述，或有数据绑定；A 的 78%/56% 仍通过，C 的无据常量被拒。
- 修复后仅重跑 C：`e71f41c7-56e0-46b0-8258-82e69785c4b3`，依旧 `FAIL` 于 VERIFICATION（`VALIDATE_GATE_FAIL:independent_status_pass, all_metrics_recalculated, declared_requirements_checked`）。真实求解程序只读取官方 CSV 表头，没有使用逐分行，却用假设标量给出 `theta=0`；结果只含 theta，模型宣称的逐分走势、游程检验、预测/留出验证及论文输出未形成可复算产物，四项自报验证要求全部 `UNCHECKED`。这是科学/产物契约缺口，不能降门限或把 `solver_success=1` 当作通过。C 的 benchmark 尚未成功，不能在论文里引用其未验证求解数字。
- 为上述生成代码故障增加通用 AST 预检：当模型有 CSV 数据绑定时，拒绝“读取 CSV reader 但只取表头”的代码，并把错误反馈给代码代理重试。仅重跑 C 的 `f5fb849a-3693-453a-bc90-6ee71e09a516` 确实遍历了官方数据，但仍在 VERIFICATION 失败：五项自报科学验证均 `UNCHECKED`。其优化目标 `p_match_win_server` 经定义方程只依赖假设参数，与唯一决策变量 `momentum_weight` 无关，属常数目标伪优化。已加通用数学门拦截此依赖缺失，A 原模型仍被接受，C 记录模型被拒。正仅重跑 C 核验。
- 数学门修复后仅重跑 C 的 `744aa7ea-adf4-45e6-bfef-25a56931b13f`：真实模型/代码/求解均落盘，仍 `FAIL` 于 VERIFICATION（`independent_status_pass`、`declared_requirements_checked`）。新目标依赖决策变量，但程序只读取 CSV 前 5 行作列名/存在性检查；核心求解和结论完全由假设常数驱动。四项自报的敏感性、留出比赛、校准、走势对照均 `UNCHECKED`。重复简单提示/重跑不能证明数据到结果的科学链；下一步需类型化数据派生输出与独立复算契约，不再盲目运行 C。
- 独立中文分析位于 `analysis/mcm2024c/`，只接受哈希固定的官方 CSV；`results.json` 与 `output/pdf/mcm2024c_chinese_independent_study.pdf` 的数值经 PDF 构建时完整复算。文献逐条核对记录在 `references.md`，论文明确不冒充通用 benchmark PASS 或正式提交。本轮后端全量 695 通过、12 跳过；前端 46 项及构建、ruff、strict mypy、GitHub CI 全部通过。
- 用户要求将独立中文研究稿扩展至约 23 页。现已增加每折/每场留出误差、校准十分位、发球分母审计和零假设模拟分位数，并据官方 CSV 重建 23 页 PDF；已逐页检查渲染和文字提取，分析回归 3 项通过，ruff、strict mypy 通过。通用 C 题 benchmark 仍未通过，事件级转势预警仍无已验证结果，不能据页数宣布科学完成。
- 本地提供方注册项已是 `http://127.0.0.1:7863/v1` 与 `global:deepseek-v4.1-flash` 的默认模型。用户现改为提供网关正在使用的原有密钥；网关配置与本项目用户级环境变量均与之相符，`/v1/models` 和实际聊天请求均返回 HTTP 200，目标模型在列表中。无需修改或重启网关；密钥不写入仓库。
- 为解除 C 题数据产物缺口，已给生成程序的 `result.json` 增加有界有限数值的 `predictions` 与命名 `series` 字段，并将其原样传入执行记录绑定的独立指标复算入口；新增回归确认不再丢弃数组，且拒绝非有限值与非法序列名。本地后端全量 695 通过、12 条件跳过，ruff 与 strict mypy 通过。这仅打通产物契约，不证明数组来自官方 CSV，也未覆盖自报的四项验证要求；仍需审定观察值与数据来源绑定后才可重跑 C。
- 新目标的第一批通用修复已落地：模型门要求目标子问题所需的官方数据被声明绑定，每个绑定进入目标/约束/状态关系，受支持的标量值与登记的确定性数据剖析一致；代码门拒绝只循环计数、不读取 CSV 行值。只读复核旧 C 模型为 5 个声明绑定、0 个参数绑定，旧程序也被新代码门拒绝；历史运行状态未修改。针对性回归 45 项、ruff、strict mypy 通过。逐分输出与官方 CSV 的独立复算、留出验证要求绑定仍缺失，因此暂不启动新的 C benchmark，也不宣称自动流程完成。
- 独立观测输入现可携带原始 CSV 的 SHA-256、标签列与二元编码；验证器重新读取登记文件并逐行核对观测数组，封闭指标计算器增加 Brier 与二元 log loss，拒绝错误的极端概率。受影响回归 36 项、ruff、strict mypy 通过；对官方 C CSV 的只读核对得到 7284 行、其中 3718 行为正类。尚未为 C 建立经审定、与模型及原始数据绑定的完整验证策略，逐分预测本身和分场留出训练仍未独立验证。
- 新增可审定的“直接从登记 CSV 读取二元标签”策略，不再要求人工复制逐分观测 JSON；官方 C CSV 经新解析器核对为 7284 行、3718 个正类。旧策略/计划未设置该字段时序列化保持不变，Case A 历史摘要需继续回归核对。此修复仅覆盖观测真值来源，预测生成和分场留出验证仍未被证明。
- 新观测来源绑定贯穿独立服务、VALIDATION 与无目标响应摘要，拒绝“报告 PASS 但策略所指 CSV 已变”的证据移接；旧 Case A 契约回归通过。最新本地后端全量 705 通过、12 跳过，覆盖率 86.72%；前端 46 项、构建、Lint、ruff、strict mypy 通过。GitHub WIP 分支 `f008e2f` 的 push CI 已成功；本轮补强待再次经远端 CI 验证。尚无新的 C 真题运行，旧 C 结论维持 FAIL。
- 新增隔离逐分留出协议：原始 CSV 仅在受信主机，按比赛 ID 选整场留出；预测程序仅在无网络、只读、非 root 容器内收到训练分标签和逐条赛前特征，不能挂载完整测试 CSV。执行记录与预测轨迹均留证，独立重放可复算留出划分、标签、基线及 Brier。官方 C 数据已通过此协议的审定基线诊断（非自动建模 benchmark）；旧 C 失败结论不变。本地全量后端 714 通过、12 跳过，覆盖率 86.48%；前端 46 项、构建与 Lint、ruff、strict mypy 通过。待把版本化隐藏输入、生成模型与该轨迹/自报验证要求接入主流程，才可重跑 C 并主张端到端通过。
- C 题现有基准入口新增版本化整场留出侧文件，完整官方 CSV 仍用于受信结构检查，建模代理与求解沙箱只接收训练比赛的派生 CSV；划分来源、策略和训练文件哈希纳入新的求解输入摘要。旧尝试与其输入摘要不变。官方缓存只读核验 31 场/7284 分，6 场整场留出、25 场训练；尚未把生成预测代码与隔离轨迹及四项验证要求绑定，故不启动新的 C benchmark，也不宣称科学链通过。
- 新增生成代码的 `causal_predictor.py` 契约和隔离评估入口：只接受与已执行模型/程序哈希一致的源码，用同一整场留出策略生成非 Mock 执行记录及宿主轨迹，执行与证据原子落库后独立重放。当前主流程在未建立正式科学验证绑定时明确失败关闭，不能把预测轨迹冒充四项自报要求均通过；未启动新的 C benchmark。后端全量 722 通过、12 跳过，覆盖率约 86%；前端 46 项、构建、Lint、ruff、strict mypy 通过；GitHub 提交 `60b48c3` 的 push/PR CI (`36095720764`/`36095724570`) 均成功。
- 对留出证据增加可重复的持久化审计：正式模型、程序、求解运行和结果 ID 必须与隔离执行相连；从数据库重新读出执行、训练 CSV 实际字节与证据文件，用官方 CSV 按固定策略重放，拒绝替换的驱动、策略和产物。真实 SQLite 与 Docker 衔接回归已通过；后端全量 726 通过、12 跳过，覆盖率约 86%，前端 46 项及构建、Lint、ruff、strict mypy 通过。它只证明该项留出预测，不覆盖走势、随机性检验等完整科学要求，也未启动新 C benchmark。
- 主验证流程现接收重新审计的留出执行、官方来源哈希与轨迹哈希，并把 Brier/基线 Brier 作为正式报告的独立输出指标；最终红队复核会再次读取持久化证据。留出指标不自动满足模型自报的校准、走势等验证要求；缺审定的全题要求策略时，报告落盘后仍阻止进入后续论文流程，修复版模型也不能沿用旧留出轨迹。后端全量 728 通过、12 跳过，新增针对性回归、ruff 与 strict mypy 通过；未重跑 C 真题，旧结论仍为 FAIL。

## 项目入口与验证

- 论文流程：`src/mathmodel_ai/paper/workflow.py`、`src/mathmodel_ai/agents/paper.py`、`src/mathmodel_ai/prompt_templates/paper_agent.prompt`。
- 提交流程：`src/mathmodel_ai/submission/workflow.py`、`src/mathmodel_ai/benchmark/profiles.py`。
- 真题单案例入口：`benchmarks/case-001-mcm-2024-c/manifest.json`、`src/mathmodel_ai/benchmark/workflow.py`、`src/mathmodel_ai/benchmark/executor.py`；C 独立复算入口为 `analysis/mcm2024c/README.md`。
- C 输入隔离及生成预测评估：`benchmarks/case-001-mcm-2024-c/causal-holdout-v1.json`、`src/mathmodel_ai/benchmark/causal_inputs.py`、`src/mathmodel_ai/benchmark/causal_evaluation.py`、`src/mathmodel_ai/benchmark/manifests.py`。
- 隔离逐分留出协议：`src/mathmodel_ai/verification/causal_binary.py`、`src/mathmodel_ai/verification/causal_holdout.py`、`src/mathmodel_ai/sandbox/causal_holdout.py`；真实容器与官方 CSV 的诊断测试在 `tests/sandbox/test_causal_holdout.py`。
- 检查：`uv run pytest -q`、`uv run ruff check src tests analysis/mcm2024c`、`uv run mypy --strict src/mathmodel_ai`；前端 `npm test -- --run` 与 `npm run build`。

## 下一步

继续在 WIP 分支处理 C 通用流水线的可复算科学链：为全题子问题及模型自报验证要求建立审定策略，把数据派生时序/统计结果和已有在线留出轨迹绑定到各项独立复算。完成后仅重跑 C，接着验证自动论文和交付物；当前独立研究稿不能代替自动流水线。A 的正式提交仍需真实队伍控制号、完整 AI 使用报告及有证据地处理最终评审意见。

当前阻塞不是提供方或 GitHub 权限，而是剩余科学契约：正式验证虽已记录并复核隔离留出轨迹，但尚无覆盖 C 题全部子问题和自报要求的审定策略及其对应的数据派生结果；旧 C 模型与代码仍会在更早阶段被拒。不能用一项留出指标或重复单题运行替代自动建模验收。用户要求完成自动建模流程，目标保持未完成；先补齐策略及复算，再进行真实 C 单题重跑。
