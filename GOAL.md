# 当前目标

在保留已完成的 Case A 诊断证据的基础上，选取第二道真实赛题（2024 MCM C）用已配置的普通 DeepSeek Flash 模型测试并修复实际问题，产出中文、数据与文献可核验的论文；通过必要检查后上传用户现有的 GitHub 仓库。

## 验收

1. Case A 的已验证科学与论文链、失败尝试保持不变；第二题仅使用官方题面与数据、真实非 Mock 模型和执行器，保存逐阶段证据并如实报告结果。
2. 对第二题暴露的通用缺陷作最小修复与回归；最终中文论文约 23 页，关键数值均有已验证数据来源，引用文献的元数据及其支持的具体论断逐条核验，PDF 逐页检查。23 页为用户的篇幅要求，不冒充 COMAP 官方最低页数。
3. 相关测试、静态检查和提交前敏感信息审查通过；上传现有 `3794708705/MathModel-AI` 仓库并核对远端状态。若认证、外部队伍编号、AI 使用报告或科学证据构成阻断，不伪造通过。

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
- 独立中文分析位于 `analysis/mcm2024c/`，只接受哈希固定的官方 CSV；`results.json` 与 `output/pdf/mcm2024c_chinese_independent_study.pdf` 的数值经 PDF 构建时完整复算。文献逐条核对记录在 `references.md`，论文明确不冒充通用 benchmark PASS 或正式提交。上轮后端全量 694 通过、12 跳过；前端 46 项及构建、ruff、strict mypy、GitHub CI 全部通过；本轮论文扩展尚需重验。
- 用户要求将独立中文研究稿扩展至约 23 页。现已增加每折/每场留出误差、校准十分位、发球分母审计和零假设模拟分位数，并据官方 CSV 重建 23 页 PDF；已逐页检查渲染和文字提取，分析回归 3 项通过，ruff、strict mypy 通过。通用 C 题 benchmark 仍未通过，事件级转势预警仍无已验证结果，不能据页数宣布科学完成。
- 本地提供方注册项已是 `http://127.0.0.1:7863/v1` 与 `global:deepseek-v4.1-flash` 的默认模型。用户提供的新密钥在当前 `workbuddy2api` 网关的 `/v1/models` 与实际聊天请求均返回 HTTP 401；为避免破坏工作连接，用户级有效旧凭据已恢复并得到 HTTP 200。待网关授权修正后再替换，不在仓库记录密钥。

## 项目入口与验证

- 论文流程：`src/mathmodel_ai/paper/workflow.py`、`src/mathmodel_ai/agents/paper.py`、`src/mathmodel_ai/prompt_templates/paper_agent.prompt`。
- 提交流程：`src/mathmodel_ai/submission/workflow.py`、`src/mathmodel_ai/benchmark/profiles.py`。
- 真题单案例入口：`benchmarks/case-001-mcm-2024-c/manifest.json`、`src/mathmodel_ai/benchmark/workflow.py`、`src/mathmodel_ai/benchmark/executor.py`；C 独立复算入口为 `analysis/mcm2024c/README.md`。
- 检查：`uv run pytest -q`、`uv run ruff check src tests analysis/mcm2024c`、`uv run mypy --strict src/mathmodel_ai`；前端 `npm test -- --run` 与 `npm run build`。

## 下一步

已按用户授权公开 WIP 独立分支/草稿 PR，保持 `main` 不变、失败标识不变。本轮先完成 23 页中文独立研究稿及其数据审计、逐页排版和回归验证，再处理 C 通用流水线的可复算科学链：现有模型/结果契约允许声明逐分及留出验证目标，却只把假设驱动标量送入验证器；需让数据派生的时序/统计输出与输入文件、验证要求建立可独立复算的契约，并明确证明主要输出确实依赖逐分数据。A 的正式提交仍需真实队伍控制号、完整 AI 使用报告及有证据地处理最终评审意见。

当前阻塞不是提供方或 GitHub 权限，而是结构性验证契约：最新 C 模型虽读了 5 行 CSV，但全部中心数值仍来自假设，四项自报验证任务未执行。简单提示、静态“读取数据”预检与重复单题运行已连续不能解除此阻塞；恢复时先设计并实现可独立计算的数据派生输出、逐分/分场验证证据绑定，再运行 C。用户已确认先公开独立中文论文草稿，明确标记 WIP 与验证失败，不改变最终科学验收标准。
