# 2024 MCM C：中文独立复算

这是对 [COMAP 官方 C 题](https://www.contest.comap.com/undergraduate/contests/mcm/contests/2024/problems/2024_MCM_Problem_C.pdf) 的独立、可复现分析，不是 MathModel AI 通用 benchmark 已通过的论文或官方参赛提交件。通用 C 题运行 `e5b27e18-f1ba-454b-9d26-412fba5ba43f` 仍为 `FAIL`：逐分时序模型与当前标量核验契约不匹配。失败证据保存在本地 `var/benchmarks/runs/`，不改写结果。

官方输入是 `Wimbledon_featured_matches.csv`，SHA-256 必须为 `b1788d0ea169b65629b0e9fb0f91d007507b306e404507bbf90bd5f700a3c229`。原始数据请从 COMAP 官方题目提供的附件取得；仓库不重复分发。

在仓库根目录运行：

```powershell
uv run python -m analysis.mcm2024c.analyze PATH_TO_OFFICIAL_CSV analysis/mcm2024c/results.json
uv run --with reportlab python -m analysis.mcm2024c.build_paper PATH_TO_OFFICIAL_CSV analysis/mcm2024c/results.json output/pdf/mcm2024c_chinese_independent_study.pdf
```

PDF 构建需要 Noto Sans SC/CJK 字体。Windows 的标准安装路径会自动检测；其他系统可用 `MM_CJK_FONT` 指向本机字体文件。构建器重新计算全部统计量，并要求与 `results.json` 完全一致。

主要方法：按比赛和发球方用过去得分更新在线 Beta(10,10) 基线；过去 8 分残差定义为走势；固定发球序列作 500 次顺序零假设模拟；按整场比赛分组做五折下一分预测测试，并比较 4/8/12 分窗口。`results.json` 还保存了每折、每场的留出误差、概率校准分组和数据分母审计，全部由原始逐分行计算。中文研究稿约 23 页，包含目录、详细方法、逐场结果和教练简报；23 页是作者的篇幅目标，不是 COMAP 的合格门槛。该研究不声称识别心理因果或可靠的“转势”报警器。出版信息和每篇文献支持的具体论断见 [references.md](references.md)。
