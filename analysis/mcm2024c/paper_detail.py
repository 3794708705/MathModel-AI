"""Evidence-rich supplementary pages for the Chinese MCM C study.

All numeric statements here are computed from the checked official CSV or from
the results object that the PDF builder independently recomputes before use.
"""

# ruff: noqa: RUF001

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, Spacer, Table, TableStyle

from analysis.mcm2024c.analyze import point_features


def _table(headers: list[str], rows: list[list[str]], widths: list[float]) -> Table:
    result = Table([headers, *rows], colWidths=widths, repeatRows=1)
    result.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "NotoSC"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.4),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F7F9")]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CDDBE2")),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return result


def add_detailed_pages(
    story: list[object],
    st: dict[str, Any],
    checked: dict[str, Any],
    matches: dict[str, dict[str, object]],
    paragraph: Callable[[str, Any], Paragraph],
    figure: Callable[..., object],
) -> None:
    """Add sixteen analytical pages without inventing benchmark acceptance."""
    cv = checked["cross_validation"]
    main = cv[1]
    audit = checked["match_audit"]
    null = checked["null_test"]
    final = checked["final_match"]

    def page(title: str, lead: str) -> None:
        story.append(PageBreak())
        story.append(paragraph(title, st["h1"]))
        story.append(paragraph(lead, st["body"]))

    def body(text: str) -> None:
        story.append(paragraph(text, st["body"]))

    def caption(text: str) -> None:
        story.append(paragraph(text, st["caption"]))

    def tab(headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
        story.append(_table(headers, rows, widths))
        story.append(Spacer(1, 4 * mm))

    # 1. Input audit and denominators.
    page(
        "3.1 数据全集与分母审计",
        "本文的观测单位是逐分记录，不是局、盘或球员。所有结论都从官方文件中实际读取的 "
        f"{checked['n_matches']} 场、{checked['n_points']:,} 分出发；"
        "哈希锁定避免同名异版 CSV 被悄悄替换。"
        "数据按文件中的比赛及逐分顺序处理，没有用赛事结果倒推前面各分的特征。",
    )
    tab(
        ["审计量", "官方 CSV 复算值", "解释"],
        [
            ["比赛数", str(checked["n_matches"]), "独立分组单位"],
            ["逐分行数", f"{checked['n_points']:,}", "预测评分分母"],
            ["每场最少分数", str(audit["min_points_per_match"]), "非零样本"],
            ["每场中位分数", str(audit["median_points_per_match"]), "按场排序中位数"],
            ["每场最多分数", str(audit["max_points_per_match"]), "不截断长比赛"],
            ["发球方赢分", f"{checked['service']['server_wins']:,}", "与总分数配对"],
        ],
        [42 * mm, 36 * mm, 78 * mm],
    )
    body(
        f"发球方赢分率为 {checked['service']['server_win_rate']:.4f}，对应 "
        f"{checked['service']['server_wins']:,}/{checked['service']['server_points']:,}。"
        "它是全样本描述量，不是未来某一分的预测概率，也不是心理势头的估计。"
        "两个发球方的赢分量分别是 "
        f"{audit['server_1_wins']:,}/{audit['server_1_points']:,} 与 "
        f"{audit['server_2_wins']:,}/{audit['server_2_points']:,}；这些分母相加严格等于全样本逐分数。"
    )
    body(
        "数据限制同样需要进入审计：这里仅使用官方 CSV 已有的男子草地比赛，"
        "没有私自补充赛前排名、伤病或女子比赛资料。原始文件不随仓库再分发；"
        "复算者须自行取得官方附件，并首先验证 SHA-256。"
    )

    # 2-3. Match inventory, one line per observed match.
    for part, rows in enumerate((main["match_results"][:16], main["match_results"][16:]), 1):
        page(
            f"3.{part + 1} 逐场测试清单（{part}/2）",
            "下表列出每场比赛在其所属留出折中的实际评分。ΔBrier 定义为"
            "修正模型减基线模型，负值代表修正后误差较低。逐场列示能避免全样本平均"
            "掩盖明显失效的比赛；表中每行的系数只由其他折的比赛拟合。",
        )
        tab(
            ["比赛 ID", "分数", "折", "基线 Brier", "修正 Brier", "ΔBrier"],
            [
                [
                    str(item["match_id"]),
                    str(item["n"]),
                    str(item["fold"]),
                    f"{item['brier_baseline']:.4f}",
                    f"{item['brier_adjusted']:.4f}",
                    f"{item['brier_adjusted'] - item['brier_baseline']:+.4f}",
                ]
                for item in rows
            ],
            [54 * mm, 17 * mm, 11 * mm, 26 * mm, 26 * mm, 22 * mm],
        )
        improved = sum(item["brier_adjusted"] < item["brier_baseline"] for item in rows)
        body(
            f"本页 {len(rows)} 场中，{improved} 场的 Brier 损失下降，"
            f"{len(rows) - improved} 场持平或上升。此计数是按比赛而非按分数加权；"
            "它用于观察跨场稳定性，不能代替全部逐分汇总损失。"
        )

    # 4. Leakage map.
    page(
        "3.4 变量时点与信息泄漏边界",
        "逐分预测最容易犯的错误，是把当前分结束后才知道的变量当作赛前输入。"
        "本研究以当前分开始前作为预测时刻；可用信息与禁止信息必须逐项区分。",
    )
    tab(
        ["字段/量", "何时可知", "本文用途"],
        [
            ["match_id / player1 / player2", "赛前", "分组及解释标签"],
            ["server", "该分开始前", "选择在线基线分支"],
            ["此前 point_victor", "此前各分结束后", "更新 W、N 与残差窗口"],
            ["当前 point_victor", "该分结束后", "仅作训练标签与评分"],
            ["最终胜负 / 后续逐分", "未来", "绝不进入当前预测"],
        ],
        [54 * mm, 40 * mm, 62 * mm],
    )
    body(
        "给定同一场比赛，程序先计算 p0_t 与 m_t，再将当前结果 y_t 写入状态。"
        "单元测试将尚未发生的结果翻转，要求较早各分的特征完全不变。"
        "跨场验证在拟合 β 时只使用训练场次；测试场次虽然可用其已发生分数"
        "更新在线状态，却不能把该场未来分数或最终结果放进训练集。"
    )
    body(
        "官方文件还包含比分、发球质量、跑动等列，但这些列可能记录当前分结果或"
        "存在复杂时点定义。为了保持最小可审计链，本版没有将它们默认为赛前可知特征。"
        "这降低预测上限，却使每一个输入的可用时间清楚可检验。"
    )

    # 5. Baseline.
    first_id = sorted(matches)[0]
    first_points = matches[first_id]["points"]
    assert isinstance(first_points, list)
    first_features = point_features(first_points, 8)
    page(
        "3.5 在线发球基线的推导与实例",
        "把 player1 赢得该分记为 y_t=1。令 s_t∈{1,2} 表示发球者；对每场、"
        "每个发球者分别维护截至 t-1 分的 player1 赢分数 W_s 与相关分数 N_s。"
        "采用固定 Beta(10,10) 平滑得到 p0_t=(10+W_s)/(20+N_s)。"
        "这只是在线估计，并未声称它等于选手真实能力或最优贝叶斯先验。",
    )
    tab(
        ["分序号", "发球方", "p0_t", "赛后 y_t", "历史走势 m_t"],
        [
            [str(i + 1), str(first_points[i][0]), f"{p0:.4f}", str(y), f"{m:+.4f}"]
            for i, (p0, m, y) in enumerate(first_features[:8])
        ],
        [24 * mm, 29 * mm, 32 * mm, 29 * mm, 42 * mm],
    )
    caption(f"表：{first_id} 的前 8 分。当前结果仅在评分及下一次更新时使用。")
    body(
        "初始两种发球状态均为 W=0、N=0，因此首个相关 p0 为 0.5。"
        "每分结束后，仅当前发球方的 W、N 变化；换发球方时选用另一条历史。"
        "这个规则纠正了把连续得分全部解释为势头的粗糙做法，但没有控制"
        "比分压力、第一发球或球员能力。"
    )

    # 6. Flow definition.
    page(
        "3.6 走势指标的数学含义",
        "定义赛后残差 r_t=y_t-p0_t，并令下一分可见的走势 m_t 为前 L 分残差"
        "的平均。主分析取 L=8；比赛开头不足 8 分时只对已有历史取均值，"
        "首分 m_1=0。不同比赛之间清空窗口，不把上一场的记录接入下一场。",
    )
    tab(
        ["性质", "操作定义", "解释边界"],
        [
            ["有方向", "m_t>0 指 player1 超额赢分", "不等于赢赛概率"],
            ["有大小", "窗口内残差平均值", "量纲是每分的概率差"],
            ["只看过去", "最晚使用 t-1 分结果", "可作下一分输入"],
            ["短期平滑", "L=4/8/12 敏感性比较", "窗口不是物理常数"],
            ["发球校正", "每分减去该发球方 p0", "仍非完整能力校正"],
        ],
        [31 * mm, 56 * mm, 69 * mm],
    )
    body(
        "如果 player1 连续赢分，但那些分数本来就在其强势发球段发生，"
        "r_t 往往小于直接的 y_t；反之接发球得分可能形成较大正残差。"
        "因而曲线衡量的是相对当前在线基线的局部偏离。"
        "窗口共享相邻分，图线天然平滑且高度相关；不能把每次穿过零线都算作"
        "一项独立的转势证据。"
    )

    # 7. Final-match flow across windows.
    final_points = matches[str(final["match_id"])]["points"]
    assert isinstance(final_points, list)
    page(
        "3.7 决赛走势的窗口敏感性",
        f"决赛共有 {final['n_points']} 分；下面三条曲线均按同一批官方逐分记录"
        "计算，只改变历史窗口长度。横轴是分序号，纵轴是 player1 的过去残差均值。"
        "图形可用于复盘，但其局部峰谷不能事后被当成预警成功。",
    )
    for window in (4, 8, 12):
        values = [row[1] for row in point_features(final_points, window)]
        story.append(figure(values, height=45 * mm, label=f"过去 {window} 分残差均值"))
        caption(f"图：决赛 L={window} 的在线走势；每一点均不包含当前分结果。")

    # 8. Final-match segments.
    final_features = point_features(final_points, 8)
    page(
        "3.8 决赛按时间片的描述",
        "为避免只挑选最显眼的峰谷，按逐分顺序把决赛等量划成五个连续区间。"
        "这些区间不是网球的五盘，也没有用最终比分定义切点。"
        "表中平均走势只描述已发生区间里的短期残差；它不能代替前瞻验证。",
    )
    segment_rows: list[list[str]] = []
    for i in range(5):
        begin = i * len(final_points) // 5
        end = (i + 1) * len(final_points) // 5
        segment = final_features[begin:end]
        segment_rows.append(
            [
                str(i + 1),
                f"{begin + 1}-{end}",
                str(len(segment)),
                str(sum(row[2] for row in segment)),
                f"{sum(row[1] for row in segment) / len(segment):+.4f}",
            ]
        )
    tab(
        ["连续片段", "分序号", "分数", "player1 赢分", "平均历史走势"],
        segment_rows,
        [30 * mm, 34 * mm, 22 * mm, 38 * mm, 32 * mm],
    )
    body(
        f"全场 player1 赢得 {final['player1_point_wins']}/{final['n_points']} 分。"
        f"从主窗口第 10 分起，最大正值出现在第 "
        f"{final['largest_prior_flow_for_player1']['point_index_1based']} 分之前，"
        f"最小值出现在第 {final['largest_prior_flow_for_player2']['point_index_1based']} 分之前。"
        "极值由全场事后搜索得到，若据此构造报警规则会产生选择偏差。"
    )
    body(
        "教练真正需要的不是看见一个已发生的波动，而是在该波动出现前"
        "给出可检验的概率预测。因此本研究把描述曲线与后续留出预测分开报告。"
    )

    # 9. Null simulation protocol.
    page(
        "3.9 随机性检验：明确零假设",
        "检验统计量 T=Σ m_t(y_t-p0_t)。正值表示近期正残差与下一分正残差"
        "同向持续，负值表示反向。零假设不是“比赛中一切都随机”，"
        "而是给定实际发球顺序后，下一分服从当前在线基线的顺序伯努利机制。",
    )
    tab(
        ["模拟步骤", "保持固定", "重新生成"],
        [
            ["1", "31 场及每场分数", "每场结果序列"],
            ["2", "各分发球顺序", "伯努利赢家"],
            ["3", "先验与 L=8", "每步 W、N、p0、m"],
            ["4", "统计量定义", "一轮 T"],
            ["5", "随机种子 2024", "500 轮独立轨迹"],
        ],
        [24 * mm, 60 * mm, 72 * mm],
    )
    body(
        "若只打乱已有胜负而不重算在线基线，便会把原序列与模拟序列的状态机制混在一起；"
        "程序因此在每次模拟中从零初始化 W、N 和残差队列，并逐分更新。"
        "零假设也保留实际发球日程，使发球优势不会被错判成异常连胜。"
    )
    body(
        "检验的计算单位是整批比赛的聚合 T。模拟次数固定为 500，"
        "若真实统计量距模拟均值至少和某个模拟值一样远，则该模拟记为极端。"
        "双侧 Monte Carlo p=(极端次数+1)/(500+1)，因此最小可报告 p 为 1/501。"
    )

    # 10. Null result.
    page(
        "3.10 零假设结果与不确定性",
        "同一份官方数据的观测统计量与 500 次模拟分布如下。"
        "分位数只描述在这个指定零假设下模拟到的 T 范围，不是对所有"
        "网球比赛或所有势头定义的置信区间。",
    )
    tab(
        ["量", "结果", "含义"],
        [
            ["观测 T", f"{null['observed']:.3f}", "原始逐分数据"],
            ["模拟均值", f"{null['null_mean']:.3f}", "500 轮平均"],
            ["模拟标准差", f"{null['null_sd']:.3f}", "500 轮离散度"],
            ["2.5% 分位", f"{null['null_q025']:.3f}", "模拟下尾"],
            ["97.5% 分位", f"{null['null_q975']:.3f}", "模拟上尾"],
            ["双侧极端轮数", str(null["extreme_count"]), "有限样本计数"],
            ["双侧 p", f"{null['two_sided_p']:.4f}", "加一修正"],
        ],
        [43 * mm, 34 * mm, 79 * mm],
    )
    body(
        f"观测值 {null['observed']:.3f} 低于模拟的 2.5% 分位 "
        f"{null['null_q025']:.3f}，方向与“正势头会继续正向延续”的简单说法相反。"
        "然而该差异也可能来自未入模的能力变化、比分情境、时间趋势或"
        "在线基线误差。小 p 值并不能把统计关联直接解释为心理因果。"
    )
    body(
        "这里对 L=8 给出一次预先声明的主检验；另外两个窗口仅供稳健性讨论，"
        "不拿多个窗口里最小的 p 值挑选结论。模拟结果可通过固定种子重新生成，"
        "但随机数发生器、程序版本和原始数据哈希仍应随复现记录保存。"
    )

    # 11. Predictive formulation.
    page(
        "3.11 从描述指标到下一分预测",
        "仅有走势曲线不足以回答“能否预测”。本研究把在线基线视为对照预测，"
        "用一个额外参数检验过去走势是否提供增量信息："
        "logit(p1_t)=logit(p0_t)+βm_t。β=0 时回到基线；β 的符号"
        "指示条件关联方向，不等于干预势头后的因果效应。",
    )
    tab(
        ["对象", "计算", "用途"],
        [
            ["训练目标", "逐分伯努利对数似然", "只拟合一个 β"],
            ["预测截距", "固定为 logit(p0_t)", "保留在线发球基线"],
            ["输入走势", "过去 L 分平均残差", "不读取当前分"],
            ["基线损失", "β=0 的 Brier / log loss", "公平对照"],
            ["模型损失", "留出场次上的同两指标", "只测未参与拟合的场次"],
        ],
        [34 * mm, 61 * mm, 61 * mm],
    )
    body(
        "拟合使用一维牛顿更新、有限步长和数值稳定限制；初值 β=0。"
        "限制只用于防止不良数值发散，不借助留出集选择 β。Brier 是"
        "平均平方概率误差，对数损失会更重地惩罚高置信度错误。"
        "两种损失都由每一条测试逐分记录实际计算。"
    )

    # 12. Fold results.
    page(
        "3.12 按整场留出的五折结果",
        "按比赛 ID 排序后，序号模 5 分配测试折；每折只用其他四折的"
        "完整比赛拟合 β。本表把折内样本量、系数与两种损失同时列出，"
        "因此可审查全局轻微改善是否被单一折驱动。",
    )
    tab(
        ["折", "场", "分", "β", "Brier 基线", "Brier 修正"],
        [
            [
                str(item["fold"]),
                str(item["n_matches"]),
                str(item["n"]),
                f"{item['beta']:+.3f}",
                f"{item['brier_baseline']:.5f}",
                f"{item['brier_adjusted']:.5f}",
            ]
            for item in main["fold_results"]
        ],
        [16 * mm, 16 * mm, 20 * mm, 22 * mm, 41 * mm, 41 * mm],
    )
    tab(
        ["折", "对数损失基线", "对数损失修正", "差值（修正-基线）"],
        [
            [
                str(item["fold"]),
                f"{item['logloss_baseline']:.5f}",
                f"{item['logloss_adjusted']:.5f}",
                f"{item['logloss_adjusted'] - item['logloss_baseline']:+.5f}",
            ]
            for item in main["fold_results"]
        ],
        [22 * mm, 42 * mm, 45 * mm, 47 * mm],
    )
    body(
        "折间训练样本不完全相同，因此 β 不是一个统一的全数据估计。"
        "汇总损失按 7,284 个测试逐分加权，不把五折的均值简单等权平均。"
        "每场比赛恰好进入一个测试折，故不存在同场分数同时出现在"
        "该折训练和测试中的问题。"
    )

    # 13-14. Match-level review sorted by impact rather than chronology.
    ranked = sorted(
        main["match_results"],
        key=lambda item: item["brier_adjusted"] - item["brier_baseline"],
    )
    for part, rows in enumerate((ranked[:16], ranked[16:]), 1):
        page(
            f"3.{12 + part} 逐场误差差异排序（{part}/2）",
            "为审查模型的适用边界，把逐场 Brier 差值从最有利到最不利排序。"
            "排序仅用于事后错误分析，不能据此筛选比赛然后重新宣称模型成功。"
            "正值表示加入走势后预测恶化。",
        )
        tab(
            ["比赛 ID", "分数", "Brier 差值", "对数损失差值"],
            [
                [
                    str(item["match_id"]),
                    str(item["n"]),
                    f"{item['brier_adjusted'] - item['brier_baseline']:+.5f}",
                    f"{item['logloss_adjusted'] - item['logloss_baseline']:+.5f}",
                ]
                for item in rows
            ],
            [56 * mm, 20 * mm, 40 * mm, 40 * mm],
        )
        body(
            "该表提示误差具有比赛间异质性：一个额外系数不可能把每场的"
            "战术、体能与对手差异都压进相同走势定义。若要研究何时失效，"
            "必须先定义新的可用特征与分析方案，再在未参与方案设计的比赛上检验。"
        )

    # 15. Calibration.
    page(
        "3.15 概率校准诊断",
        "按基线预测概率排序后，将全部留出逐分分为约等量的十组。"
        "每组报告两模型平均预测与实际 player1 赢分比例。"
        "这些是测试折的预测汇总；分组由基线概率确定，不能解释为"
        "修正模型自动校准的证据。",
    )
    tab(
        ["组", "分数", "基线均值", "修正均值", "实际比例"],
        [
            [
                str(item["decile"]),
                str(item["n"]),
                f"{item['baseline_mean']:.4f}",
                f"{item['adjusted_mean']:.4f}",
                f"{item['observed_rate']:.4f}",
            ]
            for item in main["calibration"]
        ],
        [20 * mm, 24 * mm, 38 * mm, 38 * mm, 36 * mm],
    )
    body(
        "若某组的平均预测与实际比例接近，只能说明该粗分组上的平均误差较小；"
        "不能推论每个球员或每种比分情境都已校准。相邻组的概率范围可能接近，"
        "而样本量仅约七百余分；不宜把个别组的小差异解释成稳定战术信号。"
    )

    # 16. Sensitivity and the unfulfilled swing task.
    page(
        "3.16 窗口敏感性与尚未解决的转势预测",
        "对 L=4、8、12 重复同一按场分组的下一分评估。窗口变化会改变"
        "m_t，但不改变官方输入、测试折或基线构造。对比可检验结论"
        "是否严重依赖人为窗口；它并不构成赛前确定的最优 L 搜索。",
    )
    tab(
        ["L", "β 范围", "Brier 改善", "对数损失改善"],
        [
            [
                str(item["window"]),
                f"{min(item['beta_by_fold']):+.3f} 至 {max(item['beta_by_fold']):+.3f}",
                f"{item['brier_baseline'] - item['brier_adjusted']:+.6f}",
                f"{item['logloss_baseline'] - item['logloss_adjusted']:+.6f}",
            ]
            for item in cv
        ],
        [22 * mm, 50 * mm, 42 * mm, 42 * mm],
    )
    body(
        "8 分窗口的提升很小，12 分窗口的两项指标变差；不存在跨窗口"
        "稳定的大幅预测收益。更重要的是，下一分赢球概率并不等于“走势即将"
        "从正翻负”的事件概率。要回答题目的转势任务，须先预注册事件定义、"
        "预警提前量与误报成本，并按场或按赛事做真正前瞻性测试。"
    )
    body(
        "本论文没有通过这样的事件级验证，所以不编造转势准确率、"
        "不输出虚假的报警阈值，也不把当前通用 benchmark 的失败改写为成功。"
        "现有成果可支持走势描述、限定零假设检验和下一分概率对照；"
        "对转势预警只能给出研究设计，而不能宣称已经解决。"
    )
    tab(
        ["后续验证要素", "必须事先固定的规则"],
        [
            ["事件定义", "例如走势符号持续反转，并限定连续分数；本稿尚未选定"],
            ["预警提前量", "至少提前几分发出，不能在事件后回填报警"],
            ["误报代价", "按每场报警次数及漏报次数共同评价"],
            ["外部测试", "保留完整未参与设计的比赛或新赛季数据"],
        ],
        [43 * mm, 113 * mm],
    )
