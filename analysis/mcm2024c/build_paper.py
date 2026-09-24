"""Build the Chinese independent study PDF from checked official data."""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from analysis.mcm2024c.analyze import FINAL_MATCH_ID, analyze, load_matches, point_features

FONT_NAME = "NotoSC"
INK = colors.HexColor("#17324D")
ACCENT = colors.HexColor("#0D7890")
FAINT = colors.HexColor("#E8F1F4")
MUTED = colors.HexColor("#51636F")


class FlowFigure(Flowable):
    def __init__(self, values: list[float], width: float = 166 * mm, height: float = 72 * mm):
        super().__init__()
        self.values = values
        self.width = width
        self.height = height

    def draw(self) -> None:
        canvas = self.canv
        left, bottom = 37, 20
        plot_width = self.width - 47
        plot_height = self.height - 32
        canvas.setStrokeColor(colors.HexColor("#D5E1E8"))
        for level in (-0.4, -0.2, 0.0, 0.2, 0.4):
            y = bottom + (level + 0.5) * plot_height
            canvas.line(left, y, left + plot_width, y)
            canvas.setFont(FONT_NAME, 7)
            canvas.setFillColor(MUTED)
            canvas.drawRightString(left - 5, y - 2, f"{level:+.1f}")
        canvas.setStrokeColor(INK)
        canvas.line(left, bottom, left, bottom + plot_height)
        canvas.line(left, bottom, left + plot_width, bottom)
        canvas.setFont(FONT_NAME, 7)
        for point in (1, 100, 200, 300, len(self.values)):
            x = left + (point - 1) / (len(self.values) - 1) * plot_width
            canvas.setFillColor(MUTED)
            canvas.drawCentredString(x, bottom - 12, str(point))
        canvas.setLineWidth(1.3)
        canvas.setStrokeColor(ACCENT)
        path = canvas.beginPath()
        for index, value in enumerate(self.values):
            x = left + index / (len(self.values) - 1) * plot_width
            y = bottom + (value + 0.5) * plot_height
            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        canvas.drawPath(path)
        canvas.setFont(FONT_NAME, 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(left, self.height - 8, "发球校正的过去 8 分残差均值")
        canvas.drawRightString(left + plot_width, 2, "分序号")


def styles() -> dict[str, ParagraphStyle]:
    candidates = [
        Path(os.environ["MM_CJK_FONT"]) if "MM_CJK_FONT" in os.environ else None,
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ]
    font_path = next((path for path in candidates if path is not None and path.is_file()), None)
    if font_path is None:
        raise FileNotFoundError("Noto Sans CJK font not found; set MM_CJK_FONT")
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(font_path)))
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title_sc",
            parent=sample["Title"],
            fontName=FONT_NAME,
            fontSize=20,
            leading=29,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=13,
        ),
        "subtitle": ParagraphStyle(
            "subtitle_sc",
            parent=sample["Normal"],
            fontName=FONT_NAME,
            fontSize=10,
            leading=17,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "h1_sc",
            parent=sample["Heading1"],
            fontName=FONT_NAME,
            fontSize=13,
            leading=20,
            textColor=INK,
            spaceBefore=14,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "h2_sc",
            parent=sample["Heading2"],
            fontName=FONT_NAME,
            fontSize=10.5,
            leading=17,
            textColor=ACCENT,
            spaceBefore=9,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body_sc",
            parent=sample["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.2,
            leading=16.3,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=7,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "small_sc",
            parent=sample["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.2,
            leading=13.2,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "caption_sc",
            parent=sample["Normal"],
            fontName=FONT_NAME,
            fontSize=8,
            leading=12,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
    }


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def page_decoration(canvas: object, doc: object) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(FAINT)
    canvas.line(23 * mm, height - 18 * mm, width - 23 * mm, height - 18 * mm)
    canvas.setFont(FONT_NAME, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(23 * mm, height - 15 * mm, "MathModel AI · 2024 MCM C 独立复算")
    canvas.drawRightString(width - 23 * mm, 14 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def result_table(items: list[object], text_style: ParagraphStyle) -> Table:
    header = ["历史窗口", "Brier 基线", "Brier 修正", "对数损失基线", "对数损失修正"]
    header_style = ParagraphStyle("table_header_sc", parent=text_style, textColor=colors.white)
    cells: list[list[object]] = [[paragraph(x, header_style) for x in header]]
    for item in items:
        assert isinstance(item, dict)
        cells.append(
            [
                str(item["window"]),
                f"{item['brier_baseline']:.6f}",
                f"{item['brier_adjusted']:.6f}",
                f"{item['logloss_baseline']:.6f}",
                f"{item['logloss_adjusted']:.6f}",
            ]
        )
    table = Table(cells, colWidths=[28 * mm, 30 * mm, 30 * mm, 34 * mm, 34 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 0), (-1, 0), INK),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F6FAFB")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D4E0E6")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def build(official_csv: Path, results_json: Path, output_pdf: Path) -> None:
    checked = analyze(official_csv)
    recorded = json.loads(results_json.read_text(encoding="utf-8"))
    if checked != recorded:
        raise ValueError("recorded results differ from independent recomputation")
    if checked["n_points"] != 7284 or checked["n_matches"] != 31:
        raise ValueError("official data shape differs from the reviewed source")
    st = styles()
    cv = checked["cross_validation"]
    null = checked["null_test"]
    final = checked["final_match"]
    service = checked["service"]
    assert isinstance(cv, list) and isinstance(null, dict)
    assert isinstance(final, dict) and isinstance(service, dict)
    main_cv = cv[1]
    assert isinstance(main_cv, dict)
    matches = load_matches(official_csv)
    points = matches[FINAL_MATCH_ID]["points"]
    assert isinstance(points, list)
    flow = [row[1] for row in point_features(points, 8)]

    story: list[object] = []
    add = story.append
    add(Spacer(1, 18 * mm))
    add(paragraph("网球比赛的“势头”能预测吗？", st["title"]))
    add(paragraph("2024 MCM C 题中文独立复算研究 · 非正式竞赛提交件", st["subtitle"]))
    add(paragraph("摘要", st["h1"]))
    add(
        paragraph(
            "采用 COMAP 提供的 2023 年温网男单逐分数据（31 场、7,284 分），本文先用每场比赛、"
            "每名发球方截至上一分的历史结果建立在线基线，再将过去 8 分的超额赢分均值定义为可观察的"
            "“比赛走势”。该量描述局部偏离，不等同心理动量。发球方赢得 "
            f"{service['server_wins']:,}/{service['server_points']:,} 分"
            f"（{100 * service['server_win_rate']:.1f}%）。"
            "在保留发球顺序的顺序伯努利模拟中，短程持续性统计量呈负方向偏离零动量基线"
            f"（双侧蒙特卡洛 p={null['two_sided_p']:.3f}）。按比赛分组的五折留出检验显示，"
            f"8 分窗口加入走势后的 Brier 损失从 {main_cv['brier_baseline']:.6f} 降至 "
            f"{main_cv['brier_adjusted']:.6f}，改善极小；12 分窗口并未改善。"
            "因此数据不足以支持“连胜会稳定增强下一分胜率”的教练决策规则。"
            "分析与代码可复现，但项目的通用 C 题 benchmark 仍因时序模型与标量"
            "核验契约不匹配而失败。",
            st["body"],
        )
    )
    add(paragraph("关键词：网球逐分数据；发球优势；比赛走势；分组验证；随机基线", st["small"]))
    add(paragraph("阅读须知", st["h2"]))
    add(
        paragraph(
            "这是针对真题的独立统计研究，不是 MathModel AI 通用流水线已通过科学验证的自动生成论文；"
            "不具备参赛队伍控制号和完整 AI 使用记录，不能冒充官方提交件。",
            st["body"],
        )
    )
    add(PageBreak())

    add(paragraph("1 题目、数据与可核验范围", st["h1"]))
    add(
        paragraph(
            "COMAP 2024 MCM C 要求解释温网比赛中的走势，检验连胜是否可由随机性解释，"
            "预判何时可能转势，并给教练提出建议[1]。官方 CSV 的 SHA-256 是 "
            f"{checked['source_sha256']}。数据共 31 场、7,284 分；示例决赛 "
            f"{final['player1']} 对 {final['player2']}（match_id={final['match_id']}）含 "
            f"{final['n_points']} 分。局内比分包含“AD”，不能按普通浮点数解释；"
            "本文基线只使用发球方和"
            "逐分赢家，不把局内比分的编码混入连续变量。",
            st["body"],
        )
    )
    add(paragraph("2 方法：只用过去信息", st["h1"]))
    add(
        paragraph(
            "令 y_t=1 表示该分由表中 player1 赢得，s_t 是该分发球方。对每场比赛的两种发球方"
            "分别维护截至 t-1 分的赢分数 W 和总分数 N。在线基线为 "
            "p0_t=(10+W)/(20+N)。固定的 Beta(10,10) 先验使开局估计可定义；它是建模选择，"
            "不是由本题优化出的参数。残差 r_t=y_t-p0_t，走势 m_t 为前 8 个残差的均值，"
            "不包含当前分或未来分。只在同一比赛内更新状态，比赛间不串接时间序列。",
            st["body"],
        )
    )
    add(
        paragraph(
            "随机性检验采用 T=Σ m_t r_t。零假设按观测到的发球顺序逐分生成伯努利结果，"
            "每生成一分都重算在线基线；500 次独立模拟，种子 2024。双侧 p 值采用包含观测值"
            "的 (+1)/(B+1) 规则。该检验只针对这一定义的顺序发球基线，未排除比分、疲劳、"
            "对手调整和球员状态等混杂。",
            st["body"],
        )
    )
    add(
        paragraph(
            "预测模型采用 logit(p1_t)=logit(p0_t)+βm_t，仅估计一个额外系数 β。"
            "将 31 场比赛按固定顺序分成五折，整场留出，训练与测试不共享比赛；测试场次的"
            "p0_t 只用该场已发生的分数更新，模拟现场可得信息。以 Brier 损失和对数损失同"
            "无走势的 p0_t 比较。另对 4、12 分窗口作敏感性检查，不根据最终比分调参。",
            st["body"],
        )
    )
    add(PageBreak())
    add(paragraph("3 结果：观察到起伏，但预测增益很小", st["h1"]))
    add(
        paragraph(
            "图 1 给出题目关注的阿尔卡拉斯—德约科维奇决赛。"
            "纵轴是发球校正后的过去 8 分均值，不是胜赛概率。",
            st["body"],
        )
    )
    add(FlowFigure(flow))
    add(paragraph("图 1  2023 温网男单决赛的在线走势。每一点只依赖之前的逐分结果。", st["caption"]))
    high = final["largest_prior_flow_for_player1"]
    low = final["largest_prior_flow_for_player2"]
    assert isinstance(high, dict) and isinstance(low, dict)
    add(
        paragraph(
            f"从第 10 分起看，player1 最强正向走势在第 {high['point_index_1based']} 分前"
            f"（m={high['value']:+.3f}）；最强负向走势在第 {low['point_index_1based']} 分前"
            f"（m={low['value']:+.3f}）。这只是局部描述，不能从图线的转折推断心理机制或"
            "回溯性挑选“关键时刻”的预测成功。",
            st["body"],
        )
    )
    add(paragraph("表 1  按整场比赛留出的五折逐分预测结果（越低越好）", st["caption"]))
    add(result_table(cv, st["small"]))
    add(Spacer(1, 3 * mm))
    add(
        paragraph(
            "主窗口 8 分时，Brier 仅改善 "
            f"{main_cv['brier_baseline'] - main_cv['brier_adjusted']:.6f}；"
            f"对数损失仅改善 {main_cv['logloss_baseline'] - main_cv['logloss_adjusted']:.6f}。"
            "五折 β 均为负，表示这套定义下过去正残差与下一分赢分率略呈反向关系。"
            "12 分窗口 Brier 反而增加。既然增益对窗口敏感且幅度微弱，本文不提出“高走势"
            "即继续押注”的行动阈值。",
            st["body"],
        )
    )
    add(
        paragraph(
            f"顺序零假设下 T={null['observed']:.3f}，模拟均值 {null['null_mean']:.3f}，"
            f"标准差 {null['null_sd']:.3f}，双侧蒙特卡洛 p={null['two_sided_p']:.3f}。"
            "负向偏离反驳的是本模型下的“无额外短程关联”，并不证明选手存在可利用的"
            "“反势头”，更不能证明心理因果。",
            st["body"],
        )
    )
    add(PageBreak())

    add(paragraph("4 解释、泛化与限制", st["h1"]))
    add(
        paragraph(
            "过去研究本就呈现不同层级的证据：O'Donoghue 与 Brown 的 13 场发球得分序列"
            "未发现超过随机预期的连胜[2]；Meier 等在局级破发和换边休息情境下发现动量相关"
            "效应[3]。两者研究对象与识别策略不同，不能相互简单否定。Prieto-Lage 等的"
            "2021 年大满贯数据则表明发球条件明显影响赢分率[4]，支持本文先控制发球方。"
            "本题样本发球方赢分率为 67.3%，但不能把文献的场地细分比例移植为本题结果。",
            st["body"],
        )
    )
    add(
        paragraph(
            "模型只按同场发球方估计基线，没有引入球员能力、局内比分、第一/第二发球、"
            "赛事轮次与体能。Beta 先验强度、窗口长度及 500 次模拟都影响推断；p 值不是"
            "对所有‘势头’定义的全局检验。留出检验覆盖不同男子草地比赛，但不含女子赛、"
            "其他场地或其他运动的数据，因此这些场景只能提出待验证假设。本文检验的是"
            "下一分预测，不是经独立验证的多分“转势事件”预警器；教练不应将图中极值"
            "解释为前瞻性的精确转折点。",
            st["body"],
        )
    )
    add(paragraph("5 给教练的简报", st["h1"]))
    add(
        paragraph(
            "先问谁发球，再谈连胜。发球方在本样本赢得约三分之二的分数，连续得分不应"
            "直接解释为超出发球与实力基线的“势头”。在本研究的 8 分窗口下，过去正向"
            "偏离没有稳定提高下一分的预测概率；对已出现的连胜保持战术纪律，不因图形"
            "振荡就仓促改变高质量的发球、接发球方案。",
            st["body"],
        )
    )
    add(
        paragraph(
            "如需现场辅助，可把走势曲线作为复盘提醒，而非自动换策略的指令：出现明显"
            "波动时，结合一发成功率、非受迫失误、体能和比分压力查看具体技术原因。"
            "针对转势的可执行预警还需要赛前固定事件定义、更多跨赛事数据和真正前瞻"
            "验证；现有模型不提供可靠的单点报警阈值。",
            st["body"],
        )
    )
    add(paragraph("参考文献", st["h1"]))
    references = [
        "[1] COMAP. 2024 MCM Problem C: Momentum in Tennis. 官方题面与数据说明。"
        "https://www.contest.comap.com/undergraduate/contests/mcm/contests/2024/"
        "problems/2024_MCM_Problem_C.pdf",
        "[2] O'Donoghue P, Brown E. Sequences of service points and the misperception "
        "of momentum in elite tennis. International Journal of Performance Analysis in "
        "Sport, 2009, 9(1): 113-127. DOI: 10.1080/24748668.2009.11868468.",
        "[3] Meier P, Flepp R, Ruedisser M, Franck E. Separating psychological momentum "
        "from strategic momentum: Evidence from men's professional tennis. Journal of "
        "Economic Psychology, 2020, 78: 102269. DOI: 10.1016/j.joep.2020.102269.",
        "[4] Prieto-Lage I, et al. Match analysis and probability of winning a point in "
        "elite men's singles tennis. PLOS ONE, 2023, 18(9): e0286076. "
        "DOI: 10.1371/journal.pone.0286076.",
    ]
    for reference in references:
        add(paragraph(html.escape(reference), st["small"]))
    add(paragraph("复现与证据边界", st["h1"]))
    add(
        paragraph(
            "使用 analysis/mcm2024c/analyze.py 读取原始官方 CSV，生成 results.json；"
            "build_paper.py 在成稿前重新计算全部统计量并要求与 results.json 完全一致。"
            "固定随机种子 2024。源码与数据哈希公开，原始官方数据不在本报告中重新分发。"
            "文献的出版信息及可支持的具体论断记录在 references.md。",
            st["body"],
        )
    )
    add(paragraph("AI 使用与正式提交限制", st["h1"]))
    add(
        paragraph(
            "MathModel AI 和 Codex 辅助拟定分析步骤、编写与审查程序、组织中文文字。"
            "关键统计量由公开的可运行程序从原始文件计算；本文不把模型口头输出当作数据证据。"
            "这段说明不等于 COMAP 所要求的完整 AI Use Report，也没有虚构队伍控制号。"
            "在补齐真实参赛信息、完整使用记录及正式提交审查前，不能把本 PDF 标作合规竞赛提交。",
            st["body"],
        )
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=23 * mm,
        rightMargin=23 * mm,
        topMargin=23 * mm,
        bottomMargin=21 * mm,
        title="2024 MCM C 网球势头中文独立复算研究",
        author="MathModel AI",
    )
    document.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_csv", type=Path)
    parser.add_argument("results_json", type=Path)
    parser.add_argument("output_pdf", type=Path)
    args = parser.parse_args()
    build(args.official_csv, args.results_json, args.output_pdf)


if __name__ == "__main__":
    main()
