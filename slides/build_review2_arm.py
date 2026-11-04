"""Build the Review 2 Presentation Slides for Method 1 / ARM (P2 / T11 & T12).

Generates slides/review2/review2_arm_deck.pptx — widescreen presentation slides covering:
- Slide 1: T11 Rainy vs. Dry Day Rule Evaluation & Weather Affinity Recovery
- Slide 2: T12 Rule Threshold Sensitivity Grid & Parameter Selection
- Slide 3: Synthesis & Actionable Operational Recommendations

Run from project root:
    python slides/build_review2_arm.py

Author: P2 (Order Generator, Menu Catalog & Rule Evaluation)
"""
from __future__ import annotations

import shutil
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SLIDES_DIR = ROOT / "slides/review2"
FIGS_DIR = SLIDES_DIR / "figures"
OUT_PATH = SLIDES_DIR / "review2_arm_deck.pptx"

# Colors
DARK   = RGBColor(0x18, 0x24, 0x3A)
BLUE   = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
AQUA   = RGBColor(0x1B, 0xAF, 0x7A)
GRAY   = RGBColor(0x8A, 0x89, 0x83)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
INK    = RGBColor(0x0B, 0x0B, 0x0B)
INK2   = RGBColor(0x52, 0x51, 0x4E)

W = Inches(13.33)
H = Inches(7.5)


def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    return prs


def blank_slide(prs: Presentation):
    blank_layout = prs.slide_layouts[6]
    return prs.slides.add_slide(blank_layout)


def add_rect(slide, left, top, width, height, fill_rgb: RGBColor):
    shape = slide.shapes.add_shape(1, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_rgb
    shape.line.fill.background()
    return shape


def add_text(
    slide, text: str, left, top, width, height,
    font_size: int = 18, bold: bool = False, color: RGBColor = INK,
    align=PP_ALIGN.LEFT, wrap: bool = True, italic: bool = False
):
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf = txb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txb


def add_image(slide, img_path: Path, left, top, width=None, height=None):
    if img_path.exists():
        if width and height:
            slide.shapes.add_picture(str(img_path), left, top, width, height)
        elif width:
            slide.shapes.add_picture(str(img_path), left, top, width=width)
        elif height:
            slide.shapes.add_picture(str(img_path), left, top, height=height)
        else:
            slide.shapes.add_picture(str(img_path), left, top)


def header_bar(slide, title: str, subtitle: str = "", owner_tag: str = "", color: RGBColor = DARK):
    add_rect(slide, 0, 0, W, Inches(1.15), color)
    add_text(slide, title, Inches(0.35), Inches(0.12), Inches(10), Inches(0.65),
             font_size=24, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        add_text(slide, subtitle, Inches(0.35), Inches(0.72), Inches(9.5), Inches(0.38),
                 font_size=13, bold=False, color=RGBColor(0xCC, 0xCC, 0xCC), align=PP_ALIGN.LEFT)
    if owner_tag:
        add_text(slide, owner_tag, Inches(10.8), Inches(0.22), Inches(2.2), Inches(0.55),
                 font_size=11, bold=True, color=RGBColor(0xFF, 0xD7, 0x00), align=PP_ALIGN.RIGHT)


def footer_bar(slide, note: str = ""):
    add_rect(slide, 0, H - Inches(0.35), W, Inches(0.35), DARK)
    left_text = "Weather-Driven Food Order Patterns · Surabaya, Indonesia · Review 2 (Method 1: ARM)"
    add_text(slide, left_text, Inches(0.25), H - Inches(0.32), Inches(8), Inches(0.32),
             font_size=8.5, color=RGBColor(0xAA, 0xAA, 0xAA), align=PP_ALIGN.LEFT)
    if note:
        add_text(slide, note, Inches(9.5), H - Inches(0.32), Inches(3.6), Inches(0.32),
                 font_size=8.5, color=RGBColor(0xAA, 0xAA, 0xAA), align=PP_ALIGN.RIGHT)


def bullet_block(slide, bullets: list[str], left, top, width, height,
                 font_size: int = 14, title: str = "", title_color: RGBColor = BLUE):
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf = txb.text_frame
    tf.word_wrap = True

    if title:
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run()
        r.text = title
        r.font.size = Pt(font_size + 1)
        r.font.bold = True
        r.font.color.rgb = title_color

    for i, bullet in enumerate(bullets):
        p = tf.add_paragraph() if (title or i > 0) else tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run()
        indent = "    " if bullet.startswith("  ") else ""
        r.text = indent + bullet.lstrip()
        r.font.size = Pt(font_size)
        r.font.color.rgb = INK2
        p.space_before = Pt(1.5)

    return txb


# -----------------------------------------------------------------------------
# SLIDES
# -----------------------------------------------------------------------------

def slide_01_t11_rainy_vs_dry(prs: Presentation):
    """Slide 1: T11 — Rainy vs. Dry Day Rule Evaluation & Weather Affinity Recovery."""
    sld = blank_slide(prs)
    header_bar(sld, "P2  ·  T11: Rainy vs. Dry Day Rule Evaluation",
               "Association Rule Mining across weather regimes · Lift comparison & rainy-only combo discovery",
               "P2  ·  T11", ORANGE)
    footer_bar(sld, "Method 1 Evaluation (Rubric: 3 pts)")

    # Left: Embed Fig 14
    fig14 = ROOT / "results/fig14_arm_rain_vs_dry.png"
    if fig14.exists():
        add_image(sld, fig14, Inches(0.35), Inches(1.25), width=Inches(7.2))

    # Right: Analytical findings
    findings = [
        "SEGMENTATION & SAMPLE SIZES:",
        "  Multi-item baskets (item_count >= 2): N = 8,153",
        "  Rainy Days (>=2.5mm): 3,572 orders across 123 days",
        "  Dry Days (<2.5mm):   4,581 orders across 149 days",
        "",
        "KEY FINDINGS (T11 EVALUATION):",
        "  1. Rainy-Only Combos Emerge Strongly:",
        "     408 rules appear exclusively on rainy days (min_sup=0.0008).",
        "     Top rainy-only pair: Sunset -> Gula Gula (Lift = 3.22)",
        "     Thai Tea -> Butterscotch (Lift = 3.18, Conf = 4.2%)",
        "",
        "  2. Recovery of Weather Affinities:",
        "     Hot Comfort pairs surge in rain (hot coffee + baked snack):",
        "     Pisang Coklat -> Apple Pie Latte (Lift = 2.76 in rain)",
        "     Friendly Coffee (Hot) + Biru (Lift = 2.83 in rain)",
        "",
        "  3. Cold Refreshment Expansion on Dry Days:",
        "     Candy Latte (Ice) + Donat Polos (Lift = 2.37 on dry days)",
        "     Chocolate Ice + Summer (Lift = 1.95 on dry afternoons)",
        "",
        "  4. Operational Impact:",
        "     Rain shifts demand to warm drinks and fried/baked snacks;",
        "     Enables automated rainy-day combo bundling in POS.",
    ]
    bullet_block(sld, findings, Inches(7.75), Inches(1.25), Inches(5.25), Inches(5.7),
                 font_size=11, title="Method 1 Evaluation & Empirical Recovery", title_color=ORANGE)


def slide_02_t12_threshold_sensitivity(prs: Presentation):
    """Slide 2: T12 — Rule Threshold Sensitivity Grid across min_support / min_confidence."""
    sld = blank_slide(prs)
    header_bar(sld, "P2  ·  T12: Rule Threshold Sensitivity Grid",
               "Rule counts and top rules across min_support (0.0005–0.0010) and min_confidence (0.01–0.10)",
               "P2  ·  T12", BLUE)
    footer_bar(sld, "Method 1 Evaluation (Rubric: 3 pts)")

    # Left: Embed Fig 15
    fig15 = ROOT / "results/fig15_arm_threshold_sensitivity.png"
    if fig15.exists():
        add_image(sld, fig15, Inches(0.35), Inches(1.25), width=Inches(7.2))

    # Right: Sensitivity Analysis Table & Commentary
    findings = [
        "SENSITIVITY MATRIX (min_sup x min_conf):",
        "  Support   Conf 0.01   Conf 0.02   Conf 0.03   Conf 0.04",
        "  0.0005        348         348         175          15",
        "  0.0006        348         348         175          15",
        "  0.0007        130         130         126          15",
        "  0.0008         22          22          22          14",
        "  0.0010          0           0           0           0",
        "",
        "METHODOLOGICAL INSIGHTS (T12):",
        "  1. Menu Catalog Diversity (96 items):",
        "     Mean item frequency is ~1.5%; pair co-occurrences",
        "     are spread over 3,687 unique empirical pairs.",
        "",
        "  2. Support Starvation Boundary:",
        "     Setting support >= 0.0010 (>=8 orders) starves rule",
        "     discovery completely due to basket dispersion.",
        "",
        "  3. Optimal Operational Parameter Selection:",
        "     Sweet spot: min_support = 0.0006 to 0.0008 with",
        "     min_confidence = 0.02 to 0.03.",
        "     Balances statistical reliability (n >= 5–7 orders)",
        "     while capturing high-lift pairs (Lift up to 2.37).",
    ]
    bullet_block(sld, findings, Inches(7.75), Inches(1.25), Inches(5.25), Inches(5.7),
                 font_size=11, title="Sensitivity Grid & Parameter Selection", title_color=BLUE)


def slide_03_recommendations(prs: Presentation):
    """Slide 3: Strategic Business Recommendations from Method 1."""
    sld = blank_slide(prs)
    header_bar(sld, "P2  ·  Review 2: Strategic Business Recommendations (ARM)",
               "Translating association rules & weather lift into kitchen and POS revenue strategies",
               "P2  ·  T11, T12", DARK)
    footer_bar(sld, "Review 2 Deliverable · Surabaya MSME Operations")

    col1 = [
        "RECOMMENDATION 1 — RAINY-DAY BUNDLE PROMOTIONS:",
        "  Automatically trigger dynamic POS bundles when rainfall >= 2.5mm:",
        "  • 'Monsoon Warmth': Apple Pie Latte (Hot) + Pisang Coklat (Lift 2.76)",
        "  • 'Rainy Coffee & Donut': Friendly Coffee (Hot) + Donat Polos (Lift 2.10)",
        "  • 'Afternoon Tea Combo': Thai Tea (Hot) + Butterscotch (Lift 3.18)",
        "  Provides 10–15% combo discounts to recover walk-in foot traffic drop.",
        "",
        "RECOMMENDATION 2 — KITCHEN PREP & HOLDING INVENTORY:",
        "  Warm snack orders (Pisang Coklat, Donat Polos, Croissant) spike during",
        "  monsoon afternoons. Pre-baking and holding inventory should be scaled",
        "  up 30 minutes prior to forecast rainfall events.",
    ]
    bullet_block(sld, col1, Inches(0.35), Inches(1.25), Inches(6.1), Inches(5.7),
                 font_size=11.5, title="Operational Bundling & Kitchen Workflow", title_color=ORANGE)

    col2 = [
        "RECOMMENDATION 3 — ONLINE STOREFRONT DYNAMIC BANNERS:",
        "  On food delivery applications (GoFood / GrabFood):",
        "  • Rainy Hours: Hero banner features hot artisan teas, double shot coffees,",
        "    and warm snack pairings. Suppress iced drink banners.",
        "  • Hot Dry Afternoons: Switch storefront to cold refresh banners",
        "    (Candy Latte Ice, Summer, Jus Semangka, Lemon Tea).",
        "",
        "RECOMMENDATION 4 — THRESHOLD MONITORING DASHBOARD:",
        "  Incorporate T12 parameter selection into the P4 Streamlit dashboard:",
        "  Provide managers with interactive support/confidence sliders",
        "  to explore emerging item pairings as menu SKUs rotate.",
    ]
    bullet_block(sld, col2, Inches(6.85), Inches(1.25), Inches(6.1), Inches(5.7),
                 font_size=11.5, title="Digital Channel & Dashboard Integration", title_color=BLUE)


def build_deck() -> Path:
    prs = new_prs()
    print("Building Review 2 ARM presentation slides...")

    slide_01_t11_rainy_vs_dry(prs)
    print("  [1/3] Slide 1: T11 Rainy vs Dry Rule Evaluation")

    slide_02_t12_threshold_sensitivity(prs)
    print("  [2/3] Slide 2: T12 Rule Threshold Sensitivity Grid")

    slide_03_recommendations(prs)
    print("  [3/3] Slide 3: Strategic Business Recommendations")

    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Copy presentation figures
    for fig_name in ["fig14_arm_rain_vs_dry.png", "fig15_arm_threshold_sensitivity.png"]:
        src = ROOT / "results" / fig_name
        if src.exists():
            shutil.copy(src, FIGS_DIR / fig_name)

    prs.save(str(OUT_PATH))
    print(f"\nPresentation deck saved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    return OUT_PATH


if __name__ == "__main__":
    build_deck()
