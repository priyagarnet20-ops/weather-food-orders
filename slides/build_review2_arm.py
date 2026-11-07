"""Build the Review 2 Presentation Slides for Method 1 / ARM (P3 & P2 / T16, T17, T11, T12).

Generates slides/review2/review2_arm_deck.pptx — widescreen presentation slides covering:
- Slide 1: T11 Rainy vs. Dry Day Rule Evaluation & Weather Affinity Recovery (Coimbatore)
- Slide 2: T12 Rule Threshold Sensitivity Grid & Parameter Selection
- Slide 3: Synthesis & Actionable Operational Recommendations

Target: Coimbatore, Tamil Nadu, India (11.00°N, 76.99°E)
Run from project root:
    python slides/build_review2_arm.py

Author: P3 (Data Cleaning, Feature Engineering & ARM Preparation)
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
    left_text = "Weather-Driven Food Order Patterns · Coimbatore, Tamil Nadu · Review 2 (Method 1: ARM)"
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


def slide_01_t11_rainy_vs_dry(prs: Presentation):
    """Slide 1: T11 Rainy vs Dry Rule Evaluation & Affinity Recovery."""
    sld = blank_slide(prs)
    header_bar(sld, "P3 & P2  ·  Task T11: Rainy vs. Dry Day Association Rule Mining",
               "Evaluating weather-conditioned item pairings across 82,990 multi-item transactions",
               "P3 & P2  ·  T16, T17, T11", DARK)
    footer_bar(sld, "Review 2 Deliverable · Method 1 (ARM)")

    # Left: Fig 14
    fig14 = ROOT / "results/fig14_arm_rain_vs_dry.png"
    if fig14.exists():
        add_image(sld, fig14, Inches(0.35), Inches(1.25), width=Inches(7.2))

    # Right: Analytical Commentary
    bullets = [
        "TRANSACTION UNIVERSE & ENCODING (T16 / T17):",
        "  • 82,990 multi-item customer baskets (item_count >= 2).",
        "  • Rainy segment: 23,926 orders (precip >= 2.5 mm).",
        "  • Dry segment: 59,064 orders (precip < 2.5 mm).",
        "",
        "RECOVERING THE GENERATOR'S PLANTED PAIRS (T11):",
        "  • Chai -> Pakora: Conf 0.37 rainy vs 0.24 dry | Supp 6.3% vs 1.8%",
        "    Lift: 2.56 rainy vs 4.75 dry (both > 1, proving genuine affinity).",
        "  • Chai -> Samosa: Conf 0.39 rainy vs 0.27 dry | Supp 6.6% vs 2.1%",
        "    Lift: 1.87 rainy vs 5.05 dry (strong monsoon snacking pull).",
        "  • Coffee -> Vada: Conf 0.37 rainy vs 0.33 dry | Supp 7.5% vs 2.2%",
        "    Lift: 2.85 rainy vs 3.58 dry (classic afternoon tiffin pairing).",
        "  • Ice Cream -> Cold Coffee: Conf 0.28 rainy vs 0.39 dry",
        "    (Opposite shift: stronger in heat / dry weather, as planted).",
        "  • Biryani -> Buttermilk: Conf ~0.30 on both rainy and dry days",
        "    (Controls validate model: stable across weather regimes).",
        "",
        "WHY LIFT DROPS IN RAIN (FACULTY FAQ):",
        "  Lift = Conf / P(B). On rainy days, Pakora popularity surges overall",
        "  (14% of rainy orders vs 5% dry). Dividing by that larger base rate",
        "  lowers Lift, even though Chai buyers take Pakora 1.5x more often.",
        "  Confidence & Support are the true indicators of weather affinity.",
    ]
    bullet_block(sld, bullets, Inches(7.75), Inches(1.25), Inches(5.25), Inches(5.8),
                 font_size=10, title="Key Findings: Confidence, Support & Lift", title_color=ORANGE)


def slide_02_t12_threshold_sensitivity(prs: Presentation):
    """Slide 2: T12 Rule Threshold Sensitivity Grid & Parameter Selection."""
    sld = blank_slide(prs)
    header_bar(sld, "P3 & P2  ·  Task T12: Rule Threshold Sensitivity Analysis",
               "Evaluating rule stability across a 5x5 support-confidence grid (80 menu SKUs)",
               "P3 & P2  ·  T12, T17", DARK)
    footer_bar(sld, "Review 2 Deliverable · Method 1 (ARM)")

    # Left: Fig 15
    fig15 = ROOT / "results/fig15_arm_threshold_sensitivity.png"
    if fig15.exists():
        add_image(sld, fig15, Inches(0.35), Inches(1.25), width=Inches(7.2))

    # Right: Sensitivity Analysis Table & Commentary
    findings = [
        "SENSITIVITY ANALYSIS (T12):",
        "  5x5 Grid across Support (0.004–0.015) & Confidence (0.02–0.20):",
        "  • At min_support = 0.004: 172 frequent pairs, 400+ rules.",
        "  • At min_support = 0.008: 94 frequent pairs, 180+ rules.",
        "  • At min_support = 0.015: 42 frequent pairs, high stability.",
        "",
        "METHODOLOGICAL INSIGHTS:",
        "  1. Menu Catalog Diversity (80 items):",
        "     Average SKU occurrence is ~2.4%; pair co-occurrences",
        "     follow a natural power-law distribution.",
        "",
        "  2. Support Starvation Boundary:",
        "     Setting support >= 0.020 starves rule discovery to < 10 rules.",
        "     Setting support <= 0.002 introduces low-frequency noise.",
        "",
        "  3. Optimal Operational Parameter Selection:",
        "     Sweet spot: min_support = 0.004 to 0.006 with",
        "     min_confidence = 0.05 to 0.10.",
        "     Balances statistical reliability (n >= 330–500 orders)",
        "     while capturing high-affinity weather pairs robustly.",
    ]
    bullet_block(sld, findings, Inches(7.75), Inches(1.25), Inches(5.25), Inches(5.7),
                 font_size=11, title="Sensitivity Grid & Parameter Selection", title_color=BLUE)


def slide_03_recommendations(prs: Presentation):
    """Slide 3: Strategic Business Recommendations from Method 1."""
    sld = blank_slide(prs)
    header_bar(sld, "P3  ·  Review 2: Strategic Business Recommendations (ARM)",
               "Translating association rules & weather confidence into kitchen and delivery revenue strategies",
               "P3  ·  Method 1 (ARM)", DARK)
    footer_bar(sld, "Review 2 Deliverable · Coimbatore Operations")

    col1 = [
        "RECOMMENDATION 1 — RAINY-DAY BUNDLE PROMOTIONS:",
        "  Automatically trigger dynamic delivery bundles when rainfall >= 2.5mm:",
        "  • 'Monsoon Comfort Combo': Chai + Pakora @ ₹75 (Conf 0.37, Supp 6.3%)",
        "  • 'Afternoon Tiffin Special': Coffee + Vada @ ₹65 (Conf 0.37, Supp 7.5%)",
        "  • 'Rainy Evening Snacks': Chai + Samosa @ ₹70 (Conf 0.39, Supp 6.6%)",
        "  Provides 10–14% combo discounts to lift Average Order Value (AOV).",
        "",
        "RECOMMENDATION 2 — KITCHEN PRE-BATCHING & INVENTORY:",
        "  Hot comfort snack orders spike sharply during monsoon afternoons.",
        "  Cloud kitchens should pre-fry base batches of Vada and Bajjis",
        "  between 15:30 and 16:30 on wet days, cutting prep time from 18 to 6 mins.",
    ]
    bullet_block(sld, col1, Inches(0.35), Inches(1.25), Inches(6.1), Inches(5.7),
                 font_size=11.5, title="Operational Bundling & Kitchen Workflow", title_color=ORANGE)

    col2 = [
        "RECOMMENDATION 3 — ONLINE STOREFRONT DYNAMIC HERO BANNERS:",
        "  On Swiggy / Zomato merchant homepages:",
        "  • Rainy Hours: Hero banner features hot artisan teas, filter coffee,",
        "    and warm tiffin snacks. Suppress cold beverage ads.",
        "  • Hot Dry Afternoons: Switch storefront to cold refresh banners",
        "    (Fresh Lime Juice, Tender Coconut Ice Cream, Cold Coffee).",
        "",
        "RECOMMENDATION 4 — STREAMLIT RULES EXPLORER INTEGRATION:",
        "  Incorporate T12 parameter selection into the P4 Streamlit dashboard:",
        "  Provide kitchen managers with interactive support/confidence sliders",
        "  to explore emerging item pairings as seasonal menu items rotate.",
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
