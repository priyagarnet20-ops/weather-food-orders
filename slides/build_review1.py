"""Build the Review 1 Presentation Deck for the Weather-Driven Food Orders project (P4 / T21).

Generates slides/review1/review1_deck.pptx — a 10-minute presentation with one section
per team member (P1–P5). P4's section covers EDA, key findings, and visualizations.

Run from the project root:
    python slides/build_review1.py

Author: P4 (EDA, Presentation Chart Set & Review 1 Deck)
"""
from __future__ import annotations
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SLIDES_DIR = ROOT / "slides/review1"
FIGS_DIR = SLIDES_DIR / "figures"
OUT_PATH = SLIDES_DIR / "review1_deck.pptx"

# -----------------------------------------------------------------------
# Color Palette (matched to plotstyle.py)
# -----------------------------------------------------------------------
INK    = RGBColor(0x0B, 0x0B, 0x0B)
INK2   = RGBColor(0x52, 0x51, 0x4E)
BLUE   = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
AQUA   = RGBColor(0x1B, 0xAF, 0x7A)
GRAY   = RGBColor(0x8A, 0x89, 0x83)
BG     = RGBColor(0xFC, 0xFC, 0xFB)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
DARK   = RGBColor(0x18, 0x24, 0x3A)

# Slide dimensions: Widescreen 16:9 (13.33" × 7.5")
W = Inches(13.33)
H = Inches(7.5)


def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    return prs


def blank_slide(prs: Presentation):
    blank_layout = prs.slide_layouts[6]  # blank
    return prs.slides.add_slide(blank_layout)


def add_rect(slide, left, top, width, height, fill_rgb: RGBColor, alpha: int = 255):
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
    from pptx.util import Pt
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
    """Add a colored header bar at top of slide."""
    add_rect(slide, 0, 0, W, Inches(1.15), color)
    add_text(slide, title, Inches(0.35), Inches(0.12), Inches(10), Inches(0.65),
             font_size=26, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        add_text(slide, subtitle, Inches(0.35), Inches(0.72), Inches(9.5), Inches(0.38),
                 font_size=13, bold=False, color=RGBColor(0xCC, 0xCC, 0xCC), align=PP_ALIGN.LEFT)
    if owner_tag:
        add_text(slide, owner_tag, Inches(10.8), Inches(0.22), Inches(2.2), Inches(0.55),
                 font_size=11, bold=True, color=RGBColor(0xFF, 0xD7, 0x00), align=PP_ALIGN.RIGHT)


def footer_bar(slide, note: str = ""):
    add_rect(slide, 0, H - Inches(0.35), W, Inches(0.35), DARK)
    left_text = "Weather-Driven Food Order Patterns · Coimbatore · BA Capstone 2026"
    add_text(slide, left_text, Inches(0.25), H - Inches(0.32), Inches(8), Inches(0.32),
             font_size=8.5, color=RGBColor(0xAA, 0xAA, 0xAA), align=PP_ALIGN.LEFT)
    if note:
        add_text(slide, note, Inches(9.5), H - Inches(0.32), Inches(3.6), Inches(0.32),
                 font_size=8.5, color=RGBColor(0xAA, 0xAA, 0xAA), align=PP_ALIGN.RIGHT)


def bullet_block(slide, bullets: list[str], left, top, width, height,
                 font_size: int = 14, title: str = "", title_color: RGBColor = BLUE):
    """Add a block of bullet-point text, with an optional bold title."""
    from pptx.util import Pt
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
        if title or i > 0:
            p = tf.add_paragraph()
        else:
            p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run()
        indent = "    " if bullet.startswith("  ") else ""
        r.text = indent + bullet.lstrip()
        r.font.size = Pt(font_size)
        r.font.color.rgb = INK2
        p.space_before = Pt(1.5)

    return txb


# -----------------------------------------------------------------------
# SLIDE BUILDER FUNCTIONS
# -----------------------------------------------------------------------

def slide_01_title(prs: Presentation):
    """Slide 1: Title page."""
    sld = blank_slide(prs)
    add_rect(sld, 0, 0, W, H, DARK)
    add_rect(sld, 0, Inches(2.8), W, Inches(2.0), BLUE)

    add_text(sld, "Weather-Driven Food Order Patterns",
             Inches(0.5), Inches(1.0), Inches(12.3), Inches(1.1),
             font_size=40, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(sld, "Coimbatore · Three-Year Analysis (Sep 2023 – Sep 2026)",
             Inches(0.5), Inches(1.95), Inches(12.3), Inches(0.6),
             font_size=18, bold=False, color=RGBColor(0xBB, 0xCC, 0xEE), align=PP_ALIGN.CENTER)

    add_text(sld, "REVIEW 1  ·  Business Analytics Capstone",
             Inches(0.5), Inches(2.88), Inches(12.3), Inches(0.55),
             font_size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(sld, "Real Open-Meteo Weather Data  |  Documented Synthetic Orders  |  Reproducible Pipeline",
             Inches(0.5), Inches(3.38), Inches(12.3), Inches(0.55),
             font_size=14, color=RGBColor(0xFF, 0xFF, 0xDD), align=PP_ALIGN.CENTER)

    team_text = "P1: Weather Data & Forecasting  |  P2: Menu Catalog & Order Generator  |  P3: Feature Engineering & ARM Prep\nP4: Exploratory Data Analysis & Presentation  |  P5: Predictive Modeling"
    add_text(sld, team_text, Inches(0.5), Inches(5.1), Inches(12.3), Inches(0.85),
             font_size=13, color=GRAY, align=PP_ALIGN.CENTER)
    add_text(sld, "October 2026", Inches(5.5), Inches(6.1), Inches(2.3), Inches(0.4),
             font_size=14, color=RGBColor(0xAA, 0xAA, 0xAA), align=PP_ALIGN.CENTER)


def slide_02_agenda(prs: Presentation):
    """Slide 2: Agenda / Team Overview."""
    sld = blank_slide(prs)
    header_bar(sld, "Review 1 · Agenda & Team Assignments", "10-minute capstone overview — one section per member", "TEAM OVERVIEW")
    footer_bar(sld, "Slide 2 / 14")

    add_text(sld, "Presentation Agenda (approx. 10 minutes)", Inches(0.4), Inches(1.25), Inches(12.5), Inches(0.45),
             font_size=16, bold=True, color=DARK)

    sections = [
        (DARK,   "1",  "P1  ·  Weather Data & Dataset Documentation",      "Real Open-Meteo weather collection, data quality validation, 1,097-day dataset profile", "~90 sec"),
        (BLUE,   "2",  "P2  ·  Menu Catalog & Synthetic Order Generator",   "40-item Coimbatore menu, NegBin demand model, basket generator, planted association rules", "~90 sec"),
        (ORANGE, "3",  "P3  ·  Data Cleaning, Feature Engineering & ARM",   "Weather-orders merge, lag/rolling features, rain/heat flags, monsoon phase encoding", "~90 sec"),
        (AQUA,   "4",  "P4  ·  Exploratory Data Analysis  [THIS SECTION]",  "Distribution analysis, weekday & hourly patterns, rain vs dry, correlations, 10 charts", "~90 sec"),
        (GRAY,   "5",  "P5  ·  Predictive Modeling & Evaluation",           "Ridge, Random Forest, XGBoost on order volume, MAE/RMSE/R² comparison, feature importance", "~90 sec"),
    ]

    y_start = Inches(1.72)
    h_block = Inches(0.95)
    gap = Inches(0.06)

    for color, num, title, desc, dur in sections:
        add_rect(sld, 0, y_start, Inches(0.4), h_block, color)
        add_rect(sld, Inches(0.4), y_start, W - Inches(0.4), h_block, RGBColor(0xF5, 0xF5, 0xF4))
        add_text(sld, num, Inches(0.05), y_start + Inches(0.22), Inches(0.3), Inches(0.45),
                 font_size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text(sld, title, Inches(0.5), y_start + Inches(0.05), Inches(10.5), Inches(0.4),
                 font_size=13.5, bold=True, color=DARK, align=PP_ALIGN.LEFT)
        add_text(sld, desc, Inches(0.5), y_start + Inches(0.44), Inches(10.5), Inches(0.38),
                 font_size=11, color=INK2, align=PP_ALIGN.LEFT)
        add_text(sld, dur, Inches(11.8), y_start + Inches(0.25), Inches(1.35), Inches(0.38),
                 font_size=11, bold=True, color=color, align=PP_ALIGN.RIGHT)
        y_start += h_block + gap


def slide_03_p1_dataset(prs: Presentation):
    """Slide 3: P1 — Real Weather + Synthetic Orders Dataset."""
    sld = blank_slide(prs)
    header_bar(sld, "P1  ·  Real Weather, Synthetic Orders, Fully Documented",
               "Data sources · collection method · quality validation · 3-year profile", "P1  ·  T02, T03", DARK)
    footer_bar(sld, "Slide 3 / 14")

    left_bullets = [
        "SOURCE: Open-Meteo Historical Weather API",
        "  Location: 11.00°N, 76.99°E (Coimbatore, CBE)",
        "  Date Range: 24 Sep 2023 – 24 Sep 2026 (1,097 days)",
        "  Variables: Precipitation, Rain, Temp Max/Min/Mean,",
        "    Wind, Humidity, Weather Code, Daylight Duration",
        "",
        "QUALITY CONTROLS PASSED:",
        "  No gaps · No nulls · No duplicate dates",
        "  temp_max >= temp_min on every day",
        "  24 hourly values validated per day",
        "  Offline pytest suite: 18 tests, all passing",
        "",
        "WEATHER PROFILE (3 years):",
        "  Total rain: 3,599 mm across 329 rainy days",
        "  Heavy rain days (>=15mm): 67 days",
        "  Hot days (T_max >= 33°C): 300 days",
        "  Festival days: 41 (Tamil Nadu/Coimbatore calendar)",
    ]
    bullet_block(sld, left_bullets, Inches(0.35), Inches(1.22), Inches(5.8), Inches(5.85),
                 font_size=12, title="Weather Data (REAL — Open-Meteo API)")

    right_bullets = [
        "Orders are SYNTHETIC (no real restaurant data available)",
        "Generator: src/generator.py (P2) — seed=42, reproducible",
        "",
        "DAILY DEMAND MODEL:",
        "  Y_t ~ NegativeBinomial(r=25) around mu_t",
        "  mu_t = 140 x M_weekend x M_rain x M_temp x M_festival",
        "  Weekend Multiplier: 1.30x (Fri-Sat-Sun)",
        "  Rain Multipliers: 1.20x moderate, 1.35x heavy",
        "  Temperature Multiplier: 1.10x (T_max >= 33°C)",
        "  Festival Multiplier: 1.25x",
        "",
        "DATASET SUMMARY:",
        "  193,406 total orders across 1,097 days",
        "  Mean: 176.3 orders/day | Range: 61–384",
        "  Rainy Mean: 203.1 | Hot: 171.7 | Normal: 161.2",
        "  10 weather columns match weather.csv exactly",
    ]
    bullet_block(sld, right_bullets, Inches(6.45), Inches(1.22), Inches(6.55), Inches(5.85),
                 font_size=12, title="Synthetic Orders (P2 Generator)", title_color=ORANGE)


def slide_04_p1_forecast(prs: Presentation):
    """Slide 4: P1 — Time-Series Forecasting Method & Results."""
    sld = blank_slide(prs)
    header_bar(sld, "P1  ·  Time-Series Order-Volume Forecasting",
               "SARIMAX + Prophet rolling-origin backtest — does weather knowledge help?", "P1  ·  T04, T05, T06", DARK)
    footer_bar(sld, "Slide 4 / 14")

    left_bullets = [
        "SERIES PROPERTIES:",
        "  Stationary (ADF p < 0.001, KPSS p >= 0.10)",
        "  Strong weekly pattern (Fri-Sun peaks)",
        "  No detectable yearly trend",
        "",
        "MODELS TESTED:",
        "  Seasonal Naive (same weekday 1 week earlier)",
        "  Weekday Mean baseline",
        "  SARIMAX (no weather / with weather regressors / log)",
        "  Prophet (with festival flags)",
        "",
        "BACKTEST DESIGN:",
        "  131 rolling-origin folds — 7-day forecast horizons",
        "  First 180 days training, walk-forward validation",
        "  Weather given as observed (oracle upper bound)",
    ]
    bullet_block(sld, left_bullets, Inches(0.35), Inches(1.22), Inches(5.8), Inches(5.5),
                 font_size=12, title="Method & Setup (T04, T05)")

    right_bullets = [
        "BEST MODEL: SARIMAX with Weather Regressors",
        "  MAE: 30.7 orders/day (MAPE: 18.6%)",
        "  Weekday-mean baseline: MAE = 34.2",
        "  Seasonal naive: MAE = 48.0",
        "",
        "WEATHER KNOWLEDGE EFFECT:",
        "  Adding weather lowers SARIMAX error by ~2.7 orders/day (-8%)",
        "  Statistically distinguishable, but modest",
        "  Rain/dry error: 34.0 vs 29.3 (rainy days harder to plan)",
        "",
        "OPERATIONAL PLANNING BUFFER:",
        "  Add 49 orders (29%) above forecast to cover 90% of days",
        "  Use 52 orders buffer on rainy days, 47 on dry days",
        "  Forecast is within 0.3 MAE of theoretical best possible",
    ]
    bullet_block(sld, right_bullets, Inches(6.45), Inches(1.22), Inches(6.55), Inches(5.5),
                 font_size=12, title="Results & Business Interpretation (T06)", title_color=BLUE)

    # Embed a relevant results chart
    fig_path = ROOT / "results/fig06_mae_by_model.png"
    if fig_path.exists():
        add_image(sld, fig_path, Inches(0.35), Inches(5.5), width=Inches(5.5))


def slide_05_p2_generator(prs: Presentation):
    """Slide 5: P2 — Menu Catalog & Order Generator."""
    sld = blank_slide(prs)
    header_bar(sld, "P2  ·  Menu Catalog & Weather-Driven Order Generator",
               "40-item Coimbatore menu · NegBin basket engine · planted association rules", "P2  ·  T07, T08, T09, T10", ORANGE)
    footer_bar(sld, "Slide 5 / 14")

    left_bullets = [
        "40 authentic Coimbatore/Kongu dishes across 6 categories:",
        "  Beverages (Hot)       — Filter Coffee, Masala Chai, Sukku Tea",
        "  Snacks & Quick Bites  — Pakora, Bajji, Medu Vada, Bonda",
        "  South Indian Tiffin   — Dosa, Idli, Pongal, Parotta",
        "  Biryani & Mains       — Chicken/Mutton/Veg Biryani, Curd Rice",
        "  Cold Refreshments     — Jigarthanda, Juices, Buttermilk",
        "  Desserts & Ice Cream  — Coconut/Vanilla/Butterscotch Ice Cream",
        "",
        "Price range: Rs.25 (Tea) to Rs.280 (Mutton Biryani)",
        "Weather affinity tag: hot_comfort | cold_refresh | neutral",
        "",
        "MEAL SLOT STRUCTURE:",
        "  Breakfast  07:00-10:30  (15% of orders)",
        "  Lunch      12:00-15:00  (35% of orders)",
        "  Eve Snacks 16:00-18:30  (20% of orders)",
        "  Dinner     19:00-22:30  (30% of orders)",
    ]
    bullet_block(sld, left_bullets, Inches(0.35), Inches(1.22), Inches(6.0), Inches(5.8),
                 font_size=12, title="Menu Architecture (T07)")

    right_bullets = [
        "PLANTED ASSOCIATION RULES (Ground Truth for ARM):",
        "",
        "  RAINY DAY COMBOS (hot_comfort spike):",
        "    Masala Chai -> Onion Pakora     P(rain)=0.58 vs P(dry)=0.12",
        "    Masala Chai -> Milagai Bajji    P(rain)=0.35 vs P(dry)=0.10",
        "    Filter Coffee -> Medu Vada      P(rain)=0.40 vs P(dry)=0.15",
        "",
        "  HOT DAY COMBOS (cold_refresh surge):",
        "    Fresh Lime -> Coconut Ice Cream P(hot)=0.45 vs P(rain)=0.06",
        "    Watermelon Juice -> Vanilla IC  P(hot)=0.38 vs P(rain)=0.05",
        "    Chicken Biryani -> Buttermilk   P(hot)=0.45 vs P(dry)=0.20",
        "",
        "  ALL-WEATHER STAPLE PAIRS:",
        "    Malabar Parotta -> Chicken Chukka  (stable across regimes)",
        "    Ghee Podi Dosa -> Filter Coffee    (all-weather tiffin)",
        "",
        "ARM methods (P3/P5) aim to recover these exact planted rules.",
    ]
    bullet_block(sld, right_bullets, Inches(6.45), Inches(1.22), Inches(6.55), Inches(5.8),
                 font_size=11.5, title="Planted Item Associations & Ground Truth (T09, T10)", title_color=ORANGE)


def slide_06_p3_features(prs: Presentation):
    """Slide 6: P3 — Data Cleaning, Feature Engineering & ARM Prep."""
    sld = blank_slide(prs)
    header_bar(sld, "P3  ·  Data Cleaning, Feature Engineering & ARM Preparation",
               "Weather-orders merge · lag features · rain flags · monsoon phases · FP-Growth setup", "P3  ·  T13, T14, T15, T16, T17", AQUA)
    footer_bar(sld, "Slide 6 / 14")

    left_bullets = [
        "DATA CLEANING (T13):",
        "  Joined orders.csv to weather.csv on date (1,097 rows)",
        "  Parsed date/timestamp types; checked for gaps & nulls",
        "  No duplicates found; no IQR outliers warranting removal",
        "  Log of every transformation step recorded",
        "",
        "FEATURE ENGINEERING (T14):",
        "  Lag-1, Lag-7 order counts (no leakage — past only)",
        "  7-day rolling mean of past orders",
        "  Rain flag (>=2.5mm) + heavy rain flag (>=15mm)",
        "  Rain bins: none / trace / moderate / heavy",
        "  Temperature bands: cool / warm / hot / extreme",
        "  Weekend flag (Fri-Sat-Sun) + festival flag",
        "  Monsoon phase: SW Monsoon (Jun-Sep), NE Monsoon (Oct-Dec)",
        "",
        "FEATURE SELECTION (T15):",
        "  Correlation matrix + Variance Inflation Factor (VIF)",
        "  Multicollinearity checked; PCA not required",
    ]
    bullet_block(sld, left_bullets, Inches(0.35), Inches(1.22), Inches(6.0), Inches(5.8),
                 font_size=12, title="Preprocessing & Feature Engineering")

    right_bullets = [
        "ARM STATUS (T16, T17 — OPEN):",
        "  Preparation and FP-Growth are not yet complete",
        "  No implementation PR or results are available",
        "",
        "REVIEW 2 OWNERSHIP:",
        "  P3: T16 data prep + T17 FP-Growth implementation",
        "  P2: T11 rainy/dry evaluation + T12 threshold sensitivity",
    ]
    bullet_block(sld, right_bullets, Inches(6.45), Inches(1.22), Inches(6.55), Inches(5.8),
                 font_size=12, title="ARM Preparation & Initial Results (T16, T17)", title_color=AQUA)


def slide_07_p4_intro(prs: Presentation):
    """Slide 7: P4 EDA Section Introduction."""
    sld = blank_slide(prs)
    header_bar(sld, "P4  ·  Exploratory Data Analysis — Overview & Method",
               "1,097 days · 193,406 transactions · 40 menu items · Coimbatore weather", "P4  ·  T19, T20", BLUE)
    footer_bar(sld, "Slide 7 / 14")

    add_text(sld, "Five analytical dimensions explored in this EDA:",
             Inches(0.35), Inches(1.25), Inches(12.6), Inches(0.42),
             font_size=15, bold=True, color=DARK)

    dims = [
        (BLUE,   "01", "Distribution Analysis",        "Daily order volumes, basket sizes, order values across the full 3-year sample"),
        (AQUA,   "02", "Temporal Patterns",            "Weekday-of-week demand profiles and hourly meal slot clustering"),
        (ORANGE, "03", "Rain vs. Dry Comparison",      "Quantitative lift analysis, statistical significance testing, intensity gradient"),
        (DARK,   "04", "Item & Category Substitution", "Weather-driven menu preference shifts — hot comfort vs cold refresh substitution"),
        (GRAY,   "05", "Correlation Heatmap",          "Multi-variable Pearson correlation between weather features and order metrics"),
    ]

    y = Inches(1.72)
    for color, num, title, desc in dims:
        add_rect(sld, 0, y, Inches(0.45), Inches(0.85), color)
        add_rect(sld, Inches(0.45), y, W - Inches(0.45), Inches(0.85), RGBColor(0xF4, 0xF4, 0xF2))
        add_text(sld, num, Inches(0.03), y + Inches(0.18), Inches(0.38), Inches(0.45),
                 font_size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text(sld, title, Inches(0.55), y + Inches(0.06), Inches(12.3), Inches(0.38),
                 font_size=14, bold=True, color=DARK, align=PP_ALIGN.LEFT)
        add_text(sld, desc, Inches(0.55), y + Inches(0.44), Inches(12.3), Inches(0.32),
                 font_size=11.5, color=INK2, align=PP_ALIGN.LEFT)
        y += Inches(0.85) + Inches(0.065)


def slide_08_p4_distributions(prs: Presentation):
    """Slide 8: P4 — Order Distribution & Basket Analysis."""
    sld = blank_slide(prs)
    header_bar(sld, "P4  ·  Daily Order Distributions: Volume, Baskets & Spend",
               "Right-skewed demand, 2-item modal basket, Rs.121 median order value", "P4  ·  T19 — Fig 01, 02", BLUE)
    footer_bar(sld, "Slide 8 / 14")

    # Left: Fig 01
    fig01 = FIGS_DIR / "eda_01_order_distribution.png"
    add_image(sld, fig01, Inches(0.25), Inches(1.22), width=Inches(6.5))

    # Right: Fig 02
    fig02 = FIGS_DIR / "eda_02_basket_and_value.png"
    add_image(sld, fig02, Inches(6.85), Inches(1.22), width=Inches(6.3))

    key_bullets = [
        "Mean: 176.3 orders/day  |  Median: 170.0  |  StdDev: 50.5  |  IQR: 66.0  |  Skewness: +0.80",
        "Rainy days average 203.1 orders (+23.2% over dry mean of 164.8)",
        "Modal basket: 2 items (42.0%)  |  Mean basket: 2.11 items  |  Avg spend: Rs.160.7",
    ]
    y_note = H - Inches(1.35)
    for b in key_bullets:
        add_rect(sld, Inches(0.1), y_note, W - Inches(0.2), Inches(0.28), RGBColor(0xEA, 0xF4, 0xFF))
        add_text(sld, "• " + b, Inches(0.25), y_note + Inches(0.03), W - Inches(0.5), Inches(0.24),
                 font_size=11.5, color=DARK, align=PP_ALIGN.LEFT)
        y_note += Inches(0.30)


def slide_09_p4_temporal(prs: Presentation):
    """Slide 9: P4 — Weekday & Hourly Patterns."""
    sld = blank_slide(prs)
    header_bar(sld, "P4  ·  Temporal Demand Patterns: Weekday & Hourly",
               "+33.6% weekend surge · 100% of orders in 4 meal windows", "P4  ·  T19 — Fig 03, 04", BLUE)
    footer_bar(sld, "Slide 9 / 14")

    fig03 = FIGS_DIR / "eda_03_weekday_patterns.png"
    add_image(sld, fig03, Inches(0.25), Inches(1.22), width=Inches(6.5))

    fig04 = FIGS_DIR / "eda_04_hourly_meal_slots.png"
    add_image(sld, fig04, Inches(6.85), Inches(1.22), width=Inches(6.3))

    obs = [
        "Mon-Thu baseline: 154.2 orders/day · Fri-Sun surge: 205.9 orders/day (+33.6%) — Sunday peaks at 209.9",
        "Lunch accounts for 35.0% of all transactions (67,614 orders). Dinner: 30.1%. Snacks: 19.9%. Breakfast: 15.0%.",
        "Zero orders between 22:30-07:00, 10:30-12:00, 15:00-16:00 — kitchen prep & restocking windows.",
    ]
    y_note = H - Inches(1.35)
    for b in obs:
        add_rect(sld, Inches(0.1), y_note, W - Inches(0.2), Inches(0.28), RGBColor(0xEA, 0xF4, 0xFF))
        add_text(sld, "• " + b, Inches(0.25), y_note + Inches(0.03), W - Inches(0.5), Inches(0.24),
                 font_size=11.5, color=DARK, align=PP_ALIGN.LEFT)
        y_note += Inches(0.30)


def slide_10_p4_rain_dry(prs: Presentation):
    """Slide 10: P4 — Rain vs Dry Comparison & Weekday Interaction."""
    sld = blank_slide(prs)
    header_bar(sld, "P4  ·  Rain Effect on Demand: +23.2% Surge, Statistically Significant",
               "Dry: 164.8 · Moderate Rain: 196.7 · Heavy Rain: 228.1 orders/day · Welch t = 11.24, p = 2.23×10⁻²⁶", "P4  ·  T19 — Fig 05, 06", BLUE)
    footer_bar(sld, "Slide 10 / 14")

    fig05 = FIGS_DIR / "eda_05_rain_vs_dry_comparison.png"
    add_image(sld, fig05, Inches(0.25), Inches(1.22), width=Inches(6.5))

    fig06 = FIGS_DIR / "eda_06_weekday_rain_interaction.png"
    add_image(sld, fig06, Inches(6.85), Inches(1.22), width=Inches(6.3))

    obs = [
        "Heavy rain (>=15mm) lifts demand by +38.4% to 228.1 orders/day — compared to 164.8 on dry days.",
        "Rainy-day lift varies +14.7% (Tue) to +33.4% (Mon); rainy Tue avg 174.5 vs dry Sat/Sun avg 194.4.",
        "Welch's t = 11.24, p = 2.23×10⁻²⁶ · Mann-Whitney U p = 9.98×10⁻²⁸; both reject equal distributions.",
    ]
    y_note = H - Inches(1.35)
    for b in obs:
        add_rect(sld, Inches(0.1), y_note, W - Inches(0.2), Inches(0.28), RGBColor(0xEA, 0xF4, 0xFF))
        add_text(sld, "• " + b, Inches(0.25), y_note + Inches(0.03), W - Inches(0.5), Inches(0.24),
                 font_size=11.5, color=DARK, align=PP_ALIGN.LEFT)
        y_note += Inches(0.30)


def slide_11_p4_menu_shifts(prs: Presentation):
    """Slide 11: P4 — Menu Item & Category Substitution."""
    sld = blank_slide(prs)
    header_bar(sld, "P4  ·  Monsoon Menu Effect: Hot Comfort vs Cold Refresh Substitution",
               "Pakora +106% · Vada +118% · Chai +53% on rainy days vs Ice Cream -64% · Juice -56%", "P4  ·  T19 — Fig 07, 08", BLUE)
    footer_bar(sld, "Slide 11 / 14")

    fig07 = FIGS_DIR / "eda_07_item_rain_spikers.png"
    add_image(sld, fig07, Inches(0.25), Inches(1.22), width=Inches(6.5))

    fig08 = FIGS_DIR / "eda_08_category_regime_shifts.png"
    add_image(sld, fig08, Inches(6.85), Inches(1.22), width=Inches(6.3))

    obs = [
        "Hot-comfort items (Chai, Pakora, Vada, Bajji) surge by 52–118% during rainy conditions — matching planted item affinities.",
        "Cold-refresh items (Juices, Ice Creams) collapse by 55–64% in rain — Snacks share doubles from 10.9% to 24.5%.",
        "Beverages (Hot) expand from 12.1% in hot weather to 22.3% in rainy weather. Neutral staples (Biryani, Tiffin) remain stable.",
    ]
    y_note = H - Inches(1.35)
    for b in obs:
        add_rect(sld, Inches(0.1), y_note, W - Inches(0.2), Inches(0.28), RGBColor(0xEA, 0xF4, 0xFF))
        add_text(sld, "• " + b, Inches(0.25), y_note + Inches(0.03), W - Inches(0.5), Inches(0.24),
                 font_size=11.5, color=DARK, align=PP_ALIGN.LEFT)
        y_note += Inches(0.30)


def slide_12_p4_heatmap(prs: Presentation):
    """Slide 12: P4 — Correlation Heatmap + Timeline."""
    sld = blank_slide(prs)
    header_bar(sld, "P4  ·  Weather-Orders Correlation Heatmap & Three-Year Timeline",
               "Precip r=+0.29 · Rain Hours r=+0.25 · Revenue r=+0.98 · Wind r≈0.00", "P4  ·  T19 — Fig 09, 10", BLUE)
    footer_bar(sld, "Slide 12 / 14")

    fig09 = FIGS_DIR / "eda_09_weather_orders_heatmap.png"
    add_image(sld, fig09, Inches(0.25), Inches(1.22), width=Inches(5.9))

    fig10 = FIGS_DIR / "eda_10_monsoon_timeline.png"
    add_image(sld, fig10, Inches(6.35), Inches(1.22), width=Inches(6.75))

    obs = [
        "Precipitation is the primary driver of order volume (r=+0.29, Spearman rho=+0.25). Rain duration adds further signal (r=+0.25).",
        "Daily revenue tracks orders in near-perfect synchrony (r=+0.98). Average order value slightly contracts on rainy days (r=-0.12).",
        "Wind speed and daylight duration are operationally irrelevant (r approx 0.00 and +0.06 respectively).",
    ]
    y_note = H - Inches(1.35)
    for b in obs:
        add_rect(sld, Inches(0.1), y_note, W - Inches(0.2), Inches(0.28), RGBColor(0xEA, 0xF4, 0xFF))
        add_text(sld, "• " + b, Inches(0.25), y_note + Inches(0.03), W - Inches(0.5), Inches(0.24),
                 font_size=11.5, color=DARK, align=PP_ALIGN.LEFT)
        y_note += Inches(0.30)


def slide_13_p5_model(prs: Presentation):
    """Slide 13: P5 — Predictive Modeling & Evaluation."""
    sld = blank_slide(prs)
    header_bar(sld, "P5  ·  Predictive Modeling: Ridge, Random Forest, XGBoost",
               "56-day hold-out test · MAE/RMSE/R² comparison · Feature importance & SHAP", "P5  ·  T25, T26", GRAY)
    footer_bar(sld, "Slide 13 / 14")

    left_bullets = [
        "OBJECTIVE: Predict daily order volume",
        "from weather + calendar + lag features",
        "",
        "FEATURE SET:",
        "  Lagged orders: lag1, lag7, roll7",
        "  Weather: rain flag, heavy_rain, is_hot,",
        "    precip_mm, temp_max_c, humidity_mean_pct",
        "  Calendar: dow (0-6), is_weekend, is_festival",
        "  Monsoon phase: SW / NE / other",
        "",
        "TIME-BASED SPLIT:",
        "  Training: first 1,034 feature-ready days",
        "  Test: last 56 days (8 weeks held out)",
        "  No data leakage — lag features use past only",
        "",
        "TUNING: TimeSeriesSplit cross-validation",
        "  Hyperparameters tuned on training window only",
    ]
    bullet_block(sld, left_bullets, Inches(0.35), Inches(1.22), Inches(5.8), Inches(5.5),
                 font_size=12, title="Model Design (T25)")

    # Model metrics table
    metrics_bullets = [
        "MODEL PERFORMANCE (56-day test set):",
        "",
        "  Model           MAE    RMSE   R2     MAPE",
        "  Seasonal Naive  45.5   60.4  -0.54  27.9%",
        "  Ridge           29.7   37.3  +0.41  17.4%  BEST MAE",
        "  Random Forest   31.4   40.0  +0.33  18.3%",
        "  XGBoost         30.5   37.6  +0.41  17.9%",
        "",
        "KEY FINDINGS (T26):",
        "  Ridge & XGBoost both achieve R2=0.41",
        "  Permutation importance: rain, Fri–Sun, heavy rain lead",
        "  Lag-7 importance is ~0; generator has no daily memory",
        "  Residuals are approximately zero-biased",
        "  Weather inputs alone cut error by ~3–7%; full model beats naive by 34%",
        "",
        "SHAP: Fri–Sun, rain, then temperature rank highest;",
        "festival ranks about fifth.",
    ]
    bullet_block(sld, metrics_bullets, Inches(6.45), Inches(1.22), Inches(6.55), Inches(5.5),
                 font_size=12, title="Evaluation & First Business Findings (T26)", title_color=GRAY)

    fig_path = ROOT / "results/fig12_importance.png"
    if fig_path.exists():
        add_image(sld, fig_path, Inches(9.5), Inches(4.5), width=Inches(3.6))


def slide_14_summary(prs: Presentation):
    """Slide 14: Summary, Key Findings & Next Steps."""
    sld = blank_slide(prs)
    header_bar(sld, "Review 1  ·  Summary of Key Findings & Review 2 Roadmap",
               "What we found · What we recommend · What comes next in Review 2", "ALL MEMBERS")
    footer_bar(sld, "Slide 14 / 14")

    findings_bullets = [
        "FINDING 1 — RAIN IS THE PRIMARY DEMAND DRIVER",
        "  Rainy days surge +23.2% over dry baseline (203 vs 165 orders/day)",
        "  Rain vs dry is significant (Welch p=2.23×10⁻²⁶; MW p=9.98×10⁻²⁸); heavy rain lifts volume +38.4%",
        "",
        "FINDING 2 — COMPOUNDING WEEKEND + RAIN EFFECT",
        "  Weekend + rain represents the operational stress peak (~237 orders on rainy Sundays)",
        "  Rainy Tuesday avg: 174.5; dry Saturday/Sunday avg: 194.4 orders/day",
        "",
        "FINDING 3 — MENU SUBSTITUTION IS DRAMATIC",
        "  Rain doubles Snacks share (10.9% -> 24.5%) and surges Hot Beverages to 22.3%",
        "  Cold Refresh drops from 26.0% (hot weather) to 9.1% (rain); avg ticket stays ~Rs.157–165 across regimes",
        "",
        "FINDING 4 — PREDICTIVE MODELS RECOVER RULES",
        "  Best MAE: 29.7 orders/day (Ridge, R2=0.41) vs naive 45.5 — weather flag is top-3 feature",
        "  SARIMAX + weather lowers SARIMAX error by 8% vs no-weather version",
    ]
    bullet_block(sld, findings_bullets, Inches(0.35), Inches(1.25), Inches(6.5), Inches(5.6),
                 font_size=11.5, title="Consolidated Key Findings (R1)")

    roadmap_bullets = [
        "REVIEW 2 WORK AHEAD:",
        "",
        "  P2: ARM evaluation (T11) + threshold sensitivity (T12)",
        "    Compare rainy vs dry rules; test support/confidence settings",
        "",
        "  P3: ARM preparation + FP-Growth (T16, T17)",
        "    Transaction encoding, itemsets and association rules",
        "",
        "  P1: Time-Series Forecasting — SARIMAX + Prophet",
        "    Rolling-origin backtest, 131 folds (T05, T06)",
        "",
        "  P5: Monsoon Lift Timeline",
        "    14-day rolling lift of weather-linked combos (T27)",
        "",
        "  P4: Interactive Dashboard (Streamlit)",
        "    Forecast view + ARM explorer + monsoon chart (T22, T23, T24)",
        "",
        "  ALL: Final Notebook Integration (T29)",
        "    Final report, combined insights (T18, T28, T29)",
        "",
        "GITHUB: All code in one repo — no external data used",
    ]
    bullet_block(sld, roadmap_bullets, Inches(7.05), Inches(1.25), Inches(6.0), Inches(5.6),
                 font_size=11.5, title="Review 2 Roadmap & Open Questions", title_color=BLUE)


def build_deck() -> Path:
    """Build and save the complete Review 1 presentation deck."""
    prs = new_prs()

    print("Building Review 1 deck...")
    slide_01_title(prs)
    print("  [1/14] Title page")

    slide_02_agenda(prs)
    print("  [2/14] Agenda")

    slide_03_p1_dataset(prs)
    print("  [3/14] P1 — Dataset Documentation")

    slide_04_p1_forecast(prs)
    print("  [4/14] P1 — Forecasting")

    slide_05_p2_generator(prs)
    print("  [5/14] P2 — Menu & Generator")

    slide_06_p3_features(prs)
    print("  [6/14] P3 — Feature Engineering & ARM")

    slide_07_p4_intro(prs)
    print("  [7/14] P4 — EDA Introduction")

    slide_08_p4_distributions(prs)
    print("  [8/14] P4 — Order Distributions")

    slide_09_p4_temporal(prs)
    print("  [9/14] P4 — Weekday & Hourly Patterns")

    slide_10_p4_rain_dry(prs)
    print(" [10/14] P4 — Rain vs Dry")

    slide_11_p4_menu_shifts(prs)
    print(" [11/14] P4 — Menu Substitution")

    slide_12_p4_heatmap(prs)
    print(" [12/14] P4 — Correlation Heatmap & Timeline")

    slide_13_p5_model(prs)
    print(" [13/14] P5 — Predictive Modeling")

    slide_14_summary(prs)
    print(" [14/14] Summary & Roadmap")

    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PATH))
    print(f"\nPresentation deck saved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    return OUT_PATH


if __name__ == "__main__":
    build_deck()
