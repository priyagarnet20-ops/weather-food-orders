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
    left_text = "Weather-Driven Food Order Patterns · Surabaya, Indonesia · BA Capstone 2026"
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
    add_text(sld, "Surabaya, Indonesia · F&B MSME Transaction Analysis (2025)",
             Inches(0.5), Inches(1.95), Inches(12.3), Inches(0.6),
             font_size=18, bold=False, color=RGBColor(0xBB, 0xCC, 0xEE), align=PP_ALIGN.CENTER)

    add_text(sld, "REVIEW 1  ·  Business Analytics Capstone",
             Inches(0.5), Inches(2.88), Inches(12.3), Inches(0.55),
             font_size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(sld, "Real Open-Meteo Weather Data  |  Real POS MSME Transactions (3 Outlets Combined)  |  Reproducible Pipeline",
             Inches(0.5), Inches(3.38), Inches(12.3), Inches(0.55),
             font_size=14, color=RGBColor(0xFF, 0xFF, 0xDD), align=PP_ALIGN.CENTER)

    team_text = "P1: Weather Data & Forecasting  |  P2: Menu Catalog & Preprocessing  |  P3: Feature Engineering & Clean Pipeline\nP4: Exploratory Data Analysis & Visualizations  |  P5: Predictive Modeling"
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
        (DARK,   "1",  "P1  ·  Weather Data & Dataset Documentation",      "Real Open-Meteo weather collection for Surabaya (-7.26°N, 112.75°E), 3-year profile", "~90 sec"),
        (BLUE,   "2",  "P2  ·  Menu Catalog & Multi-Outlet Aggregation",  "96-item Surabaya menu, combining 3 outlets into single-store stream, meal slots", "~90 sec"),
        (ORANGE, "3",  "P3  ·  Data Cleaning, Feature Engineering & Selection", "Weather-orders merge, lag/rolling features, rain/heat flags, NW monsoon phase", "~90 sec"),
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
    """Slide 3: P1 — Real Weather + Real POS Orders Dataset."""
    sld = blank_slide(prs)
    header_bar(sld, "P1  ·  Real Weather & Real Store Transactions, Fully Documented",
               "Data sources · collection method · quality validation · dataset profile", "P1  ·  T02, T03", DARK)
    footer_bar(sld, "Slide 3 / 14")

    left_bullets = [
        "SOURCE: Open-Meteo Historical Weather API",
        "  Location: -7.26°N, 112.75°E (Surabaya, Indonesia)",
        "  Timezone: Asia/Jakarta (WIB, UTC+7)",
        "  Date Range: 24 Sep 2023 – 24 Sep 2026 (1,097 days)",
        "  Variables: Precipitation, Rain, Temp Max/Min/Mean,",
        "    Wind, Humidity, Weather Code, Daylight Duration",
        "",
        "QUALITY CONTROLS PASSED:",
        "  No gaps · No nulls · No duplicate dates",
        "  temp_max >= temp_min on every day",
        "  24 hourly values validated per day",
        "  Offline pytest suite: all tests passing",
        "",
        "WEATHER PROFILE (Surabaya):",
        "  Monsoon Seasonality: NW Monsoon (wet, Nov–Apr)",
        "    and SE Monsoon (dry, May–Oct)",
        "  Average Max Temp: ~31.8°C | Rain Days: ~123/272",
    ]
    bullet_block(sld, left_bullets, Inches(0.35), Inches(1.22), Inches(5.8), Inches(5.85),
                 font_size=12, title="Weather Data (REAL — Open-Meteo API)")

    right_bullets = [
        "Orders are REAL Point-of-Sale Transactions",
        "Source: Surabaya F&B MSME (Mendeley Data / Data in Brief)",
        "",
        "DATA PREPROCESSING & COMBINATION:",
        "  Three Outlets: SHOP001, SHOP002, SHOP003",
        "  Combined into a single restaurant demand stream",
        "  Meal Slots: Breakfast, Lunch, Snacks, Dinner",
        "  Catalog: 96 products across 6 menu categories",
        "",
        "DATASET SUMMARY:",
        "  53,820 total orders across 272 days (Jan–Sep 2025)",
        "  Mean: 197.9 orders/day | Range: 126–282",
        "  Average Order Value: Rp 25,261 (Modal: Rp 23,000)",
        "  100% matched item names between POS and Catalog",
    ]
    bullet_block(sld, right_bullets, Inches(6.45), Inches(1.22), Inches(6.55), Inches(5.85),
                 font_size=12, title="Real Food Orders (Surabaya MSME POS)", title_color=ORANGE)


def slide_04_p1_forecast(prs: Presentation):
    """Slide 4: P1 — Time-Series Forecasting Method & Results."""
    sld = blank_slide(prs)
    header_bar(sld, "P1  ·  Time-Series Demand Profile & Forecasting Setup",
               "Surabaya daily order series profile · SARIMAX & Prophet rolling-origin framework", "P1  ·  T04, T05, T06", DARK)
    footer_bar(sld, "Slide 4 / 14")

    left_bullets = [
        "SERIES PROPERTIES (Surabaya 2025):",
        "  272 consecutive days (Jan–Sep 2025)",
        "  Stationary daily series (mean: 197.9 orders/day, std: 33.0)",
        "  Weekly pattern: Friday–Sunday peak (198.8 vs 197.1 Mon–Thu)",
        "  Weather modulation: Dry days avg 204.5 orders vs Rainy days avg 189.8 (-7.2%)",
        "",
        "MODELS TESTED (Review 2 Framework):",
        "  Seasonal Naive (same weekday 1 week earlier)",
        "  Weekday Mean baseline",
        "  SARIMAX with weather regressors (precip, temp, humidity)",
        "  Prophet with Indonesian holiday & festival regressors",
        "",
        "BACKTEST DESIGN:",
        "  Rolling-origin walk-forward validation (7-day forecast horizons)",
        "  First 216 days training, last 56 days held out for testing",
    ]
    bullet_block(sld, left_bullets, Inches(0.35), Inches(1.22), Inches(5.8), Inches(5.5),
                 font_size=12, title="Method & Setup (T04, T05)")

    right_bullets = [
        "BASELINE VS PREDICTIVE GAINS (56-day hold-out):",
        "  Seasonal Naive: MAE = 17.34 orders/day (MAPE: 9.30%)",
        "  Ridge / ML Baseline: MAE = 13.62 orders/day (MAPE: 7.36%)",
        "  XGBoost: MAE = 14.39 orders/day (MAPE: 7.75%)",
        "  Model outperforms naive baseline by 21.5% MAE reduction",
        "",
        "WEATHER KNOWLEDGE EFFECT:",
        "  Rain suppresses daily foot traffic by 7.2% overall (189.8 vs 204.5 orders/day)",
        "  Moderate rain (2.5–15mm): 190.8 orders/day | Heavy rain (>=15mm): 186.8 orders/day",
        "  Statistically significant shift (Welch t = -3.79, p < 0.001)",
        "",
        "OPERATIONAL PLANNING BUFFER:",
        "  Dry days require higher prep buffers (mean 204.5 orders)",
        "  Rainy days see shift toward hot comfort beverages (Hot Coffee, Artisan Tea)",
        "  Staffing & restocking tailored to weather forecast and meal windows",
    ]
    bullet_block(sld, right_bullets, Inches(6.45), Inches(1.22), Inches(6.55), Inches(5.5),
                 font_size=12, title="Demand Profile & Operational Insights (T06)", title_color=BLUE)

    fig_path = ROOT / "results/fig09_model_metrics.png"
    if fig_path.exists():
        add_image(sld, fig_path, Inches(0.35), Inches(5.2), width=Inches(5.5))


def slide_05_p2_generator(prs: Presentation):
    """Slide 5: P2 — Menu Catalog & Multi-Outlet Aggregation."""
    sld = blank_slide(prs)
    header_bar(sld, "P2  ·  Menu Catalog & Multi-Outlet Aggregation",
               "96-item Surabaya menu · 3 outlets combined into single restaurant · meal slots & price imputation", "P2  ·  T07, T08, T09, T10", ORANGE)
    footer_bar(sld, "Slide 5 / 14")

    left_bullets = [
        "96 authentic Surabaya F&B items across 6 categories:",
        "  Coffee (36.2%)     — Espresso Houseblend, Friendly Coffee, Americano, Latte",
        "  Menu Gedhe (20.3%) — Signature Rice Dishes, Bowls, Savory Main Courses",
        "  Other (16.1%)      — Artisan Drinks, Black Rice, Syrups & Specialties",
        "  Tea (13.7%)        — Artisan Teas, Moody, Fruit Teas, Herbal Brews",
        "  Snacks (9.4%)      — Donat Polos, Croissant, Pastries, Fried Finger Foods",
        "  Add on (4.3%)      — Extra Shots, Syrups, Toppings, Milk Options",
        "",
        "Price range: Rp 3,000 (Add-ons) to Rp 40,000 (Specialty Mains), Median: Rp 20,000",
        "Weather affinity tag: hot_comfort | cold_refresh | neutral",
        "",
        "MEAL SLOT STRUCTURE:",
        "  Breakfast  07:00-11:00  (20.8% of orders, 11,169 txns)",
        "  Lunch      11:00-15:00  (23.7% of orders, 12,752 txns)",
        "  Snacks     15:00-18:00  (19.2% of orders, 10,307 txns)",
        "  Dinner     18:00-23:00  (36.4% of orders, 19,592 txns)",
    ]
    bullet_block(sld, left_bullets, Inches(0.35), Inches(1.22), Inches(6.0), Inches(5.8),
                 font_size=12, title="Menu Architecture & Meal Windows (T07)")

    right_bullets = [
        "COMBINING 3 OUTLETS INTO SINGLE RESTAURANT STREAM (T08):",
        "  Three Outlets: SHOP001, SHOP002, SHOP003 (53,820 orders)",
        "  Aggregated into one unified demand stream for single-store forecasting",
        "  Standardized semicolon-delimited item strings into clean transaction baskets",
        "",
        "CATALOG MAPPING & PRICE IMPUTATION (T09):",
        "  100% of 94 unique transaction item strings matched to official catalog",
        "  1,733 promotional zero-amount transactions imputed from catalog prices",
        "  100% positive order values (Mean: Rp 25,261, Median: Rp 23,000)",
        "",
        "WEATHER & BASKET DYNAMICS (T10):",
        "  Hot coffee and bakery snacks (Donat Polos) gain share during heavy monsoon rain",
        "  Iced coffee and fruit tea expand during hot humid afternoons",
        "  Transaction baskets preprocessed and ready for Review 2 ARM",
    ]
    bullet_block(sld, right_bullets, Inches(6.45), Inches(1.22), Inches(6.55), Inches(5.8),
                 font_size=11.5, title="Multi-Outlet Aggregation & Basket Processing (T08, T09, T10)", title_color=ORANGE)


def slide_06_p3_features(prs: Presentation):
    """Slide 6: P3 — Data Cleaning, Feature Engineering & Pipeline Architecture."""
    sld = blank_slide(prs)
    header_bar(sld, "P3  ·  Data Cleaning, Feature Engineering & Pipeline Architecture",
               "Weather-orders merge · lag features · Indonesian calendar · NW monsoon phases", "P3  ·  T13, T14, T15, T16, T17", AQUA)
    footer_bar(sld, "Slide 6 / 14")

    left_bullets = [
        "DATA CLEANING & MERGING (T13):",
        "  Joined daily POS orders to Surabaya weather on date (272 days, Jan–Sep 2025)",
        "  Parsed timestamps, verified 0 missing dates, zero duplicates, zero IQR errors",
        "  Clean reproducible pipeline saved to data/processed/daily_merged.csv",
        "",
        "FEATURE ENGINEERING (T14):",
        "  Lags (1, 7) and 7-day rolling mean (strictly chronological, zero data leakage)",
        "  Rain flag (>=2.5mm) + Heavy rain flag (>=15mm) + Intensity tiers",
        "  Temperature bands: Cool / Warm / Hot (>=33°C) / Extreme",
        "  Indonesian Public Holidays (holidays.Indonesia()) + Nyepi, Waisak, Idul Fitri",
        "  Surabaya Monsoon Phase: NW Monsoon (wet, Nov–Apr) vs SE Monsoon (dry, May–Oct)",
        "",
        "FEATURE SELECTION (T15):",
        "  Correlation matrix + Variance Inflation Factor (VIF) checked",
        "  Multicollinearity controlled; 28 curated modeling features",
    ]
    bullet_block(sld, left_bullets, Inches(0.35), Inches(1.22), Inches(6.0), Inches(5.8),
                 font_size=12, title="Preprocessing & Feature Engineering (T13, T14, T15)")

    right_bullets = [
        "PIPELINE REPRODUCIBILITY & TEST SUITE:",
        "  62/62 automated unit tests passing in pytest (tests/ directory)",
        "  Modular architecture in src/ (weather, preprocess, features, eda, predict)",
        "  Notebooks execute cleanly end-to-end (03_preprocess, 04_eda, 05_model)",
        "",
        "ARM PREPARATION (T16, T17 — REVIEW 2 ROADMAP):",
        "  Order transactions preprocessed into basket itemsets",
        "  Ready for TransactionEncoder and weather regime segmentation",
        "  FP-Growth and Apriori implementations scheduled for Review 2",
    ]
    bullet_block(sld, right_bullets, Inches(6.45), Inches(1.22), Inches(6.55), Inches(5.8),
                 font_size=12, title="Pipeline Architecture & ARM Preparation (T16, T17)", title_color=AQUA)


def slide_07_p4_intro(prs: Presentation):
    """Slide 7: P4 EDA Section Introduction."""
    sld = blank_slide(prs)
    header_bar(sld, "P4  ·  Exploratory Data Analysis — Overview & Method",
               "272 days · 53,820 real POS orders · 96 menu items · Surabaya tropical weather", "P4  ·  T19, T20", BLUE)
    footer_bar(sld, "Slide 7 / 14")

    add_text(sld, "Five analytical dimensions explored in this EDA:",
             Inches(0.35), Inches(1.25), Inches(12.6), Inches(0.42),
             font_size=15, bold=True, color=DARK)

    dims = [
        (BLUE,   "01", "Distribution Analysis",        "Daily order volumes, basket sizes, Rp spend across the 2025 observation window"),
        (AQUA,   "02", "Temporal Patterns",            "Day-of-week demand profiles and 4 operational meal windows (Breakfast, Lunch, Snacks, Dinner)"),
        (ORANGE, "03", "Rain vs. Dry Comparison",      "Quantitative lift/suppression analysis, Welch t-test & Mann-Whitney U significance, rain intensity tiers"),
        (DARK,   "04", "Item & Category Substitution", "Weather-driven shifts — hot comfort coffee/tea vs iced refreshments and snack pairings"),
        (GRAY,   "05", "Correlation Heatmap",          "Multi-variable Pearson correlation matrix linking rainfall, humidity, wind, and revenue"),
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
               "Symmetric demand (mean 197.9, median 194.0), Rp 25,261 avg spend, 84.9% single-item quick grabs", "P4  ·  T19 — Fig 01, 02", BLUE)
    footer_bar(sld, "Slide 8 / 14")

    fig01 = FIGS_DIR / "eda_01_order_distribution.png"
    add_image(sld, fig01, Inches(0.25), Inches(1.22), width=Inches(6.5))

    fig02 = FIGS_DIR / "eda_02_basket_and_value.png"
    add_image(sld, fig02, Inches(6.85), Inches(1.22), width=Inches(6.3))

    key_bullets = [
        "Mean: 197.9 orders/day  |  Median: 194.0  |  StdDev: 33.0  |  Min: 126  |  Max: 282  |  IQR: 43.2",
        "Dry days average 204.5 orders vs Rainy days average 189.8 (-7.2% foot traffic shift; Welch t = -3.79, p < 0.001)",
        "Modal basket: 1 item (84.9%)  |  2-item baskets: 15.1%  |  Mean order value: Rp 25,261 (Median: Rp 23,000)",
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
    header_bar(sld, "P4  ·  Temporal Demand Patterns: Weekday & Meal Windows",
               "Consistent weekly demand · Peak meal slots: Dinner (36.4%) & Lunch (23.7%)", "P4  ·  T19 — Fig 03, 04", BLUE)
    footer_bar(sld, "Slide 9 / 14")

    fig03 = FIGS_DIR / "eda_03_weekday_patterns.png"
    add_image(sld, fig03, Inches(0.25), Inches(1.22), width=Inches(6.5))

    fig04 = FIGS_DIR / "eda_04_hourly_meal_slots.png"
    add_image(sld, fig04, Inches(6.85), Inches(1.22), width=Inches(6.3))

    obs = [
        "Mon-Thu baseline: 197.1 orders/day · Fri-Sun: 198.8 orders/day — steady demand with weekend leisure visits.",
        "Dinner is the dominant window (36.4%, 19,592 orders), followed by Lunch (23.7%), Breakfast (20.8%), and Snacks (19.2%).",
        "Store operating hours run 07:00–23:00 WIB, with zero transactions during late night prep and closing hours.",
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
    header_bar(sld, "P4  ·  Precipitation Effect: Moderate Foot-Traffic Suppression",
               "Dry: 204.5 · Moderate Rain: 190.8 · Heavy Rain: 186.8 orders/day · Welch t = -3.79, p = 1.89×10⁻⁴", "P4  ·  T19 — Fig 05, 06", BLUE)
    footer_bar(sld, "Slide 10 / 14")

    fig05 = FIGS_DIR / "eda_05_rain_vs_dry_comparison.png"
    add_image(sld, fig05, Inches(0.25), Inches(1.22), width=Inches(6.5))

    fig06 = FIGS_DIR / "eda_06_weekday_rain_interaction.png"
    add_image(sld, fig06, Inches(6.85), Inches(1.22), width=Inches(6.3))

    obs = [
        "Rain moderately suppresses physical foot traffic by -7.2% overall (189.8 vs 204.5 orders/day); Heavy Rain (>=15mm) drops to 186.8.",
        "Rain suppression is consistent across all weekdays (e.g., Wednesday 209.8 down to 182.7, Friday 211.5 down to 186.8).",
        "Welch's t = -3.79, p = 1.89×10⁻⁴ · Mann-Whitney U p = 4.66×10⁻⁴; statistically significant rejection of equal distributions.",
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
    header_bar(sld, "P4  ·  Weather Menu Shifts: Hot Comfort vs Cold Refresh Substitution",
               "Hot coffee & bakery snacks surge on rainy days · Cold drinks expand on hot dry afternoons", "P4  ·  T19 — Fig 07, 08", BLUE)
    footer_bar(sld, "Slide 11 / 14")

    fig07 = FIGS_DIR / "eda_07_item_rain_spikers.png"
    add_image(sld, fig07, Inches(0.25), Inches(1.22), width=Inches(6.5))

    fig08 = FIGS_DIR / "eda_08_category_regime_shifts.png"
    add_image(sld, fig08, Inches(6.85), Inches(1.22), width=Inches(6.3))

    obs = [
        "Coffee is the core anchor (36.1% to 38.1% of items), with hot specialty coffee and warm bakery snacks (Donat Polos) spiking in rainfall.",
        "Hot comfort beverages (Friendly Coffee Hot, Houseblend, Artisan Teas) gain relative share during monsoon downpours.",
        "Menu Gedhe (20.3%) and tea staples maintain strong volume consistency across all weather regimes.",
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
    header_bar(sld, "P4  ·  Weather-Orders Correlation Heatmap & Monsoon Timeline",
               "Revenue r=+0.98 · Wind r=+0.26 · Precip r=-0.17 · Humidity r=-0.20 · Daily spend tracks orders", "P4  ·  T19 — Fig 09, 10", BLUE)
    footer_bar(sld, "Slide 12 / 14")

    fig09 = FIGS_DIR / "eda_09_weather_orders_heatmap.png"
    add_image(sld, fig09, Inches(0.25), Inches(1.22), width=Inches(5.9))

    fig10 = FIGS_DIR / "eda_10_monsoon_timeline.png"
    add_image(sld, fig10, Inches(6.35), Inches(1.22), width=Inches(6.75))

    obs = [
        "Daily revenue tracks order volume in near-perfect synchrony (r = +0.983).",
        "Weather variables show negative correlation with walk-in foot traffic: Precipitation (r = -0.175), Precip Hours (r = -0.180), Humidity (r = -0.199).",
        "Northwest Monsoon phase (Nov–Apr) concentrated rainy days in early 2025; dry season (May–Oct) exhibits higher average daily volumes.",
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
        "  Monsoon phase: NW Monsoon (wet) vs SE Monsoon (dry)",
        "",
        "TIME-BASED SPLIT:",
        "  Training: first 216 days (Jan–Aug 2025)",
        "  Test: last 56 days (Aug–Sep 2025 held out)",
        "  No data leakage — lag features use past only",
        "",
        "TUNING: TimeSeriesSplit cross-validation",
        "  Hyperparameters tuned on training window only",
    ]
    bullet_block(sld, left_bullets, Inches(0.35), Inches(1.22), Inches(5.8), Inches(5.5),
                 font_size=12, title="Model Design (T25)")

    metrics_bullets = [
        "MODEL PERFORMANCE (56-day test set):",
        "  Model           MAE    RMSE   MAPE",
        "  Seasonal Naive  17.34  21.81  9.30%",
        "  Ridge           13.62  17.31  7.36%  BEST MAE (-21.5%)",
        "  XGBoost         14.39  18.79  7.75%",
        "  Random Forest   15.00  20.30  8.12%",
        "",
        "KEY FINDINGS (T26):",
        "  Ridge & XGBoost achieve lowest error (13.62 & 14.39 MAE)",
        "  Importance: lag7, Sunday, Friday, heavy_rain, is_rainy",
        "  Weather signals provide actionable guidance for kitchen prep",
        "  SHAP confirms heavy rain moderates foot traffic volume",
    ]
    bullet_block(sld, metrics_bullets, Inches(6.45), Inches(1.22), Inches(6.55), Inches(3.55),
                 font_size=11, title="Evaluation & First Business Findings (T26)", title_color=GRAY)

    fig_path = ROOT / "results/fig12_importance.png"
    if fig_path.exists():
        add_image(sld, fig_path, Inches(8.6), Inches(4.85), width=Inches(4.1))


def slide_14_summary(prs: Presentation):
    """Slide 14: Summary, Key Findings & Next Steps."""
    sld = blank_slide(prs)
    header_bar(sld, "Review 1  ·  Summary of Key Findings & Review 2 Roadmap",
               "What we found · What we recommend · What comes next in Review 2", "ALL MEMBERS")
    footer_bar(sld, "Slide 14 / 14")

    findings_bullets = [
        "FINDING 1 — RAIN MODERATES FOOT TRAFFIC (-7.2%)",
        "  Rainy days average 189.8 orders vs 204.5 on dry days (p = 1.89×10⁻⁴)",
        "  Heavy rain (>=15mm) drops to 186.8 orders/day; physical store foot traffic contracts",
        "",
        "FINDING 2 — MEAL WINDOWS SHOW DISTINCT DEMAND PROFILE",
        "  Dinner is the peak operational window (36.4%), followed by Lunch (23.7%)",
        "  High demand stability across weekdays (197.1 Mon–Thu vs 198.8 Fri–Sun)",
        "",
        "FINDING 3 — MENU SUBSTITUTION FAVORS HOT COMFORT & SNACKS",
        "  Coffee anchors 36.2% of orders; hot coffee and pastries surge during rainfall",
        "  Cold beverages peak during hot dry afternoons; avg ticket stays ~Rp 25,000",
        "",
        "FINDING 4 — MACHINE LEARNING OUTPERFORMS NAIVE BASELINE",
        "  Ridge achieves 13.62 MAE (7.36% MAPE), beating Seasonal Naive (17.34 MAE) by 21.5%",
        "  Real weather and lag features provide significant operational value for kitchen prep",
    ]
    bullet_block(sld, findings_bullets, Inches(0.35), Inches(1.25), Inches(6.5), Inches(5.6),
                 font_size=11.5, title="Consolidated Key Findings (R1)")

    roadmap_bullets = [
        "REVIEW 2 WORK AHEAD:",
        "",
        "  P2: ARM evaluation (T11) & threshold sensitivity (T12)",
        "    Compare rainy vs dry rules; test support/confidence settings",
        "",
        "  P3: ARM preparation + FP-Growth (T16, T17)",
        "    Transaction encoding, itemsets and association rules",
        "",
        "  P1: Time-Series Forecasting — SARIMAX + Prophet",
        "    Rolling-origin backtest, 7-day forecast horizons (T05, T06)",
        "",
        "  P5: Monsoon Lift Timeline & Business Strategy",
        "    Track lift of weather-linked combos; combo promotions (T27, T28)",
        "",
        "  P4: Interactive Dashboard (Streamlit)",
        "    Forecast view + ARM explorer + monsoon chart (T22, T23, T24)",
        "",
        "  ALL: Final Report & Notebook Integration (T18, T29)",
        "    Complete documentation and top-to-bottom reproducibility",
        "",
        "GITHUB: 100% real data, fully reproducible pipeline, 62/62 tests passing",
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
