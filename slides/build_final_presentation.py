"""Build the Final Presentation Deck for 23CSE452 Business Analytics (Group 9).

Generates Final_Project_Presentation.pptx in the repository root and slides/
Widescreen 16:9 presentation (13.33 x 7.5 in) covering:
- Cover slide (Team members, roll numbers, links)
- Executive Summary & Problem Statement
- Dataset Provenance, Calibration & Validation
- Unit 1: Exploratory Data Analysis & Weekday/Weather Dynamics
- Unit 1: Hourly Meal Slots & Monsoon Menu Shifts
- Unit 2: Feature Engineering & Preprocessing
- Unit 2: Predictive Modelling & Supervised ML
- Unit 2 / Comparative: 11-Model Selection & Statistical Ties
- Unit 3 (Method 1): Time-Series Forecasting (SARIMAX & Prophet)
- Unit 3 (Method 2): Weather-Segmented Association Rule Mining
- Unit 3 Comparison & Integrated Business Insights
- Interactive Application (Streamlit Dashboard)
- Operational Buffer & Strategic Business Recommendations
- Project Management, GitHub Projects Board & Individual Contributions
- Limitations, Conclusion & Academic References
"""
from __future__ import annotations

import shutil
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
VIS_DIR = ROOT / "visualizations"
OUT_ROOT = ROOT / "Final_Project_Presentation.pptx"
OUT_SLIDES = ROOT / "slides/Final_Project_Presentation.pptx"

# Colors
DARK   = RGBColor(0x18, 0x24, 0x3A)
BLUE   = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
AQUA   = RGBColor(0x1B, 0xAF, 0x7A)
GRAY   = RGBColor(0x8A, 0x89, 0x83)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
INK    = RGBColor(0x0B, 0x0B, 0x0B)
INK2   = RGBColor(0x3A, 0x39, 0x36)
CARD_BG = RGBColor(0xF5, 0xF6, 0xF8)
GOLD   = RGBColor(0xF1, 0xC4, 0x0F)

W = Inches(13.33)
H = Inches(7.5)


def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    return prs


def blank_slide(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


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


def add_image(slide, img_name: str, left, top, width=None, height=None):
    img_path = VIS_DIR / img_name
    if not img_path.exists():
        img_path = ROOT / "results" / img_name
    if img_path.exists():
        if width and height:
            slide.shapes.add_picture(str(img_path), left, top, width, height)
        elif width:
            slide.shapes.add_picture(str(img_path), left, top, width=width)
        elif height:
            slide.shapes.add_picture(str(img_path), left, top, height=height)
        else:
            slide.shapes.add_picture(str(img_path), left, top)


def header_bar(slide, title: str, subtitle: str = "", unit_tag: str = "", color: RGBColor = DARK):
    add_rect(slide, 0, 0, W, Inches(1.15), color)
    add_text(slide, title, Inches(0.4), Inches(0.12), Inches(10.2), Inches(0.65),
             font_size=24, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        add_text(slide, subtitle, Inches(0.4), Inches(0.72), Inches(10.0), Inches(0.38),
                 font_size=13, bold=False, color=RGBColor(0xCC, 0xCC, 0xCC), align=PP_ALIGN.LEFT)
    if unit_tag:
        add_text(slide, unit_tag, Inches(10.6), Inches(0.22), Inches(2.35), Inches(0.55),
                 font_size=11, bold=True, color=GOLD, align=PP_ALIGN.RIGHT)


def footer_bar(slide, note: str = ""):
    add_rect(slide, 0, H - Inches(0.35), W, Inches(0.35), DARK)
    left_text = "Weather-Driven Food Order Patterns (Coimbatore) · 23CSE452 Business Analytics · Group 9"
    add_text(slide, left_text, Inches(0.35), H - Inches(0.30), Inches(8.5), Inches(0.30),
             font_size=8.5, color=RGBColor(0xAA, 0xAA, 0xAA), align=PP_ALIGN.LEFT)
    right_text = note if note else "Live App: weather-food-ord-smj647r3xjpffhdkb4j2df.streamlit.app"
    add_text(slide, right_text, Inches(8.8), H - Inches(0.30), Inches(4.2), Inches(0.30),
             font_size=8.5, color=RGBColor(0xAA, 0xAA, 0xAA), align=PP_ALIGN.RIGHT)


def bullet_block(slide, bullets: list[str], left, top, width, height,
                 font_size: int = 13, title: str = "", title_color: RGBColor = BLUE):
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf = txb.text_frame
    tf.word_wrap = True

    if title:
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run()
        r.text = title
        r.font.size = Pt(font_size + 1.5)
        r.font.bold = True
        r.font.color.rgb = title_color
        p.space_after = Pt(4)

    for i, bullet in enumerate(bullets):
        p = tf.add_paragraph() if (title or i > 0) else tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        is_sub = bullet.startswith("  ")
        r = p.add_run()
        r.text = ("    • " if is_sub else "• ") + bullet.lstrip()
        r.font.size = Pt(font_size - 1 if is_sub else font_size)
        r.font.color.rgb = INK2
        p.space_before = Pt(1.5)
        p.space_after = Pt(2.5)
    return txb


# -----------------------------------------------------------------------
# SLIDES BUILDER
# -----------------------------------------------------------------------

def slide_01_cover(prs: Presentation):
    s = blank_slide(prs)
    add_rect(s, 0, 0, W, H, DARK)
    add_rect(s, 0, 0, Inches(0.4), H, BLUE)

    add_text(s, "23CSE452 BUSINESS ANALYTICS · CAPSTONE PROJECT 2026",
             Inches(1.0), Inches(0.8), Inches(11.5), Inches(0.4),
             font_size=13, bold=True, color=GOLD)
    add_text(s, "Weather-Driven Food Order Patterns in Coimbatore",
             Inches(1.0), Inches(1.2), Inches(11.5), Inches(1.1),
             font_size=32, bold=True, color=WHITE)
    add_text(s, "A Data-Driven Framework for Restaurant Demand Forecasting & Menu Basket Dynamics",
             Inches(1.0), Inches(2.3), Inches(11.5), Inches(0.5),
             font_size=16, italic=True, color=RGBColor(0xDD, 0xDD, 0xDD))

    # Team Card
    add_rect(s, Inches(1.0), Inches(3.1), Inches(6.0), Inches(3.4), RGBColor(0x22, 0x30, 0x4B))
    add_text(s, "STUDENT TEAM (GROUP 9)", Inches(1.2), Inches(3.3), Inches(5.6), Inches(0.35),
             font_size=12, bold=True, color=GOLD)
    members = [
        "P1 Priyadharshini (CB.SC.U4CSE23438) — Weather API & Time Series",
        "P2 Edavalapati Aashiq (CB.SC.U4CSE23560) — Order Synthesis & Calibration",
        "P3 Sai Sri Satvic L (CB.SC.U4CSE23225) — Preprocessing, Features & ARM",
        "P4 Swasthika P S (CB.SC.U4CSE23350) — EDA & Interactive Dashboard",
        "P5 Deepak S (CB.SC.U4CSE23267) — Predictive Models & Validation",
    ]
    for i, m in enumerate(members):
        add_text(s, m, Inches(1.2), Inches(3.7 + i * 0.5), Inches(5.6), Inches(0.4),
                 font_size=12, color=WHITE)

    # Project Links & Metadata Card
    add_rect(s, Inches(7.3), Inches(3.1), Inches(5.0), Inches(3.4), RGBColor(0x22, 0x30, 0x4B))
    add_text(s, "FACULTY GUIDE & VERIFIED DELIVERABLES", Inches(7.5), Inches(3.3), Inches(4.6), Inches(0.35),
             font_size=12, bold=True, color=GOLD)
    add_text(s, "Faculty Guide: Mrs. Rohini S\nAmrita School of Computing, Coimbatore",
             Inches(7.5), Inches(3.7), Inches(4.6), Inches(0.7),
             font_size=13, color=WHITE)
    add_text(s, "Live Interactive Streamlit Dashboard:\nhttps://weather-food-ord-smj647r3xjpffhdkb4j2df.streamlit.app/",
             Inches(7.5), Inches(4.5), Inches(4.6), Inches(0.8),
             font_size=11, color=RGBColor(0x70, 0xB5, 0xFF))
    add_text(s, "GitHub Projects Agile Board:\nhttps://github.com/users/priyagarnet20-ops/projects/1/views/2",
             Inches(7.5), Inches(5.4), Inches(4.6), Inches(0.8),
             font_size=11, color=RGBColor(0x70, 0xB5, 0xFF))


def slide_02_executive_summary(prs: Presentation):
    s = blank_slide(prs)
    header_bar(s, "Executive Summary & Business Objectives",
               "Can weather intelligence improve daily restaurant demand planning and kitchen operations?",
               "Overview")
    footer_bar(s)

    col1 = [
        "BUSINESS PROBLEM IN COIMBATORE:",
        "  Restaurants face high demand volatility from sudden monsoon showers,",
        "  extreme summer heatwaves, and weekend dining habits.",
        "  Unanticipated demand shifts cause costly food waste (over-prep) or",
        "  stockouts and lost revenue during order surges.",
        "",
        "THREE CORE RESEARCH QUESTIONS:",
        "  1. How does weather interact with calendar cycles (weekday vs. weekend)?",
        "  2. Does adding weather covariates improve daily order predictions?",
        "  3. Can weather-segmented item associations optimize menu bundling and prep?",
    ]
    bullet_block(s, col1, Inches(0.4), Inches(1.35), Inches(6.0), Inches(5.6),
                 font_size=13, title="Business Problem & Research Scope", title_color=BLUE)

    col2 = [
        "KEY QUANTITATIVE FINDINGS:",
        "  • Dual Weather Effect: Rain lifts weekday orders by +54% (delivery convenience)",
        "    but suppresses weekend orders by -47% (dining out cancellations).",
        "  • Superior Forecasting: SARIMAX with weather cuts MAE by 60% vs naive",
        "    baselines (saving 23.5 orders/day of error; 95% CI: 21.5–25.5).",
        "  • Model Equivalence: Ridge, Count models, and SARIMAX are statistically tied",
        "    over a 365-day backtest (MAE 14.99–15.08).",
        "  • Weather Basket Affinity: Chai + Pakora/Samosa confidence surges in rain,",
        "    while Ice Cream and Fresh Juices decline by over 50%.",
        "  • Operational Planning Buffer: A buffer of +25 orders covers 90% of days.",
    ]
    bullet_block(s, col2, Inches(6.8), Inches(1.35), Inches(6.1), Inches(5.6),
                 font_size=13, title="Core Empirical Findings & Impact", title_color=ORANGE)


def slide_03_dataset(prs: Presentation):
    s = blank_slide(prs)
    header_bar(s, "Dataset Collection, Calibration & Quality Audit",
               "1,097 days of observed Open-Meteo weather paired with evidence-calibrated synthetic orders",
               "P1 & P2 · Data")
    footer_bar(s)

    bullets = [
        "PRIMARY WEATHER HARVESTING (Open-Meteo API):",
        "  • Daily & hourly data for Coimbatore (11.00°N, 76.99°E) from Sep 2023 to Sep 2026 (1,097 days).",
        "  • Variables: Precipitation (mm), rain hours, max/min temperature, humidity, wind speed, weather code.",
        "",
        "CALIBRATION ON REAL EVIDENCE:",
        "  • Ground Truth 1: 464 actual Swiggy orders from a Coimbatore restaurant (Aug–Sep 2026).",
        "  • Ground Truth 2: 27-person structured consumer dining survey in Coimbatore.",
        "  • Evidence Derivations: Menu pricing, item catalog (80 items), hourly meal slots, and weather multipliers.",
        "",
        "RIGOROUS VALIDATION SUITE (T10):",
        "  • 25 of 26 empirical checks pass against the Swiggy report and survey.",
        "  • All 5 volume effects recovered: weekday rain x1.56, weekend rain x0.53, weekend x2.45, heat x1.06, festival x0.87.",
    ]
    bullet_block(s, bullets, Inches(0.4), Inches(1.35), Inches(6.3), Inches(5.6),
                 font_size=12, title="Data Sources & Validation", title_color=BLUE)
    add_image(s, "fig_v4_volume_effects.png", Inches(7.0), Inches(1.6), width=Inches(5.9))


def slide_04_unit1_eda(prs: Presentation):
    s = blank_slide(prs)
    header_bar(s, "Unit 1: Exploratory Analysis — Rain × Weekday Interaction",
               "Overall correlation hides opposite weather behavior on weekdays vs weekends",
               "Unit 1 · P4")
    footer_bar(s)

    bullets = [
        "DEMAND DISTRIBUTION (182,696 orders across 1,097 days):",
        "  • Mean 166.5 orders/day, median 157.0 (range 73 to 354). Right-skewed volume.",
        "  • Two-humped distribution driven by heavy weekend dining surges.",
        "",
        "THE SIMPSON'S-LIKE INTERACTION EFFECT:",
        "  • Pooled Rain Correlation: r = -0.10, falsely suggesting rain decreases orders.",
        "  • Stratified Reality:",
        "    - Weekdays: Rain increases orders from 116.5 to 179.3 (+53.9%).",
        "    - Weekends: Rain decreases orders from 285.8 to 151.7 (-46.9%).",
        "",
        "CRITICAL BUSINESS TAKEAWAY:",
        "  • Predictive models MUST include the interaction term (Rain x Is_Weekend);",
        "    ignoring it biases demand estimates by up to 134 orders on wet Saturdays.",
    ]
    bullet_block(s, bullets, Inches(0.4), Inches(1.35), Inches(6.3), Inches(5.6),
                 font_size=12.5, title="Key Findings from Exploratory Analysis", title_color=BLUE)
    add_image(s, "eda_05_rain_vs_dry_comparison.png", Inches(7.0), Inches(1.6), width=Inches(5.9))


def slide_05_unit1_menu_shifts(prs: Presentation):
    s = blank_slide(prs)
    header_bar(s, "Unit 1: Hourly Meal Slots & Monsoon Menu Shifts",
               "Four distinct meal slots and clear hot-comfort vs cold-refresh substitution",
               "Unit 1 · P4")
    footer_bar(s)

    bullets = [
        "TEMPORAL DIURNAL WINDOWS:",
        "  • 100% of daily transactions fall into 4 distinct ordering slots:",
        "    - Breakfast (07:30–10:30): 11.2% volume (Idli, Dosa, Pongal).",
        "    - Lunch (12:00–15:00): 74.0% of all orders (Meals, Biryani, Thali).",
        "    - Tea / Snack (16:30–18:30): 8.5% volume (Chai, Samosa, Pakora).",
        "    - Dinner (19:30–22:00): 6.3% volume (Parotta, Fried Rice, Dosa).",
        "",
        "MONSOON REPERTOIRE SHIFTS (Category & SKU Level):",
        "  • Hot Comfort Snacks Surge in Rain: Samosa (+335%), Pakora (+200%),",
        "    Medu Vada (+118%), Hot Masala Chai (+155%).",
        "  • Cold Refreshments Plunge in Rain: Tender Coconut Ice Cream (-64%),",
        "    Fresh Lime Juice (-56%), Cold Badam Milk (-52%).",
    ]
    bullet_block(s, bullets, Inches(0.4), Inches(1.35), Inches(6.3), Inches(5.6),
                 font_size=12.5, title="Meal Windows & Category Shifts", title_color=ORANGE)
    add_image(s, "eda_08_category_regime_shifts.png", Inches(7.0), Inches(1.6), width=Inches(5.9))


def slide_06_unit2_features(prs: Presentation):
    s = blank_slide(prs)
    header_bar(s, "Unit 2: Feature Engineering & Preprocessing",
               "Constructing past-only lag features, weather interaction terms, and multicollinearity audits",
               "Unit 2 · P3")
    footer_bar(s)

    bullets = [
        "STRICT LEAKAGE AUDIT & TRAIN/TEST SPLIT:",
        "  • Chronological split preserving time arrow (no shuffling or future leakage).",
        "  • 56-day final hold-out evaluation set for supervised ML.",
        "",
        "ENGINEERED FEATURE MATRIX (T14):",
        "  • Autoregressive Lags: Orders_lag1, Orders_lag7 (capturing weekly inertia).",
        "  • Rolling Statistics: 7-day rolling mean & rolling standard deviation.",
        "  • Calendar Terms: Day-of-week one-hot, is_weekend, Tamil festival flag.",
        "  • Weather Exogenous: Rain mm, rain hours, apparent temp, heavy rain flag.",
        "  • Interaction Terms: rain_x_weekend, heat_x_weekend.",
        "",
        "MULTICOLLINEARITY & VIF AUDIT (T15):",
        "  • Temperature & humidity were collinear with apparent temp; redundant raw fields pruned.",
        "  • All final features have VIF < 4.0; PCA not required, preserving full interpretability.",
    ]
    bullet_block(s, bullets, Inches(0.4), Inches(1.35), Inches(6.3), Inches(5.6),
                 font_size=12, title="Feature Engineering & Data Hygiene", title_color=BLUE)
    add_image(s, "correlation_heatmap.png", Inches(7.0), Inches(1.5), width=Inches(5.9))


def slide_07_unit2_models(prs: Presentation):
    s = blank_slide(prs)
    header_bar(s, "Unit 2: Predictive Modelling & Supervised ML",
               "Evaluating Ridge, Random Forest, and XGBoost on the 56-day hold-out",
               "Unit 2 · P5")
    footer_bar(s)

    bullets = [
        "MODEL PERFORMANCE COMPARISON (56-Day Holdout):",
        "  • Random Forest: MAE 14.61 orders/day, RMSE 19.34, R² = 0.908",
        "  • XGBoost Regressor: MAE 14.73 orders/day, RMSE 19.52, R² = 0.906",
        "  • Ridge Regression: MAE 14.94 orders/day, RMSE 19.78, R² = 0.903",
        "  • Baseline (Weekday Mean): MAE 38.99 orders/day, R² = 0.421",
        "",
        "FEATURE IMPORTANCE & SHAP VALUES:",
        "  • Top predictive drivers: Orders_lag7, Orders_lag1, rain_x_weekend, is_weekend.",
        "  • SHAP analysis confirms strong positive attribution of rain on weekdays,",
        "    and strong negative attribution on weekends.",
        "  • Residual diagnostics confirm homoscedasticity and zero-mean normality.",
    ]
    bullet_block(s, bullets, Inches(0.4), Inches(1.35), Inches(6.3), Inches(5.6),
                 font_size=12.5, title="Model Evaluation & Interpretability", title_color=BLUE)
    add_image(s, "fig13_shap.png", Inches(7.0), Inches(1.6), width=Inches(5.9))


def slide_08_unit2_selection(prs: Presentation):
    s = blank_slide(prs)
    header_bar(s, "Unit 2 / Comparative: Full-Year Model Selection (11 Models)",
               "365-day backtest proves multiple models are statistically tied; weather adds 20 orders/day accuracy",
               "Unit 2 · P5")
    footer_bar(s)

    bullets = [
        "365-DAY COMPREHENSIVE BACKTEST (T25):",
        "  • Evaluated 11 model families over 365 test days (Oct 2025 – Sep 2026).",
        "  • Includes Linear, Ridge, Count models (Poisson, NegBin), Tree ensembles, and SARIMAX.",
        "",
        "STATISTICAL TIE DISCOVERY (Paired Bootstrap, B=10,000):",
        "  • Ridge (MAE 14.99), SARIMAX (15.00), Poisson (15.03), Negative Binomial (15.04),",
        "    and Extra Trees (15.08) are STATISTICALLY TIED (p > 0.45 vs Ridge).",
        "  • Best possible theoretical score (generator expectation) = 14.77.",
        "",
        "VALUE OF WEATHER ABLATION (T26):",
        "  • Stripping weather features causes MAE to spike from 15.0 to 33.2–37.4.",
        "  • Weather saves 18.2 to 22.4 orders/day across every single model family.",
    ]
    bullet_block(s, bullets, Inches(0.4), Inches(1.35), Inches(6.3), Inches(5.6),
                 font_size=12, title="365-Day Selection & Weather Value", title_color=ORANGE)
    add_image(s, "fig20_mae_difference_vs_ridge.png", Inches(7.0), Inches(1.4), width=Inches(5.9))
    add_image(s, "fig21_weather_effect.png", Inches(7.0), Inches(4.3), width=Inches(5.9))


def slide_09_unit3_forecasting(prs: Presentation):
    s = blank_slide(prs)
    header_bar(s, "Unit 3 (Method 1): Time-Series Forecasting (SARIMAX & Prophet)",
               "131 rolling folds across 917 days; SARIMAX beats baselines by 60%",
               "Unit 3 · P1")
    footer_bar(s)

    bullets = [
        "TIME-SERIES PROPERTIES & STATIONARITY:",
        "  • Series is stationary: ADF = -6.08 (p < 0.001); KPSS confirms stationarity (p >= 0.10).",
        "  • STL decomposition exhibits high seasonal strength (0.56, 7-day periodicity).",
        "",
        "ROLLING-ORIGIN BACKTEST (131 folds, 7-day horizon, 917 test days):",
        "  • SARIMAX with Weather: MAE 15.75 orders/day (10.1% MAPE).",
        "  • Prophet with Weather: MAE 21.40 orders/day.",
        "  • Weekday-Mean Baseline: MAE 38.99 orders/day.",
        "  • Seasonal Naive Baseline: MAE 48.20 orders/day.",
        "",
        "OPERATIONAL BENEFIT OF WEATHER:",
        "  • Adding weather reduces SARIMAX MAE by 23.5 orders/day (95% CI: 21.5–25.5).",
        "  • Rainy-day MAE (15.4) is comparable to dry-day MAE (15.9): weather removes surprise.",
    ]
    bullet_block(s, bullets, Inches(0.4), Inches(1.35), Inches(6.3), Inches(5.6),
                 font_size=12, title="Time-Series Forecasting Framework", title_color=BLUE)
    add_image(s, "fig05_forecast_vs_actual.png", Inches(7.0), Inches(1.4), width=Inches(5.9))
    add_image(s, "fig06_mae_by_model.png", Inches(7.0), Inches(4.3), width=Inches(5.9))


def slide_10_unit3_arm(prs: Presentation):
    s = blank_slide(prs)
    header_bar(s, "Unit 3 (Method 2): Weather-Segmented Association Rule Mining",
               "Discovering weather-conditioned basket affinities across 182,696 transactions",
               "Unit 3 · P3 & P2")
    footer_bar(s)

    bullets = [
        "SEGMENTED BASKET MINING METHODOLOGY:",
        "  • Apriori algorithm evaluated separately on Rainy Days (n=33,628) vs Dry Days (n=149,068).",
        "  • Thresholds: Minimum support = 0.005, minimum confidence = 0.05.",
        "",
        "PLANTED PAIR RECOVERY & WEATHER AFFINITY (T11):",
        "  • Chai + Pakora: Support 6.3%, Confidence 0.37, Lift 2.8 in rain (surges +12% conf vs dry).",
        "  • Coffee + Vada: Support 7.5%, Confidence 0.37, Lift 2.9 in rain.",
        "  • Cold Coffee + Vanilla Ice Cream: Active on dry hot afternoons (Lift 2.4, suppressed in rain).",
        "",
        "SENSITIVITY ANALYSIS GRID (T12):",
        "  • Evaluated 5x5 grid across support (0.004–0.015) and confidence (0.02–0.20).",
        "  • Optimal balance: min_support = 0.004–0.006 preserves high-affinity rules without noise.",
    ]
    bullet_block(s, bullets, Inches(0.4), Inches(1.35), Inches(6.3), Inches(5.6),
                 font_size=12, title="Weather-Segmented Item Pairings", title_color=BLUE)
    add_image(s, "fig14_arm_rain_vs_dry.png", Inches(7.0), Inches(1.5), width=Inches(5.9))


def slide_11_comparison(prs: Presentation):
    s = blank_slide(prs)
    header_bar(s, "Comparison of Unit 3 Methods & Integrated Insights",
               "Synergizing macro aggregate demand forecasting with micro basket-level co-purchasing",
               "Synthesis")
    footer_bar(s)

    col1 = [
        "METHOD 1: TIME-SERIES FORECASTING (SARIMAX):",
        "  • Granularity: Aggregate store-level daily volume (total orders/day).",
        "  • Time Horizon: 1 to 7 days forward planning horizon.",
        "  • Primary Value: Accurate staffing, total raw ingredient procurement,",
        "    and delivery fleet scheduling.",
        "  • Key KPI: MAE = 15.75 orders/day (10.1% MAPE).",
        "",
        "METHOD 2: ASSOCIATION RULE MINING (ARM):",
        "  • Granularity: Micro transaction-level basket co-occurrence.",
        "  • Time Horizon: Real-time dynamic bundling and prep slot scheduling.",
        "  • Primary Value: Increasing Average Order Value (AOV) and reducing prep wait.",
        "  • Key KPI: High-lift rules (Lift 2.4–2.9, +12% confidence shift under rain).",
    ]
    bullet_block(s, col1, Inches(0.4), Inches(1.35), Inches(6.0), Inches(5.6),
                 font_size=12, title="Methodological Comparison", title_color=BLUE)

    col2 = [
        "INTEGRATED DUAL-HORIZON OPERATIONS:",
        "  1. Macro Procurement:",
        "     Use SARIMAX forecast + buffer to set morning bulk prep quantities",
        "     (rice, gravies, packaging containers).",
        "",
        "  2. Micro Station Prep:",
        "     When rainfall >= 2.5mm is registered, use ARM insights to pre-fry",
        "     vada/pakora snack bases between 15:30 and 16:30,",
        "     slashing preparation bottleneck from 18 to 6 minutes.",
        "",
        "  3. Dynamic Digital Storefront:",
        "     Synchronize Swiggy/Zomato hero banners with weather:",
        "     push 'Monsoon Combos' in rain; 'Cold Refreshers' in heatwaves.",
    ]
    bullet_block(s, col2, Inches(6.8), Inches(1.35), Inches(6.1), Inches(5.6),
                 font_size=12, title="Integrated Commercial Value", title_color=ORANGE)


def slide_12_dashboard(prs: Presentation):
    s = blank_slide(prs)
    header_bar(s, "Interactive Application: Streamlit Dashboard",
               "A deployed 5-tab analytics application integrating results from Units 1, 2, and 3",
               "Dashboard · P4")
    footer_bar(s)

    bullets = [
        "DEPLOYMENT & ARCHITECTURE:",
        "  • Public Cloud URL: https://weather-food-ord-smj647r3xjpffhdkb4j2df.streamlit.app/",
        "  • Local Execution: streamlit run dashboard/app.py (port 8501).",
        "  • Clean light-themed responsive UX with Vega-Lite / Altair charts and cached data pipelines.",
        "",
        "FIVE COMPREHENSIVE TABS:",
        "  1. Overview: Executive summary KPIs, order volume trends, and trailing 90-day trajectory.",
        "  2. Dataset: 4-step collection pipeline, Open-Meteo API config, field dictionary, validation metrics.",
        "  3. Demand Patterns: Weekday distributions, weather bivariate heatmap, monsoon timeline.",
        "  4. Forecasts: Future projections (recursive Ridge) + retrospective full-year 11-model selection",
        "     with automated statistical tie detection and weather ablation metrics.",
        "  5. Menu Insights: Interactive ARM rule explorer with support/confidence sliders and rain/dry diffs.",
    ]
    bullet_block(s, bullets, Inches(0.4), Inches(1.35), Inches(12.5), Inches(5.6),
                 font_size=13, title="Streamlit Dashboard Architecture & User Experience", title_color=BLUE)


def slide_13_recommendations(prs: Presentation):
    s = blank_slide(prs)
    header_bar(s, "Strategic Recommendations & Operational Buffer",
               "Actionable kitchen policies, buffer sizes, and revenue optimization strategies",
               "Strategy")
    footer_bar(s)

    col1 = [
        "POLICY 1: PLANNING BUFFER IMPLEMENTATION:",
        "  • Planning buffer = Forecast + 25 orders (~17% buffer).",
        "  • Empirically covers 90% of days without stockouts.",
        "  • Regime-adjusted buffer: +27 on rainy days; +24 on dry days.",
        "  • Prevents panic re-orders and delivery driver bottlenecks.",
        "",
        "POLICY 2: KITCHEN PRE-BATCHING SCHEDULE:",
        "  • Snack orders surge abruptly between 16:00 and 18:30 on rain alerts.",
        "  • Cloud kitchens pre-fry base batches of Vada and Pakora at 15:30,",
        "    cutting customer delivery times by 12 minutes.",
    ]
    bullet_block(s, col1, Inches(0.4), Inches(1.35), Inches(6.0), Inches(5.6),
                 font_size=12, title="Inventory & Kitchen Policies", title_color=ORANGE)

    col2 = [
        "POLICY 3: DYNAMIC DIGITAL BUNDLING:",
        "  • Trigger automated platform combo promotions during rain:",
        "    - 'Monsoon Comfort Combo' (Chai + Pakora @ ₹75; Conf 0.37).",
        "    - 'Afternoon Tiffin Special' (Coffee + Vada @ ₹65; Conf 0.37).",
        "  • 10–14% combo discounts boost Average Order Value by 18%.",
        "",
        "POLICY 4: REAL-TIME STOREFRONT MERCHANDISING:",
        "  • Automatically update Swiggy/Zomato merchant banner:",
        "    - Wet weather: Warm artisan teas and hot tiffin combos.",
        "    - Hot afternoons (>33°C): Tender coconut ice cream and cold drinks.",
    ]
    bullet_block(s, col2, Inches(6.8), Inches(1.35), Inches(6.1), Inches(5.6),
                 font_size=12, title="Commercial & Promotion Policies", title_color=BLUE)


def slide_14_project_mgmt(prs: Presentation):
    s = blank_slide(prs)
    header_bar(s, "Project Management & Individual Contributions",
               "Kanban agile execution, 26 verified tasks, and documented individual ownership",
               "Management")
    footer_bar(s)

    bullets = [
        "AGILE TRACKING & VERIFICATION:",
        "  • GitHub Projects Board: https://github.com/users/priyagarnet20-ops/projects/1/views/2",
        "  • 26 verified tasks (T01–T26) executed across 5 sprint weeks, logged in github/issues.csv.",
        "  • Branch protection, continuous pytest regression testing (173 tests passing), and documentation audits.",
        "",
        "INDIVIDUAL ROLES & OWNERSHIP:",
        "  • P1 Priyadharshini (CB.SC.U4CSE23438): Open-Meteo weather fetcher, SARIMAX & Prophet forecasting (T01–T06).",
        "  • P2 Edavalapati Aashiq (CB.SC.U4CSE23560): Menu catalog, synthetic order generator & evidence calibration (T07–T12).",
        "  • P3 Sai Sri Satvic L (CB.SC.U4CSE23225): Feature engineering, multicollinearity VIF audits, ARM pipeline (T13–T17).",
        "  • P4 Swasthika P S (CB.SC.U4CSE23350): Exploratory analysis, visual chart sets, Streamlit dashboard (T18–T24).",
        "  • P5 Deepak S (CB.SC.U4CSE23267): Supervised ML models, 11-model selection backtest, validation suite (T25–T26).",
    ]
    bullet_block(s, bullets, Inches(0.4), Inches(1.35), Inches(12.5), Inches(5.6),
                 font_size=12.5, title="Agile Task Distribution & Team Ownership", title_color=BLUE)


def slide_15_conclusion(prs: Presentation):
    s = blank_slide(prs)
    header_bar(s, "Limitations, Conclusion & References",
               "Summary of accomplishments, methodological boundaries, and academic citations",
               "Conclusion")
    footer_bar(s)

    col1 = [
        "METHODOLOGICAL LIMITATIONS:",
        "  • Orders are synthetic: calibrated on 464 real Swiggy orders (51-day overlap)",
        "    and a 27-person survey; reflects generator rules rather than living demand.",
        "  • Exogenous Weather Assumptions: Models use observed historical weather;",
        "    a live system must ingest forecasted weather, adding ~1.5–2.0 orders MAE.",
        "",
        "CORE CONCLUSIONS:",
        "  • Weather integration yields significant demand planning improvements",
        "    (saving 23.5 orders/day of forecasting error).",
        "  • Dual macro forecasting + micro basket mining provides a complete,",
        "    reproducible operating framework for restaurant managers.",
    ]
    bullet_block(s, col1, Inches(0.4), Inches(1.35), Inches(6.0), Inches(5.6),
                 font_size=12, title="Project Limitations & Conclusion", title_color=BLUE)

    col2 = [
        "KEY ACADEMIC REFERENCES:",
        "  [1] Liu, D., Wang, W., & Zhao, Y. (2022). Effect of weather on online",
        "      food ordering. Kybernetes, 51(1), 163–182.",
        "  [2] Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining",
        "      association rules. Proc. 20th Int. Conf. Very Large Data Bases (VLDB).",
        "  [3] Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).",
        "      Time series analysis: forecasting and control. John Wiley & Sons.",
        "  [4] Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles",
        "      and practice. OTexts.",
        "  [5] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting",
        "      model predictions. NeurIPS 30.",
    ]
    bullet_block(s, col2, Inches(6.8), Inches(1.35), Inches(6.1), Inches(5.6),
                 font_size=11.5, title="Academic References", title_color=ORANGE)


def build_final_deck():
    prs = new_prs()
    print("Building Final Project Presentation Deck (15 Slides)...")

    slide_01_cover(prs)
    print("  [01/15] Cover Slide")
    slide_02_executive_summary(prs)
    print("  [02/15] Executive Summary & Business Objectives")
    slide_03_dataset(prs)
    print("  [03/15] Dataset Collection, Calibration & Validation")
    slide_04_unit1_eda(prs)
    print("  [04/15] Unit 1: Exploratory Analysis (Rain x Weekday)")
    slide_05_unit1_menu_shifts(prs)
    print("  [05/15] Unit 1: Meal Slots & Monsoon Menu Shifts")
    slide_06_unit2_features(prs)
    print("  [06/15] Unit 2: Feature Engineering & Preprocessing")
    slide_07_unit2_models(prs)
    print("  [07/15] Unit 2: Predictive Modelling & Supervised ML")
    slide_08_unit2_selection(prs)
    print("  [08/15] Unit 2: 11-Model Selection & Statistical Ties")
    slide_09_unit3_forecasting(prs)
    print("  [09/15] Unit 3: Time-Series Forecasting (SARIMAX & Prophet)")
    slide_10_unit3_arm(prs)
    print("  [10/15] Unit 3: Weather-Segmented ARM")
    slide_11_comparison(prs)
    print("  [11/15] Comparison & Integrated Business Insights")
    slide_12_dashboard(prs)
    print("  [12/15] Interactive Dashboard (Streamlit App)")
    slide_13_recommendations(prs)
    print("  [13/15] Strategic Recommendations & Planning Buffer")
    slide_14_project_mgmt(prs)
    print("  [14/15] Project Management & Individual Contributions")
    slide_15_conclusion(prs)
    print("  [15/15] Limitations, Conclusion & References")

    prs.save(str(OUT_ROOT))
    shutil.copy(OUT_ROOT, OUT_SLIDES)
    print(f"\nFinal presentation saved:")
    print(f"  -> {OUT_ROOT} ({OUT_ROOT.stat().st_size / 1024:.0f} KB)")
    print(f"  -> {OUT_SLIDES}")


if __name__ == "__main__":
    build_final_deck()
