"""Build the Review 2 forecasting slides (P1 / T04-T06).

Writes slides/review2/review2_forecasting_deck.pptx (2 slides, same style as the ARM deck).
Run from the project root:  python slides/build_review2_forecasting.py
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "slides/review2/review2_forecasting_deck.pptx"

DARK = RGBColor(0x18, 0x24, 0x3A)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK2 = RGBColor(0x52, 0x51, 0x4E)
W, H = Inches(13.33), Inches(7.5)


def rect(s, l, t, w, h, rgb):
    sh = s.shapes.add_shape(1, l, t, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = rgb; sh.line.fill.background()


def text(s, t, l, top, w, h, size, color, bold=False, align=PP_ALIGN.LEFT):
    tb = s.shapes.add_textbox(l, top, w, h)
    tb.text_frame.word_wrap = True
    p = tb.text_frame.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = t
    r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color


def bullets(s, items, l, top, w, h, size):
    tb = s.shapes.add_textbox(l, top, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    for i, b in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = "• " + b
        r.font.size = Pt(size); r.font.color.rgb = INK2
        p.space_after = Pt(7)


def frame(prs, title, subtitle):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, Inches(1.15), DARK)
    text(s, title, Inches(0.35), Inches(0.12), Inches(10.5), Inches(0.65), 24, WHITE, True)
    text(s, subtitle, Inches(0.35), Inches(0.72), Inches(10.5), Inches(0.38), 13, RGBColor(0xCC, 0xCC, 0xCC))
    text(s, "P1 · T04–T06", Inches(10.8), Inches(0.22), Inches(2.2), Inches(0.5), 11,
         RGBColor(0xFF, 0xD7, 0x00), True, PP_ALIGN.RIGHT)
    rect(s, 0, H - Inches(0.35), W, Inches(0.35), DARK)
    text(s, "Weather-Driven Food Order Patterns · Coimbatore, Tamil Nadu · Review 2 (Time-series forecasting)",
         Inches(0.25), H - Inches(0.32), Inches(9), Inches(0.32), 8.5, RGBColor(0xAA, 0xAA, 0xAA))
    return s


def pic(s, name, l, t, w):
    s.shapes.add_picture(str(RES / name), l, t, width=w)


prs = Presentation(); prs.slide_width, prs.slide_height = W, H

a = frame(prs, "Forecasting daily orders: method and setup",
          "Weekly pattern + festival days; weather given as observed")
bullets(a, [
    "Series is stationary (ADF −6.08, p < 0.001; KPSS p ≥ 0.10).",
    "Strong weekly pattern (seasonal strength 0.56) plus festival days; no yearly term, because the generator has none.",
    "Models: seasonal naive, weekday mean, SARIMAX (with and without weather), SARIMAX on log orders, Prophet.",
    "Festival flag is given to every regression model.",
    "Rolling-origin backtest: 131 folds, 7-day forecasts, 917 forecast days.",
    "Weather is given as observed (best case); a real forecast would carry weather-forecast error.",
], Inches(0.4), Inches(1.4), Inches(6.0), Inches(5.4), 16)
pic(a, "fig03_decomposition.png", Inches(6.7), Inches(1.5), Inches(6.2))

b = frame(prs, "Forecasting daily orders: results",
          "Knowing the weather cuts the error by 60%; plan forecast + 25 orders")
bullets(b, [
    "Best MAE 15.8 orders/day (10.1% MAPE), against 39.0 for the weekday-mean baseline and 48.2 for seasonal naive.",
    "Weather lowers SARIMAX error by 23.5 orders/day (95% CI 21.5 to 25.5).",
    "Rainy-day error 15.4 vs dry-day 15.9: once weather is known, rainy days are no harder to plan.",
    "Within about 0.2 orders/day of the best possible score (generator expected value, MAE 15.5).",
    "Planning buffer: forecast + 25 orders (about 17%) covers 9 days in 10; 27 on rainy days, 24 on dry days.",
    "Caveat: the data is synthetic, so this shows the generator's rules are recovered, not real customer behaviour.",
], Inches(0.4), Inches(1.4), Inches(6.2), Inches(5.6), 15)
pic(b, "fig06_mae_by_model.png", Inches(6.9), Inches(1.35), Inches(6.0))
pic(b, "fig07_error_rain_dry.png", Inches(6.9), Inches(4.0), Inches(5.9))

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print("wrote", OUT)
