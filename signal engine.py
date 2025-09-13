# signal_engine.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta

# Fields returned in signal JSON
SIGNAL_FIELDS = ["id", "asset", "time_utc6", "direction", "confidence", "logic", "score"]

def compute_indicators_and_signal(df):
    """
    Input: df with index datetime and columns ['open','high','low','close'] (float)
    Returns a dict: {direction, logic, score, components...}
    """
    out = df.copy().sort_index()
    if "close" not in out:
        raise ValueError("DF missing close")
    # compute indicators robustly with try/except to avoid runtime failure
    try:
        out["ma8"] = ta.sma(out["close"], length=8)
        out["ma21"] = ta.sma(out["close"], length=21)
        bb = ta.bbands(out["close"], length=20, std=2)
        out = pd.concat([out, bb], axis=1)
        out["rsi14"] = ta.rsi(out["close"], length=14)
        macd = ta.macd(out["close"])
        out = pd.concat([out, macd], axis=1)
    except Exception:
        # fallback: compute only smas
        out["ma8"] = out["close"].rolling(8).mean()
        out["ma21"] = out["close"].rolling(21).mean()
    latest = out.iloc[-1]
    prev = out.iloc[-2] if len(out) >= 2 else latest

    votes = []
    reasons = []

    # 1) Bollinger band breakout
    if not pd.isna(latest.get("BBU_20_2.0")) and not pd.isna(latest.get("BBL_20_2.0")):
        if latest["close"] > latest["BBU_20_2.0"]:
            votes.append(1 * 1.5); reasons.append("BB breakout up")
        elif latest["close"] < latest["BBL_20_2.0"]:
            votes.append(-1 * 1.5); reasons.append("BB breakout down")

    # 2) MA cross
    if not pd.isna(latest["ma8"]) and not pd.isna(latest["ma21"]):
        if prev["ma8"] < prev["ma21"] and latest["ma8"] > latest["ma21"]:
            votes.append(1 * 1.2); reasons.append("MA8 crossed above MA21")
        elif prev["ma8"] > prev["ma21"] and latest["ma8"] < latest["ma21"]:
            votes.append(-1 * 1.2); reasons.append("MA8 crossed below MA21")

    # 3) RSI extremes
    if not pd.isna(latest.get("rsi14")):
        if latest["rsi14"] < 28:
            votes.append(1 * 0.9); reasons.append("RSI oversold")
        elif latest["rsi14"] > 72:
            votes.append(-1 * 0.9); reasons.append("RSI overbought")

    # 4) MACD direction
    macdh = latest.get("MACDh_12_26_9", None)
    if macdh is not None and not pd.isna(macdh):
        votes.append((1 if macdh > 0 else -1) * 0.8); reasons.append("MACDh sign")

    # 5) Price action: last candle body and wick ratio
    try:
        body = abs(latest["close"] - latest["open"])
        total = latest["high"] - latest["low"] if (latest["high"] - latest["low"]) > 0 else 1e-6
        body_ratio = body / total
        if latest["close"] > latest["open"]:
            votes.append(1 * (0.5 + 0.5 * body_ratio)); reasons.append(f"Bullish candle (body_ratio={body_ratio:.2f})")
        else:
            votes.append(-1 * (0.5 + 0.5 * body_ratio)); reasons.append(f"Bearish candle (body_ratio={body_ratio:.2f})")
    except Exception:
        pass

    # Aggregation: weighted average
    if not votes:
        score = 0.0
    else:
        score = sum(votes) / (len(votes) if len(votes) else 1)

    direction = "Up" if score > 0 else "Down"
    logic = "; ".join(reasons) if reasons else "No clear signal"

    return {
        "direction": direction,
        "logic": logic,
        "score": float(score),
        # provide debug components for UI or logging
        "components": {"votes": votes}
    }

def calibrate_confidence(signal_result, df):
    """
    Convert raw score + image/data quality heuristics into 0-100 confidence.
    - agreement among strategies increases confidence
    - larger body ratios give more confidence
    - low volatility or low sample size reduce confidence
    """
    base = abs(signal_result.get("score", 0))
    # scale base into 0..1
    norm = min(1.0, base / 2.0) if base != 0 else 0.0

    # candle body heuristic
    latest = df.iloc[-1]
    body = abs(latest.get("close", 0) - latest.get("open", 0))
    candle_range = (latest.get("high", 0) - latest.get("low", 0)) or 1.0
    body_ratio = min(1.0, body / candle_range) if candle_range != 0 else 0.0

    # volatility (higher volatility = more uncertainty) -> reduce confidence a bit
    vol = df["close"].pct_change().rolling(10).std().iloc[-1]
    vol = float(vol) if not pd.isna(vol) else 0.0

    # sample size penalty
    n = len(df)
    size_factor = min(1.0, n / 60.0)  # prefer at least 60 candles

    # combine heuristics
    confidence = 0.0
    confidence += norm * 60        # main weight
    confidence += body_ratio * 25  # clear candle adds confidence
    confidence *= size_factor
    # volatility penalty
    confidence *= max(0.6, 1 - vol * 5)

    # clamp 0..100
    conf_pct = int(max(0, min(100, confidence)))
    return conf_pct