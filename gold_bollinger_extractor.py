#!/usr/bin/env python3
"""
Gold (XAUUSD) Bollinger Swing Extractor
=======================================
For every timeframe (1Y, 6M, 3M, 1M, Weekly, 4H, 2H) it finds:

  HIGH  - swing high: the highest high among at least N candles on each side
          AND the high reaches/crosses the UPPER Bollinger band.
  MID   - the candle where price crosses the MIDDLE band (SMA) between
          the two swings (the "transition").
  LOW   - swing low: the lowest low among at least N candles on each side
          AND the low reaches/crosses the LOWER Bollinger band.

Output = one row per leg (High -> Mid -> Low, or Low -> Mid -> High), saved to
CSV and printed. With --live it re-downloads and refreshes every few seconds.

Install:   pip install yfinance pandas numpy
Optional:  pip install --upgrade git+https://github.com/rongardF/tvdatafeed.git   (TradingView feed)

Run:
    python gold_bollinger_extractor.py                  # once, Yahoo data
    python gold_bollinger_extractor.py --live 60        # refresh every 60 s
    python gold_bollinger_extractor.py --source tv      # TradingView (OANDA:XAUUSD)
    python gold_bollinger_extractor.py --source demo    # offline test with fake data
"""
import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# SETTINGS  (edit freely)
# --------------------------------------------------------------------------
BB_PERIOD = 20          # Bollinger SMA length
BB_STD = 2.0            # Bollinger standard deviations
MIN_CANDLES = 4         # swing must be the extreme of >= 4 candles each side
OUTPUT_CSV = "gold_bollinger_transitions.csv"

# name: candle interval used, and how far back to look (calendar days)
TIMEFRAMES = {
    "1Y":     dict(interval="1d", days=365),
    "6M":     dict(interval="1d", days=182),
    "3M":     dict(interval="1d", days=91),
    "1M":     dict(interval="1d", days=30),
    "Weekly": dict(interval="1w", days=365 * 3),
    "4H":     dict(interval="4h", days=60),
    "2H":     dict(interval="2h", days=30),
}


# --------------------------------------------------------------------------
# DATA SOURCES
# --------------------------------------------------------------------------
def _clean(df):
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def fetch_yahoo(interval, days):
    """Yahoo Finance. GC=F = COMEX gold futures (close to spot XAUUSD, but a few $ apart)."""
    import yfinance as yf
    warm = BB_PERIOD * 3                      # extra candles so bands are warmed up
    if interval in ("2h", "4h"):
        raw = yf.download("GC=F", period=f"{min(days + 40, 700)}d", interval="1h",
                          progress=False, auto_adjust=False)
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        raw = _clean(raw)
        rule = interval.upper()
        return raw.resample(rule).agg({"open": "first", "high": "max",
                                       "low": "min", "close": "last"}).dropna()
    yf_int = {"1d": "1d", "1w": "1wk"}[interval]
    start_days = days + (warm * (7 if interval == "1w" else 1.5))
    raw = yf.download("GC=F", period=f"{int(start_days)}d", interval=yf_int,
                      progress=False, auto_adjust=False)
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    return _clean(raw)


_tv = None
def fetch_tradingview(interval, days):
    """TradingView via the unofficial tvdatafeed package (no login = limited, but works)."""
    global _tv
    from tvDatafeed import TvDatafeed, Interval
    if _tv is None:
        _tv = TvDatafeed()
    imap = {"1d": Interval.in_daily, "1w": Interval.in_weekly,
            "4h": Interval.in_4_hour, "2h": Interval.in_2_hour}
    bars_per_day = {"1d": 1, "1w": 1 / 7, "4h": 6, "2h": 12}[interval]
    n = int(days * bars_per_day * (1.0 if interval == "1d" else 1.0)) + BB_PERIOD * 3
    df = _tv.get_hist("XAUUSD", "OANDA", interval=imap[interval], n_bars=min(max(n, 120), 5000))
    return _clean(df)


def fetch_demo(interval, days):
    """Random-walk gold-like data so you can test the logic offline."""
    freq = {"1d": "D", "1w": "W", "4h": "4h", "2h": "2h"}[interval]
    n = 600
    idx = pd.date_range(end=datetime.now(), periods=n, freq=freq)
    rng = np.random.default_rng(abs(hash(interval)) % 2**32)
    close = 3800 + np.cumsum(rng.normal(0, 18, n)) + 120 * np.sin(np.linspace(0, 14, n))
    op = close + rng.normal(0, 6, n)
    hi = np.maximum(op, close) + rng.uniform(2, 14, n)
    lo = np.minimum(op, close) - rng.uniform(2, 14, n)
    return pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close}, index=idx)


SOURCES = {"yahoo": fetch_yahoo, "tv": fetch_tradingview, "demo": fetch_demo}


# --------------------------------------------------------------------------
# INDICATORS + SWING LOGIC
# --------------------------------------------------------------------------
def add_bollinger(df):
    mid = df["close"].rolling(BB_PERIOD).mean()
    sd = df["close"].rolling(BB_PERIOD).std(ddof=0)
    df = df.copy()
    df["mid"], df["upper"], df["lower"] = mid, mid + BB_STD * sd, mid - BB_STD * sd
    return df.dropna()


def find_swings(df, n=MIN_CANDLES):
    """Swing high = highest high in +/-n candles that touches the upper band.
    Swing low  = lowest low  in +/-n candles that touches the lower band."""
    w = 2 * n + 1
    is_hi = (df["high"] == df["high"].rolling(w, center=True).max()) & (df["high"] >= df["upper"])
    is_lo = (df["low"] == df["low"].rolling(w, center=True).min()) & (df["low"] <= df["lower"])

    events = []
    for i in np.where(is_hi.values)[0]:
        events.append((i, "H", df["high"].iat[i]))
    for i in np.where(is_lo.values)[0]:
        events.append((i, "L", df["low"].iat[i]))
    events.sort()

    # enforce alternation H, L, H, L ... keeping the more extreme of consecutive duplicates
    clean = []
    for ev in events:
        if clean and clean[-1][1] == ev[1]:
            better = ev[2] > clean[-1][2] if ev[1] == "H" else ev[2] < clean[-1][2]
            if better:
                clean[-1] = ev
        else:
            clean.append(ev)
    return clean


def build_transitions(df, swings, tf_name):
    rows = []
    for a, b in zip(swings[:-1], swings[1:]):
        i0, t0, p0 = a
        i1, t1, p1 = b
        seg = df.iloc[i0:i1 + 1]
        down = t0 == "H"
        # first candle in the segment whose close crosses the middle band
        if down:
            cross = (seg["close"] < seg["mid"]) & (seg["close"].shift(1) >= seg["mid"].shift(1))
        else:
            cross = (seg["close"] > seg["mid"]) & (seg["close"].shift(1) <= seg["mid"].shift(1))
        if cross.any():
            ct = cross.idxmax()
            mid_price, mid_time = round(float(seg.loc[ct, "mid"]), 2), ct
        else:
            mid_price, mid_time = np.nan, pd.NaT

        hi_i, lo_i = (i0, i1) if down else (i1, i0)
        rows.append({
            "Timeframe": tf_name,
            "Direction": "High > Mid > Low" if down else "Low > Mid > High",
            "Highest Price": round(float(df["high"].iat[hi_i]), 2),
            "Highest Time": df.index[hi_i],
            "Mid Price (BB basis)": mid_price,
            "Mid Time": mid_time,
            "Lowest Price": round(float(df["low"].iat[lo_i]), 2),
            "Lowest Time": df.index[lo_i],
            "Range": round(float(df["high"].iat[hi_i] - df["low"].iat[lo_i]), 2),
        })
    return rows


def analyse(fetch, live_status):
    all_rows = []
    cache = {}
    for name, cfg in TIMEFRAMES.items():
        key = (cfg["interval"], cfg["days"] if cfg["interval"] != "1d" else 365)
        try:
            if cfg["interval"] == "1d":
                if "daily" not in cache:
                    cache["daily"] = fetch("1d", 365)
                raw = cache["daily"]
            else:
                raw = fetch(cfg["interval"], cfg["days"])
        except Exception as exc:                                  # noqa: BLE001
            print(f"[{name}] data error: {exc}")
            continue

        df = add_bollinger(raw)
        window_start = df.index[-1] - pd.Timedelta(days=cfg["days"])
        df = df[df.index >= window_start]           # bands already warmed up before slicing
        if len(df) < 2 * MIN_CANDLES + 2:
            print(f"[{name}] not enough candles")
            continue

        all_rows += build_transitions(df, find_swings(df), name)
        last = df.iloc[-1]
        live_status[name] = (df.index[-1], last["close"], last["upper"], last["mid"], last["lower"])
    return pd.DataFrame(all_rows)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def run_once(fetch):
    status = {}
    out = analyse(fetch, status)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print("\n" + "=" * 100)
    print("Updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("-" * 100)
    for tf, (t, c, u, m, l) in status.items():
        print(f"{tf:>6} | last {t:%Y-%m-%d %H:%M}  close {c:,.2f} | upper {u:,.2f}  mid {m:,.2f}  lower {l:,.2f}")
    if out.empty:
        print("No transitions found.")
        return
    out = out.sort_values(["Timeframe", "Highest Time"], ascending=[True, False])
    out.to_csv(OUTPUT_CSV, index=False)
    for tf in TIMEFRAMES:
        sub = out[out["Timeframe"] == tf]
        if not sub.empty:
            print(f"\n### {tf}  (latest 5 legs)")
            print(sub.head(5).drop(columns="Timeframe").to_string(index=False))
    print(f"\nSaved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES, default="yahoo")
    ap.add_argument("--live", type=int, default=0, help="refresh every N seconds (0 = run once)")
    args = ap.parse_args()
    fetcher = SOURCES[args.source]

    if args.live:
        while True:
            try:
                run_once(fetcher)
            except Exception as exc:                              # noqa: BLE001
                print("Update failed:", exc)
            time.sleep(args.live)
    else:
        run_once(fetcher)
