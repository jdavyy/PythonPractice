import yfinance as yf
import pandas as pd
import numpy as np
from datetime import time

# =============================
# CONFIG
# =============================
TICKER = "AAPL"
PERIOD = "60d"
INTERVAL = "5m"

# Risk/Reward
STOP_LOSS_PCT = 0.012   # 1.2% stop
TAKE_PROFIT_PCT = 0.020 # 2.0% target

# Filters
RVOL_THRESHOLD = 1.0    # at least average volume
ATR_QUANTILE = 0.20     # use bars with ATR in top 80% (avoid super low-vol chop)

# Trading session (US/Eastern)
SESSION_START = time(9, 31)
SESSION_END   = time(15, 55)


# =============================
# HELPER: FLATTEN YFINANCE COLS
# =============================
def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


# =============================
# DOWNLOAD DATA
# =============================
print(f"Downloading {TICKER} {INTERVAL} data for last {PERIOD}...")
nvda = yf.download(
    TICKER,
    period=PERIOD,
    interval=INTERVAL,
    auto_adjust=False,
    progress=False,
)

if nvda.empty:
    raise SystemExit("Download failed or returned no data. Check internet or ticker symbol.")

nvda = flatten_cols(nvda).dropna()

print("NVDA columns:", nvda.columns.tolist())
print("First few NVDA rows:")
print(nvda.head())


# =============================
# TIMEZONE → US/EASTERN
# =============================
def to_eastern_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert("America/New_York")
    df = df.copy()
    df.index = idx
    return df

nvda = to_eastern_index(nvda)


# =============================
# INDICATORS
# =============================

# EMAs
nvda["EMA20"] = nvda["Close"].ewm(span=20, adjust=False).mean()
nvda["EMA50"] = nvda["Close"].ewm(span=50, adjust=False).mean()

# Session VWAP (resets each day)
nvda["Date"] = nvda.index.date
nvda["PV"] = nvda["Close"] * nvda["Volume"]
nvda["CumVol"] = nvda.groupby("Date")["Volume"].cumsum()
nvda["CumPV"] = nvda.groupby("Date")["PV"].cumsum()
nvda["VWAP"] = nvda["CumPV"] / nvda["CumVol"]

# ATR (14)
high_low = nvda["High"] - nvda["Low"]
high_close = (nvda["High"] - nvda["Close"].shift()).abs()
low_close = (nvda["Low"] - nvda["Close"].shift()).abs()
tr_df = pd.DataFrame({"HL": high_low, "HC": high_close, "LC": low_close})
true_range = tr_df.max(axis=1)
nvda["ATR"] = true_range.rolling(14).mean()

# ATR threshold (avoid lowest 20% vol)
atr_threshold = nvda["ATR"].quantile(ATR_QUANTILE)

# RVOL based on time-of-day vs last 20 days
nvda["TimeIdx"] = nvda.index.strftime("%H:%M")
nvda["RVOL"] = nvda["Volume"] / nvda.groupby("TimeIdx")["Volume"].transform(
    lambda x: x.rolling(20, min_periods=5).mean()
)


# =============================
# BACKTEST LOOP
# =============================

in_trade = False
trades = []

entry_price = None
entry_time = None
stop_price = None
take_profit = None

# Debug counters
cnt_vwap_cross = 0
cnt_vwap_retest = 0
cnt_full_signal = 0

for i in range(1, len(nvda)):
    row = nvda.iloc[i]
    prev = nvda.iloc[i - 1]
    idx_ts = nvda.index[i]
    t = idx_ts.time()

    # Only trade during RTH window
    if t < SESSION_START or t > SESSION_END:
        # If somehow still in trade and we've passed end, flatten
        if in_trade and t > SESSION_END:
            exit_price = row["Close"]
            trades.append({
                "Entry Time": entry_time,
                "Exit Time": idx_ts,
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "Return %": (exit_price - entry_price) / entry_price * 100.0,
            })
            in_trade = False
        continue

    # =========================
    # MANAGE OPEN TRADE
    # =========================
    if in_trade:
        low = row["Low"]
        high = row["High"]

        # Stop-loss
        if low <= stop_price:
            trades.append({
                "Entry Time": entry_time,
                "Exit Time": idx_ts,
                "Entry Price": entry_price,
                "Exit Price": stop_price,
                "Return %": (stop_price - entry_price) / entry_price * 100.0,
            })
            in_trade = False
            continue

        # Take-profit
        if high >= take_profit:
            trades.append({
                "Entry Time": entry_time,
                "Exit Time": idx_ts,
                "Entry Price": entry_price,
                "Exit Price": take_profit,
                "Return %": (take_profit - entry_price) / entry_price * 100.0,
            })
            in_trade = False
            continue

        # End-of-day flatten at SESSION_END
        if t >= SESSION_END:
            exit_price = row["Close"]
            trades.append({
                "Entry Time": entry_time,
                "Exit Time": idx_ts,
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "Return %": (exit_price - entry_price) / entry_price * 100.0,
            })
            in_trade = False
            continue

    # =========================
    # LOOK FOR NEW ENTRY
    # =========================
    if not in_trade:
        # Need valid VWAP, ATR, RVOL
        if pd.isna(prev["VWAP"]) or pd.isna(row["VWAP"]):
            continue
        if pd.isna(row["ATR"]) or pd.isna(row["RVOL"]):
            continue

        # VWAP cross & retest logic
        vwap_cross = (prev["Close"] < prev["VWAP"]) and (row["Close"] > row["VWAP"])
        vwap_retest = (prev["Low"] < prev["VWAP"]) and (row["Close"] > row["VWAP"])

        if vwap_cross:
            cnt_vwap_cross += 1
        if vwap_retest:
            cnt_vwap_retest += 1

        # Soft bullish EMA condition
        emas_bullish = row["EMA20"] >= row["EMA50"] * 0.995

        # RVOL filter (at least average volume)
        rvol_ok = row["RVOL"] >= RVOL_THRESHOLD

        # ATR filter (avoid super low-vol chop)
        atr_ok = row["ATR"] >= atr_threshold

        # Final entry signal
        signal = (vwap_cross or vwap_retest) and emas_bullish and rvol_ok and atr_ok

        if signal:
            cnt_full_signal += 1

            # Improved entry price: mid-candle
            entry_price = (row["Open"] + row["Close"]) / 2.0
            entry_time = idx_ts
            stop_price = entry_price * (1.0 - STOP_LOSS_PCT)
            take_profit = entry_price * (1.0 + TAKE_PROFIT_PCT)
            in_trade = True
            continue


# =============================
# RESULTS
# =============================

df_trades = pd.DataFrame(trades)

print("\n---- DEBUG COUNTS ----")
print("VWAP cross signals:", cnt_vwap_cross)
print("VWAP retest signals:", cnt_vwap_retest)
print("Full entry signals:", cnt_full_signal)

print("\n---- BACKTEST RESULTS (NVDA VWAP loose) ----")
if df_trades.empty:
    print("No trades triggered. Try loosening filters or changing parameters.")
else:
    wins = df_trades[df_trades["Return %"] > 0]
    losses = df_trades[df_trades["Return %"] <= 0]

    total_trades = len(df_trades)
    win_rate = len(wins) / total_trades * 100.0 if total_trades > 0 else 0.0
    avg_win = wins["Return %"].mean() if not wins.empty else 0.0
    avg_loss = losses["Return %"].mean() if not losses.empty else 0.0
    net_return = df_trades["Return %"].sum()

    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Avg win: {avg_win:.2f}%")
    print(f"Avg loss: {avg_loss:.2f}%")
    print(f"Net return (sum of %): {net_return:.2f}%")

    print("\nSample trades:")
    print(df_trades.head())

    out_file = "results.csv"
    df_trades.to_csv(out_file, index=False)
    print(f"\nTrade log saved → {out_file}")
