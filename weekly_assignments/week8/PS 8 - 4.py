#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""PS8 Q4: Gamma-theta-variance identity using real BTC market data.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from IPython.display import display
from scipy.optimize import brentq
from scipy.stats import norm


torch.set_default_dtype(torch.float64)
plt.rcParams["figure.figsize"] = (12, 5)


OPTION_FILE = Path("BTCUSD-3JUL26-65000-C.parquet")
SPOT_OUTPUT_FILE = Path("BTC-USDT_spot_2026-06-16_14h_1s.parquet")
RESULT_FILE = Path("ps8_q4_gamma_theta_variance_results.csv")
SUMMARY_FILE = Path("ps8_q4_summary.csv")

SPOT_SYMBOL = "BTC-USDT"
STRIKE = 65_000.0
EXPIRY = pd.Timestamp("2026-07-03 08:00:00", tz="UTC")
OPTION_TYPE = "call"

RISK_FREE_RATE = 0.0
DIVIDEND_YIELD = 0.0
CONTRACT_MULTIPLIER = 1.0

# Short one call. Its Delta is negative, so the hedge holds +Delta BTC.
OPTION_POSITION = -1.0 * CONTRACT_MULTIPLIER

# Set to zero for the theoretical benchmark.
TRANSACTION_COST_BPS = 0.0

# IV calibration: median of the first 60 valid one-second observations.
IV_CALIBRATION_OBSERVATIONS = 60

OKX_URL = "https://www.okx.com/api/v5/market/history-candles"
BAR_SIZE = "1s"
API_LIMIT = 100
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0



def fetch_okx_history_candles(
    inst_id: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    bar: str = "1s",
    limit: int = 100,
    pause_seconds: float = 0.12,
) -> pd.DataFrame:
    """Download OKX historical candles between two UTC timestamps."""

    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)

    start_time = (
        start_time.tz_localize("UTC")
        if start_time.tzinfo is None
        else start_time.tz_convert("UTC")
    )
    end_time = (
        end_time.tz_localize("UTC")
        if end_time.tzinfo is None
        else end_time.tz_convert("UTC")
    )

    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    cursor_ms = end_ms + 1

    all_rows: list[list[str]] = []
    seen_timestamps: set[int] = set()
    session = requests.Session()

    for page_number in range(1, 201):
        response = session.get(
            OKX_URL,
            params={
                "instId": inst_id,
                "bar": bar,
                "after": str(cursor_ms),
                "limit": str(limit),
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("code") != "0":
            raise RuntimeError(f"OKX API error: {payload}")

        rows = payload.get("data", [])
        if not rows:
            break

        page_timestamps = [int(row[0]) for row in rows]

        for row in rows:
            row_ts = int(row[0])
            if start_ms <= row_ts <= end_ms and row_ts not in seen_timestamps:
                all_rows.append(row)
                seen_timestamps.add(row_ts)

        oldest_ts = min(page_timestamps)
        newest_ts = max(page_timestamps)
        print(
            f"Page {page_number:02d}: "
            f"{pd.to_datetime(oldest_ts, unit='ms', utc=True)} to "
            f"{pd.to_datetime(newest_ts, unit='ms', utc=True)}"
        )

        if oldest_ts <= start_ms:
            break

        cursor_ms = oldest_ts
        time.sleep(pause_seconds)
    else:
        raise RuntimeError("Pagination exceeded 200 pages.")

    if not all_rows:
        raise ValueError("No OKX spot data returned for the requested period.")

    columns = [
        "ts_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "volume_ccy",
        "volume_quote",
        "confirm",
    ]
    spot = pd.DataFrame(all_rows, columns=columns)

    spot["ts_ms"] = pd.to_numeric(spot["ts_ms"], errors="coerce")
    for column in columns[1:-1]:
        spot[column] = pd.to_numeric(spot[column], errors="coerce")

    spot["time"] = pd.to_datetime(spot["ts_ms"], unit="ms", utc=True)
    spot = (
        spot.dropna(subset=["time", "close"])
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .query("@start_time <= time <= @end_time")
        .reset_index(drop=True)
    )
    return spot


def load_option_quotes(path: Path) -> pd.DataFrame:
    """Load and clean Coincall best bid/ask quotes."""

    if not path.exists():
        raise FileNotFoundError(f"Option file not found: {path.resolve()}")

    raw = pd.read_parquet(path)
    required = {"recv_ts_ms", "bid1_px", "ask1_px"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing option columns: {sorted(missing)}")

    option = raw.copy()
    option["time"] = pd.to_datetime(option["recv_ts_ms"], unit="ms", utc=True)
    option["bid"] = pd.to_numeric(option["bid1_px"], errors="coerce")
    option["ask"] = pd.to_numeric(option["ask1_px"], errors="coerce")
    option = option.dropna(subset=["time", "bid", "ask"])
    option = option[
        (option["bid"] > 0)
        & (option["ask"] > 0)
        & (option["ask"] >= option["bid"])
    ].copy()

    option["option_mid"] = 0.5 * (option["bid"] + option["ask"])
    option["option_spread"] = option["ask"] - option["bid"]
    return (
        option.sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .reset_index(drop=True)
    )


# Black-Scholes and Greeks

def time_to_expiry_years(current_time: pd.Timestamp) -> float:
    seconds = (EXPIRY - pd.Timestamp(current_time)).total_seconds()
    return max(seconds / SECONDS_PER_YEAR, 1e-10)


def black_scholes_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> float:
    if spot <= 0 or strike <= 0 or maturity <= 0 or volatility <= 0:
        return np.nan

    sqrt_t = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    if option_type == "call":
        return float(
            spot * np.exp(-dividend_yield * maturity) * norm.cdf(d1)
            - strike * np.exp(-rate * maturity) * norm.cdf(d2)
        )
    if option_type == "put":
        return float(
            strike * np.exp(-rate * maturity) * norm.cdf(-d2)
            - spot * np.exp(-dividend_yield * maturity) * norm.cdf(-d1)
        )
    raise ValueError("option_type must be 'call' or 'put'.")


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
    lower_vol: float = 1e-4,
    upper_vol: float = 5.0,
) -> float:
    if not all(np.isfinite([market_price, spot, maturity])):
        return np.nan
    if market_price <= 0 or spot <= 0 or maturity <= 0:
        return np.nan

    if option_type == "call":
        lower_bound = max(
            spot * np.exp(-dividend_yield * maturity)
            - strike * np.exp(-rate * maturity),
            0.0,
        )
        upper_bound = spot * np.exp(-dividend_yield * maturity)
    else:
        lower_bound = max(
            strike * np.exp(-rate * maturity)
            - spot * np.exp(-dividend_yield * maturity),
            0.0,
        )
        upper_bound = strike * np.exp(-rate * maturity)

    if market_price < lower_bound - 1e-6 or market_price > upper_bound + 1e-6:
        return np.nan

    def objective(volatility: float) -> float:
        return black_scholes_price(
            spot,
            strike,
            maturity,
            rate,
            volatility,
            option_type,
            dividend_yield,
        ) - market_price

    try:
        return float(brentq(objective, lower_vol, upper_vol, maxiter=200))
    except (ValueError, RuntimeError):
        return np.nan


def bs_call_torch(
    spot: torch.Tensor,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> torch.Tensor:
    k = torch.as_tensor(strike)
    t = torch.as_tensor(maturity)
    r = torch.as_tensor(rate)
    sigma = torch.as_tensor(volatility)
    normal = torch.distributions.Normal(0.0, 1.0)

    d1 = (torch.log(spot / k) + (r + 0.5 * sigma**2) * t) / (
        sigma * torch.sqrt(t)
    )
    d2 = d1 - sigma * torch.sqrt(t)
    return spot * normal.cdf(d1) - k * torch.exp(-r * t) * normal.cdf(d2)


def autograd_price_delta_gamma(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> tuple[float, float, float]:
    spot_tensor = torch.tensor(float(spot), requires_grad=True)
    price = bs_call_torch(spot_tensor, strike, maturity, rate, volatility)
    delta = torch.autograd.grad(price, spot_tensor, create_graph=True)[0]
    gamma = torch.autograd.grad(delta, spot_tensor)[0]
    return float(price.detach()), float(delta.detach()), float(gamma.detach())


# Self-financing portfolio helper

def build_hedged_portfolio(
    frame: pd.DataFrame,
    option_price_column: str,
    output_prefix: str,
) -> pd.DataFrame:
    """Create a self-financing delta-hedged portfolio marked by one price series."""

    option_value_col = f"{output_prefix}_option_value"
    cash_col = f"{output_prefix}_cash"
    portfolio_col = f"{output_prefix}_portfolio_value"
    pnl_col = f"{output_prefix}_hedged_pnl"
    increment_col = f"{output_prefix}_pnl_increment"

    frame[option_value_col] = OPTION_POSITION * frame[option_price_column]
    frame[cash_col] = 0.0

    initial_hedge_value = frame.loc[0, "hedge_position"] * frame.loc[0, "spot"]
    frame.loc[0, cash_col] = -(
        frame.loc[0, option_value_col]
        + initial_hedge_value
        + frame.loc[0, "transaction_cost"]
    )

    for index in range(1, len(frame)):
        frame.loc[index, cash_col] = (
            frame.loc[index - 1, cash_col]
            - frame.loc[index, "hedge_trade"] * frame.loc[index, "spot"]
            - frame.loc[index, "transaction_cost"]
        )

    frame[portfolio_col] = (
        frame[option_value_col]
        + frame["hedge_position"] * frame["spot"]
        + frame[cash_col]
    )
    frame[pnl_col] = frame[portfolio_col] - frame[portfolio_col].iloc[0]
    frame[increment_col] = frame[pnl_col].diff().fillna(0.0)
    return frame


def safe_correlation(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].std() == 0 or valid.iloc[:, 1].std() == 0:
        return np.nan
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))


# Load and align real data

option = load_option_quotes(OPTION_FILE)
print("Option rows:", len(option))
print("Option start:", option["time"].min())
print("Option end:  ", option["time"].max())
display(option[["time", "bid", "ask", "option_mid", "option_spread"]].head())

option_1s = (
    option[["time", "bid", "ask", "option_mid", "option_spread"]]
    .set_index("time")
    .resample("1s")
    .last()
    .ffill()
    .dropna(subset=["option_mid"])
    .reset_index()
)

START_TIME = option_1s["time"].min().floor("s")
END_TIME = option_1s["time"].max().ceil("s")
print("Backtest start:       ", START_TIME)
print("Backtest end:         ", END_TIME)
print("Option 1-second rows:", len(option_1s))

try:
    spot_api = fetch_okx_history_candles(
        SPOT_SYMBOL,
        START_TIME,
        END_TIME,
        bar=BAR_SIZE,
        limit=API_LIMIT,
    )
    spot_api.to_parquet(SPOT_OUTPUT_FILE, index=False)
    print("Saved spot data to:", SPOT_OUTPUT_FILE.resolve())
except Exception as exc:
    if not SPOT_OUTPUT_FILE.exists():
        raise
    print(f"OKX download failed ({exc}); using local spot file instead.")
    spot_api = pd.read_parquet(SPOT_OUTPUT_FILE)
    spot_api["time"] = pd.to_datetime(spot_api["time"], utc=True)

spot_1s = (
    spot_api[["time", "open", "high", "low", "close", "volume"]]
    .set_index("time")
    .resample("1s")
    .last()
    .ffill()
    .reset_index()
    .rename(columns={"close": "spot"})
    .dropna(subset=["spot"])
)

# Backward matching avoids using future spot observations (look-ahead bias).
market = pd.merge_asof(
    option_1s.sort_values("time"),
    spot_1s.sort_values("time"),
    on="time",
    direction="backward",
    tolerance=pd.Timedelta("2s"),
)
market = (
    market.dropna(subset=["option_mid", "spot"])
    .sort_values("time")
    .reset_index(drop=True)
)

print("Merged rows: ", len(market))
print("Merged start:", market["time"].min())
print("Merged end:  ", market["time"].max())
display(market[["time", "spot", "bid", "ask", "option_mid"]].head())


# Market IV calibration and fixed-IV model Greeks

market["T"] = market["time"].map(time_to_expiry_years)
market["market_implied_vol"] = [
    implied_volatility(
        market_price=mid,
        spot=spot,
        strike=STRIKE,
        maturity=t,
        rate=RISK_FREE_RATE,
        option_type=OPTION_TYPE,
        dividend_yield=DIVIDEND_YIELD,
    )
    for mid, spot, t in zip(market["option_mid"], market["spot"], market["T"])
]
market["market_implied_vol"] = (
    market["market_implied_vol"]
    .replace([np.inf, -np.inf], np.nan)
    .interpolate(limit_direction="both")
)
market = market.dropna(subset=["market_implied_vol"]).reset_index(drop=True)

calibration_sample = market["market_implied_vol"].iloc[:IV_CALIBRATION_OBSERVATIONS]
if calibration_sample.empty:
    raise ValueError("No valid implied-volatility observations for calibration.")
SIGMA_IMPLIED = float(calibration_sample.median())
print(f"Fixed implied volatility used in backtest: {SIGMA_IMPLIED:.6f}")

model_outputs = [
    autograd_price_delta_gamma(
        spot=spot,
        strike=STRIKE,
        maturity=t,
        rate=RISK_FREE_RATE,
        volatility=SIGMA_IMPLIED,
    )
    for spot, t in zip(market["spot"], market["T"])
]
market[["model_option_price", "delta", "gamma"]] = pd.DataFrame(
    model_outputs,
    index=market.index,
)

print(
    market[
        [
            "spot",
            "option_mid",
            "model_option_price",
            "T",
            "market_implied_vol",
            "delta",
            "gamma",
        ]
    ].describe()
)
display(
    market[
        [
            "time",
            "spot",
            "option_mid",
            "model_option_price",
            "market_implied_vol",
            "delta",
            "gamma",
        ]
    ].head()
)


# Delta hedge: primary model portfolio and secondary market portfolio

backtest = market.copy()
backtest["option_position"] = OPTION_POSITION

# Total option Delta = OPTION_POSITION * delta. The BTC hedge is its negative.
backtest["hedge_position"] = -OPTION_POSITION * backtest["delta"]
backtest["hedge_trade"] = backtest["hedge_position"].diff()
backtest.loc[0, "hedge_trade"] = backtest.loc[0, "hedge_position"]

transaction_cost_rate = TRANSACTION_COST_BPS / 10_000.0
backtest["transaction_cost"] = (
    backtest["hedge_trade"].abs() * backtest["spot"] * transaction_cost_rate
)

# Main identity verification: fixed-IV model price.
backtest = build_hedged_portfolio(backtest, "model_option_price", "model")

# Secondary comparison only: observed Coincall mid-quote.
backtest = build_hedged_portfolio(backtest, "option_mid", "market")


# Gamma-theta-variance identity

backtest["dt_years"] = (
    backtest["time"].diff().dt.total_seconds() / SECONDS_PER_YEAR
)
backtest["spot_change"] = backtest["spot"].diff()
backtest["realized_variance_term"] = backtest["spot_change"] ** 2
backtest["implied_variance_term"] = (
    SIGMA_IMPLIED**2
    * backtest["spot"].shift(1) ** 2
    * backtest["dt_years"]
)
backtest["position_gamma"] = OPTION_POSITION * backtest["gamma"]
backtest["identity_pnl_increment"] = (
    0.5
    * backtest["position_gamma"].shift(1)
    * (
        backtest["realized_variance_term"]
        - backtest["implied_variance_term"]
    )
)
backtest["identity_pnl_increment"] = (
    backtest["identity_pnl_increment"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)
backtest["identity_cumulative_pnl"] = backtest["identity_pnl_increment"].cumsum()

# Increment-level residuals are the primary diagnostic.
backtest["model_identity_residual"] = (
    backtest["model_pnl_increment"] - backtest["identity_pnl_increment"]
)
backtest["market_identity_residual"] = (
    backtest["market_pnl_increment"] - backtest["identity_pnl_increment"]
)



# Statistics

log_returns = np.log(backtest["spot"] / backtest["spot"].shift(1)).dropna()
median_dt_seconds = backtest["time"].diff().dt.total_seconds().median()
periods_per_year = SECONDS_PER_YEAR / median_dt_seconds
realized_volatility = float(log_returns.std(ddof=1) * np.sqrt(periods_per_year))
average_market_iv = float(backtest["market_implied_vol"].mean())

model_increment_corr = safe_correlation(
    backtest["model_pnl_increment"].iloc[1:],
    backtest["identity_pnl_increment"].iloc[1:],
)
market_increment_corr = safe_correlation(
    backtest["market_pnl_increment"].iloc[1:],
    backtest["identity_pnl_increment"].iloc[1:],
)

model_final_pnl = float(backtest["model_hedged_pnl"].iloc[-1])
market_final_pnl = float(backtest["market_hedged_pnl"].iloc[-1])
identity_final_pnl = float(backtest["identity_cumulative_pnl"].iloc[-1])
model_tracking_error = model_final_pnl - identity_final_pnl
market_tracking_error = market_final_pnl - identity_final_pnl

model_increment_rmse = float(
    np.sqrt(np.mean(backtest["model_identity_residual"].iloc[1:] ** 2))
)
model_increment_mae = float(
    np.mean(np.abs(backtest["model_identity_residual"].iloc[1:]))
)
model_max_abs_increment_error = float(
    np.max(np.abs(backtest["model_identity_residual"].iloc[1:]))
)

print(f"Annualized realized volatility:       {realized_volatility:.6f}")
print(f"Fixed implied volatility:            {SIGMA_IMPLIED:.6f}")
print(f"Average market implied volatility:   {average_market_iv:.6f}")
print(f"Final model hedged P&L:               {model_final_pnl:.6f}")
print(f"Final market-quote hedged P&L:        {market_final_pnl:.6f}")
print(f"Final identity P&L:                   {identity_final_pnl:.6f}")
print(f"Model vs identity tracking error:     {model_tracking_error:.12f}")
print(f"Model incremental correlation:        {model_increment_corr:.12f}")
print(f"Market incremental correlation:       {market_increment_corr:.12f}")
print(f"Model incremental RMSE:               {model_increment_rmse:.12e}")
print(f"Model incremental MAE:                {model_increment_mae:.12e}")
print(f"Model maximum absolute step error:    {model_max_abs_increment_error:.12e}")

summary = pd.DataFrame(
    {
        "Metric": [
            "Option contract",
            "Position",
            "Strike",
            "Backtest observations",
            "Initial BTC spot",
            "Final BTC spot",
            "Fixed implied volatility (sigma_imp)",
            "Average market implied volatility",
            "Annualized realized volatility",
            "Final P&L - fixed-IV model",
            "Final P&L - observed market mid",
            "Final P&L - identity",
            "Tracking error - model vs identity",
            "Tracking error - market vs identity",
            "Incremental corr - model vs identity",
            "Incremental corr - market vs identity",
            "Incremental RMSE - model vs identity",
            "Incremental MAE - model vs identity",
            "Maximum absolute step error - model vs identity",
            "Total transaction costs",
        ],
        "Value": [
            "BTCUSD-3JUL26-65000-C",
            "Short 1 call, delta hedged",
            STRIKE,
            len(backtest),
            backtest["spot"].iloc[0],
            backtest["spot"].iloc[-1],
            SIGMA_IMPLIED,
            average_market_iv,
            realized_volatility,
            model_final_pnl,
            market_final_pnl,
            identity_final_pnl,
            model_tracking_error,
            market_tracking_error,
            model_increment_corr,
            market_increment_corr,
            model_increment_rmse,
            model_increment_mae,
            model_max_abs_increment_error,
            backtest["transaction_cost"].sum(),
        ],
    }
)
display(summary)



plt.figure()
plt.plot(backtest["time"], backtest["spot"], linewidth=1)
plt.title("BTC-USDT Spot Price")
plt.xlabel("Time")
plt.ylabel("Spot Price")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(backtest["time"], backtest["option_mid"], label="Observed mid", linewidth=1)
plt.plot(
    backtest["time"],
    backtest["model_option_price"],
    label="Fixed-IV model price",
    linewidth=1,
)
plt.title("Observed Option Mid vs Fixed-IV Model Price")
plt.xlabel("Time")
plt.ylabel("Option Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(
    backtest["time"],
    backtest["model_hedged_pnl"],
    label="Fixed-IV model hedged P&L",
    linewidth=1.5,
)
plt.plot(
    backtest["time"],
    backtest["identity_cumulative_pnl"],
    label="Gamma-theta-variance identity",
    linewidth=1.5,
)
plt.title("Primary Verification: Model P&L vs Identity")
plt.xlabel("Time")
plt.ylabel("P&L")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(
    backtest["time"],
    backtest["market_hedged_pnl"],
    label="Observed market-mid hedged P&L",
    linewidth=1.2,
)
plt.plot(
    backtest["time"],
    backtest["identity_cumulative_pnl"],
    label="Gamma-theta-variance identity",
    linewidth=1.2,
)
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Secondary Comparison: Market Mid vs Identity")
plt.xlabel("Time")
plt.ylabel("P&L")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(
    backtest["identity_pnl_increment"].iloc[1:],
    backtest["model_pnl_increment"].iloc[1:],
    s=8,
    alpha=0.5,
)
minimum = min(
    backtest["identity_pnl_increment"].iloc[1:].min(),
    backtest["model_pnl_increment"].iloc[1:].min(),
)
maximum = max(
    backtest["identity_pnl_increment"].iloc[1:].max(),
    backtest["model_pnl_increment"].iloc[1:].max(),
)
plt.plot([minimum, maximum], [minimum, maximum], linestyle="--", linewidth=1)
plt.title("Incremental Model P&L vs Identity")
plt.xlabel("Identity P&L increment")
plt.ylabel("Model hedged P&L increment")
plt.grid(True)
plt.tight_layout()
plt.show()


backtest.to_csv(RESULT_FILE, index=False)
summary.to_csv(SUMMARY_FILE, index=False)
print("Saved:")
print(" -", RESULT_FILE.resolve())
print(" -", SUMMARY_FILE.resolve())


# In[ ]:




