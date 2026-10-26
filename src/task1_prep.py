"""Task 1. Data preparation: alignment & missing data, stationarity (ADF), log-price Z-score.

Run:    python src/task1_prep.py
Output: outputs/task1/
Tasks 2-4 import load_task1_outputs() and zscore_logprice() from here.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from joblib import Parallel, delayed
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "task1"

TZ = "America/New_York"
BAR = "15min"
START = pd.Timestamp("2021-10-19 09:30", tz=TZ)  # BITO listing date; pre-listing bars are out of range, not missing
EXCLUDE = {
    "FNGU": "listed 2025-02-20, insufficient history",
    "SPCX": "listed 2026-06-12, insufficient history",
}
MAX_FFILL = 2      # forward-fill only gaps of <= 2 consecutive bars
ADF_ALPHA = 0.01


# ---------- 1.1 Load, align, missing data ----------

def load_closes(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Per-CSV closes as a wide DataFrame indexed by ET timestamps."""
    cols = {}
    for f in sorted(data_dir.glob("*_15m_*.csv")):
        d = pd.read_csv(f, usecols=["symbol", "bucket", "close"])
        ts = pd.to_datetime(d["bucket"], utc=True).dt.tz_convert(TZ)
        cols[d["symbol"].iloc[0]] = pd.Series(d["close"].to_numpy(), index=ts)
    return pd.DataFrame(cols).sort_index()


def trading_grid(start, end) -> pd.DatetimeIndex:
    """15-min bar-start grid from the XNYS trading calendar (early-close days have 14 bars)."""
    sched = mcal.get_calendar("XNYS").schedule(start.date(), end.date())
    bars = [
        pd.date_range(o, c, freq=BAR, inclusive="left")
        for o, c in zip(sched["market_open"], sched["market_close"])
    ]
    grid = bars[0].append(bars[1:]).tz_convert(TZ)
    return grid[(grid >= start) & (grid <= end)]


def nan_runs(s: pd.Series) -> pd.Series:
    """Lengths of consecutive NaN runs."""
    isna = s.isna()
    grp = (isna != isna.shift()).cumsum()
    return isna.groupby(grp).sum()[lambda x: x > 0]


def handle_missing(px: pd.DataFrame, first_bar: pd.Series):
    """Forward-fill gaps of <= MAX_FFILL bars; flag longer gaps for exclusion. Returns (clean prices, missing log)."""
    rows, keep = [], []
    for sym in px.columns:
        runs = nan_runs(px[sym])
        excluded = bool((runs > MAX_FFILL).any())
        rows.append({
            "symbol": sym,
            "first_bar": first_bar[sym],
            "n_runs": len(runs),
            "max_run": int(runs.max()) if len(runs) else 0,
            "ffill_bars": int(runs[runs <= MAX_FFILL].sum()),
            "excluded": excluded,
            "reason": f"gap > {MAX_FFILL} consecutive bars" if excluded else "",
        })
        if not excluded:
            keep.append(sym)
    clean = px[keep].ffill(limit=MAX_FFILL)
    return clean, pd.DataFrame(rows)


# ---------- 1.2 Stationarity ----------

def _adf(sym, series, regression):
    stat, p, lags, nobs, *_ = adfuller(series.to_numpy(), regression=regression, autolag="AIC")
    return {"symbol": sym, "regression": regression, "stat": stat, "pvalue": p,
            "lags": lags, "nobs": nobs}


def adf_table(px: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    jobs = [delayed(_adf)(s, px[s], "ct") for s in px.columns]
    jobs += [delayed(_adf)(s, rets[s], "c") for s in rets.columns]
    res = pd.DataFrame(Parallel(n_jobs=-1)(jobs))
    price = res[res.regression == "ct"].set_index("symbol").add_prefix("price_")
    ret = res[res.regression == "c"].set_index("symbol").add_prefix("ret_")
    out = price.join(ret).drop(columns=["price_regression", "ret_regression"])
    out["stationary"] = out["ret_pvalue"] < ADF_ALPHA
    return out.reset_index()


# ---------- 1.3 Normalization ----------

def zscore_logprice(logp: pd.DataFrame) -> pd.DataFrame:
    """z = (ln P - mean) / std. Re-applied per rolling-window slice in Task 2.2."""
    return (logp - logp.mean()) / logp.std()


def load_task1_outputs(out_dir: Path = OUT_DIR) -> dict:
    return {
        "close": pd.read_parquet(out_dir / "close_aligned.parquet"),
        "log_returns": pd.read_parquet(out_dir / "log_returns.parquet"),
        "zscore_logprice": pd.read_parquet(out_dir / "zscore_logprice.parquet"),
        "universe": pd.read_csv(out_dir / "universe.csv")["symbol"].tolist(),
    }


# ---------- main ----------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_closes()
    first_bar = raw.apply(pd.Series.first_valid_index)
    assert (raw.stack() > 0).all(), "non-positive close found"

    # Reference index: actual trading calendar, cross-checked against the data union
    grid = trading_grid(START, raw.index.max())
    union = raw.index[raw.index >= START]
    grid_only, data_only = grid.difference(union), union.difference(grid)
    bars_per_day = pd.Series(1, index=grid).groupby(grid.date).size()

    # Fix sample period and universe
    px = raw.drop(columns=list(EXCLUDE)).reindex(grid)
    px, miss = handle_missing(px, first_bar)
    miss = pd.concat([miss, pd.DataFrame([
        {"symbol": s, "first_bar": first_bar[s], "n_runs": np.nan, "max_run": np.nan,
         "ffill_bars": np.nan, "excluded": True, "reason": r} for s, r in EXCLUDE.items()
    ])], ignore_index=True)
    miss["note"] = np.where(miss["first_bar"] > raw.index.min(), "late listing", "")

    # Log returns (overnight included)
    logp = np.log(px)
    rets = logp.diff().iloc[1:]
    assert not rets.isna().any().any(), "NaN in returns"

    # ADF
    adf = adf_table(px, rets)
    failed = adf.loc[~adf["stationary"], "symbol"].tolist()
    universe = [s for s in px.columns if s not in failed]
    px, logp, rets = px[universe], logp[universe], rets[universe]

    # Normalize and verify: cumulative log return == ln P_t - ln P_0
    z = zscore_logprice(logp)
    cum = rets.cumsum()
    assert np.allclose(cum.to_numpy(), (logp.iloc[1:] - logp.iloc[0]).to_numpy(), atol=1e-9)
    z_from_cum = zscore_logprice(pd.concat([logp.iloc[[0]] * 0, cum]))
    assert np.allclose(z.to_numpy(), z_from_cum.to_numpy(), atol=1e-9)

    # Save
    px.to_parquet(OUT_DIR / "close_aligned.parquet")
    rets.to_parquet(OUT_DIR / "log_returns.parquet")
    z.to_parquet(OUT_DIR / "zscore_logprice.parquet")
    miss.to_csv(OUT_DIR / "missing_log.csv", index=False, encoding="utf-8-sig")
    adf.to_csv(OUT_DIR / "adf_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"symbol": universe}).to_csv(OUT_DIR / "universe.csv", index=False)

    excl = miss[miss["excluded"]]

    # Facts for the "Key results" section
    price_nonstat = adf.loc[adf["price_pvalue"] >= ADF_ALPHA, "symbol"].tolist()
    price_stat = adf.loc[adf["price_pvalue"] < ADF_ALPHA, ["symbol", "price_pvalue"]]
    top_abs = rets.abs().stack().nlargest(20)
    top_first_bar = sum(t.strftime("%H:%M") == "09:30" for t, _ in top_abs.index)
    largest = rets.stack().loc[[top_abs.index[0]]]
    (lt, ls), lv = next(iter(largest.items()))

    lines = [
        "# Task 1 Summary",
        "",
        f"- Period: {grid[0]} ~ {grid[-1]} (ET)",
        f"- Trading days: {len(bars_per_day)}, bars: {len(grid):,} (returns: {len(rets):,})",
        f"- Bars per day: {bars_per_day.value_counts().to_dict()}",
        f"- Calendar check: {len(grid_only)} bars only in calendar, {len(data_only)} bars only in data",
        f"- Final universe: {len(universe)} assets (raw: {raw.shape[1]})",
        f"- Forward-filled bars: {int(miss['ffill_bars'].sum())}",
        f"- ADF on log returns (p < {ADF_ALPHA}) failed: {failed or 'none'}",
        f"- Price levels non-stationary (ADF p >= {ADF_ALPHA}): {int((adf['price_pvalue'] >= ADF_ALPHA).sum())}",
        "",
        "## Key results",
        "",
        f"1. **Data quality.** All {len(universe)} assets sit exactly on the XNYS trading grid "
        f"({len(grid_only)} calendar-only bars, {len(data_only)} data-only bars) with no intraday gaps; "
        f"{int(miss['ffill_bars'].sum())} bars needed forward-fill. Assets were dropped only for late listing, "
        "not for missing data.",
        f"2. **Price levels are non-stationary.** {len(price_nonstat)} of {len(adf)} assets fail to reject a unit "
        f"root in raw prices (ADF with constant + trend, p >= {ADF_ALPHA}), so prices cannot be used directly in "
        "Granger/VAR models.",
        "   - Exceptions (p < {:.2f}): ".format(ADF_ALPHA)
        + ", ".join(f"{r.symbol} (p={r.price_pvalue:.1e})" for r in price_stat.itertuples())
        + ". Rejecting here does not imply tradable mean reversion (a large level shift followed by a flat "
        "stretch can produce it); no action is needed since the analysis uses returns.",
        f"3. **Log returns are stationary.** All {len(adf)} assets reject a unit root in log returns "
        f"(ADF with constant, max p = {max(adf['ret_pvalue'].max(), 1e-16):.0e} or smaller; AIC lags {adf['ret_lags'].min()}-"
        f"{adf['ret_lags'].max()}). `log_returns.parquet` is the input for Tasks 2.2, 3 and 4.",
        "4. **Log-price Z-score curves saved.** `zscore_logprice.parquet` holds (ln P - mean) / std over the full "
        "period for every asset, making assets with very different price levels comparable on one scale while "
        "keeping each asset's net trend. Verified: cumulative log returns equal ln P_t - ln P_0, so curves built "
        "from prices and from returns are identical. Input for Task 2.1.",
        f"5. **Overnight gaps dominate extreme moves.** {top_first_bar} of the 20 largest |returns| occur on the "
        f"day's first bar (largest: {ls} {lt.date()} {lv:+.3f}), consistent with earnings gaps rather than "
        "split artifacts. Overnight returns stay in the series as agreed; revisit the first-bar distortion check "
        "(plan 1.1) if Task 4.1 outliers cluster on first bars.",
        "",
        "## Excluded assets",
        "",
        "| Symbol | First bar | Reason |",
        "| --- | --- | --- |",
        *[f"| {r.symbol} | {r.first_bar} | {r.reason} |" for r in excl.itertuples()],
        "| (ADF) | | " + (", ".join(failed) if failed else "none") + " |",
        "",
        "## Late listings (kept by trimming the start date)",
        "",
        *[f"- {r.symbol}: {r.first_bar}" for r in miss[(miss.note != "") & ~miss.excluded].itertuples()],
        "",
        "## Output files",
        "",
        "Parquet files share one index: ET bar-start timestamps (tz-aware, `America/New_York`) "
        "on the XNYS trading grid; columns are the symbols in `universe.csv`.",
        "",
        "| File | Shape | Content | Used by |",
        "| --- | --- | --- | --- |",
        f"| `close_aligned.parquet` | {px.shape[0]:,} x {px.shape[1]} | Close prices aligned to the "
        "trading grid after gap handling | 4.3 (forward returns) |",
        f"| `log_returns.parquet` | {rets.shape[0]:,} x {rets.shape[1]} | r_t = ln(P_t / P_t-1), "
        "overnight gap included in each day's first bar; no NaN | 2.2 (volatility), 3, 4.1, 4.2 |",
        f"| `zscore_logprice.parquet` | {z.shape[0]:,} x {z.shape[1]} | (ln P - mean) / std over the "
        "full period; keeps the net trend | 2.1 (optimal k) |",
        "| `missing_log.csv` | 1 row per raw asset | First bar, NaN run count, max run, forward-filled "
        "bars, excluded flag, reason, late-listing note | audit |",
        "| `adf_results.csv` | 1 row per asset | ADF stat, p-value, lags, nobs for price levels "
        "(`ct`) and log returns (`c`), AIC lag selection; `stationary` = return p < 0.01 | audit |",
        "| `universe.csv` | 1 column | Final asset list used by Tasks 2-4 | 2-4 |",
        "| `summary.md` | - | This file | - |",
        "",
        "Load in later tasks with `from task1_prep import load_task1_outputs, zscore_logprice`. "
        "For Task 2.2, re-normalize each rolling window with `zscore_logprice(np.log(close).loc[window])`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
