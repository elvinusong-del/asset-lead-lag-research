# Task 1 Summary

- Period: 2021-10-19 09:30:00-04:00 ~ 2026-09-02 15:45:00-04:00 (ET)
- Trading days: 1223, bars: 31,678 (returns: 31,677)
- Bars per day: {26: 1213, 14: 10}
- Calendar check: 0 bars only in calendar, 0 bars only in data
- Final universe: 50 assets (raw: 52)
- Forward-filled bars: 0
- ADF on log returns (p < 0.01) failed: none
- Price levels non-stationary (ADF p >= 0.01): 47

## Key results

1. **Data quality.** All 50 assets sit exactly on the XNYS trading grid (0 calendar-only bars, 0 data-only bars) with no intraday gaps; 0 bars needed forward-fill. Assets were dropped only for late listing, not for missing data.
2. **Price levels are non-stationary.** 47 of 50 assets fail to reject a unit root in raw prices (ADF with constant + trend, p >= 0.01), so prices cannot be used directly in Granger/VAR models.
   - Exceptions (p < 0.01): AGQ (p=2.4e-03), SNAP (p=8.3e-08), XYZ (p=1.4e-04). Rejecting here does not imply tradable mean reversion (a large level shift followed by a flat stretch can produce it); no action is needed since the analysis uses returns.
3. **Log returns are stationary.** All 50 assets reject a unit root in log returns (ADF with constant, max p = 1e-16 or smaller; AIC lags 0-51). `log_returns.parquet` is the input for Tasks 2.2, 3 and 4.
4. **Log-price Z-score curves saved.** `zscore_logprice.parquet` holds (ln P - mean) / std over the full period for every asset, making assets with very different price levels comparable on one scale while keeping each asset's net trend. Verified: cumulative log returns equal ln P_t - ln P_0, so curves built from prices and from returns are identical. Input for Task 2.1.
5. **Overnight gaps dominate extreme moves.** 20 of the 20 largest |returns| occur on the day's first bar (largest: SNAP 2022-05-24 -0.493), consistent with earnings gaps rather than split artifacts. Overnight returns stay in the series as agreed; revisit the first-bar distortion check (plan 1.1) if Task 4.1 outliers cluster on first bars.

## Excluded assets

| Symbol | First bar | Reason |
| --- | --- | --- |
| FNGU | 2025-02-20 09:45:00-05:00 | listed 2025-02-20, insufficient history |
| SPCX | 2026-06-12 11:45:00-04:00 | listed 2026-06-12, insufficient history |
| (ADF) | | none |

## Late listings (kept by trimming the start date)

- BITO: 2021-10-19 09:30:00-04:00
- COIN: 2021-04-14 13:15:00-04:00
- CPNG: 2021-03-11 12:15:00-05:00
- ONON: 2021-09-15 13:15:00-04:00

## Output files

Parquet files share one index: ET bar-start timestamps (tz-aware, `America/New_York`) on the XNYS trading grid; columns are the symbols in `universe.csv`.

| File | Shape | Content | Used by |
| --- | --- | --- | --- |
| `close_aligned.parquet` | 31,678 x 50 | Close prices aligned to the trading grid after gap handling | 4.3 (forward returns) |
| `log_returns.parquet` | 31,677 x 50 | r_t = ln(P_t / P_t-1), overnight gap included in each day's first bar; no NaN | 2.2 (volatility), 3, 4.1, 4.2 |
| `zscore_logprice.parquet` | 31,678 x 50 | (ln P - mean) / std over the full period; keeps the net trend | 2.1 (optimal k) |
| `missing_log.csv` | 1 row per raw asset | First bar, NaN run count, max run, forward-filled bars, excluded flag, reason, late-listing note | audit |
| `adf_results.csv` | 1 row per asset | ADF stat, p-value, lags, nobs for price levels (`ct`) and log returns (`c`), AIC lag selection; `stationary` = return p < 0.01 | audit |
| `universe.csv` | 1 column | Final asset list used by Tasks 2-4 | 2-4 |
| `summary.md` | - | This file | - |

Load in later tasks with `from task1_prep import load_task1_outputs, zscore_logprice`. For Task 2.2, re-normalize each rolling window with `zscore_logprice(np.log(close).loc[window])`.