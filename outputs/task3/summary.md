# Task 3 Summary

## Overview

Task 3 found no persistent lead-lag relation. Lead-lag appears more often than chance, but it comes and goes across windows. Significant results also cluster in a few high-volatility periods, which may be an artifact of F-test over-rejection (not yet verified).

## Settings

- 50 assets: 1,225 pairs, 2,450 directions; 116 rolling windows (same as Task 2, 2022-01-27 ~ 2026-08-28)
- Bivariate VAR, constant only, all lags t-1..t-p*; p* = argmin BIC over p = 2-8 per pair and window
- Per-window significance: raw p < 0.01; BH-FDR q = 0.05 across all directions in the window
- Consensus: A_ij = 1 if the direction's share of significant windows is in the top 5% of all directions (computed separately for raw and FDR; ties at the cut kept)

## Key results

### 1. Lag order

BIC picks the lower bound p = 2 in 99.9% of pair-windows, so it would prefer fewer lags if allowed.

| p* | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pair-windows | 283,810 | 352 | 24 | 6 | 0 | 4 | 4 |

### 2. Significant directions per window (of 2,450)

| Test | Median | Range |
| --- | --- | --- |
| raw (p < 0.01) | 112 | 37 - 968 |
| FDR (q = 0.05) | 16 | 0 - 1,171 |

- Chance level for raw is about 24 per window; a placebo check (all series shuffled, one window) gives a raw rejection rate of 1.55%.
- Largest windows (raw count, window end date): 2025-04-07 (968), 2025-07-03 (567), 2025-06-18 (501).
- These contain the April 2025 sell-off. With volatility clustering the homoskedastic F test can over-reject, so spikes need not mean stronger lead-lag.

### 3. Consensus network

| | raw | FDR |
| --- | --- | --- |
| Cut (top 5% frequency) | >= 15.5% | >= 8.6% |
| Edges | 124 | 140 |
| Bidirectional pairs | 2 | 5 |
| Hubs (out-degree) | SNAP 13, TQQQ 8, SOXL 8, SMH 7, BITO 7 | SNAP 15, BITO 11, TQQQ 11, SOXL 8, SMH 7 |

Top 10 directions by frequency:

| Rank | raw | FDR |
| --- | --- | --- |
| 1 | SLV -> AGQ (28%) | SLV -> PPLT (16%) |
| 2 | MSFT -> GOOGL (25%) | SNAP -> UBER (16%) |
| 3 | MSFT -> CCJ (23%) | AAPL -> SMCI (15%) |
| 4 | NFLX -> PG (23%) | SOXL -> MSTR (14%) |
| 5 | SLV -> PPLT (23%) | BAC -> JPM (13%) |
| 6 | SMH -> SMCI (23%) | BITO -> XBI (13%) |
| 7 | SMH -> XBI (23%) | BITO -> XLF (13%) |
| 8 | SNAP -> CPNG (23%) | SLV -> AGQ (13%) |
| 9 | WDAY -> XBI (23%) | TQQQ -> XBI (13%) |
| 10 | NVDA -> SMCI (22%) | WDAY -> IWM (13%) |

### 4. Decay check

Criterion pending. Per-window p*, cross-lag coefficients, standard errors, F and p-values are stored in `var_results.parquet` for the later decision.

## Current findings

1. **Lead-lag relations are significant in too few windows.**
   - A direction is significant (raw) in a median 6.0% of the 116 windows (FDR: 2.6%); the most frequent direction reaches only 28% (FDR: 16%).
   - Directions significant in >= 50% of windows: 0 (raw), 0 (FDR). No direction is persistently significant.
   - With no lead-lag at all, raw frequency would center on 1%. Observed frequencies are above that, so relations exist more often than chance, but they appear and disappear across windows rather than holding steadily.
   - The top 5% consensus therefore selects relatively frequent directions, not stable ones.
2. **Significant results cluster in specific periods.**
   - 22 of 116 windows (19%) have more than twice the median number of significant directions (median 112); together they hold 44% of all raw significant results.

   | Period (window end dates) | Windows | Share of all raw significant results | Peak window (count) |
   | --- | --- | --- | --- |
   | 2025-04-07 ~ 2025-07-03 | 7 | 21% | 2025-04-07 (968) |
   | 2022-11-11 ~ 2023-06-06 | 11 | 18% | 2023-01-11 (367) |
   | 2026-02-23 ~ 2026-03-09 | 2 | 3% | 2026-02-23 (287) |
   | 2022-06-07 ~ 2022-06-07 | 1 | 1% | 2022-06-07 (245) |
   | 2022-02-10 ~ 2022-02-10 | 1 | 1% | 2022-02-10 (226) |

   - Each window spans the 63 trading days before its end date, so a period's results reflect market conditions up to about 3 months earlier.
   - The most extreme window contains the April 2025 sell-off. In high-volatility periods the homoskedastic F test can over-reject, so part of this clustering may reflect volatility rather than lead-lag.

## Output files

| File | Content |
| --- | --- |
| `adjacency_raw.csv`, `adjacency_fdr.csv` | 50 x 50 consensus adjacency, row -> column |
| `frequency_raw.csv`, `frequency_fdr.csv` | 50 x 50 share of windows significant |
| `hubs.csv` | Top 5 by out-degree per test, with in-degree |
| `var_results.parquet` | Window x direction (116 x 2,450 rows): window_end, source, target, p_star, F, pvalue, sig_raw, sig_fdr, coef_1..8, se_1..8 (cross-lag terms of source in the target equation; NaN beyond p*). Used for the decay check and Task 4.2 |
| `causality_topology.pdf` | Hubs and the assets they lead (raw, FDR), frequency histogram, network graphs, significant directions per window |

Coefficients for a window are known only at its end date; Task 4.2 must use the window that ended before time t.