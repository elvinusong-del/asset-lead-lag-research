# Task 2 Summary

## Settings

- Clustering: TimeSeriesKMeans, k in 3-8, seed 42, n_init 10; DTW with Sakoe-Chiba radius 10 bars
- Rolling: 63-trading-day window, 10-trading-day step; 116 windows (2022-01-27 ~ 2026-08-28), 122 blocks
- Volatility event: median across assets of each asset's RV percentile vs its previous 26 blocks >= 70%; first crossing only; first 26 blocks not judged

## Key results

1. **Optimal k.** DTW k = 3, Euclidean k = 3 (max Silhouette). Cluster sizes at the optimum: DTW [14, 8, 28], Euclidean [34, 8, 8].

   | k | dtw | euclidean |
   | --- | --- | --- |
   | 3 | 0.266 | 0.322 |
   | 4 | 0.245 | 0.242 |
   | 5 | 0.212 | 0.227 |
   | 6 | 0.195 | 0.196 |
   | 7 | 0.185 | 0.204 |
   | 8 | 0.187 | 0.208 |

2. **DTW vs Euclidean agreement.** ARI on the full-period fit = 0.648, median rolling-window ARI = 0.891. Downstream analysis is run and reported for both metrics regardless of ARI.
3. **Volatility-expansion events.** 12 events: 2024-02-09, 2024-04-09, 2024-07-19, 2024-08-30, 2024-10-28, 2025-01-27, 2025-02-25, 2026-01-26, 2026-02-24, 2026-03-24, 2026-05-06, 2026-06-04.
4. **Migration activity** across 115 window transitions:
   - DTW: 1368 migrations. Most active: SNAP (45), WDAY (39), INTC (37), CPNG (36), COIN (35). Least active: QQQ (14), SMH (15), TQQQ (18), NVDA (20), SOXL (20).
   - Euclidean: 1338 migrations. Most active: ARKK (38), UBER (38), SNAP (37), PG (36), WDAY (36). Least active: QQQ (13), TQQQ (15), NVDA (18), XLK (19), SMH (20).
5. **Pre-event migrations (>= 5).** DTW: 11 assets, Euclidean: 12 assets; in both: ARKK, BAC, COIN, CPNG, MSFT, ONON, SNAP, UBER, XYZ. No highlighting rule is applied yet (pending).

   DTW

   | Symbol | Pre-event moves | Pre / total |
   | --- | --- | --- |
   | CPNG | 8/12 | 8/36 (22%) |
   | ONON | 7/12 | 7/30 (23%) |
   | SNAP | 6/12 | 6/45 (13%) |
   | XYZ | 5/12 | 5/24 (21%) |
   | BAC | 5/12 | 5/30 (17%) |
   | BITO | 5/12 | 5/30 (17%) |
   | ARKK | 5/12 | 5/31 (16%) |
   | PG | 5/12 | 5/31 (16%) |
   | MSFT | 5/12 | 5/33 (15%) |
   | COIN | 5/12 | 5/35 (14%) |
   | UBER | 5/12 | 5/35 (14%) |

   Euclidean

   | Symbol | Pre-event moves | Pre / total |
   | --- | --- | --- |
   | ARKK | 8/12 | 8/38 (21%) |
   | SNAP | 7/12 | 7/37 (19%) |
   | ONON | 6/12 | 6/32 (19%) |
   | CPNG | 6/12 | 6/34 (18%) |
   | SOXL | 5/12 | 5/20 (25%) |
   | MSFT | 5/12 | 5/23 (22%) |
   | SMCI | 5/12 | 5/26 (19%) |
   | XYZ | 5/12 | 5/26 (19%) |
   | INTC | 5/12 | 5/32 (16%) |
   | BAC | 5/12 | 5/33 (15%) |
   | COIN | 5/12 | 5/33 (15%) |
   | UBER | 5/12 | 5/38 (13%) |

## Output files

| File | Content |
| --- | --- |
| `k_selection.csv` | Silhouette and inertia per metric and k (2.1) |
| `labels_full.csv` | Full-period cluster label per asset at each metric's optimal k |
| `membership_{metric}.csv` | Migration matrix: asset x window end date, Hungarian-aligned labels |
| `migrations_{metric}.csv` | 1 if the asset changed cluster vs the previous window |
| `migration_counts_{metric}.csv` | Per asset: total moves, pre-event moves, ratio, event count |
| `pre_event_migrations_{metric}.csv` | Subset with >= 5 pre-event moves |
| `ari_windows.csv` | DTW vs Euclidean ARI per rolling window |
| `vol_blocks.csv` | Per 2-week block: dates, indicator, hot, event |
| `cluster_drift_report.pdf` | Silhouette curves, volatility indicator, migrations per asset |

Membership is recorded at each window's end date (no look-ahead). A pre-event migration is one recorded by the window ending at the event block's start; windows overlapping the event block are excluded.