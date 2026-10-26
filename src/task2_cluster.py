"""Task 2. Behavioral clustering: optimal k (2.1), rolling drift (2.2), volatility-expansion migrations.

Run:    python src/task2_cluster.py
Input:  outputs/task1/
Output: outputs/task2/

DTW k-means is implemented here with a banded (Sakoe-Chiba) DTW in numba. tslearn's DTW builds
full T x T matrices even with a band, which is infeasible at T = 31,678. The algorithm is the same
as tslearn's TimeSeriesKMeans(metric="dtw"): k-means++ init, DTW assignment, DBA barycenters.
Euclidean clustering uses tslearn directly.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
import numba
from numba import njit, prange
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, silhouette_score
from tslearn.clustering import TimeSeriesKMeans

from task1_prep import load_task1_outputs, zscore_logprice

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "task2"

BAND = 10              # Sakoe-Chiba radius in bars (max 2.5h shift), shared by 2.1 and 2.2
SEED = 42
N_INIT = 10
K_RANGE = range(3, 9)
MAX_ITER = 50          # k-means iterations
DBA_ITER = 10          # barycenter refinement steps per k-means iteration

BLOCK_DAYS = 10        # 2-week step / volatility block
WINDOW_DAYS = 63       # 3-month rolling window
LOOKBACK_BLOCKS = 26   # 1 year of blocks for the RV percentile
HOT_LEVEL = 0.70       # block indicator threshold
MIN_PRE_MOVES = 5      # report assets with at least this many pre-event migrations


# ---------- banded DTW (numba) ----------

@njit(cache=True)
def _dtw_band_matrix(x, y, r):
    """Accumulated squared-cost matrix stored as (n, 2r+1); column k = j - i + r."""
    n = x.shape[0]
    w = 2 * r + 1
    D = np.full((n, w), np.inf)
    for i in range(n):
        for j in range(max(0, i - r), min(n, i + r + 1)):
            k = j - i + r
            c = (x[i] - y[j]) ** 2
            if i == 0 and j == 0:
                D[i, k] = c
                continue
            best = np.inf
            if i > 0 and j > 0:
                best = min(best, D[i - 1, k])          # (i-1, j-1)
            if i > 0 and k + 1 < w:
                best = min(best, D[i - 1, k + 1])      # (i-1, j)
            if j > 0 and k - 1 >= 0:
                best = min(best, D[i, k - 1])          # (i, j-1)
            D[i, k] = c + best
    return D


@njit(cache=True)
def dtw_band(x, y, r):
    """DTW distance (sqrt of summed squared cost, same convention as tslearn)."""
    D = _dtw_band_matrix(x, y, r)
    return np.sqrt(D[x.shape[0] - 1, r])


@njit(cache=True)
def _dtw_band_path(x, y, r):
    D = _dtw_band_matrix(x, y, r)
    n = x.shape[0]
    w = 2 * r + 1
    pi = np.empty(2 * n, np.int64)
    pj = np.empty(2 * n, np.int64)
    i, j, L = n - 1, n - 1, 0
    while True:
        pi[L], pj[L] = i, j
        L += 1
        if i == 0 and j == 0:
            break
        k = j - i + r
        best, bi, bj = np.inf, -1, -1
        if i > 0 and j > 0 and D[i - 1, k] < best:
            best, bi, bj = D[i - 1, k], i - 1, j - 1
        if i > 0 and k + 1 < w and D[i - 1, k + 1] < best:
            best, bi, bj = D[i - 1, k + 1], i - 1, j
        if j > 0 and k - 1 >= 0 and D[i, k - 1] < best:
            best, bi, bj = D[i, k - 1], i, j - 1
        i, j = bi, bj
    return pi[:L], pj[:L]


@njit(cache=True)
def _dba(center, members, r, n_iter):
    """DTW Barycenter Averaging starting from `center`."""
    n = center.shape[0]
    c = center.copy()
    for _ in range(n_iter):
        s = np.zeros(n)
        cnt = np.zeros(n)
        for m in range(members.shape[0]):
            pi, pj = _dtw_band_path(c, members[m], r)
            for t in range(pi.shape[0]):
                s[pi[t]] += members[m, pj[t]]
                cnt[pi[t]] += 1
        c = s / cnt
    return c


@njit(cache=True)
def _dist_to_centers(X, C, r):
    out = np.empty((X.shape[0], C.shape[0]))
    for a in range(X.shape[0]):
        for b in range(C.shape[0]):
            out[a, b] = dtw_band(X[a], C[b], r)
    return out


@njit(parallel=True, cache=True)
def cdist_dtw_band(X, r):
    """Symmetric pairwise DTW distance matrix."""
    n = X.shape[0]
    out = np.zeros((n, n))
    for a in prange(n):
        for b in range(a + 1, n):
            out[a, b] = dtw_band(X[a], X[b], r)
    return out + out.T


def _kmeans_dtw_once(X, k, D, seed):
    """One DTW k-means run. X: (n, T). D: pairwise DTW matrix for k-means++ init."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = [rng.integers(n)]
    for _ in range(1, k):
        d2 = (D[:, idx].min(axis=1)) ** 2
        idx.append(rng.choice(n, p=d2 / d2.sum()))
    C = X[idx].copy()
    labels = None
    for _ in range(MAX_ITER):
        dist = _dist_to_centers(X, C, BAND)
        new = dist.argmin(axis=1)
        for c in range(k):                      # refill empty clusters with the worst-fit series
            if not (new == c).any():
                new[dist.min(axis=1).argmax()] = c
        if labels is not None and (new == labels).all():
            break
        labels = new
        for c in range(k):
            C[c] = _dba(C[c], X[labels == c], BAND, DBA_ITER)
    dist = _dist_to_centers(X, C, BAND)
    inertia = float((dist[np.arange(n), labels] ** 2).sum())
    return labels, inertia


def kmeans_dtw(X, k, D, n_jobs=1):
    """Best of N_INIT runs by inertia."""
    runs = Parallel(n_jobs=n_jobs)(
        delayed(_kmeans_dtw_once)(X, k, D, [SEED, i]) for i in range(N_INIT)
    )
    return min(runs, key=lambda t: t[1])


def kmeans_euclidean(X, k):
    m = TimeSeriesKMeans(n_clusters=k, metric="euclidean", n_init=N_INIT, random_state=SEED)
    labels = m.fit_predict(X[:, :, None])
    return labels, float(m.inertia_)


def check_dtw_against_tslearn(X):
    from tslearn.metrics import dtw
    a, b = X[0, :2000], X[1, :2000]
    ref = dtw(a, b, global_constraint="sakoe_chiba", sakoe_chiba_radius=BAND)
    assert np.isclose(dtw_band(a, b, BAND), ref), "banded DTW differs from tslearn"


# ---------- 2.1 optimal k ----------

def select_k(X):
    D = cdist_dtw_band(X, BAND)
    rows, labels = [], {}
    for k in K_RANGE:
        lab_d, in_d = kmeans_dtw(X, k, D, n_jobs=N_INIT)
        lab_e, in_e = kmeans_euclidean(X, k)
        rows.append({"metric": "dtw", "k": k, "inertia": in_d,
                     "silhouette": silhouette_score(D, lab_d, metric="precomputed")})
        rows.append({"metric": "euclidean", "k": k, "inertia": in_e,
                     "silhouette": silhouette_score(X, lab_e)})
        labels[("dtw", k)], labels[("euclidean", k)] = lab_d, lab_e
        print(f"k={k}: silhouette dtw={rows[-2]['silhouette']:.3f} euclidean={rows[-1]['silhouette']:.3f}")
    ks = pd.DataFrame(rows)
    best = ks.loc[ks.groupby("metric")["silhouette"].idxmax()].set_index("metric")["k"].to_dict()
    return ks, best, labels


# ---------- calendar: blocks and windows ----------

def build_calendar(index: pd.DatetimeIndex):
    days = pd.Index(index.normalize().unique())
    n_blocks = len(days) // BLOCK_DAYS
    blocks = pd.DataFrame({
        "block": range(n_blocks),
        "start": days[np.arange(n_blocks) * BLOCK_DAYS],
        "end": days[np.arange(n_blocks) * BLOCK_DAYS + BLOCK_DAYS - 1],
    })
    # Window ending at block b covers the last WINDOW_DAYS trading days up to the end of block b
    end_pos = (blocks["block"] + 1) * BLOCK_DAYS
    win = blocks[end_pos >= WINDOW_DAYS].copy()
    win["win_start"] = days[end_pos[win.index] - WINDOW_DAYS]
    return days, blocks, win.reset_index(drop=True)


def window_slice(logp, day_index, start, end):
    mask = (day_index >= start) & (day_index <= end)
    assert logp.index[mask].max().normalize() <= end, "window uses data after its record date"
    return zscore_logprice(logp[mask]).to_numpy().T


# ---------- 2.2 rolling drift ----------

def _fit_window(Xw, k_dtw, k_euc):
    numba.set_num_threads(1)                    # avoid oversubscription inside joblib workers
    D = cdist_dtw_band(Xw, BAND)
    lab_d, _ = kmeans_dtw(Xw, k_dtw, D)
    lab_e, _ = kmeans_euclidean(Xw, k_euc)
    return lab_d, lab_e


def align_labels(raw: np.ndarray) -> np.ndarray:
    """Chain Hungarian matching: relabel each window to maximize overlap with the previous aligned one."""
    out = raw.copy()
    k = raw.max() + 1
    for w in range(1, raw.shape[0]):
        C = np.zeros((k, k))
        np.add.at(C, (out[w - 1], raw[w]), 1)
        rows, cols = linear_sum_assignment(-C)
        mapping = np.empty(k, int)
        mapping[cols] = rows
        out[w] = mapping[raw[w]]
        assert sorted(np.bincount(out[w], minlength=k)) == sorted(np.bincount(raw[w], minlength=k))
    return out


# ---------- volatility-expansion events ----------

def vol_blocks(rets, blocks):
    day = rets.index.normalize()
    rv = pd.DataFrame({
        b.block: np.sqrt((rets[(day >= b.start) & (day <= b.end)] ** 2).sum())
        for b in blocks.itertuples()
    }).T                                                   # blocks x assets
    pct = pd.DataFrame(np.nan, index=rv.index, columns=rv.columns)
    for b in range(LOOKBACK_BLOCKS, len(rv)):
        past = rv.iloc[b - LOOKBACK_BLOCKS:b]
        pct.iloc[b] = (past < rv.iloc[b]).mean()
    out = blocks.copy()
    out["indicator"] = pct.median(axis=1).to_numpy()
    out["hot"] = out["indicator"] >= HOT_LEVEL
    prev_judged_cold = out["indicator"].shift().notna() & ~out["hot"].shift(fill_value=False)
    out["event"] = out["hot"] & prev_judged_cold
    assert out["indicator"].iloc[:LOOKBACK_BLOCKS].isna().all()
    return out


def migration_analysis(aligned, win, vb, symbols):
    """aligned: windows x assets. Migration at window w = label change vs window w-1."""
    moves = np.zeros_like(aligned, dtype=bool)
    moves[1:] = aligned[1:] != aligned[:-1]
    win_block = win["block"].to_numpy()
    events = vb.loc[vb["event"], "block"].to_numpy()
    # Pre-event migration: recorded by the window ending at the end of block e-1 (event block start)
    pre_rows = [np.where(win_block == e - 1)[0] for e in events]
    pre_rows = np.concatenate([r for r in pre_rows if len(r)]) if events.size else np.array([], int)
    total = moves.sum(axis=0)
    pre = moves[pre_rows].sum(axis=0) if len(pre_rows) else np.zeros(len(symbols), int)
    table = pd.DataFrame({
        "symbol": symbols,
        "total_moves": total,
        "pre_event_moves": pre,
        "pre_over_total": np.where(total > 0, pre / np.maximum(total, 1), np.nan),
        "n_events": len(events),
        "n_events_with_prior_window": len(pre_rows),
    })
    return moves, table


# ---------- report ----------

def make_pdf(path, ks, best, vb, tables):
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(7, 4))
        for m, g in ks.groupby("metric"):
            ax.plot(g["k"], g["silhouette"], marker="o", label=f"{m} (best k={best[m]})")
        ax.set(xlabel="k", ylabel="Silhouette", title="2.1 Optimal k (full-period z-score curves)")
        ax.legend()
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(vb["start"], vb["indicator"], marker=".")
        ax.axhline(HOT_LEVEL, color="red", ls="--", label=f"threshold {HOT_LEVEL:.0%}")
        ax.scatter(vb.loc[vb["event"], "start"], vb.loc[vb["event"], "indicator"], color="red", zorder=3,
                   label=f"events (n={int(vb['event'].sum())})")
        ax.set(ylabel="median RV percentile", title="Volatility-expansion indicator per 2-week block")
        ax.legend()
        pdf.savefig(fig); plt.close(fig)

        for m, t in tables.items():
            t = t.sort_values("total_moves")
            fig, ax = plt.subplots(figsize=(8, 10))
            ax.barh(t["symbol"], t["total_moves"], label="total")
            ax.barh(t["symbol"], t["pre_event_moves"], label="pre-event")
            ax.tick_params(axis="y", labelsize=6)
            ax.set(xlabel="migrations", title=f"Migrations per asset ({m})")
            ax.legend()
            pdf.savefig(fig); plt.close(fig)


# ---------- main ----------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t1 = load_task1_outputs()
    symbols = t1["universe"]
    z, rets = t1["zscore_logprice"][symbols], t1["log_returns"][symbols]
    logp = np.log(t1["close"][symbols])
    X = np.ascontiguousarray(z.to_numpy().T)

    check_dtw_against_tslearn(X)

    # 2.1
    ks, best, labels_full = select_k(X)
    k_dtw, k_euc = best["dtw"], best["euclidean"]
    full = pd.DataFrame({"symbol": symbols, "dtw": labels_full[("dtw", k_dtw)],
                         "euclidean": labels_full[("euclidean", k_euc)]})
    ari_full = adjusted_rand_score(full["dtw"], full["euclidean"])

    # 2.2
    days, blocks, win = build_calendar(logp.index)
    day_index = logp.index.normalize()
    slices = [np.ascontiguousarray(window_slice(logp, day_index, w.win_start, w.end)) for w in win.itertuples()]
    fits = Parallel(n_jobs=-1)(delayed(_fit_window)(Xw, k_dtw, k_euc) for Xw in slices)
    raw = {"dtw": np.array([f[0] for f in fits]), "euclidean": np.array([f[1] for f in fits])}
    ari_win = np.array([adjusted_rand_score(d, e) for d, e in zip(raw["dtw"], raw["euclidean"])])
    ari_med = float(np.median(ari_win))
    metrics = ["dtw", "euclidean"]              # downstream analysis always runs for both metrics

    for m, k in (("dtw", k_dtw), ("euclidean", k_euc)):
        assert raw[m].min() >= 0 and raw[m].max() < k

    vb = vol_blocks(rets, blocks)
    aligned, tables = {}, {}
    for m in metrics:
        aligned[m] = align_labels(raw[m])
        moves, tables[m] = migration_analysis(aligned[m], win, vb, symbols)
        cols = win["end"].dt.strftime("%Y-%m-%d")
        pd.DataFrame(aligned[m].T, index=symbols, columns=cols).to_csv(OUT_DIR / f"membership_{m}.csv")
        pd.DataFrame(moves.T.astype(int), index=symbols, columns=cols).to_csv(OUT_DIR / f"migrations_{m}.csv")
        tables[m].to_csv(OUT_DIR / f"migration_counts_{m}.csv", index=False)
        tables[m][tables[m]["pre_event_moves"] >= MIN_PRE_MOVES].sort_values(
            "pre_event_moves", ascending=False).to_csv(OUT_DIR / f"pre_event_migrations_{m}.csv", index=False)

    # Save
    ks.to_csv(OUT_DIR / "k_selection.csv", index=False)
    full.to_csv(OUT_DIR / "labels_full.csv", index=False)
    pd.DataFrame({"window_end": win["end"], "ari_dtw_vs_euclidean": ari_win}).to_csv(
        OUT_DIR / "ari_windows.csv", index=False)
    vb.to_csv(OUT_DIR / "vol_blocks.csv", index=False)
    make_pdf(OUT_DIR / "cluster_drift_report.pdf", ks, best, vb, tables)

    write_summary(ks, best, ari_full, ari_med, win, blocks, vb, tables, full)


def write_summary(ks, best, ari_full, ari_med, win, blocks, vb, tables, full):
    n_events = int(vb["event"].sum())
    ev = vb.loc[vb["event"], "start"].dt.strftime("%Y-%m-%d").tolist()
    flagged = {
        m: t[t["pre_event_moves"] >= MIN_PRE_MOVES].sort_values(["pre_event_moves", "pre_over_total"], ascending=False)
        for m, t in tables.items()
    }
    both = sorted(set(flagged["dtw"]["symbol"]) & set(flagged["euclidean"]["symbol"]))
    labels = {"dtw": "DTW", "euclidean": "Euclidean"}
    sil = ks.pivot(index="k", columns="metric", values="silhouette")
    sizes = {m: full[m].value_counts().sort_index().tolist() for m in ("dtw", "euclidean")}

    lines = [
        "# Task 2 Summary",
        "",
        "## Settings",
        "",
        f"- Clustering: TimeSeriesKMeans, k in {K_RANGE.start}-{K_RANGE.stop - 1}, seed {SEED}, n_init {N_INIT}; "
        f"DTW with Sakoe-Chiba radius {BAND} bars",
        f"- Rolling: {WINDOW_DAYS}-trading-day window, {BLOCK_DAYS}-trading-day step; "
        f"{len(win)} windows ({win['end'].iloc[0].date()} ~ {win['end'].iloc[-1].date()}), {len(blocks)} blocks",
        f"- Volatility event: median across assets of each asset's RV percentile vs its previous "
        f"{LOOKBACK_BLOCKS} blocks >= {HOT_LEVEL:.0%}; first crossing only; first {LOOKBACK_BLOCKS} blocks not judged",
        "",
        "## Key results",
        "",
        f"1. **Optimal k.** DTW k = {best['dtw']}, Euclidean k = {best['euclidean']} (max Silhouette). "
        f"Cluster sizes at the optimum: DTW {sizes['dtw']}, Euclidean {sizes['euclidean']}.",
        "",
        "   | k | " + " | ".join(sil.columns) + " |",
        "   | --- | " + " | ".join("---" for _ in sil.columns) + " |",
        *[f"   | {k} | " + " | ".join(f"{v:.3f}" for v in r) + " |" for k, r in sil.iterrows()],
        "",
        f"2. **DTW vs Euclidean agreement.** ARI on the full-period fit = {ari_full:.3f}, median rolling-window "
        f"ARI = {ari_med:.3f}. Downstream analysis is run and reported for both metrics regardless of ARI.",
        f"3. **Volatility-expansion events.** {n_events} events: {', '.join(ev) if ev else 'none'}.",
        f"4. **Migration activity** across {len(win) - 1} window transitions:",
        *[f"   - {labels[m]}: {int(t['total_moves'].sum())} migrations. Most active: "
          + ", ".join(f"{r.symbol} ({r.total_moves})" for r in t.nlargest(5, "total_moves").itertuples())
          + ". Least active: "
          + ", ".join(f"{r.symbol} ({r.total_moves})" for r in t.nsmallest(5, "total_moves").itertuples()) + "."
          for m, t in tables.items()],
        f"5. **Pre-event migrations (>= {MIN_PRE_MOVES}).** DTW: {len(flagged['dtw'])} assets, Euclidean: "
        f"{len(flagged['euclidean'])} assets; in both: {', '.join(both) if both else 'none'}. "
        "No highlighting rule is applied yet (pending).",
    ]
    for m, f in flagged.items():
        lines += [
            "",
            f"   {labels[m]}",
            "",
            "   | Symbol | Pre-event moves | Pre / total |",
            "   | --- | --- | --- |",
            *[f"   | {r.symbol} | {r.pre_event_moves}/{n_events} | {r.pre_event_moves}/{r.total_moves} "
              f"({r.pre_over_total:.0%}) |" for r in f.itertuples()],
        ]
    if n_events < 10:
        lines.append(f"\n> **Caution:** only {n_events} events; one event shifts any ratio by "
                     f"{100 / max(n_events, 1):.0f}%p. Read all ratios together with the event count.")
    lines += [
        "",
        "## Output files",
        "",
        "| File | Content |",
        "| --- | --- |",
        "| `k_selection.csv` | Silhouette and inertia per metric and k (2.1) |",
        "| `labels_full.csv` | Full-period cluster label per asset at each metric's optimal k |",
        "| `membership_{metric}.csv` | Migration matrix: asset x window end date, Hungarian-aligned labels |",
        "| `migrations_{metric}.csv` | 1 if the asset changed cluster vs the previous window |",
        "| `migration_counts_{metric}.csv` | Per asset: total moves, pre-event moves, ratio, event count |",
        f"| `pre_event_migrations_{{metric}}.csv` | Subset with >= {MIN_PRE_MOVES} pre-event moves |",
        "| `ari_windows.csv` | DTW vs Euclidean ARI per rolling window |",
        "| `vol_blocks.csv` | Per 2-week block: dates, indicator, hot, event |",
        "| `cluster_drift_report.pdf` | Silhouette curves, volatility indicator, migrations per asset |",
        "",
        "Membership is recorded at each window's end date (no look-ahead). A pre-event migration is one "
        "recorded by the window ending at the event block's start; windows overlapping the event block are excluded.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
