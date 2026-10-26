"""Task 3. Rolling Granger causality over all asset pairs.

Run:    python src/task3_granger.py
Input:  outputs/task1/ (log returns); windows from Task 2's calendar
Output: outputs/task3/

Bivariate VAR per pair and window, constant only, lag p* = argmin BIC over p = 4..8 on a common sample.
Equation-by-equation OLS is used (identical to VAR OLS); results are cross-checked against statsmodels.
Direction convention: row i -> column j means i Granger-causes (leads) j.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

from task1_prep import load_task1_outputs
from task2_cluster import WINDOW_DAYS, build_calendar

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "task3"

P_RANGE = range(2, 9)
P_MAX = max(P_RANGE)
ALPHA_RAW = 0.01
FDR_Q = 0.05
TOP_SHARE = 0.05       # A_ij = 1 for directions whose significance frequency is in the top 5%
N_HUBS = 5
SEED = 42


# ---------- per-pair VAR ----------

def _lagmat(Y, p, start):
    """Regressors [1, y1_{t-1}, y2_{t-1}, ..., y1_{t-p}, y2_{t-p}] for rows t = start..T-1."""
    T = Y.shape[0]
    cols = [np.ones(T - start)]
    for k in range(1, p + 1):
        cols += [Y[start - k:T - k, 0], Y[start - k:T - k, 1]]
    return np.column_stack(cols)


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def bic_values(Y):
    """VAR BIC per p on the common sample t = P_MAX..T-1 (same sample as statsmodels select_order)."""
    out = {}
    for p in P_RANGE:
        _, E = _ols(_lagmat(Y, p, P_MAX), Y[P_MAX:])
        n = E.shape[0]
        out[p] = np.linalg.slogdet(E.T @ E / n)[1] + np.log(n) / n * p * 4
    return out


def select_p(Y):
    bic = bic_values(Y)
    return min(bic, key=bic.get)


def fit_pair(Y):
    """Returns p*, and for directions 0->1 and 1->0: F, p-value, cross-lag coefs, std errors."""
    p = select_p(Y)
    X = _lagmat(Y, p, p)
    Yt = Y[p:]
    n, m = X.shape
    beta, E = _ols(X, Yt)
    XtX_inv = np.linalg.inv(X.T @ X)
    out = {}
    for src, dst in ((0, 1), (1, 0)):
        cross = 1 + src + 2 * np.arange(p)              # columns of src lags
        own = np.setdiff1d(np.arange(m), cross)
        rss_u = E[:, dst] @ E[:, dst]
        _, e_r = _ols(X[:, own], Yt[:, dst])
        rss_r = e_r @ e_r
        df = n - m
        F = (rss_r - rss_u) / p / (rss_u / df)
        se = np.sqrt(np.diag(XtX_inv)[cross] * rss_u / df)
        out[(src, dst)] = (F, stats.f.sf(F, p, df), beta[cross, dst], se)
    return p, out


def check_against_statsmodels(R):
    Y = R[:, :2]
    p = select_p(Y)
    ours = bic_values(Y)
    sm = VAR(Y).select_order(P_MAX).ics["bic"]           # index = lag 0..P_MAX; same additive constant
    assert np.allclose([ours[q] - ours[P_RANGE.start] for q in P_RANGE],
                       [sm[q] - sm[P_RANGE.start] for q in P_RANGE]), "BIC differs from statsmodels"
    _, out = fit_pair(Y)
    res = VAR(Y).fit(p, trend="c")
    F, pv, coef, se = out[(0, 1)]                       # column 0 -> column 1
    assert np.allclose(coef, res.params[1 + 2 * np.arange(p), 1])
    assert np.allclose(se, res.stderr[1 + 2 * np.arange(p), 1])
    g = grangercausalitytests(Y[:, [1, 0]], maxlag=[p], verbose=False)[p][0]["ssr_ftest"]
    assert np.isclose(F, g[0]) and np.isclose(pv, g[1])


# ---------- per-window ----------

def fit_window(R):
    """R: (T, N) returns. Arrays indexed [source, target]."""
    N = R.shape[1]
    p_star = np.zeros((N, N), int)
    F = np.full((N, N), np.nan)
    pv = np.full((N, N), np.nan)
    coef = np.full((N, N, P_MAX), np.nan)
    se = np.full((N, N, P_MAX), np.nan)
    for a in range(N):
        for b in range(a + 1, N):
            p, out = fit_pair(R[:, [a, b]])
            for (s, d), (f, q, c, e) in out.items():
                i, j = (a, b)[s], (a, b)[d]
                p_star[i, j], F[i, j], pv[i, j] = p, f, q
                coef[i, j, :p], se[i, j, :p] = c, e
    return p_star, F, pv, coef, se


def placebo_rate(R):
    """Share of raw rejections after independently shuffling every series (should be ~ALPHA_RAW)."""
    rng = np.random.default_rng(SEED)
    Rs = np.column_stack([rng.permutation(R[:, c]) for c in range(R.shape[1])])
    pv = fit_window(Rs)[2]
    off = ~np.eye(R.shape[1], dtype=bool)
    return float((pv[off] < ALPHA_RAW).mean())


# ---------- aggregation ----------

def bh_mask(pv):
    off = ~np.eye(pv.shape[0], dtype=bool)
    sig = np.zeros_like(pv, dtype=bool)
    sig[off] = multipletests(pv[off], alpha=FDR_Q, method="fdr_bh")[0]
    return sig


def hubs_table(A, Fq, symbols, kind):
    t = pd.DataFrame({"symbol": symbols, "out_degree": A.sum(axis=1), "in_degree": A.sum(axis=0),
                      "out_freq_sum": Fq.sum(axis=1)})
    t = t.sort_values(["out_degree", "out_freq_sum"], ascending=False).head(N_HUBS)
    t.insert(0, "test", kind)
    return t


def top_directions(Fq, symbols, n=10):
    N = len(symbols)
    idx = [(i, j) for i in range(N) for j in range(N) if i != j]
    idx.sort(key=lambda ij: Fq[ij], reverse=True)
    return [(symbols[i], symbols[j], Fq[i, j]) for i, j in idx[:n]]


# ---------- report ----------

def _hub_page(pdf, kind, adj, freq, hubs, cut, symbols):
    """Hubs on the left, the assets they lead on the right; line width = significance frequency."""
    hub_syms = hubs.loc[hubs["test"] == kind, "symbol"].tolist()
    idx = {s: i for i, s in enumerate(symbols)}
    edges = [(h, symbols[j], freq[idx[h], j]) for h in hub_syms for j in np.nonzero(adj[idx[h]])[0]]
    led_by = pd.Series([t for _, t, _ in edges]).value_counts()
    targets = sorted(led_by.index, key=lambda t: (-led_by[t], t))
    fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(targets) + 2)))
    y_hub = {h: y for h, y in zip(hub_syms, np.linspace(len(targets) - 1, 0, len(hub_syms)))}
    y_tgt = {t: len(targets) - 1 - i for i, t in enumerate(targets)}
    colors = dict(zip(hub_syms, plt.cm.tab10.colors))
    fmin, fmax = min(f for *_, f in edges), max(f for *_, f in edges)
    for h, t, f in edges:
        w = 0.8 + 3.5 * (f - fmin) / max(fmax - fmin, 1e-9)
        ax.annotate("", xy=(1, y_tgt[t]), xytext=(0, y_hub[h]),
                    arrowprops=dict(arrowstyle="-|>", color=colors[h], lw=w, alpha=0.65, shrinkA=6, shrinkB=4))
    out_deg = pd.Series([h for h, *_ in edges]).value_counts()
    for h, y in y_hub.items():
        ax.text(-0.03, y, f"{h} ({out_deg.get(h, 0)})", ha="right", va="center", fontsize=11,
                fontweight="bold", color=colors[h])
    for t, y in y_tgt.items():
        ax.text(1.03, y, f"{t}" + (f"  x{led_by[t]}" if led_by[t] > 1 else ""), ha="left", va="center", fontsize=8)
    ax.set_xlim(-0.35, 1.3)
    ax.set_ylim(-1, len(targets))
    ax.axis("off")
    ax.set_title(f"Hubs and the assets they lead ({kind}; edges = top {TOP_SHARE:.0%}, frequency >= {cut:.0%})\n"
                 f"line width = share of windows significant ({fmin:.0%}-{fmax:.0%}); "
                 "(n) = out-degree, xN = led by N hubs", fontsize=10)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def make_pdf(path, freq, adj, cut, hubs, n_sig, win, symbols):
    with PdfPages(path) as pdf:
        for kind in ("raw", "fdr"):
            _hub_page(pdf, kind, adj[kind], freq[kind], hubs, cut[kind], symbols)

        fig, ax = plt.subplots(figsize=(8, 4))
        off = ~np.eye(len(symbols), dtype=bool)
        for kind in ("raw", "fdr"):
            ax.hist(freq[kind][off], bins=40, range=(0, 1), alpha=0.6, label=kind)
        for kind, color in (("raw", "red"), ("fdr", "purple")):
            ax.axvline(cut[kind], color=color, ls="--", label=f"top {TOP_SHARE:.0%} cut ({kind}) {cut[kind]:.0%}")
        ax.set(xlabel="share of windows significant", ylabel="directions", yscale="log",
               title="Distribution of per-direction significance frequency")
        ax.legend()
        pdf.savefig(fig); plt.close(fig)

        for kind in ("raw", "fdr"):
            G = nx.DiGraph()
            G.add_nodes_from(symbols)
            G.add_edges_from((symbols[i], symbols[j]) for i, j in zip(*np.nonzero(adj[kind])))
            fig, ax = plt.subplots(figsize=(10, 10))
            pos = nx.circular_layout(G)
            out_deg = dict(G.out_degree())
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[100 + 120 * out_deg[s] for s in symbols],
                                   node_color="tab:blue", alpha=0.7)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
            nx.draw_networkx_edges(G, pos, ax=ax, arrowsize=8, alpha=0.5, connectionstyle="arc3,rad=0.1")
            ax.set_title(f"Causality network ({kind}, top {TOP_SHARE:.0%}, frequency >= {cut[kind]:.0%}): "
                         f"{G.number_of_edges()} edges; node size = out-degree")
            ax.axis("off")
            pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 4))
        for kind in ("raw", "fdr"):
            ax.plot(win["end"], n_sig[kind], label=kind)
        ax.set(ylabel="significant directions", title=f"Significant directions per window (of {off.sum()})")
        ax.legend()
        pdf.savefig(fig); plt.close(fig)


# ---------- main ----------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t1 = load_task1_outputs()
    symbols = t1["universe"]
    rets = t1["log_returns"][symbols]
    N = len(symbols)

    _, _, win = build_calendar(t1["close"].index)
    day = rets.index.normalize()
    windows = [rets[(day >= w.win_start) & (day <= w.end)].to_numpy() for w in win.itertuples()]

    check_against_statsmodels(windows[0])

    fits = Parallel(n_jobs=-1)(delayed(fit_window)(R) for R in windows)
    p_star, F, pv, coef, se = (np.stack(x) for x in zip(*fits))     # (W, N, N[, P_MAX])
    W = len(windows)

    sig = {"raw": pv < ALPHA_RAW, "fdr": np.stack([bh_mask(pv[w]) for w in range(W)])}
    off = ~np.eye(N, dtype=bool)
    freq = {k: s.mean(axis=0) * off for k, s in sig.items()}
    # Ties at the cut are kept, so the edge count can exceed TOP_SHARE x directions
    cut = {k: max(np.quantile(f[off], 1 - TOP_SHARE), 1 / W) for k, f in freq.items()}
    adj = {k: ((f >= cut[k]) & off).astype(int) for k, f in freq.items()}
    n_sig = {k: s[:, off].sum(axis=1) for k, s in sig.items()}
    placebo = placebo_rate(windows[0])

    assert p_star[:, off].min() >= P_RANGE.start and p_star[:, off].max() <= P_MAX
    assert all((np.diag(a) == 0).all() for a in adj.values())

    # Long table: window x direction
    w_idx, i_idx, j_idx = np.nonzero(np.broadcast_to(off, (W, N, N)))
    long = pd.DataFrame({
        "window_end": win["end"].to_numpy()[w_idx],
        "source": np.array(symbols)[i_idx],
        "target": np.array(symbols)[j_idx],
        "p_star": p_star[w_idx, i_idx, j_idx],
        "F": F[w_idx, i_idx, j_idx],
        "pvalue": pv[w_idx, i_idx, j_idx],
        "sig_raw": sig["raw"][w_idx, i_idx, j_idx],
        "sig_fdr": sig["fdr"][w_idx, i_idx, j_idx],
    })
    for k in range(P_MAX):
        long[f"coef_{k + 1}"] = coef[w_idx, i_idx, j_idx, k]
    for k in range(P_MAX):
        long[f"se_{k + 1}"] = se[w_idx, i_idx, j_idx, k]
    assert len(long) == W * N * (N - 1)

    long.to_parquet(OUT_DIR / "var_results.parquet", index=False)
    for k in ("raw", "fdr"):
        pd.DataFrame(adj[k], index=symbols, columns=symbols).to_csv(OUT_DIR / f"adjacency_{k}.csv")
        pd.DataFrame(freq[k], index=symbols, columns=symbols).to_csv(OUT_DIR / f"frequency_{k}.csv")
    hubs = pd.concat([hubs_table(adj[k], freq[k], symbols, k) for k in ("raw", "fdr")])
    hubs.to_csv(OUT_DIR / "hubs.csv", index=False)
    make_pdf(OUT_DIR / "causality_topology.pdf", freq, adj, cut, hubs, n_sig, win, symbols)

    write_summary(p_star[:, off], n_sig, adj, cut, freq, hubs, placebo, win, symbols)


def current_findings(n_sig, freq, win, N):
    """Plain statement of the two main caveats: low per-direction frequency and time concentration."""
    off = ~np.eye(N, dtype=bool)
    W = len(win)
    f_raw, f_fdr = freq["raw"][off], freq["fdr"][off]
    raw = pd.Series(n_sig["raw"], index=win["end"].dt.date)
    busy = raw[raw > 2 * raw.median()]                     # windows with more than twice the typical count
    busy_share = busy.sum() / raw.sum()
    # Group busy windows into periods: windows at most 3 steps apart belong to the same period
    pos = np.flatnonzero(raw > 2 * raw.median())
    groups = np.split(pos, np.flatnonzero(np.diff(pos) > 3) + 1) if len(pos) else []
    periods = [(raw.index[g[0]], raw.index[g[-1]], len(g), raw.iloc[g].sum() / raw.sum(),
                raw.index[g[np.argmax(raw.iloc[g])]], raw.iloc[g].max()) for g in groups]
    periods.sort(key=lambda x: -x[3])
    return [
        "## Current findings",
        "",
        "1. **Lead-lag relations are significant in too few windows.**",
        f"   - A direction is significant (raw) in a median {np.median(f_raw):.1%} of the {W} windows "
        f"(FDR: {np.median(f_fdr):.1%}); the most frequent direction reaches only {f_raw.max():.0%} "
        f"(FDR: {f_fdr.max():.0%}).",
        f"   - Directions significant in >= 50% of windows: {int((f_raw >= 0.5).sum())} (raw), "
        f"{int((f_fdr >= 0.5).sum())} (FDR). No direction is persistently significant.",
        f"   - With no lead-lag at all, raw frequency would center on {ALPHA_RAW:.0%}. Observed frequencies are "
        "above that, so relations exist more often than chance, but they appear and disappear across windows "
        "rather than holding steadily.",
        "   - The top 5% consensus therefore selects relatively frequent directions, not stable ones.",
        "2. **Significant results cluster in specific periods.**",
        f"   - {len(busy)} of {W} windows ({len(busy) / W:.0%}) have more than twice the median number of "
        f"significant directions (median {raw.median():.0f}); together they hold {busy_share:.0%} of all raw "
        "significant results.",
        "",
        "   | Period (window end dates) | Windows | Share of all raw significant results | Peak window (count) |",
        "   | --- | --- | --- | --- |",
        *[f"   | {a} ~ {b} | {n} | {sh:.0%} | {pk} ({pn}) |" for a, b, n, sh, pk, pn in periods],
        "",
        f"   - Each window spans the {WINDOW_DAYS} trading days before its end date, so a period's results reflect market "
        "conditions up to about 3 months earlier.",
        "   - The most extreme window contains the April 2025 sell-off. In high-volatility periods the "
        "homoskedastic F test can over-reject, so part of this clustering may reflect volatility rather than "
        "lead-lag.",
    ]


def write_summary(p_off, n_sig, adj, cut, freq, hubs, placebo, win, symbols):
    N = len(symbols)
    n_dir = N * (N - 1)
    p_dist = pd.Series(p_off.ravel()).value_counts().sort_index()
    off = ~np.eye(N, dtype=bool)
    spikes = pd.Series(n_sig["raw"], index=win["end"].dt.date).nlargest(3)
    lines = [
        "# Task 3 Summary",
        "",
        "## Settings",
        "",
        f"- {N} assets: {N * (N - 1) // 2:,} pairs, {n_dir:,} directions; {len(win)} rolling windows "
        f"(same as Task 2, {win['end'].iloc[0].date()} ~ {win['end'].iloc[-1].date()})",
        f"- Bivariate VAR, constant only, all lags t-1..t-p*; p* = argmin BIC over p = {P_RANGE.start}-{P_MAX} "
        "per pair and window",
        f"- Per-window significance: raw p < {ALPHA_RAW}; BH-FDR q = {FDR_Q} across all directions in the window",
        f"- Consensus: A_ij = 1 if the direction's share of significant windows is in the top {TOP_SHARE:.0%} "
        "of all directions (computed separately for raw and FDR; ties at the cut kept)",
        "",
        "## Key results",
        "",
        "1. **Lag order.** p* counts over all pair-windows: " + ", ".join(f"p={p}: {c:,}" for p, c in p_dist.items())
        + (f". BIC picks the lower bound {P_RANGE.start} in {p_dist.iloc[0] / p_dist.sum():.0%} of cases, "
           "i.e. it would prefer fewer lags if allowed." if p_dist.index[0] == P_RANGE.start
           and p_dist.iloc[0] / p_dist.sum() > 0.9 else "."),
        f"2. **Significant directions per window** (of {n_dir:,}): raw median {np.median(n_sig['raw']):.0f} "
        f"(range {n_sig['raw'].min()}-{n_sig['raw'].max()}), FDR median {np.median(n_sig['fdr']):.0f} "
        f"(range {n_sig['fdr'].min()}-{n_sig['fdr'].max()}). Chance level for raw is about "
        f"{ALPHA_RAW * n_dir:.0f}; placebo check (all series shuffled, one window) gives a raw rejection rate of "
        f"{placebo:.2%}. Spikes: "
        + ", ".join(f"{d} ({n})" for d, n in spikes.items())
        + " (raw count, window end date). The largest windows contain the April 2025 sell-off; "
        "with volatility clustering the homoskedastic F test can over-reject, so spikes need not mean "
        "stronger lead-lag.",
    ]
    for k in ("raw", "fdr"):
        A = adj[k]
        bidir = int(((A == 1) & (A.T == 1)).sum() // 2)
        h = hubs[hubs["test"] == k]
        lines.append(
            f"{3 if k == 'raw' else 4}. **Consensus network ({k}).** Cut = frequency >= {cut[k]:.1%} "
            f"(top {TOP_SHARE:.0%}); {int(A.sum())} edges, {bidir} bidirectional "
            "pairs. Hubs (out-degree): "
            + (", ".join(f"{r.symbol} ({r.out_degree})" for r in h.itertuples()) if A.sum() else "none")
            + ". Top directions by frequency: "
            + ", ".join(f"{s}->{t} ({f:.0%})" for s, t, f in top_directions(freq[k], symbols)) + "."
        )
    lines += [
        "5. **Decay check.** Criterion pending; per-window p*, cross-lag coefficients, standard errors, F and "
        "p-values are stored in `var_results.parquet` for the later decision.",
        "",
        *current_findings(n_sig, freq, win, N),
        "",
        "## Output files",
        "",
        "| File | Content |",
        "| --- | --- |",
        "| `adjacency_raw.csv`, `adjacency_fdr.csv` | 50 x 50 consensus adjacency, row -> column |",
        "| `frequency_raw.csv`, `frequency_fdr.csv` | 50 x 50 share of windows significant |",
        f"| `hubs.csv` | Top {N_HUBS} by out-degree per test, with in-degree |",
        f"| `var_results.parquet` | Window x direction ({len(win)} x {n_dir:,} rows): window_end, source, target, "
        "p_star, F, pvalue, sig_raw, sig_fdr, coef_1..8, se_1..8 (cross-lag terms of source in the target "
        "equation; NaN beyond p*). Used for the decay check and Task 4.2 |",
        "| `causality_topology.pdf` | Hubs and the assets they lead (raw, FDR), frequency histogram, network "
        "graphs, significant directions per window |",
        "",
        "Coefficients for a window are known only at its end date; Task 4.2 must use the window that ended "
        "before time t.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
