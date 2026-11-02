"""Figures of the paper (Figs. 3–6, S14–S15)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import mae, r2  # noqa: E402

BLUE = "#6F8FB5"
ROSE = "#A8687A"
GREY = "#9AA0A6"
PALE_BLUE = "#CED8E6"
DARK = "#2B2B2B"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "figure.dpi": 110, "font.family": "DejaVu Sans"})


def savefig(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _fit_line(ax, x, y, color):
    if len(x) < 3:
        return
    b, a = np.polyfit(x, y, 1)
    xx = np.linspace(0, max(np.max(x), np.max(y)) * 1.05, 50)
    ax.plot(xx, a + b * xx, color=color, lw=1.2)


def measured_vs_predicted(ax, y, p, *, label: str = "", y_train=None, p_train=None, prefix: str = "Summer ",
                          unit: str = "ng L$^{-1}$", lim: float | None = None):
    lim = lim or float(np.nanmax(np.r_[y, p] if y_train is None else np.r_[y, p, y_train, p_train])) * 1.08
    if y_train is not None:
        ax.scatter(y_train, p_train, s=10, color=ROSE, alpha=0.55, lw=0, label="Train")
        _fit_line(ax, np.asarray(y_train), np.asarray(p_train), ROSE)
    ax.scatter(y, p, s=12, color=BLUE, alpha=0.8, lw=0, label="Held-out predictions")
    _fit_line(ax, np.asarray(y), np.asarray(p), BLUE)
    ax.plot([0, lim], [0, lim], ls="--", color=GREY, lw=0.9)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"Measured ΣPAHs ({unit})")
    ax.set_ylabel(f"Predicted ΣPAHs ({unit})")
    ax.text(0.97, 0.05, f"{prefix}$R^2$ = {r2(y, p):.2f}\n$MAE$ = {mae(y, p):.1f} {unit}", transform=ax.transAxes,
            ha="right", va="bottom")
    if label:
        ax.text(0.03, 0.95, label, transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="square,pad=0.3", fc="#EEEEEE", ec="none"))


def subset_benchmark(ax, res: pd.DataFrame):
    """Fig. 3a: mean held-out R² of every sensor subset × algorithm against subset size."""
    rng = np.random.default_rng(0)
    sizes = sorted(res["subset_size"].unique())
    med, p90 = [], []
    for k in sizes:
        v = res.loc[res["subset_size"] == k, "mean_R2"].to_numpy()
        ax.scatter(k - 0.18 + 0.12 * rng.standard_normal(len(v)), v, s=8, color=ROSE, alpha=0.25, lw=0)
        parts = ax.violinplot([v], positions=[k + 0.22], widths=0.4, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(PALE_BLUE)
            pc.set_alpha(0.9)
        med.append(np.median(v))
        p90.append(np.percentile(v, 90))
    ax.plot(sizes, med, "-o", color=BLUE, ms=4, label="Median of subset × model mean across splits")
    ax.plot(sizes, p90, "-^", color=ROSE, ms=4, label="90th percentile of subset × model mean across splits")
    ax.set_xlabel("Number of sensors (subset size)")
    ax.set_ylabel("Mean test $R^2$ across splits")
    ax.legend(loc="lower right", frameon=False)


def r2_mae_by_algorithm(ax, table: pd.DataFrame):
    """Fig. 4a: summer R² (points) and MAE (bars) of sensor-only vs -P models."""
    algs = table["algorithm"].tolist()
    x = np.arange(len(algs))
    ax2 = ax.twinx()
    ax2.bar(x - 0.18, table["MAE_sensor"], 0.36, color=PALE_BLUE, label="MAE, sensor only")
    ax2.bar(x + 0.18, table["MAE_P"], 0.36, color=ROSE, label="MAE, privileged information (-P)")
    ax2.set_ylabel("MAE (ng L$^{-1}$)")
    for i in range(len(algs)):
        ax.plot([x[i] - 0.18, x[i] + 0.18], [table["R2_sensor"].iloc[i], table["R2_P"].iloc[i]], color=DARK, lw=1)
    ax.scatter(x - 0.18, table["R2_sensor"], s=40, facecolor="white", edgecolor=DARK, zorder=3, label="$R^2$, sensor only")
    ax.scatter(x + 0.18, table["R2_P"], s=40, color=DARK, zorder=3, label="$R^2$, privileged information (-P)")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xticks(x, algs)
    ax.set_ylabel("Summer $R^2$")
    ax.set_xlabel("Algorithm")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)


def shap_beeswarm(ax, shap_table: pd.DataFrame, features: list[str], labels: dict | None = None,
                  max_features: int | None = None):
    """SHAP strip plot coloured by the (within-variable scaled) feature value (Fig. 5a–c)."""
    labels = labels or {}
    imp = {f: np.mean(np.abs(shap_table[f"shap_{f}"])) for f in features}
    order = sorted(features, key=lambda f: -imp[f])[: max_features or len(features)]
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("coolwarm")
    for i, f in enumerate(order):
        s = shap_table[f"shap_{f}"].to_numpy()
        v = shap_table[f].to_numpy(float)
        vs = (v - np.nanpercentile(v, 5)) / (np.nanpercentile(v, 95) - np.nanpercentile(v, 5) + 1e-12)
        ax.scatter(s, np.full(len(s), -i) + 0.15 * rng.standard_normal(len(s)), c=cmap(np.clip(vs, 0, 1)), s=9, lw=0)
    ax.axvline(0, color=DARK, lw=0.8)
    ax.set_yticks(-np.arange(len(order)), [labels.get(f, f) for f in order])
    ax.set_xlabel("SHAP value (ng L$^{-1}$)")
    ax.grid(axis="x", ls="--", alpha=0.3)


def descriptor_r2_bars(ax, table: pd.DataFrame, labels: dict | None = None):
    labels = labels or {}
    t = table.sort_values("R2")
    colors = np.where(t["R2"] >= 0, "#337A83", "#AD6363")
    ax.barh([labels.get(d, d) for d in t["descriptor"]], t["R2"], color=colors)
    ax.axvline(0, color=DARK, lw=0.8)
    ax.set_xlabel("Date-block held-out $R^2$")


def importance_bars(ax, imp: pd.DataFrame, color=ROSE):
    t = imp.sort_values("median")
    ax.barh(t["feature"], t["median"], xerr=[t["median"] - t["q1"], t["q3"] - t["median"]], color=color,
            error_kw=dict(lw=0.8))
    ax.set_xlim(0, 100)
    ax.set_xlabel("Relative feature importance (%)")


def application_timeseries(ax, app: pd.DataFrame, rain: pd.DataFrame, colors=None):
    """Fig. 6g: rainfall bars + measured and estimated ΣPAHs with 90 % prediction intervals."""
    colors = colors or {"ShenzhenBay": "#2F3F9E", "YangyongRiver": "#F0A050"}
    axr = ax
    axr.bar(rain["date"], rain["rain_target_mm"], color="#A9D3E0", width=0.8, label="Rainfall")
    axr.set_ylabel("Daily rainfall (mm)", color="#3F8FA8")
    axp = ax.twinx()
    for site, g in app.groupby("site"):
        c = colors.get(site, BLUE)
        axp.fill_between(g["date"], g["PI90_low"], g["PI90_high"], color=c, alpha=0.18, lw=0)
        axp.plot(g["date"], g["SumPAHs"], "-o", color=c, ms=3, lw=1.2, label=f"{site}: measured")
        axp.plot(g["date"], g["SumPAHs_hat"], "--s", color=c, ms=3, lw=1, alpha=0.8, label=f"{site}: SOLACE")
    axp.set_ylabel("ΣPAHs (ng L$^{-1}$)")
    axp.spines["right"].set_visible(True)
    axp.legend(loc="upper right", frameon=False, fontsize=7)
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%d"))


def calibration_intervals(ax, summ: pd.DataFrame, model: str = "SOLACE"):
    """Fig. S14: median, IQR and 2.5–97.5 % range of validation R² per calibration setting."""
    t = summ[summ["model"] == model].reset_index(drop=True)
    y = np.arange(len(t))[::-1]
    ax.hlines(y, t["R2_p2.5"], t["R2_p97.5"], color=BLUE, lw=1)
    ax.hlines(y, t["R2_q1"], t["R2_q3"], color=BLUE, lw=4)
    ax.plot(t["R2_median"], y, "o", color=BLUE)
    ax.set_yticks(y, [f"{s} ({n})" for s, n in zip(t["site_setting"], t["n_val"])])
    ax.set_xlabel("Validation $R^2$")


def hf_map(ax, rec: pd.DataFrame, vmin=None, vmax=None):
    ok = ~rec["extrapolation"]
    sc = ax.scatter(rec.loc[ok, "lon"], rec.loc[ok, "lat"], c=rec.loc[ok, "SumPAHs_hat"], s=4, cmap="viridis",
                    vmin=vmin, vmax=vmax)
    ax.scatter(rec.loc[~ok, "lon"], rec.loc[~ok, "lat"], marker="x", s=8, color=GREY, lw=0.6, label="extrapolation")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    return sc
