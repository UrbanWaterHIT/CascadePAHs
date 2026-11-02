"""Dataset construction checks and descriptive statistics (Sections 2.2–2.5, Notes S1.6–S1.7).

    python scripts/01_data_qc_and_descriptive_stats.py --config configs/fast.yaml

Outputs (results/01_data/):
  hf_qaqc_summary.csv            QA/QC flags of the high-frequency records
  onstation_matching_check.csv   paired sensor vectors re-derived from on-station readings
  dataset_accounting.csv         records / paired samples per site and campaign (Tables S1, S12)
  summary_statistics.csv         variable summaries per site (Tables S3–S5)
  sumpahs_distribution.csv       ΣPAHs distribution per dataset (Table S23)
  composition_kruskal.csv        LMW/MMW/HMW shares and Kruskal–Wallis tests (Fig. S6)
  detection_frequency.csv        detection frequency of each PAH by dataset (Table S29)
  eqs_exceedance.csv             EU EQS comparison for fluoranthene and benzo[a]pyrene
  date_dependence.json           ANOVA by date, ICC, lag-1 correlation of daily means
  rainfall_lag_correlation.csv   daily rainfall vs daily mean ΣPAHs, lags 0–3 (Table S27)
  manova_site_heterogeneity.csv  multivariate tests vs the Dasha River (Table S9)
"""
from _common import Timer, get_config, save_json  # noqa: I001

import numpy as np
import pandas as pd

from cascadepahs import pahs, sonde, stats
from cascadepahs.config import resolve_path
from cascadepahs.features import OPTICAL_VARS, SENSOR_VARS, TARGET
from cascadepahs.io import load_hf, load_paired
from cascadepahs.pipeline import results_dir


def main():
    cfg = get_config(__doc__)
    out = results_dir(cfg, "01_data")
    paired = load_paired(resolve_path(cfg, "paired"))
    lod = pd.read_csv(resolve_path(cfg, "paired").parent / "pah_lod.csv").set_index("congener")["LOD_ng_L"].to_dict()

    with Timer("high-frequency QA/QC"):
        hf = load_hf(resolve_path(cfg, "hf"))
        q = sonde.qaqc(hf)
        summ = q.groupby("site")[["flag_stabilization", "flag_air", "flag_range", "flag_spike", "qc_pass"]].sum()
        summ["n_records"] = q.groupby("site").size()
        summ.to_csv(out / "hf_qaqc_summary.csv")
        q[q["qc_pass"]].drop(columns=[c for c in q if c.startswith("flag_")]).to_csv(out / "hf_records_qc_passed.csv",
                                                                                     index=False)
        print(summ)

    with Timer("sensor–sample matching (mean of on-station readings)"):
        rd = pd.read_csv(resolve_path(cfg, "onstation"))
        m = sonde.match_onstation(rd, SENSOR_VARS).set_index("sample_id")
        p = paired.set_index("sample_id")
        diff = (m[SENSOR_VARS] - p.loc[m.index, SENSOR_VARS]).abs().max()
        chk = pd.DataFrame({"max_abs_difference": diff})
        chk.to_csv(out / "onstation_matching_check.csv")
        m.describe()[["n_sonde_readings", "window_min"]].to_csv(out / "onstation_window_summary.csv")
        print(f"  max |re-derived − stored| sensor value: {diff.max():.2e}")

    with Timer("dataset accounting and summary statistics"):
        acc = paired.groupby(["site", "campaign"]).agg(paired_samples=(TARGET, "size"), survey_days=("date", "nunique"),
                                                       stations=("station", "nunique")).reset_index()
        hf_n = hf.groupby(["site", "campaign"]).size().rename("hf_records_in_demo_file").reset_index()
        acc.merge(hf_n, how="left").to_csv(out / "dataset_accounting.csv", index=False)
        rows = []
        for site, g in paired.groupby("site"):
            for v in SENSOR_VARS + ["Cond", "TDS"] + OPTICAL_VARS + [TARGET]:
                if v in g:
                    rows.append({"site": site, "variable": v, "N": len(g), "max": g[v].max(), "mean": g[v].mean(),
                                 "min": g[v].min(), "sd": g[v].std()})
        pd.DataFrame(rows).to_csv(out / "summary_statistics.csv", index=False)
        dist = paired.groupby(["site", "campaign"])[TARGET].describe(percentiles=[0.25, 0.5, 0.75])
        dist.to_csv(out / "sumpahs_distribution.csv")
        print(dist.round(2))

    with Timer("PAH composition, detection frequency, EQS"):
        grp = pahs.group_sums(paired, lod)
        comp = pd.concat([paired[["sample_id", "site", "campaign"]], grp], axis=1)
        comp["dataset"] = comp["site"] + "/" + comp["campaign"]
        comp.to_csv(out / "pah_group_shares.csv", index=False)
        kw = []
        for g in ["LMW", "MMW", "HMW"]:
            r = stats.kruskal_epsilon2(comp[f"{g}_share"], comp["site"])          # Dasha campaigns pooled
            kw.append({"group": g, **r, **{f"mean_share_{k}": v for k, v in
                                           comp.groupby("dataset")[f"{g}_share"].mean().items()}})
        pd.DataFrame(kw).to_csv(out / "composition_kruskal.csv", index=False)
        det = pahs.detection_frequency(paired.assign(dataset=paired["site"] + "/" + paired["campaign"]), lod, by="dataset")
        det.to_csv(out / "detection_frequency.csv")
        pahs.eqs_exceedance(paired, lod).to_csv(out / "eqs_exceedance.csv", index=False)

    with Timer("dependence among Dasha River samples and rainfall associations"):
        d = paired[paired["site"] == "DashaRiver"]
        res = {}
        for camp, g in d.groupby("campaign"):
            res[camp] = stats.anova_icc(g[TARGET], g["date"].astype(str))
        s = d[d["campaign"].str.startswith("summer")]
        daily = s.groupby("date")[TARGET].mean()
        res["summer_daily_mean_lag1_r"] = stats.lag_correlation(daily, 1)
        save_json(res, out / "date_dependence.json")
        print(res)
        rain = pd.read_csv(resolve_path(cfg, "rainfall"), parse_dates=["date"]).set_index("date")["rain_dasha_mm"]
        stats.rainfall_lag_correlations(rain, daily).to_csv(out / "rainfall_lag_correlation.csv", index=False)

    with Timer("lost summer samples: rain days vs other days (Fisher) and by station (χ²)"):
        days = pd.date_range(s["date"].min(), s["date"].max(), freq="D")
        stations = sorted(s["station"].unique())
        rain_days = pd.to_datetime(cfg.get("loss_analysis", {}).get(
            "rain_days", ["2024-07-27", "2024-07-28", "2024-07-31", "2024-08-01"]))
        analysed_rain = int(s["date"].isin(rain_days).sum())
        sched_rain = len(rain_days) * len(stations)
        fisher = stats.fisher_loss_test(analysed_rain, sched_rain, len(s) - analysed_rain,
                                        len(days) * len(stations) - sched_rain)
        per_station = s.groupby("station").size().reindex(stations, fill_value=0).to_dict()
        chi2 = stats.chi2_loss_by_station(per_station, len(days))
        loss = {"scheduled": len(days) * len(stations), "analysed": len(s),
                "rain_days_analysed": f"{analysed_rain}/{sched_rain}", "fisher": fisher,
                "analysed_per_station": per_station, "chi2_by_station": chi2,
                "min_samples_per_day": int(s.groupby("date").size().min())}
        save_json(loss, out / "sample_loss_tests.json")
        print(loss)

    with Timer("multivariate heterogeneity between the source and target domains"):
        rows = []
        cols = SENSOR_VARS + [v for v in OPTICAL_VARS if v not in ("C1", "C2", "C3")] + [TARGET]
        for tgt in ["YangyongRiver", "ShenzhenBay"]:
            sub = paired[paired["site"].isin(["DashaRiver", tgt])]
            Xs = sub[cols].to_numpy(float)
            sd = Xs.std(0)
            Xs = (Xs - Xs.mean(0)) / np.where(sd > 0, sd, 1)
            keep = sub[cols].std().to_numpy() > 0
            t = stats.manova(Xs[:, keep], sub["site"].to_numpy())
            t.insert(0, "comparison", f"DashaRiver vs {tgt}")
            rows.append(t)
        pd.concat(rows).to_csv(out / "manova_site_heterogeneity.csv", index=False)


if __name__ == "__main__":
    main()
