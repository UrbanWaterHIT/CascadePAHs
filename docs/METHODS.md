# Methods implemented in CascadePAHs

This document describes how each method of the paper is implemented and where it lives in the code.
Section numbers refer to the main text (M) and the Supplementary Information (S).

---

## 1. Data structure (M 2.2; S1.6; Tables S1, S2, S12, S13)

| symbol | content | columns |
|---|---|---|
| **X** | seven deployment-available sonde variables | `T, pH, DO, Sal, Turb, Chla, FDOM` |
| **O** | 17 laboratory optical descriptors (privileged, training only) | EEM: `C1–C3, TotalF, ProteinF, HumicF, FI, BIX, HIX, BetaAlpha`; UV–vis: `a254, a350, S275_295, S350_400, SR`; DOC: `DOC, SUVA254` |
| **y** | dissolved ΣPAHs (ng L⁻¹) | `SumPAHs` |
| **g** | survey date (validation groups) | `date` |

Conductivity and TDS are excluded before modelling (collinear with salinity). In the cross-site refit the PARAFAC components (C1–C3) are excluded, leaving 14 descriptors (`CROSS_SITE_OPTICAL_VARS`).

**Sensor–sample matching** (`sonde.match_onstation`): the paired sensor vector is the mean of the on-station readings of each sampling event. For riverine samples these are the two deployments, within about 10–30 min.
**QA/QC** (`sonde.qaqc`) flags sensor stabilisation, exposure to air, implausible values and spikes (rolling median ± k·MAD).

---

## 2. SORFE (M 2.4) — `models/sorfe.py`

A random forest maps **X → y**. Hyper-parameters are tuned only inside the training data by grid search (Table S28) with folds grouped by survey date (`models/base.tune`). `SensorOnlyModel(algorithm=...)` gives the other five sensor-only algorithms:

* **SVR, MLP, DNN:** X standardised; target standardised during fitting (TransformedTargetRegressor).
* **LR:** Ridge with α ∈ logspace(−4, 3, 20).
* **XGBoost:** `xgboost.XGBRegressor`, or sklearn `GradientBoostingRegressor` with the same four hyper-parameters if xgboost is not installed.

---

## 3. SOLACE (M 2.4; S3.1, S4.2) — `models/solace.py`

### Training — `SOLACE.fit(X, y, groups, O)`

```
inner folds  = GroupKFold(n_inner) over survey dates           (shared by all stages)
scaler       = StandardScaler fitted on X

# stage 1 – optical forest (leaf vocabulary)
F_ref        = RandomForestRegressor(O → y)                     (tuned over the RF grid if tune=True)
L_ref        = F_ref.apply(O)                                   (n × T terminal-node ids)

# cross-fitted leaf targets
for each inner fold k:
    F_k      = RandomForestRegressor(same params).fit(O[-k], y[-k])
    for each tree t:
        for every sample i in fold k:
            co-members  = training samples of F_k that share i's leaf in tree t of F_k
            L[i, t]     = most frequent L_ref[·, t] among the co-members
(optional) shuffled control: permute the rows of L

# stage 2 – multi-output sensor-to-leaf classifier
C            = RandomForestClassifier(X → L)                    (multi-output; tuned with mean per-tree accuracy)
for each inner fold k:
    L̂_oof[k] = clone(C).fit(X[-k], L[-k]).predict(X[k])

# stage 3 – final regressor
Z            = [scaler(X), one-hot(L̂_oof)]                      (sparse; categories = leaves present in L)
R            = final_algorithm(Z → y)                           (RF for SOLACE; RF quantile forest if quantile=True)
```

### Deployment — `SOLACE.predict(X)`

```
L̂ = C.predict(X);   ŷ = R.predict([scaler(X), one-hot(L̂)])
```

`O` is never used by `predict`. `leaf_source="sensor"` builds the leaves from **X** instead of **O**; this is the capacity control.

### Cross-fitted leaf targets

The optical forest is cross-fitted with inner date-grouped folds. Node ids of forests fitted on different folds are not comparable, whereas the multi-output classifier needs one set of classes per output and the deployed model a fixed one-hot design. The reference forest `F_ref` therefore defines the leaf vocabulary, and each sample takes the reference leaf held by the majority of its co-members in the inner-fold forest fitted without its own date group. `leaf_targets="insample"` (targets = `F_ref.apply(O)`) is available for comparison.

### Diagnostics (Table S18)

* `leaf_agreement(X, O)`: per sample, the share of trees whose predicted leaf equals the leaf of the measured descriptors in `F_ref`.
* `chance_agreement_`: mean over trees of Σ_l p_l², with p_l the training frequency of leaf l.
* `predict_oracle(X, O)`: the final regressor given the measured-optics leaves (not deployable).

### "-P" variants (Table 1)

`SOLACE(final_algorithm=alg)` uses the same optical forest and classifier with any of the six algorithms as the final stage.

---

## 4. Validation (M 2.5; S3.5; Tables S19, S20, S30) — `validation.py`

| scheme | function | notes |
|---|---|---|
| (a) stratified random 80:20 × 50 | `folds_random` | strata: campaign × within-campaign ΣPAHs quintile; medians and 2.5–97.5 % across splits |
| (b) leave-one-date-out | `folds_lodo` | |
| (c) contiguous blocks (primary) | `folds_blocks(block_days=4)` | summer blocks S1–S8; winter block W trained on summer only |
| (d) train winter → test summer | `folds_winter_to_summer` | |
| block lengths / buffers / offsets | `folds_blocks(L, buffer_days, offset, include_winter=False)` | winter always in training |
| station exclusion | `folds_station_exclusion` | test station removed from training |

`run_cv` fits every model on every outer training set; tuning and cross-fitting happen inside `fit`. Pooled out-of-fold predictions give R², MAE, RMSE and bias. `metrics.paired_date_bootstrap` resamples survey dates with replacement and gives 95 % CIs of R² and ΔR² = R²(SOLACE) − R²(SORFE). `metrics.paired_block_tests` gives the Wilcoxon signed-rank and sign tests on block MAEs. `validation.leakage_invariance` re-runs the CV after replacing held-out ΣPAHs and optical data with random values.

---

## 5. Controls and alternative uses of the optical data (M 2.4; S4.3; Table S17) — `models/controls.py`, `models/svr_plus.py`

* **Capacity control:** `SOLACE(leaf_source="sensor")`.
* **Shuffled control:** `SOLACE(shuffle_leaf_targets=True)`.
* **Two-stage regression:** a multi-output RF predicts X → Ô, using cross-fitted Ô for training; then an RF predicts [X, Ô] → y.
* **Generalized distillation** (Lopez-Paz et al., 2016): a teacher RF predicts O → y and gives out-of-fold soft labels s. The student RF predicts X → (1−λ)y + λs, with λ ∈ {0.25, 0.5, 0.75} tuned by date-grouped CV.
* **SVR+** (Vapnik & Vashist, 2009), regression form. The slack variables are modelled by correcting functions in the optical space. The dual (4n variables, 3 equality constraints, non-negativity) is solved with SLSQP, with variables scaled by C. RBF kernels are used in both spaces and γ is fixed (`gamma_priv`). C and ε are tuned by date-grouped CV. Prediction uses the sensor space only.
* **Multi-task learning** (Caruana, 1997): one MLP with shared hidden layers maps X to [y, O], all standardised. Each auxiliary output has loss weight 1/17 by default. The hidden size is tuned by date-grouped CV.
* **Ablations:** SOLACE with the EEM, UV–vis or DOC group removed from **O**.
* **References (not deployable):** an RF on O, and an RF on [X, O].
* **Negative control:** remove the three least important optical descriptors, compared with 100 random triplets.

---

## 6. Prediction intervals (M 2.5; S3.6) — `models/qrf.py`, `models/conformal.py`

The final stage is a **quantile regression forest** (Meinshausen, 2006). It uses forest weights w_i(x) = mean_t 1[x, x_i share a leaf]/|leaf| and returns the 5th and 95th conditional percentiles. **Conformalized quantile regression** (Romano et al., 2019) then calibrates these: the conformity scores E_i = max(q̂₀.₀₅ − y_i, y_i − q̂₀.₉₅) are out-of-fold predictions of the *complete pipeline* with date-grouped folds. Q is the ⌈(1−α)(n+1)⌉/n empirical quantile of E, and the interval is [q̂₀.₀₅ − Q, q̂₀.₉₅ + Q]. The calibration is repeated inside every training set. At the target sites the scores are recomputed on the pooled source + calibration samples.
`sonde.calibration_domain` flags records with any sensor outside the training range.
`metrics.screening` and `metrics.interval_triage` compute the screening statistics for the upper-tertile and median thresholds.

---

## 7. SHAP and descriptor predictability (M 2.5; S3.4; Table S22) — `interpret.py`

* Panels a and b use `shap.TreeExplainer` (interventional, background = fold training samples).
* Panel c uses `shap.PermutationExplainer` on the function X ↦ SOLACE.predict(X).
* Without `shap`, exact interventional Shapley values are computed over all 2^M coalitions for M ≤ 12. For larger M they are estimated by antithetic permutation sampling.
* Panel d: an RF predicts each optical descriptor from X, and its R² is computed over the summer blocks.

---

## 8. Few-shot cross-site calibration (M 2.5, 3.5; S6; Tables S21, S25, S26) — `calibration.py`

* **Day-separated draws** (primary): for each sampling day d of a target site, draw 10 calibration samples from day d, 200 times. The validation set is all samples from the other days.
* **Random reference:** 10 samples are drawn from all days. This is optimistic.
* In every draw, SORFE and SOLACE are refitted on source (142) + calibration samples with hyper-parameters fixed from the source fit (`frozen_clone`). SOLACE uses 14 descriptors. Validation uses sensors only.
* Summaries: median, IQR and 2.5–97.5 % of R², RMSE and MAE across draws; paired ΔR²; % of draws where SOLACE is better; Spearman ρ between validation R² and the share of the concentration range covered by the calibration samples.
* `calibration_size_curve`: n = 3, 5, 10, 15, 20, comparing the SOLACE pooled refit, a local RF trained on the n samples only, and a linear correction y ≈ a + b·ŷ_source.
* `permutation_importance_pct`: the drop in validation R² after permuting each sonde variable, as % of the total drop (Fig. 6e–f).

---

## 9. Optical descriptors and PARAFAC (S1.3, S2.1) — `optics.py`, `parafac.py`

* **UV–vis:** a(λ) = 2.303·A(λ)/L. Spectral slopes come from a linear fit of ln a(λ) over 275–295 nm and 350–400 nm, and SR = S275–295/S350–400. SUVA254 = a254/DOC; `suva_decadic=True` gives A254/L/DOC instead.
* **EEM pre-processing steps:**
  * blank subtraction;
  * inner-filter correction F·10^((A_ex+A_em)/2);
  * Raman normalisation by the integrated water Raman peak (Ex 350 nm, Em 371–428 nm);
  * masking of 1st/2nd-order Rayleigh and Raman scatter and of Em < Ex, then interpolation along emission.
* **Indices:**
  * FI = F(370/470)/F(370/520);
  * BIX = F(310/380)/F(310/430);
  * β:α = F(310/380)/max F(310, 420–435);
  * HIX (Ohno) = ΣF(254; 438–480)/[ΣF(254; 300–345) + ΣF(254; 438–480)];
  * protein-like (Ex 220–300 nm / Em 280–380 nm) and humic-like (Ex 220–400 nm / Em 380–550 nm) regional fluorescence.
* **PARAFAC:** non-negative CP by HALS, with EM imputation of missing values and several random starts. Loadings are scaled to max 1, so the sample mode gives Fmax. Split-half validation uses Tucker congruence (> 0.95). **TotalF** = ΣFmax of the components.

---

## 10. ΣPAHs and descriptive statistics (M 2.3, 2.5; S1.5–S1.7) — `pahs.py`, `stats.py`

* `pahs.sum_pahs`: ΣPAHs is the sum of the 16 congeners, with non-detects set to 0.5 × LOD (`recovery_correct` is optional).
* `pahs.group_sums` gives the LMW (2–3 rings), MMW (4 rings) and HMW (5–6 rings) sums and shares. `pahs.eqs_exceedance` compares fluoranthene and benzo[a]pyrene with the EU EQS (Directive 2013/39/EU).
* `stats.anova_icc` gives the one-way ANOVA by survey date and ICC(1). `stats.lag_correlation`, `stats.rainfall_lag_correlations` and `stats.residual_acf` (with the ±1.96/√n bound) cover the time-dependence analyses.
* `stats.kruskal_epsilon2` (ε² = H/(n−1)), `stats.fisher_loss_test` and `stats.chi2_loss_by_station` support the composition and sample-loss analyses. `stats.manova` gives Wilks, Pillai, Hotelling–Lawley and Roy statistics with F approximations.

---

## 11. Congener and molecular-weight-group models (M 2.3; S4.5; Table S16) — `scripts/11_congener_and_group_models.py`

Each congener (non-detects at ½·LOD) and each of the LMW, MMW and HMW sums replaces ΣPAHs as the response. The model is the same SOLACE configuration, evaluated on the summer blocks of the primary scheme.
