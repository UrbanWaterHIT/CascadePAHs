# Data

## `data/demo/` — demo data

Demo data for running and testing the code (`scripts/00_demo_data.py`). The values are simulated (not measurements) and follow the sampling design of the study; station coordinates are approximate. For data requests, please contact congchaozhang@stu.hit.edu.cn.

## Using the measured data

The measured data are in the Supplementary Data of the article. To run the analysis on them, write CSV files with the columns below and set the paths in a copy of `configs/default.yaml`.

### 1. `paired_samples.csv` — complete PAH–optical–sensor paired samples (242 rows)

| column | unit | description |
|---|---|---|
| `sample_id` | – | unique ID |
| `site` | – | `DashaRiver`, `ShenzhenBay` or `YangyongRiver` |
| `campaign` | – | `summer2024`, `winter2025` (Dasha River) or `calib2024` (target sites, May–June 2024) |
| `date` | YYYY-MM-DD | survey date (**validation groups**) |
| `station` | – | station number |
| `lat`, `lon` | ° (WGS 84) | station coordinates |
| `T` | °C | temperature |
| `pH` | – | pH |
| `DO` | mg L⁻¹ | dissolved oxygen |
| `Sal` | PSU | salinity |
| `Turb` | FNU | turbidity |
| `Chla` | µg L⁻¹ | chlorophyll-a fluorescence |
| `FDOM` | QSU | sonde FDOM (Ex 365 nm / Em 480 nm) |
| `Cond`, `TDS` | µS cm⁻¹, mg L⁻¹ | recorded but **not** used (collinear with salinity) – optional |
| `C1`, `C2`, `C3` | R.U. | PARAFAC components (Fmax; humic-, tryptophan-, tyrosine-like) |
| `TotalF` | R.U. | total fluorescence |
| `ProteinF`, `HumicF` | R.U. | protein-like / humic-like fluorescence |
| `FI`, `BIX`, `HIX`, `BetaAlpha` | – | fluorescence, biological, humification and freshness (β:α) indices |
| `a254`, `a350` | m⁻¹ | Napierian absorption coefficients |
| `S275_295`, `S350_400` | nm⁻¹ | spectral slopes |
| `SR` | – | slope ratio |
| `SUVA254` | L mg C⁻¹ m⁻¹ | specific UV absorbance |
| `DOC` | mg L⁻¹ | dissolved organic carbon |
| `SumPAHs` | ng L⁻¹ | **target**: dissolved ΣPAHs (16 US EPA PAHs, 0.7 µm GF filtrate, recovery-corrected) |
| `NaP` … `BghiP` | ng L⁻¹ | 16 congeners (optional; empty = below LOD). Needed for composition / EQS analyses |
| `n_sonde_readings` | – | number of on-station readings averaged (optional) |

The sensor columns of the paired file are the **mean of the on-station readings** of each sampling event (`cascadepahs.sonde.match_onstation`).

### 2. `onstation_readings.csv` (optional) — raw sonde readings recorded on station during sampling
`sample_id, site, station, timestamp, deployment, depth_m, T, pH, DO, Sal, Turb, Chla, FDOM, Cond, TDS`

### 3. `hf_sonde_records.csv` — high-frequency survey records (unlabelled)
`timestamp, site, campaign, survey_id, lat, lon, depth_m, T, pH, DO, Sal, Turb, Chla, FDOM[, Cond, TDS]`

### 4. `application_period.csv` — 22 Jul–22 Aug 2024 target-site samples (ΣPAHs only)
`site, date, station, T, pH, DO, Sal, Turb, Chla, FDOM, SumPAHs`

### 5. `rainfall_daily.csv`
`date, rain_dasha_mm, rain_target_mm` (daily rainfall, Shenzhen Meteorological Bureau)

### 6. `pah_lod.csv`
`congener, LOD_ng_L` – limits of detection used for ½·LOD substitution.

### 7. `raw_spectra/` (optional) — input of `02_optical_descriptors.py`
* `absorbance_doc.csv`: `sample_id, DOC_mg_L, A240 … A600` (decadic absorbance, 1-cm cell, 1-nm steps)
* `eem_cube.npz`: arrays `ex` (nm), `em` (nm), `eem` (samples × ex × em, raw intensities), `blank` (ex × em, ultrapure water), `sample_id`
