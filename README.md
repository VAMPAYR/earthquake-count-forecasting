# Forecasting Annual Global Earthquake Counts

Reproduction package for Albaraa Mesawi's study of worldwide counts of earthquakes with preferred USGS magnitude at least 7.0. Release v1.1.0 adds an exploratory reassessment of Gaussian autoregression (AR) and a constant-weight Gaussian AR mixture. It retains the earlier six-model count analysis. [Analysis protocol](PLAN.md)

## AR and training history

The reassessment compares 48 matched one-year forecasts for 1978–2025. Gaussian AR's root mean squared error (RMSE) falls from 4.396 with expanding training to 3.890 using the most recent 30 observations. The mean of the same recent response years has RMSE 3.895. At the main standard-deviation lower bound of 1, the mixture has RMSE 4.421 and 3.899 with expanding and recent training. Changing the lower bound changes the ranking of the two Gaussian models. The 30-year window was fixed before these fits. The outcomes were already known when the reassessment was designed.

The new Gaussian fits use constrained conditional likelihood and reproduce the original model families. The original Bayesian estimates and posterior draws are unavailable. The printed-value audit transcribes Tables 2 and 4 of the original manuscript dated 8 August 2024 and scores its 2021–2025 values against the reconstructed USGS series. The audit does not establish a verified 2020 forecast issuance. The original describes a threshold above 7 on the Richter scale, and the reconstructed series uses USGS preferred magnitude at least 7.0. The original query is unavailable, and differences in threshold and catalogue version limit comparability.

The new negative-binomial AR fit uses a log1p formulation with analytic derivatives and an exact Poisson limit. This corrects numerical cancellation at dispersion near zero. The earlier count-study files retain their original bytes.

## Earlier count-model comparison

Six count models were evaluated at 36 historical origins (1977–2012) over horizons of one to eight years. Negative-binomial AR had the lowest pooled negative log predictive probability (NLL, 2.9853) and ranked probability score (RPS, 2.6775). Poisson AR had the lowest RMSE of forecast means (4.5878). Rankings varied by horizon and period. Updating combination weights increased mean one-year NLL from 2.9169 to 2.9186 and decreased mean one-year RPS from 2.5100 to 2.5031. The results describe small differences on one revised catalogue with overlapping forecast cases. Prediction intervals condition on fitted parameters.

The separate supplement uses five models fitted through 2020 and scores the five annual counts from 2021 to 2025. The 2020-origin distributions were computed retrospectively in November 2026 and frozen before the supplementary catalogue query. The supplement contains one forecast origin and five outcomes.

## Verify offline

The verified environment was CPython 3.14.0 on Windows x64. Install the pinned dependencies once, then run either command from this directory. If the dependencies are already available, use that Python interpreter. Both verification modes use the included snapshots and run offline.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B verify.py
.\.venv\Scripts\python.exe -B verify.py --full
```

- The default check verifies the release hashes, recounts both catalogues, and checks scores without refitting. It checks 1,825 count-model forecast records and the corresponding CSV rows, plus 990 AR prediction rows and 86 AR summaries. It recomputes fitted AR objectives, the Gaussian fixed-origin mean recursion, and high-precision negative-binomial limit cases.
- The full run repeats these checks and refits in a temporary copy using one OpenBLAS thread. It runs `analysis.py`, `extension.py`, `model_extension.py`, `build_assets.py`, `ar_reassessment.py`, and `build_ar_assets.py` in order. It checks exact scientific results and compares the frozen forecasts, CSV files, tables, and figures. Exact agreement can depend on numerical libraries, CPU behavior, serialization, and the rendering environment.

The comparison excludes historical elapsed time. For the model extension it excludes elapsed time, creation time, the historical results file's byte hash, and the plan hash. It verifies the results-file byte hash separately for each run and retains the canonical scientific digest. The earlier saved model-extension result records the digest of its original plan. For the AR reassessment, the scientific digest excludes runtime metadata and its own digest field, and includes the normalized scientific protocol from this package's `PLAN.md`.

## Files and data

- `analysis.py` and `results.json` contain the historical analysis for the initial five models, with all optimizer starts and forecast scores.
- `forecast_2020_frozen.json`, `extension.py`, and `extension_results.json` contain the saved 2020-origin distributions and supplementary evaluation.
- `model_extension.py` and `model_extension_results.json` contain the added negative-binomial autoregression and one-year combinations, including probability distributions and the weight sequence.
- `observed_vs_predicted.csv` contains 1,728 individual historical forecasts, 72 combination forecasts, and 25 supplementary forecasts. `forecast_diagnostics.json` describes signed one-year mean errors (forecast minus observation).
- `build_assets.py` regenerates the two exports, the forecast figure, and the three standalone table fragments. `SHA256SUMS` lists the original byte hashes of the release files other than the manifest itself.
- `usgs_1918_2020_snapshot.json` contains the parsed USGS response and 1,275 events. `usgs_2021_2025_snapshot.json` contains 75 events and the original UTF-8 response. Each snapshot records the query URL and retrieval time. The historical snapshot carries a hash of the raw response, and the package verifies the included snapshot bytes. Both use preferred catalogue magnitudes, which include different magnitude types. The analysis uses no declustering or magnitude conversion. [USGS ComCat event service](https://earthquake.usgs.gov/fdsnws/event/1/)
- `ar_reassessment.py` and `ar_reassessment_results.json` contain the reassessment, including fitting starts, outcomes, and summaries. `ar_reassessment_predictions.csv` contains 960 rolling rows, 20 fixed-2020 rows, and 10 printed-value audit rows. `build_ar_assets.py` regenerates the AR figure and its two tables.
