# Forecasting Annual Global Earthquake Counts

Reproduction package for Albaraa Mesawi's study of annual worldwide counts of large earthquakes. Release v1.2.0 compares fixed, expanding, and adaptively selected training histories in the USGS and ISC-GEM catalogues. It retains the Gaussian autoregression (AR) reassessment and the earlier six-model count analysis. [Analysis protocol](PLAN.md)

## Training history and catalogue sensitivity

The extension scores one-year forecasts for USGS targets in 1978–2025 and ISC-GEM targets in 1978–2022. The matched catalogue comparison uses 1978–2022. The primary USGS series uses preferred magnitude at least 7.0. The primary ISC-GEM series uses main-catalogue moment magnitude at least 7.0, including direct and proxy estimates. Supplementary ISC-GEM records enter a separate sensitivity comparison. Both catalogues can include records for the same earthquake.

The analysis compares 20-, 30-, and 40-observation histories, expansion from 1918, and exponentially discounted means. At each origin the two selectors use only the prior 20 realized forecast errors. Matched comparisons hold the training responses fixed when testing lag dependence or negative-binomial dispersion. The extension also records paired loss paths, approximate dependence-adjusted intervals, magnitude-threshold sensitivities, and an event-link audit. The earlier USGS results were already known when this exploratory protocol was set.

Over the 45 matched targets, shortening the response mean's training history from expansion since 1918 to 30 observations reduced USGS RMSE from 4.406 to 3.898 and increased ISC-GEM RMSE from 3.700 to 3.772. Both approximate 95% intervals for the paired mean squared-error difference included zero.

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

- The default check verifies the release hashes, rebuilds the ISC-GEM snapshot from the raw archive, and replays the event-link audit. It verifies the earlier forecast records and summaries. It recalculates the extension's forecasts from saved parameters, reconstructs count probabilities by recurrence, and checks selector histories, scores, and paired losses without refitting.
- The full run repeats these checks and refits in a temporary copy using one OpenBLAS thread. It reruns the earlier analyses and asset builders, then runs `isc_gem_data.py`, `time_series_extension.py`, and `build_time_series_assets.py`. It checks exact scientific results and compares the snapshots, frozen forecasts, CSV files, tables, and figures. Exact agreement can depend on numerical libraries, CPU behavior, serialization, and the rendering environment.

The comparison excludes historical elapsed time. For the model extension it excludes elapsed time, creation time, the historical results file's byte hash, and the plan hash. It verifies the results-file byte hash separately for each run and retains the canonical scientific digest. The earlier saved model-extension result records the digest of its original plan. The AR and time-series scientific digests exclude runtime metadata and their own digest fields. Each includes its normalized scientific protocol from this package's `PLAN.md`.

## Files and data

- `isc-gem.zip` contains the frozen ISC-GEM version 13.0 main catalogue, supplementary catalogue, and KMZ file. `isc_gem_data.py` rebuilds `isc_gem_snapshot.json` and verifies the archive and member hashes. [ISC-GEM attribution and data license](DATA-LICENSE.md)
- `time_series_extension.py`, `time_series_results.json`, and `time_series_predictions.csv` contain the time-series extension. The JSON includes fitted models, selector histories, paired losses, and the event-link audit. `build_time_series_assets.py` regenerates its tables and figures.
- `analysis.py` and `results.json` contain the historical analysis for the initial five models, with all optimizer starts and forecast scores.
- `forecast_2020_frozen.json`, `extension.py`, and `extension_results.json` contain the saved 2020-origin distributions and supplementary evaluation.
- `model_extension.py` and `model_extension_results.json` contain the added negative-binomial autoregression and one-year combinations, including probability distributions and the weight sequence.
- `observed_vs_predicted.csv` contains 1,728 individual historical forecasts, 72 combination forecasts, and 25 supplementary forecasts. `forecast_diagnostics.json` describes signed one-year mean errors (forecast minus observation).
- `build_assets.py` regenerates the two exports, the forecast figure, and the three standalone table fragments. `SHA256SUMS` lists the original byte hashes of the release files other than the manifest itself.
- `usgs_1918_2020_snapshot.json` contains the parsed USGS response and 1,275 events. `usgs_2021_2025_snapshot.json` contains 75 events and the original UTF-8 response. Each snapshot records the query URL and retrieval time. The historical snapshot carries a hash of the raw response, and the package verifies the included snapshot bytes. Both use preferred catalogue magnitudes, which include different magnitude types. The analysis uses no declustering or magnitude conversion. [USGS ComCat event service](https://earthquake.usgs.gov/fdsnws/event/1/)
- `ar_reassessment.py` and `ar_reassessment_results.json` contain the reassessment, including fitting starts, outcomes, and summaries. `ar_reassessment_predictions.csv` contains 960 rolling rows, 20 fixed-2020 rows, and 10 printed-value audit rows. `build_ar_assets.py` regenerates the AR figure and its two tables.
