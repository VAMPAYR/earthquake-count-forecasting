# Forecasting Annual Global Earthquake Counts

Reproduction package for Albaraa Mesawi's exploratory study of annual worldwide counts of earthquakes with preferred USGS magnitude at least 7.0. Six established count models were evaluated at 36 historical forecast origins (1977–2012) over horizons of one to eight years. The study also compares two combinations of one-year forecasts. [Analysis protocol](PLAN.md)

Negative-binomial autoregression had the lowest pooled negative log predictive probability (NLL, 2.9853) and ranked probability score (RPS, 2.6775). Poisson autoregression had the lowest root mean squared error of forecast means (4.5878). Rankings varied by horizon and period. Updating combination weights increased mean one-year NLL from 2.9169 to 2.9186 and decreased mean one-year RPS from 2.5100 to 2.5031. The results describe small differences on one revised catalogue with overlapping forecast cases. Prediction intervals condition on fitted parameters.

The separate supplement uses five models fitted through 2020 and scores the five annual counts from 2021 to 2025. The 2020-origin distributions were computed retrospectively in November 2026 and frozen before the supplementary catalogue query. The supplement contains one forecast origin and five outcomes.

## Verify offline

The verified environment was CPython 3.14.0 on Windows x64. Install the pinned dependencies once, then run either command from this directory. If the dependencies are already available, use that Python interpreter. Both verification modes use the included snapshots and run offline.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B verify.py
.\.venv\Scripts\python.exe -B verify.py --full
```

- The default check verifies the release hashes, recounts events by UTC year from the two catalogues, and recalculates forecast scores without refitting. It reconstructs the original five historical models' probability distributions from saved fits and uses stored probability distributions for the added model, combinations, and 2020-origin supplement. It checks all 1,825 CSV rows, loss and interval summaries, and one-year diagnostics. Numeric scores use an absolute tolerance of 1e-10.
- The full run repeats those checks and refits in a temporary copy using one OpenBLAS thread. It runs `analysis.py`, `extension.py`, `model_extension.py`, and `build_assets.py` in that order. It compares the full scientific JSON content, frozen forecasts, CSV, tables, and figure. Exact scientific comparisons and frozen-output guards can fail if numerical libraries, CPU behavior, or serialization differ. The rendered figure can also differ with the rendering environment.

The refit comparison excludes historical elapsed time. For the model extension it excludes elapsed time, creation time, the historical results file's byte hash, and the plan hash. It verifies the results-file byte hash against each run's file and retains the canonical scientific digest. `PLAN.md` is a public description of the protocol. The released model-extension JSON retains the recorded plan hash from the original run, and the refit verifies its newly recorded hash against this public plan.

## Files and data

- `analysis.py` and `results.json` contain the historical analysis for the initial five models, with all optimizer starts and forecast scores.
- `forecast_2020_frozen.json`, `extension.py`, and `extension_results.json` contain the saved 2020-origin distributions and supplementary evaluation.
- `model_extension.py` and `model_extension_results.json` contain the added negative-binomial autoregression and one-year combinations, including probability distributions and the weight sequence.
- `observed_vs_predicted.csv` contains 1,728 individual historical forecasts, 72 combination forecasts, and 25 supplementary forecasts. `forecast_diagnostics.json` describes signed one-year mean errors (forecast minus observation).
- `build_assets.py` regenerates the two exports, the forecast figure, and the three standalone table fragments. `SHA256SUMS` lists the original byte hashes of the release files other than the manifest itself.
- `usgs_1918_2020_snapshot.json` contains the parsed USGS response and 1,275 events. `usgs_2021_2025_snapshot.json` contains 75 events and the original UTF-8 response. Each snapshot records the query URL and retrieval time. The historical snapshot carries a hash of the raw response, and the package verifies the included snapshot bytes. Both use preferred catalogue magnitudes, which include different magnitude types. The analysis uses no declustering or magnitude conversion. [USGS ComCat event service](https://earthquake.usgs.gov/fdsnws/event/1/)
