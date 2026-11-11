"""Check frozen earthquake forecasts offline. Pass --full to refit in a temporary copy."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
from itertools import accumulate
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ORIGINS = range(1977, 2013)
HORIZONS = range(1, 9)
TOLERANCE = 1e-10
CSV_FIELDS = (
    'evaluation model origin target_year horizon observed_count forecast_mean forecast_median '
    'error_mean_minus_observed absolute_error_of_mean absolute_error_of_median squared_error_of_mean '
    'lower95 upper95 covered95 observed_count_probability nll rps'
).split()
REFIT_JSON = ('results.json', 'model_extension_results.json', 'extension_results.json',
              'forecast_diagnostics.json')
AR_RESULT = 'ar_reassessment_results.json'
AR_CSV = 'ar_reassessment_predictions.csv'
TS_RESULT = 'time_series_results.json'
TS_CSV = 'time_series_predictions.csv'
ISC_SNAPSHOT = 'isc_gem_snapshot.json'
REFIT_TABLES = ('tables.tex', 'extension_table.tex', 'combination_table.tex',
                'ar_comparison_table.tex', 'ar_recent_table.tex')
REFIT_EXPORTS = ('observed_vs_predicted.csv', 'origin_comparison.png', AR_CSV, 'ar_comparison.png',
                 ISC_SNAPSHOT, TS_CSV, 'time_series_forecasts.png', 'time_series_catalogues.png',
                 'time_series_point_table.tex', 'time_series_contrast_table.tex',
                 'time_series_count_table.tex', 'time_series_sensitivity_table.tex',
                 'time_series_recent_table.tex')
TS_FIELDS = (
    'catalogue model history applied_history half_life sd_floor origin target_year '
    'training_start_year first_response_year training_observations training_responses observed '
    'predicted_mean negative_mean signed_error squared_error absolute_error effective_slope fit_source '
    'dispersion probability nll rps cdf_left cdf_right median lower80 upper80 covered80 width80 '
    'lower95 upper95 covered95 width95 cap comparison_cap tail_mass_upper tail_first_moment_upper '
    'mass_discrepancy mean_discrepancy'
).split()
TS_INTEGER = set(('origin target_year training_start_year first_response_year training_observations '
                  'training_responses observed median lower80 upper80 width80 lower95 upper95 width95 '
                  'cap comparison_cap').split())
TS_BOOLEAN = {'negative_mean', 'covered80', 'covered95'}
TS_TEXT = {'catalogue', 'model', 'history', 'applied_history', 'half_life', 'fit_source'}
AR_FIELDS = (
    'protocol model window sd_floor origin target_year horizon training_start_year '
    'first_response_year training_responses observed predicted_mean signed_error '
    'squared_error absolute_error'
).split()
AR_WINDOWS = ('expanding', 'recent_30')
GAUSSIAN_MODELS = ('gaussian_ar', 'gaussian_ar_mix')
AR_FLOORS = (1.0, .5, 2.0)
AR_PERIODS = ((1978, 2025), (1978, 2000), (2001, 2020), (2021, 2025))
PRINTED_MEANS = {
    'printed_ar': (11.73, 10.36, 11.73, 11.15, 9.39),
    'printed_ar_mix': (9.0, 10.65, 8.98, 8.28, 12.41),
}


def require(condition, message):
    if not condition:
        raise SystemExit(f'Verification failed: {message}')


def close(actual, expected, label, tolerance=TOLERANCE):
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance),
            f'{label}: {actual!r} differs from {expected!r}')


def load(name, directory=ROOT):
    return json.loads((directory / name).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest():
    files = set()
    for line in (ROOT / 'SHA256SUMS').read_text(encoding='ascii').splitlines():
        expected, name = line.split('  ', 1)
        path = ROOT / name
        require(Path(name).name == name and name not in files, f'manifest entry {name}')
        require(path.is_file() and digest(path) == expected, f'SHA-256 for {name}')
        files.add(name)
    require(bool(files), 'empty manifest')
    print(f'Checked SHA-256 for {len(files)} release files.', flush=True)
    return files


def catalogue_counts(name, years, size):
    snapshot = load(name)
    catalogue = snapshot['geojson']
    require(catalogue['type'] == 'FeatureCollection'
            and len(catalogue['features']) == catalogue['metadata']['count'] == size,
            f'{name} response type and event count')
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    counts = {str(year): 0 for year in years}
    identities, magnitude_types = set(), Counter()
    for event in catalogue['features']:
        properties = event['properties']
        year = str((epoch + timedelta(milliseconds=properties['time'])).year)
        require(event['id'] not in identities and year in counts and properties['type'] == 'earthquake'
                and math.isfinite(properties['mag']) and properties['mag'] >= 7,
                f'{name} event {event["id"]}')
        identities.add(event['id'])
        counts[year] += 1
        magnitude_types[properties['magType']] += 1
    recorded = snapshot['annual_counts']
    if isinstance(recorded, list):
        recorded = {str(row['year']): row['count'] for row in recorded}
    require(recorded == counts and snapshot['magnitude_type_counts'] == magnitude_types,
            f'{name} catalogue aggregates')
    if 'raw_response_utf8' in snapshot:
        raw = snapshot['raw_response_utf8'].encode('utf-8')
        require(len(raw) == snapshot['raw_response_bytes'] and json.loads(raw) == catalogue
                and hashlib.sha256(raw).hexdigest() == snapshot['raw_response_sha256'],
                f'{name} raw response')
    return counts


def score(pmf, target, label):
    require(all(math.isfinite(p) and p >= 0 for p in pmf), f'{label} probability mass')
    close(math.fsum(pmf), 1, f'{label} normalization', 1e-12)
    require(0 <= target < len(pmf) and pmf[target] > 0, f'{label} observed probability')
    cdf = list(accumulate(pmf))
    lo95, lo80, median, hi80, hi95 = (bisect_left(cdf, q) for q in (.025, .10, .5, .90, .975))
    mean = math.fsum(k * p for k, p in enumerate(pmf))
    return {
        'probability': pmf[target], 'nll': -math.log(pmf[target]),
        'rps': math.fsum((p - (k >= target)) ** 2 for k, p in enumerate(cdf)),
        'mean': mean, 'median': median, 'interval80': [lo80, hi80], 'interval95': [lo95, hi95],
        'squared_error': (mean - target) ** 2, 'absolute_error': abs(median - target),
        'coverage80': lo80 <= target <= hi80, 'coverage95': lo95 <= target <= hi95,
        'width80': hi80 - lo80, 'width95': hi95 - lo95,
    }


def verify_row(row, pmf, counts, label):
    year = row['origin'] + row['horizon']
    require(year == row['target_year'] and row['target'] == counts[str(year)], f'{label} target')
    for metric, actual in score(pmf, row['target'], label).items():
        if isinstance(actual, (int, bool, list)):
            require(actual == row[metric], f'{label} {metric}')
        else:
            close(actual, row[metric], f'{label} {metric}')


def by_case(rows, expected):
    indexed = {(r['model'], r['origin'], r['horizon']): r for r in rows}
    require(len(indexed) == len(rows) and indexed.keys() == expected, 'forecast case keys')
    return indexed


def verify_summary(rows, expected, label):
    require(set(expected) == {r['model'] for r in rows}, f'{label} models')
    fields = ('nll', 'rps', 'absolute_error', 'coverage80', 'coverage95', 'width80', 'width95')
    for model, summary in expected.items():
        subset = [r for r in rows if r['model'] == model]
        require(summary['cases'] == len(subset), f'{label} {model} cases')
        for field in fields:
            close(math.fsum(r[field] for r in subset) / len(subset), summary[field],
                  f'{label} {model} {field}')
        rmse = math.sqrt(math.fsum(r['squared_error'] for r in subset) / len(subset))
        close(rmse, summary['rmse'], f'{label} {model} rmse')


def verify_summaries(result, label):
    rows = result['rows']
    verify_summary(rows, result['summary'], label)
    for h, summary in result.get('by_horizon', {}).items():
        verify_summary([r for r in rows if r['horizon'] == int(h)], summary, f'{label} horizon {h}')
    for period, summary in result.get('fixed_origin_split', {}).items():
        verify_summary([r for r in rows if (r['origin'] <= 1994) == (period == 'first18')],
                       summary, f'{label} {period}')


def verify_export(groups, historical):
    expected = []
    for evaluation, rows in groups:
        for r in rows:
            error = r['mean'] - r['target']
            expected.append([str(value) for value in [
                evaluation, r['model'], r['origin'], r['target_year'], r['horizon'], r['target'],
                r['mean'], r['median'], error, abs(error), r['absolute_error'], r['squared_error'],
                *r['interval95'], r['coverage95'], r['probability'], r['nll'], r['rps']
            ]])
    with (ROOT / 'observed_vs_predicted.csv').open(newline='', encoding='utf-8-sig') as stream:
        header, *actual = csv.reader(stream)
    require(header == CSV_FIELDS and len(actual) == len(expected) == 1825, 'CSV dimensions')
    for line, (actual_row, expected_row) in enumerate(zip(actual, expected), 2):
        require(actual_row == expected_row, f'CSV line {line}')
    diagnostics = load('forecast_diagnostics.json')
    require(diagnostics['csv_rows'] == 1825, 'diagnostic CSV row count')
    for model, expected_diagnostics in diagnostics['one_year'].items():
        rows = [r for r in historical if r['model'] == model and r['horizon'] == 1]
        means = [r['mean'] for r in rows]
        targets = [r['target'] for r in rows]
        actual_diagnostics = {
            'cases': len(rows), 'mean_error_forecast_minus_observed':
                math.fsum(mean - target for mean, target in zip(means, targets)) / len(rows),
            'mean_forecast_min': min(means), 'mean_forecast_max': max(means),
            'observed_min': min(targets), 'observed_max': max(targets),
        }
        for field, value in actual_diagnostics.items():
            close(value, expected_diagnostics[field], f'{model} diagnostic {field}')


def verify_forecasts():
    import analysis

    base = load('results.json')
    added = load('model_extension_results.json')
    supplement = load('extension_results.json')
    frozen = load('forecast_2020_frozen.json')
    counts = catalogue_counts('usgs_1918_2020_snapshot.json', range(1918, 2021), 1275)
    later = catalogue_counts('usgs_2021_2025_snapshot.json', range(2021, 2026), 75)
    require(base['annual_counts'] == counts and supplement['annual_counts'] == later, 'evaluation counts')
    old_rows = by_case(base['rows'], {(m, o, h) for m in analysis.MODELS for o in ORIGINS for h in HORIZONS})
    nb_rows = by_case(added['rows'], {('nb2_ar', o, h) for o in ORIGINS for h in HORIZONS})
    combo = added['aggregation']
    combo_models = ['uniform_six', 'likelihood_weighted_six']
    combo_rows = by_case(combo['rows'], {(m, o, 1) for m in combo_models for o in ORIGINS})
    supplement_rows = by_case(supplement['rows'],
                              {(m, 2020, h) for m in analysis.MODELS for h in range(1, 6)})
    for origin in ORIGINS:
        lag = counts[str(origin)]
        for model in analysis.MODELS:
            params = base['fits'][str(origin)][model]['params']
            forecasts = analysis.forecast(model, params, lag, len(HORIZONS))
            for h, forecast in zip(HORIZONS, forecasts, strict=True):
                case = (model, origin, h)
                verify_row(old_rows[case], forecast['pmf'], counts, case)
        for h in HORIZONS:
            case = ('nb2_ar', origin, h)
            verify_row(nb_rows[case], nb_rows[case]['pmf'], counts, case)
        for model in combo_models:
            case = (model, origin, 1)
            verify_row(combo_rows[case], combo_rows[case]['pmf'], counts, case)
    for model in analysis.MODELS:
        for h, forecast in zip(HORIZONS, frozen['forecasts'][model], strict=True):
            case = (model, 2020, h)
            require(forecast['horizon'] == h and forecast['target_year'] == 2020 + h,
                    f'{case} frozen calendar')
            if h <= 5:
                verify_row(supplement_rows[case], forecast['pmf'], later, case)
            else:
                score(forecast['pmf'], 0, case)
    for result, label in [(base, 'historical'), (added, 'NB2 AR'), (combo, 'combinations'),
                          (supplement, 'supplement')]:
        verify_summaries(result, label)
    historical = base['rows'] + added['rows']
    verify_export([('historical', historical), ('combination', combo['rows']),
                   ('frozen_2020_supplement', supplement['rows'])], historical)
    print('Checked both catalogues, 1,825 score records and CSV rows, summaries, and one-year diagnostics.',
          flush=True)
    return counts | later


def ar_scientific_digest(result):
    stable = {k: v for k, v in result.items() if k not in ('elapsed_seconds', 'runtime', 'scientific_sha256')}
    payload = json.dumps(stable, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def gaussian_affine(model, params, pairs, objective, label):
    def log_density(y, x, a, b, sd):
        return -.5 * ((y - a - b * x) / sd) ** 2 - math.log(sd * math.sqrt(2 * math.pi))

    a, b, sd = params[:3]
    if model == 'gaussian_ar':
        log_terms = [log_density(y, x, a, b, sd) for x, y in pairs]
    else:
        c, d, other_sd, logit = params[3:]
        w = 1 / (1 + math.exp(-logit))
        v = 1 / (1 + math.exp(logit))
        log_w, log_v = math.log(w), math.log(v)
        log_terms = []
        for x, y in pairs:
            left = log_w + log_density(y, x, a, b, sd)
            right = log_v + log_density(y, x, c, d, other_sd)
            hi, lo = max(left, right), min(left, right)
            log_terms.append(hi + math.log1p(math.exp(lo - hi)))
        a, b = w * a + v * c, w * b + v * d
    close(-math.fsum(log_terms), objective, f'{label} Gaussian training objective')
    return a, b


def count_objective(a, b, alpha, pairs):
    terms = []
    for x, y in pairs:
        eta = a + b * math.log1p(x)
        mu = math.exp(eta)
        if alpha == 0:
            normalizer = mu
        else:
            normalizer = (1 / alpha + y) * math.log1p(alpha * mu)
            normalizer -= math.fsum(math.log1p(alpha * k) for k in range(y))
        terms.append(math.lgamma(y + 1) - y * eta + normalizer)
    return math.fsum(terms)


def verify_nb_limit():
    import numpy as np
    from ar_reassessment import nb_training, stable_nb_ar_objective

    lag = [5, 12, 24]
    observed = [0, 19, 10]
    log_lag = np.log1p(lag)
    training = nb_training(np.asarray(observed, dtype=float))
    intercept, slope = 2.5, -.4
    with localcontext() as ctx:
        ctx.prec = 60
        means = [(Decimal('2.5') - Decimal('0.4') * Decimal(1 + x).ln()).exp() for x in lag]

        def high_precision(alpha):
            value = Decimal(0)
            for mu, y in zip(means, observed, strict=True):
                if alpha == 0:
                    probability = (-mu).exp() * mu ** y / Decimal(math.factorial(y))
                else:
                    r = 1 / alpha
                    p = r / (r + mu)
                    rising = math.prod(r + k for k in range(y))
                    probability = rising * p ** r * (1 - p) ** y / Decimal(math.factorial(y))
                value -= probability.ln()
            return value

        step = Decimal('1e-20')
        for alpha in map(Decimal, ('0', '1e-8', '1e-4', '0.2', '2')):
            obj, gradient = stable_nb_ar_objective([intercept, slope, float(alpha)], log_lag, training)
            ref = high_precision(alpha)
            low = max(Decimal(0), alpha - step)
            d_alpha = (high_precision(alpha + step) - high_precision(low)) / (alpha + step - low)
            close(obj, float(ref), f'NB mass at alpha={alpha}')
            close(gradient[2], float(d_alpha), f'NB dispersion derivative at alpha={alpha}')


def expected_ar_predictions(result, counts):
    expected = {}
    fitted = result['fits']
    require(set(fitted) == {str(year) for year in range(1977, 2025)}, 'AR fitted origins')
    for year, fit in fitted.items():
        origin = int(year)
        target = origin + 1
        lag = counts[year]
        require(fit['origin'] == origin and fit['lag_count'] == lag, f'AR fitted origin {origin}')
        trailing = [counts[str(t)] for t in range(origin - 9, origin + 1)]
        for model, start, value in [('persistence', origin, lag),
                                    ('trailing_10_mean', origin - 9, math.fsum(trailing) / 10)]:
            expected['rolling', model, 'common', None, target] = (origin, 1, start, None, None, value)
        for window in AR_WINDOWS:
            fit_window = fit['windows'][window]
            start = 1918 if window == 'expanding' else origin - 29
            n = origin - start
            training_fields = ('training_start_year', 'first_response_year', 'last_response_year',
                               'training_observations', 'training_responses')
            require(tuple(fit_window[k] for k in training_fields) == (start, start + 1, origin, n + 1, n),
                    f'AR training window {origin} {window}')
            sample = [counts[str(t)] for t in range(start, origin + 1)]
            pairs = list(zip(sample[:-1], sample[1:], strict=True))
            calendar = (origin, 1, start, start + 1, n)
            mean = math.fsum(sample[1:]) / n
            close(fit_window['response_mean'], mean, f'AR response mean {origin} {window}')
            expected['rolling', 'response_mean', window, None, target] = (*calendar, mean)
            for model, model_fit in fit_window['count_fits'].items():
                if model == 'iid_nb2_nested_candidate':
                    a, alpha = model_fit['params']
                    b = 0
                else:
                    a, b, *dispersion = model_fit['params']
                    alpha = dispersion[0] if dispersion else 0
                    mean = math.exp(a + b * math.log1p(lag))
                    expected['rolling', model, window, None, target] = (*calendar, mean)
                objective = count_objective(a, b, alpha, pairs)
                close(objective, model_fit['objective'], f'{model} count objective {origin} {window}')
            for floor in AR_FLOORS:
                gaussians = fit_window['gaussian_fits_by_sd_floor'][str(floor)]
                for model in GAUSSIAN_MODELS:
                    gauss = gaussians[model]
                    label = (origin, window, model, floor)
                    a, b = gaussian_affine(model, gauss['params'], pairs, gauss['objective'], label)
                    expected['rolling', model, window, floor, target] = (*calendar, a + b * lag)
                    if origin == 2020 and floor == 1.0:
                        for horizon in range(1, 6):
                            mean = b ** horizon * lag + a * math.fsum(b ** h for h in range(horizon))
                            expected['fixed_2020', model, window, floor, origin + horizon] = (
                                origin, horizon, start, start + 1, n, mean
                            )
    for model, means in PRINTED_MEANS.items():
        for year, mean in enumerate(means, 2021):
            expected['printed', model, None, None, year] = (None, None, None, None, None, mean)
    return expected


def verify_ar_summaries(summaries, rows):
    count = 0
    for protocol, periods in [('rolling', AR_PERIODS), ('fixed_2020', AR_PERIODS[-1:]),
                              ('printed', AR_PERIODS[-1:])]:
        groups = {(m, w, f) for p, m, w, f, _ in rows if p == protocol}
        expected = {(f'{start}-{end}', m, w, f) for start, end in periods for m, w, f in groups}
        seen = set()
        for entry in summaries[protocol]:
            period, model, window, floor = (entry[k] for k in ('period', 'model', 'window', 'sd_floor'))
            key = (period, model, window, floor)
            require(key in expected - seen, f'AR summary key {key}')
            seen.add(key)
            start, end = map(int, period.split('-'))
            errors = [rows[protocol, model, window, floor, t] for t in range(start, end + 1)]
            n = len(errors)
            require(entry['n'] == n and entry['negative_forecast_means'] == sum(mean < 0 for mean, _ in errors),
                    f'AR summary counts {key}')
            close(entry['rmse'], math.sqrt(math.fsum(e ** 2 for _, e in errors) / n), f'AR RMSE {key}')
            close(entry['mae'], math.fsum(abs(e) for _, e in errors) / n, f'AR MAE {key}')
            close(entry['mean_signed_error'], math.fsum(e for _, e in errors) / n, f'AR signed error {key}')
        require(seen == expected, f'{protocol} AR summary coverage')
        count += len(seen)
    require(count == 86, 'AR summary total')


def verify_ar(counts):
    from ar_reassessment import plan_digest

    result = load(AR_RESULT)
    require(result['scientific_sha256'] == ar_scientific_digest(result), 'AR scientific SHA-256')
    require(result['code_sha256'] == digest(ROOT / 'ar_reassessment.py')
            and result['design_section_sha256'] == plan_digest()
            and all(value == digest(ROOT / name) for name, value in result['input_sha256'].items()),
            'AR input, code, and design provenance')
    require(result['annual_counts'] == counts and result['predictions_csv_sha256'] == digest(ROOT / AR_CSV),
            'AR catalogue counts and prediction export digest')
    snapshot = load('usgs_2021_2025_snapshot.json')
    boundary = result['original_printed_forecast_source']['benchmark_magnitude_boundary_counts']
    for year in range(2021, 2026):
        equal = sum(event['time_utc'].startswith(str(year)) and event['magnitude'] == 7.0
                    for event in snapshot['event_records'])
        require(boundary['preferred_magnitude_equal_7'][str(year)] == equal
                and boundary['preferred_magnitude_above_7'][str(year)] == counts[str(year)] - equal,
                f'AR supplementary magnitude threshold {year}')
    expected = expected_ar_predictions(result, counts)
    rows = {}
    with (ROOT / AR_CSV).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == AR_FIELDS, 'AR CSV columns')
        for line, row in enumerate(reader, 2):
            target = int(row['target_year'])
            floor = float(row['sd_floor']) if row['sd_floor'] else None
            key = (row['protocol'], row['model'], row['window'] or None, floor, target)
            require(key in expected and key not in rows, f'AR CSV line {line} case')
            *calendar, mean = expected[key]
            actual_calendar = [int(row[k]) if row[k] else None for k in AR_FIELDS[4:5] + AR_FIELDS[6:10]]
            observed = counts[str(target)]
            require(actual_calendar == calendar and int(row['observed']) == observed, f'AR CSV line {line} calendar')
            actual_mean = float(row['predicted_mean'])
            close(actual_mean, mean, f'AR CSV line {line} mean')
            error = actual_mean - observed
            for field, value in [('signed_error', error), ('squared_error', error ** 2), ('absolute_error', abs(error))]:
                close(float(row[field]), value, f'AR CSV line {line} {field}')
            rows[key] = (actual_mean, error)
    require(rows.keys() == expected.keys() and len(rows) == 990, 'AR CSV case coverage')
    verify_ar_summaries(result['summaries'], rows)
    verify_nb_limit()
    print('Checked 990 AR prediction rows, 86 summaries, training objectives, and the NB limit.', flush=True)


def ts_catalogues(ts, usgs_counts, result):
    from isc_gem_data import build_snapshot

    study_events = {'main': [], 'supplementary': []}
    isc = build_snapshot(ROOT / 'isc-gem.zip', study_events=study_events)
    payload = (json.dumps(isc, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode()
    require(payload == (ROOT / ISC_SNAPSHOT).read_bytes(), 'ISC-GEM snapshot replay from raw archive')
    usgs, _ = ts.load_usgs()
    values = {'usgs_ge_7': usgs_counts}
    for catalogue, minimum, strict in [('usgs_gt_7', 7, True), ('usgs_ge_7_5', 7.5, False)]:
        counts = Counter(str(event['when'].year) for event in usgs
                         if event['mag'] > minimum or (not strict and event['mag'] == minimum))
        values[catalogue] = {year: counts[year] for year in usgs_counts}
    for key, catalogue in [('main', 'isc_main_ge_7'),
                           ('main_plus_supplementary', 'isc_main_plus_supplementary_ge_7')]:
        annual = isc['annual']
        values[catalogue] = dict(zip(map(str, annual['years']), annual[key]['mw_ge_7'], strict=True))
    isc_matching = [
        {'id': event['eventid'], 'when': datetime.fromisoformat(event['time_utc']),
         'lat': event['lat'], 'lon': event['lon'], 'mag': event['mw'],
         'magnitude_type': event['mw_source'], 'catalogue': label}
        for label, events in study_events.items() for event in events
    ]
    audit = ts.match_events([event for event in usgs if event['when'].year <= ts.COMMON_END], isc_matching)
    require(audit == result['data']['isc_usgs_event_audit'], 'replayed cross-catalogue event audit')
    return values


def ts_reference_mean(counts, origin, length=None, half_life=None):
    start = 1919 if length is None else origin - length + 2
    values = [counts[str(year)] for year in range(start, origin + 1)]
    if half_life is None:
        return math.fsum(values) / len(values)
    weights = [2 ** (-age / half_life) for age in range(len(values) - 1, -1, -1)]
    return math.fsum(w * y for w, y in zip(weights, values, strict=True)) / math.fsum(weights)


def ts_selections(ts, catalogue, counts, audits):
    expected_origins = range(1977, max(map(int, counts)))
    require([audit['origin'] for audit in audits] == list(expected_origins), f'{catalogue} selector origins')
    selected = {}
    for audit in audits:
        origin = audit['origin']
        inner = audit['inner_forecasts']
        require([r['origin'] for r in inner] == list(range(origin - 20, origin)),
                f'{catalogue} inner origins at {origin}')
        for row in inner:
            o = row['origin']
            require(row['target_year'] == o + 1 and row['observed'] == counts[str(o + 1)],
                    f'{catalogue} selector calendar at {o}')
        for selector, field, choices in [('history', 'history_means', ts.HISTORIES),
                                         ('half_life', 'discounted_means', ts.HALF_LIVES)]:
            losses = {}
            for choice, setting in choices.items():
                args = {'length' if selector == 'history' else 'half_life': setting}
                predictions = [ts_reference_mean(counts, row['origin'], **args) for row in inner]
                for forecast, row in zip(predictions, inner, strict=True):
                    close(forecast, row[field][choice], f'{catalogue} {field} {row["origin"]} {choice}')
                losses[choice] = math.fsum((prediction - row['observed']) ** 2
                                           for prediction, row in zip(predictions, inner, strict=True)) / 20
                close(losses[choice], audit[f'{selector}_inner_mse'][choice],
                      f'{catalogue} {selector} inner MSE at {origin} {choice}')
            best = next(key for key in reversed(losses) if losses[key] <= min(losses.values()) + 1e-10)
            require(best == audit[f'selected_{selector}'], f'{catalogue} {selector} choice at {origin}')
        selected[origin] = (audit['selected_history'], audit['selected_half_life'])
    return selected


def ts_check_count_row(row, mean, alpha, label):
    observed = row['observed']
    cap = row['cap']
    require(alpha >= 0 and mean >= 0 and 0 <= observed <= cap <= 8192, f'{label} count support')
    close(row['dispersion'], alpha, f'{label} dispersion')
    mass = [math.exp(-math.log1p(alpha * mean) / alpha if alpha else -mean)]
    for count in range(cap):
        mass.append(mass[-1] * mean * (1 + alpha * count) / ((1 + alpha * mean) * (count + 1)))
    cdf = list(accumulate(mass))
    lo95, lo80, median, hi80, hi95 = (bisect_left(cdf, p) for p in (.025, .1, .5, .9, .975))
    expected = {'probability': mass[observed], 'nll': -math.log(mass[observed]),
                'rps': math.fsum((p - (k >= observed)) ** 2 for k, p in enumerate(cdf)),
                'cdf_left': min(cdf[observed - 1], 1) if observed else 0,
                'cdf_right': min(cdf[observed], 1), 'median': median,
                'lower80': lo80, 'upper80': hi80, 'covered80': lo80 <= observed <= hi80, 'width80': hi80 - lo80,
                'lower95': lo95, 'upper95': hi95, 'covered95': lo95 <= observed <= hi95, 'width95': hi95 - lo95}
    for field, value in expected.items():
        close(row[field], value, f'{label} {field}')
    close(math.fsum(mass), 1, f'{label} probability sum')
    close(math.fsum(k * p for k, p in enumerate(mass)), mean, f'{label} probability mean')


def ts_check_row(ts, row, counts, selections, fits, label):
    model, history = row['model'], row['history']
    origin, target = row['origin'], row['target_year']
    chosen, half_life = selections[origin]
    applied = row['applied_history']
    if history == 'selected':
        require(applied == chosen, f'{label} applied selected history')
    lag_count = counts[str(origin)]
    if model in ('persistence', 'trailing_10_mean'):
        start = origin if model == 'persistence' else origin - 9
        first = start
        mean = math.fsum(counts[str(y)] for y in range(start, origin + 1)) / (origin - start + 1)
        require(applied == 'baseline', f'{label} baseline history')
    elif model == 'discounted_mean':
        require(applied == 'expanding', f'{label} discounted history')
        if history == 'selected_discount':
            require(row['half_life'] == half_life, f'{label} applied half-life')
        start, first = 1918, 1919
        mean = ts_reference_mean(counts, origin, half_life=ts.HALF_LIVES[row['half_life']])
    else:
        require(applied == (chosen if history == 'selected' else history), f'{label} applied history')
        start = 1964 if applied == 'expanding_from_1964' else 1918
        if applied.startswith('recent_'):
            start = origin - ts.HISTORIES[applied] + 1
        first = start + 1
        train = [counts[str(year)] for year in range(start, origin + 1)]
        pairs = list(zip(train[:-1], train[1:], strict=True))
        window = fits[str(origin)][applied]
        if model == 'response_mean':
            mean = math.fsum(train[1:]) / (origin - start)
        elif model in GAUSSIAN_MODELS:
            fit = window['gaussian_fits_by_sd_floor'][str(row['sd_floor'])][model]
            a, b = gaussian_affine(model, fit['params'], pairs, fit['objective'], label)
            mean = a + b * lag_count
        else:
            fit = window['count_fits'][model]
            params = fit['params']
            a = params[0]
            b = params[1] if model in ('poisson_ar', 'nb2_ar') else 0.0
            alpha = params[-1] if model in ('iid_nb2', 'nb2_ar') else 0.0
            mean = math.exp(a + b * math.log1p(lag_count))
            close(count_objective(a, b, alpha, pairs), fit['objective'], f'{label} fitted likelihood')
            ts_check_count_row(row, mean, alpha, label)
        if model != 'response_mean':
            close(row['effective_slope'], b, f'{label} fitted slope')
            require(row['fit_source'] == fit['selected_source'], f'{label} fitted source')
    dates = ('training_start_year', 'first_response_year', 'training_observations', 'training_responses')
    require(tuple(row[k] for k in dates) == (start, first, origin - start + 1, origin - first + 1),
            f'{label} training calendar')
    require(origin + 1 == target and row['observed'] == counts[str(target)], f'{label} observation')
    close(row['predicted_mean'], mean, f'{label} mean')
    error = row['predicted_mean'] - row['observed']
    for field, value in [('signed_error', error), ('squared_error', error ** 2), ('absolute_error', abs(error))]:
        close(row[field], value, f'{label} {field}')
    require(row['negative_mean'] == (row['predicted_mean'] < 0), f'{label} negative-mean flag')


def verify_time_series(usgs_counts):
    import numpy as np
    import time_series_extension as ts

    result = load(TS_RESULT)
    require(result['scientific_sha256'] == ar_scientific_digest(result), 'time-series scientific SHA-256')
    require(result['code_sha256'] == digest(ROOT / 'time_series_extension.py')
            and result['design_section_sha256'] == ts.plan_digest()
            and result['input_sha256'] == ts.frozen_inputs()
            and result['predictions_csv_sha256'] == digest(ROOT / TS_CSV), 'time-series inputs and protocol')
    counts = ts_catalogues(ts, usgs_counts, result)
    rows = defaultdict(list)
    with (ROOT / TS_CSV).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == TS_FIELDS, 'time-series CSV columns')
        for record in reader:
            typed = {k: (v if k in TS_TEXT else int(v) if k in TS_INTEGER
                         else {'True': True, 'False': False}[v] if k in TS_BOOLEAN else float(v))
                     for k, v in record.items() if v}
            for key in ('half_life', 'sd_floor', 'effective_slope', 'fit_source'):
                typed.setdefault(key, None)
            rows[typed['catalogue']].append(typed)
    total = sum(map(len, rows.values()))
    require(total == result['forecast_rows'] and rows.keys() == counts.keys() == result['catalogues'].keys(),
            'time-series case coverage')
    for catalogue, data in result['catalogues'].items():
        catalogue_counts = counts[catalogue]
        require(catalogue_counts == data['raw_series']['annual_counts'], f'{catalogue} annual counts')
        selections = ts_selections(ts, catalogue, catalogue_counts, data['selection_audit'])
        groups = defaultdict(list)
        for row in rows[catalogue]:
            groups[ts.group_key(row)].append(row)
            label = (catalogue, row['model'], row['history'], row['origin'])
            ts_check_row(ts, row, catalogue_counts, selections, data['fits_by_origin_history'], label)
        last = max(map(int, catalogue_counts))
        series = np.array([catalogue_counts[str(year)] for year in range(1918, last + 1)])
        modern = catalogue in ts.MAIN_CATALOGUES
        require(ts.validate_rows(rows[catalogue], groups, series, modern) == data['validation'],
                f'{catalogue} forecast coverage and diagnostics')
        require(ts.summarize(groups, last) == data['scores'], f'{catalogue} score summaries')
        contrasts = json.loads(json.dumps(ts.contrasts(groups, last, modern), allow_nan=False))
        require(contrasts == data['paired_contrasts'],
                f'{catalogue} paired losses, influence and HAC intervals')
    print(f'Checked ISC-GEM raw data and event matching, {total:,} time-series forecasts, selectors, and paired scores.',
          flush=True)


def compare_refit(name, directory):
    recorded, replayed = load(name), load(name, directory)
    if name == 'results.json':
        for result in (recorded, replayed):
            del result['elapsed_seconds']
    if name == 'model_extension_results.json':
        require(recorded['input_sha256']['results.json'] == digest(ROOT / 'results.json')
                and replayed['input_sha256']['results.json'] == digest(directory / 'results.json'),
                'model extension results-file provenance')
        require(replayed['plan_sha256'] == digest(directory / 'PLAN.md'), 'replayed public plan provenance')
        for result in (recorded, replayed):
            for field in ('elapsed_seconds', 'created_at_utc', 'plan_sha256'):
                del result[field]
            del result['input_sha256']['results.json']
    canonical = dict(sort_keys=True, allow_nan=False, separators=(',', ':'))
    require(json.dumps(recorded, **canonical) == json.dumps(replayed, **canonical), f'full refit {name}')


def verify_full(files):
    with tempfile.TemporaryDirectory(prefix='earthquake-reproduction-') as scratch:
        directory = Path(scratch)
        for name in files.difference(REFIT_JSON + REFIT_TABLES + REFIT_EXPORTS + (AR_RESULT, TS_RESULT)):
            shutil.copyfile(ROOT / name, directory / name)
        env = os.environ | {'OPENBLAS_NUM_THREADS': '1', 'MPLCONFIGDIR': str(directory / 'matplotlib')}
        for name in ('analysis.py', 'extension.py', 'model_extension.py', 'build_assets.py',
                     'ar_reassessment.py', 'build_ar_assets.py', 'isc_gem_data.py',
                     'time_series_extension.py', 'build_time_series_assets.py'):
            print(f'Rerunning {name} in a temporary directory.', flush=True)
            result = subprocess.run([sys.executable, '-B', name], cwd=directory, env=env,
                                    capture_output=True, text=True)
            if result.returncode:
                print(result.stdout, file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                raise SystemExit(result.returncode)
        for name in REFIT_JSON:
            compare_refit(name, directory)
        recorded, replayed = load(AR_RESULT), load(AR_RESULT, directory)
        require(recorded['scientific_sha256'] == replayed['scientific_sha256'] == ar_scientific_digest(replayed),
                'full refit AR scientific SHA-256')
        recorded, replayed = load(TS_RESULT), load(TS_RESULT, directory)
        require(recorded['scientific_sha256'] == replayed['scientific_sha256'] == ar_scientific_digest(replayed),
                'full refit time-series scientific SHA-256')
        for name in ('forecast_2020_frozen.json',) + REFIT_EXPORTS:
            require(digest(ROOT / name) == digest(directory / name), f'full refit {name}')
        for name in REFIT_TABLES:
            require((ROOT / name).read_text(encoding='utf-8') ==
                    (directory / name).read_text(encoding='utf-8'), f'regenerated {name}')
    print('Full refit, supplementary replay, and export regeneration matched the released results.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='refit every model in a temporary copy')
    args = parser.parse_args()
    files = verify_manifest()
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    counts = verify_forecasts()
    verify_ar(counts)
    verify_time_series(counts)
    if args.full:
        verify_full(files)


if __name__ == '__main__':
    main()
