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
from collections import Counter
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
REFIT_TABLES = ('tables.tex', 'extension_table.tex', 'combination_table.tex',
                'ar_comparison_table.tex', 'ar_recent_table.tex')
REFIT_EXPORTS = ('observed_vs_predicted.csv', 'origin_comparison.png', AR_CSV, 'ar_comparison.png')
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
        for name in files.difference(REFIT_JSON + REFIT_TABLES + REFIT_EXPORTS + (AR_RESULT,)):
            shutil.copyfile(ROOT / name, directory / name)
        env = os.environ | {'OPENBLAS_NUM_THREADS': '1', 'MPLCONFIGDIR': str(directory / 'matplotlib')}
        for name in ('analysis.py', 'extension.py', 'model_extension.py', 'build_assets.py',
                     'ar_reassessment.py', 'build_ar_assets.py'):
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
    verify_ar(verify_forecasts())
    if args.full:
        verify_full(files)


if __name__ == '__main__':
    main()
