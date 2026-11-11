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
REFIT_TABLES = ('tables.tex', 'extension_table.tex', 'combination_table.tex')
REFIT_EXPORTS = ('observed_vs_predicted.csv', 'origin_comparison.png')


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
        for name in files.difference(REFIT_JSON + REFIT_TABLES + REFIT_EXPORTS):
            shutil.copyfile(ROOT / name, directory / name)
        env = os.environ | {'OPENBLAS_NUM_THREADS': '1', 'MPLCONFIGDIR': str(directory / 'matplotlib')}
        for name in ('analysis.py', 'extension.py', 'model_extension.py', 'build_assets.py'):
            print(f'Rerunning {name} in a temporary directory.', flush=True)
            result = subprocess.run([sys.executable, '-B', name], cwd=directory, env=env,
                                    capture_output=True, text=True)
            if result.returncode:
                print(result.stdout, file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                raise SystemExit(result.returncode)
        for name in REFIT_JSON:
            compare_refit(name, directory)
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
    verify_forecasts()
    if args.full:
        verify_full(files)


if __name__ == '__main__':
    main()
