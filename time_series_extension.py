"""Replay the locked 2026 time-series extension from frozen catalogue snapshots.

Set OPENBLAS_NUM_THREADS=1 for a deterministic replay. Use --checks-only for
synthetic and numerical checks without fitting any catalogue data.
"""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.special import gammaln
from scipy.stats import nbinom, poisson

import ar_reassessment as ar


ROOT = Path(__file__).resolve().parent
FIRST = 1918
FIRST_TARGET = 1978
COMMON_END = 2022
MODERN_START = 1964
MODERN_FIRST_TARGET = 1994
HISTORIES = {'recent_20': 20, 'recent_30': 30, 'recent_40': 40, 'expanding': None}
HALF_LIVES = {'5': 5, '10': 10, '20': 20, '40': 40, 'infinity': None}
FLOORS = (0.5, 1.0, 2.0)
INNER_LENGTH = 20
TIE = 1e-10
HAC_LAGS = (3, 0, 1, 5)
QUANTILES = (0.025, 0.1, 0.5, 0.9, 0.975)
TAIL_TOL = 1e-12
PMF_TOL = 2e-10
COUNT_MODELS = ('iid_poisson', 'poisson_ar', 'iid_nb2', 'nb2_ar')
CORE_MODELS = ('response_mean', 'gaussian_ar', *COUNT_MODELS)
PLAN_HEADING = '## Time-series strengthening protocol, 11 November 2026'
PLAN_START = 'Data: preserve both existing USGS snapshots'
PLAN_END = 'This audit describes recorded measurement differences; it cannot assign a cause or force a complete mapping.'
PLAN_PRIVATE_ATTRIBUTION_SHA256 = 'a4cfbbbcf4a0e2769072b035811d8c5e2c5501d045657a55108cd23aaa7e542a'
PLAN_SHA256 = 'e0d71aee0de0eded04de61484ac1edbe95be716e6853bb126cd13a17ae53a1e4'
ORIGINAL_PRIVATE_PROTOCOL_SHA256 = 'de2e7146255c52b4f213a543247fec9c260a26c0869e4ba186891cd4d5a9aecd'
ISC_ARCHIVE_SHA256 = '12f255cef062e3ee51345b91dc0431b161d9a9af865cc09bae3a5ad39fdab0db'
INPUT_HASHES = {
    **ar.INPUT_HASHES,
    'ar_reassessment.py': 'ff7e5b45ef1fbc49f3fa1be8846d06ce49e7edf6c602d4be288b54bd8e5bb651',
    'isc_gem_data.py': '1e8832c2b8e6795b061e86fd7006c7d2bbd85b66b1371c3ef7ef545a1ae95bd9',
    'isc_gem_snapshot.json': '60f9c6788ce4250509199c3b949d26beb6e19162c9a24bf45d79b61099c96e2d',
    'isc-gem.zip': ISC_ARCHIVE_SHA256,
}
MAIN_CATALOGUES = ('usgs_ge_7', 'isc_main_ge_7')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def plan_digest():
    plan = (ROOT / 'PLAN.md').read_text(encoding='utf-8-sig')
    start = plan.index(PLAN_START, plan.index(PLAN_HEADING))
    end = plan.index(PLAN_END, start) + len(PLAN_END)
    scientific_text = plan[start:end]
    before = 'Record every candidate inner loss and selected history. '
    after = 'There is no performance-driven grid expansion or retrospective best-history selection.'
    left = scientific_text.index(before) + len(before)
    right = scientific_text.index(after, left)
    # Omit the one known private attribution without storing it in the public code.
    if sha256(scientific_text[left:right].encode()) == PLAN_PRIVATE_ATTRIBUTION_SHA256:
        scientific_text = scientific_text[:left] + scientific_text[right:]
    return sha256(f'{PLAN_HEADING}\n\n{scientific_text}\n'.encode())


def frozen_inputs():
    hashes = {name: sha256((ROOT / name).read_bytes()) for name in INPUT_HASHES}
    if hashes != INPUT_HASHES or plan_digest() != PLAN_SHA256:
        raise RuntimeError('Frozen inputs or time-series protocol differ')
    return hashes


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def annual_counts(events, last, predicate):
    counts = Counter(event['when'].year for event in events if predicate(event['mag']))
    return np.array([counts[year] for year in range(FIRST, last + 1)], dtype=int)


def load_usgs():
    events = []
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    for name, first, last in (('usgs_1918_2020_snapshot.json', FIRST, 2020),
                              ('usgs_2021_2025_snapshot.json', 2021, 2025)):
        geojson = read_json(name)['geojson']
        features = geojson['features']
        if geojson['type'] != 'FeatureCollection' or len(features) != geojson['metadata']['count']:
            raise RuntimeError('USGS feature count mismatch')
        for feature in features:
            props = feature['properties']
            when = epoch + timedelta(milliseconds=props['time'])
            lon, lat, _ = feature['geometry']['coordinates']
            if (props['type'] != 'earthquake' or not first <= when.year <= last
                    or props['mag'] is None or not math.isfinite(props['mag']) or props['mag'] < 7
                    or feature['geometry']['type'] != 'Point'
                    or not (-90 <= lat <= 90 and -180 <= lon <= 180)):
                raise RuntimeError(f"Invalid frozen USGS record {feature['id']}")
            aliases = sorted(set(filter(None, props['ids'].split(','))) | {feature['id']})
            events.append({'id': feature['id'], 'when': when, 'lat': lat, 'lon': lon,
                           'mag': props['mag'], 'magnitude_type': props['magType'], 'aliases': aliases})
    if len(events) != 1350 or len({event['id'] for event in events}) != len(events):
        raise RuntimeError('Unexpected USGS total or duplicate event identifiers')
    series = {key: annual_counts(events, 2025, predicate) for key, predicate in (
        ('usgs_ge_7', lambda mag: mag >= 7), ('usgs_gt_7', lambda mag: mag > 7),
        ('usgs_ge_7_5', lambda mag: mag >= 7.5))}
    np.testing.assert_array_equal(series['usgs_ge_7'][:103], ar.original.read_counts())
    return events, series


def load_isc():
    snapshot = read_json('isc_gem_snapshot.json')
    annual = snapshot['annual']
    if (annual['years'] != list(range(FIRST, COMMON_END + 1))
            or snapshot['provenance']['archive']['sha256'] != ISC_ARCHIVE_SHA256):
        raise RuntimeError('ISC calendar mismatch')
    events_by_catalogue = {}
    for label in ('main', 'supplementary'):
        records = snapshot['catalogues'][label]['events']
        events = [{'id': event['eventid'], 'when': datetime.fromisoformat(event['time_utc']),
                   'lat': event['lat'], 'lon': event['lon'], 'mag': event['mw'],
                   'magnitude_type': event['mw_source']}
                  for event in records if event['mw'] is not None and event['mw'] >= 7
                  and FIRST <= int(event['time_utc'][:4]) <= COMMON_END]
        np.testing.assert_array_equal(annual_counts(events, COMMON_END, lambda mag: mag >= 7),
                                      annual[label]['mw_ge_7'])
        if len({event['id'] for event in events}) != len(events):
            raise RuntimeError(f'Duplicate qualifying ISC {label} IDs')
        events_by_catalogue[label] = events
    main, supplementary = (events_by_catalogue[key] for key in ('main', 'supplementary'))
    if {event['id'] for event in main} & {event['id'] for event in supplementary}:
        raise RuntimeError('Overlapping main and supplementary ISC IDs')
    combined = np.array(annual['main_plus_supplementary']['mw_ge_7'], dtype=int)
    main_counts = np.array(annual['main']['mw_ge_7'], dtype=int)
    np.testing.assert_array_equal(combined, main_counts + annual['supplementary']['mw_ge_7'])
    return {'isc_main_ge_7': main_counts, 'isc_main_plus_supplementary_ge_7': combined}, snapshot


def training(counts, origin, history):
    end = origin - FIRST + 1
    if history == 'expanding_from_1964':
        start = MODERN_START - FIRST
    elif history == 'expanding':
        start = 0
    else:
        start = end - HISTORIES[history]
    if start < 0 or not start + 2 <= end <= len(counts):
        raise ValueError(f'Unavailable training history {history} at {origin}')
    values = counts[start:end].astype(float)
    return FIRST + start, values[:-1], values[1:]


def response_mean(counts, origin, history):
    return float(training(counts, origin, history)[2].mean())


def discounted_mean(counts, origin, half_life):
    response = training(counts, origin, 'expanding')[2]
    if half_life is None:
        return float(response.mean())
    ages = np.arange(len(response) - 1, -1, -1, dtype=float)
    weights = np.exp2(-ages / half_life)
    return float(weights @ response / weights.sum())


def choose_longer_tie(losses):
    minimum = min(losses.values())
    return next(key for key in reversed(losses) if losses[key] - minimum <= TIE)


def select_history(counts, origin, fixed_means, discounted_means):
    inner = []
    for inner_origin in range(origin - INNER_LENGTH, origin):
        target = inner_origin + 1
        observed = int(counts[target - FIRST])
        inner.append({'origin': inner_origin, 'target_year': target, 'observed': observed,
                      'history_means': fixed_means[inner_origin],
                      'discounted_means': discounted_means[inner_origin]})
    audit = {'origin': origin, 'inner_forecasts': inner}
    for selector, mean_key, choices in (('history', 'history_means', HISTORIES),
                                        ('half_life', 'discounted_means', HALF_LIVES)):
        losses = {key: math.fsum((row[mean_key][key] - row['observed']) ** 2 for row in inner) / INNER_LENGTH
                  for key in choices}
        audit[f'{selector}_inner_mse'] = losses
        audit[f'selected_{selector}'] = choose_longer_tie(losses)
    return audit


def validate_candidates(fit, expected_count):
    starts = fit['starts']
    successful = [start for start in starts if start['success']]
    best = min(successful, key=lambda start: start['objective'])
    if (len(starts) != expected_count or fit['params'] != best['params']
            or fit['objective'] != best['objective'] or fit['selected_source'] != best['source']
            or fit['failed_starts'] != len(starts) - len(successful)
            or not math.isfinite(best['objective'])):
        raise RuntimeError('Fitted candidate audit failed')


def fit_history(lag, response, mixtures):
    counts = ar.fit_count_models(response, lag)
    counts['iid_nb2'] = counts.pop('iid_nb2_nested_candidate')
    nested = counts['poisson_ar']['starts'][0]
    counts['iid_poisson'] = ar.select_fit([
        ar.exact_candidate('closed_form_response_mean', nested['params'][:1], nested['objective'], [])])
    for model, expected in zip(COUNT_MODELS, (1, 7, 9, 2 + len(ar.START_SLOPES) * len(ar.START_ALPHAS))):
        validate_candidates(counts[model], expected)
    if (counts['nb2_ar']['objective'] > min(counts['iid_nb2']['objective'], counts['poisson_ar']['objective']) + 1e-10
            or counts['iid_nb2']['objective'] > counts['iid_poisson']['objective'] + 1e-10
            or counts['poisson_ar']['objective'] > counts['iid_poisson']['objective'] + 1e-10):
        raise RuntimeError('Count fit exceeded a nested model likelihood')
    gaussians = {}
    for floor in FLOORS if mixtures else (1.0,):
        single = ar.fit_single(response, lag, floor)
        validate_candidates(single, 1)
        gaussians[str(floor)] = {'gaussian_ar': single}
        if mixtures:
            mixture = ar.fit_mixture(response, lag, floor, single)
            validate_candidates(mixture, 13)
            if mixture['objective'] > single['objective'] + 1e-10:
                raise RuntimeError('Mixture fit exceeded the nested Gaussian likelihood')
            gaussians[str(floor)]['gaussian_ar_mix'] = mixture
    return {'count_fits': counts, 'gaussian_fits_by_sd_floor': gaussians}


def log_count_mass(grid, mean, alpha):
    if mean == 0:
        return np.where(grid == 0, 0.0, -np.inf)
    base = grid * math.log(mean) - gammaln(grid + 1)
    if alpha == 0:
        return base - mean
    z = alpha * mean
    rising = np.concatenate(([0.0], np.cumsum(np.log1p(alpha * grid[:-1]))))
    return base + rising - (1 + alpha * grid) * (mean * (math.log1p(z) / z))


def pmf_at_cap(mean, alpha, cap):
    grid = np.arange(cap + 1, dtype=float)
    log_mass = log_count_mass(grid, mean, alpha)
    mass = np.exp(log_mass)
    ratio = mean / (1 + alpha * mean) * max((1 + alpha * cap) / (cap + 1), alpha)
    if ratio >= 1:
        tail_mass = tail_moment = math.inf
    else:
        tail_mass = mass[-1] * ratio / (1 - ratio)
        tail_moment = cap * tail_mass + mass[-1] * ratio / (1 - ratio) ** 2
    cdf = np.cumsum(mass)
    quantiles = tuple(int(np.searchsorted(cdf, q)) for q in QUANTILES)
    return {'cap': cap, 'grid': grid, 'log_mass': log_mass, 'mass': mass, 'cdf': cdf,
            'tail_mass_upper': float(tail_mass), 'tail_first_moment_upper': float(tail_moment),
            'mass_discrepancy': float(abs(1 - mass.sum())),
            'mean_discrepancy': float(abs(mean - mass @ grid)), 'quantiles': quantiles}


def count_pmf(mean, alpha):
    if not math.isfinite(mean) or not math.isfinite(alpha) or min(mean, alpha) < 0:
        raise ValueError('Invalid count distribution parameters')
    smaller = pmf_at_cap(mean, alpha, 32)
    for cap in (64, 128, 256, 512, 1024, 2048, 4096, 8192):
        larger = pmf_at_cap(mean, alpha, cap)
        if (smaller['tail_first_moment_upper'] <= TAIL_TOL
                and smaller['quantiles'] == larger['quantiles']):
            if (not np.all(np.isfinite(larger['mass'])) or np.any(larger['mass'] < 0)
                    or not np.all(np.isfinite(larger['log_mass']) | (mean == 0))
                    or larger['mass_discrepancy'] > PMF_TOL
                    or larger['mean_discrepancy'] > PMF_TOL * max(1, mean)
                    or larger['quantiles'][-1] > cap):
                raise RuntimeError('Count PMF normalization, mean, support or finiteness check failed')
            larger['comparison_cap'] = smaller['cap']
            return larger
        smaller = larger
    raise RuntimeError(f'Count tail bound or quantile stability failed for mean={mean}, alpha={alpha}')


def count_scores(mean, alpha, observed):
    pmf = count_pmf(mean, alpha)
    if observed > pmf['cap']:
        raise RuntimeError('Observation exceeds the independently determined count-score support')
    k = pmf['grid']
    cdf = pmf['cdf']
    lower95, lower80, median, upper80, upper95 = pmf['quantiles']
    log_prob = float(pmf['log_mass'][observed])
    if not math.isfinite(log_prob):
        raise RuntimeError('Nonfinite observed log probability')
    return {'dispersion': alpha, 'probability': float(pmf['mass'][observed]), 'nll': -log_prob,
            'rps': float(np.sum((cdf - (k >= observed)) ** 2)),
            'cdf_left': float(min(cdf[observed - 1], 1)) if observed else 0.0,
            'cdf_right': float(min(cdf[observed], 1)), 'median': median,
            'lower80': lower80, 'upper80': upper80, 'covered80': lower80 <= observed <= upper80,
            'width80': upper80 - lower80, 'lower95': lower95, 'upper95': upper95,
            'covered95': lower95 <= observed <= upper95, 'width95': upper95 - lower95,
            **{key: pmf[key] for key in ('cap', 'comparison_cap', 'tail_mass_upper',
                                        'tail_first_moment_upper', 'mass_discrepancy', 'mean_discrepancy')}}


def row(catalogue, model, history, applied, origin, observed, mean, start, first_response,
        fit=None, floor=None, half_life=None, alpha=None, effective_slope=None):
    if not math.isfinite(mean):
        raise RuntimeError('Nonfinite forecast mean')
    error = mean - observed
    return {'catalogue': catalogue, 'model': model, 'history': history, 'applied_history': applied,
            'half_life': half_life, 'sd_floor': floor, 'origin': origin, 'target_year': origin + 1,
            'training_start_year': start, 'first_response_year': first_response,
            'training_observations': origin - start + 1, 'training_responses': origin - first_response + 1,
            'observed': observed, 'predicted_mean': mean, 'negative_mean': mean < 0,
            'signed_error': error, 'squared_error': error ** 2, 'absolute_error': abs(error),
            'effective_slope': effective_slope, 'fit_source': fit['selected_source'] if fit else None,
            **(count_scores(mean, alpha, observed) if alpha is not None else {})}


def fitted_rows(catalogue, history, origin, observed, start, response, fit):
    last = float(response[-1])
    rows = [row(catalogue, 'response_mean', history, history, origin, observed,
                float(response.mean()), start, start + 1)]
    for floor, gaussians in fit['gaussian_fits_by_sd_floor'].items():
        for model, fitted in gaussians.items():
            intercept, slope = ar.affine_mean(model, fitted['params'])
            rows.append(row(catalogue, model, history, history, origin, observed, intercept + slope * last,
                            start, start + 1, fitted, floor=float(floor), effective_slope=slope))
    for model in COUNT_MODELS:
        fitted = fit['count_fits'][model]
        params = fitted['params']
        slope = params[1] if model in ('poisson_ar', 'nb2_ar') else 0.0
        alpha = params[-1] if model in ('iid_nb2', 'nb2_ar') else 0.0
        mean = float(response.mean()) if model == 'iid_poisson' else math.exp(params[0] + slope * math.log1p(last))
        if model == 'iid_nb2' and alpha == 0:
            np.testing.assert_allclose(mean, response.mean(), atol=1e-12, rtol=0)
            mean = float(response.mean())
        rows.append(row(catalogue, model, history, history, origin, observed, mean,
                        start, start + 1, fitted, alpha=alpha, effective_slope=slope))
    return rows


def primary_floor(prediction):
    return prediction['sd_floor'] in (None, 1.0)


def all_forecasts(catalogue, counts, cache):
    last = FIRST + len(counts) - 1
    outer_origins = range(FIRST_TARGET - 1, last)
    mean_origins = range(outer_origins.start - INNER_LENGTH, outer_origins.stop)
    fixed_means = {origin: {key: response_mean(counts, origin, key) for key in HISTORIES}
                   for origin in mean_origins}
    discounted_means = {origin: {key: discounted_mean(counts, origin, half) for key, half in HALF_LIVES.items()}
                        for origin in mean_origins}
    rows = []
    audits = []
    fits = {}
    for origin in outer_origins:
        observed = int(counts[origin + 1 - FIRST])
        lag_count = float(counts[origin - FIRST])
        trailing = float(counts[origin - FIRST - 9:origin - FIRST + 1].mean())
        rows.extend((row(catalogue, 'persistence', 'baseline', 'baseline', origin, observed,
                         lag_count, origin, origin),
                     row(catalogue, 'trailing_10_mean', 'baseline', 'baseline', origin, observed,
                         trailing, origin - 9, origin - 9)))
        selection = select_history(counts, origin, fixed_means, discounted_means)
        audits.append(selection)
        for key, prediction in discounted_means[origin].items():
            rows.append(row(catalogue, 'discounted_mean', 'fixed_discount', 'expanding', origin, observed,
                            prediction, FIRST, FIRST + 1, half_life=key))
        half = selection['selected_half_life']
        rows.append(row(catalogue, 'discounted_mean', 'selected_discount', 'expanding', origin, observed,
                        discounted_means[origin][half], FIRST, FIRST + 1, half_life=half))
        histories = list(HISTORIES)
        if catalogue in MAIN_CATALOGUES and MODERN_FIRST_TARGET <= origin + 1 <= COMMON_END:
            histories.append('expanding_from_1964')
        fixed_rows = {}
        for history in histories:
            start, lag, response = training(counts, origin, history)
            mixtures = history in ('recent_30', 'expanding')
            # A modern history can duplicate the recent-30 core fit. Cache every actual
            # count training input separately from the optional mixture/sensitivity fits.
            key = (start, origin, int(lag[0]), *map(int, response))
            if key not in cache:
                cache[key] = fit_history(lag, response, mixtures)
            fit = cache[key]
            if mixtures and len(fit['gaussian_fits_by_sd_floor']) != len(FLOORS):
                raise RuntimeError('Mixture history encountered after its core-only cache entry')
            if not mixtures and len(fit['gaussian_fits_by_sd_floor']) != 1:
                fit = {**fit, 'gaussian_fits_by_sd_floor': {'1.0': {
                    'gaussian_ar': fit['gaussian_fits_by_sd_floor']['1.0']['gaussian_ar']}}}
            fits.setdefault(str(origin), {})[history] = {
                'training_start_year': start, 'first_response_year': start + 1,
                'last_response_year': origin, 'training_responses': len(response), **fit}
            fixed_rows[history] = fitted_rows(catalogue, history, origin, observed, start, response, fit)
            rows.extend(fixed_rows[history])
        selected = selection['selected_history']
        rows.extend({**forecast, 'history': 'selected'} for forecast in fixed_rows[selected]
                    if forecast['model'] in CORE_MODELS and primary_floor(forecast))
    return rows, fits, audits


def group_key(prediction):
    return (prediction['model'], prediction['history'], prediction['sd_floor'],
            prediction['half_life'] if prediction['history'] == 'fixed_discount' else None)


def periods(last):
    return list(dict.fromkeys(((FIRST_TARGET, last), (FIRST_TARGET, 2000), (2001, 2020),
                               (2021, last), (FIRST_TARGET, COMMON_END))))


def acf(values, limit):
    centered = np.asarray(values, dtype=float) - np.mean(values)
    sum_sq = centered @ centered
    return {str(lag): float(centered[:-lag] @ centered[lag:] / sum_sq) if sum_sq else None
            for lag in range(1, min(limit, len(values) - 1) + 1)}


def pit_bins(left, right):
    if not 0 <= left <= right <= 1:
        raise RuntimeError('Invalid observation CDF interval')
    bins = np.linspace(0, 1, 11)
    if left == right:
        # A rounded CDF interval of zero representable width contributes a point mass.
        result = np.zeros(10)
        result[min(int(left * 10), 9)] = 1.0
        return result
    return np.maximum(0, np.minimum(right, bins[1:]) - np.maximum(left, bins[:-1])) / (right - left)


def avg(values):
    return float(np.mean(values))


def score_group(rows):
    result = {'n': len(rows), 'rmse': math.sqrt(avg([entry['squared_error'] for entry in rows])),
              'mae': avg([entry['absolute_error'] for entry in rows]),
              'mean_signed_error': avg([entry['signed_error'] for entry in rows]),
              'negative_forecast_means': sum(entry['negative_mean'] for entry in rows),
              'forecast_error_acf': acf([entry['signed_error'] for entry in rows], 4)}
    if 'nll' in rows[0]:
        hist = np.sum([pit_bins(entry['cdf_left'], entry['cdf_right']) for entry in rows], axis=0)
        np.testing.assert_allclose(hist.sum(), len(rows), atol=1e-12, rtol=0)
        result.update({key: avg([entry[key] for entry in rows]) for key in
                       ('nll', 'rps', 'covered80', 'width80', 'covered95', 'width95')})
        result['pit_bin_counts'] = hist.tolist()
    return result


def group_fields(key):
    return dict(zip(('model', 'history', 'sd_floor', 'half_life'), key))


def summarize(groups, last):
    results = []
    for key, rows in groups.items():
        applicable = periods(last) if key[1] != 'expanding_from_1964' else []
        if key[0] in CORE_MODELS and key[1] in ('expanding', 'recent_30', 'expanding_from_1964') and key[2] in (None, 1.0):
            applicable = [*applicable, (MODERN_FIRST_TARGET, COMMON_END)]
        for start, end in applicable:
            subset = [entry for entry in rows if start <= entry['target_year'] <= end]
            results.append({**group_fields(key), 'period': f'{start}-{end}', **score_group(subset)})
    return results


def hac(differences, lag):
    d = np.asarray(differences, dtype=float)
    n = len(d)
    if lag >= n:
        raise ValueError('HAC lag must be smaller than the sample size')
    mean = d.mean()
    centered = d - mean
    gammas = [float(centered @ centered / n)]
    gammas.extend(float(centered[k:] @ centered[:-k] / n) for k in range(1, lag + 1))
    variance = (gammas[0] + 2 * math.fsum((1 - k / (lag + 1)) * gammas[k]
                                        for k in range(1, lag + 1))) / n
    distance = np.abs(np.arange(n)[:, None] - np.arange(n))
    weights = np.maximum(0, 1 - distance / (lag + 1))
    matrix = float(centered @ weights @ centered / n ** 2)
    tolerance = 64 * np.finfo(float).eps * max(gammas[0] / n, np.finfo(float).tiny)
    if variance < -tolerance or abs(variance - matrix) > tolerance:
        raise RuntimeError('Bartlett HAC variance failed independent matrix check')
    variance = max(0.0, variance)
    radius = 1.96 * math.sqrt(variance)
    return {'lag': lag, 'variance': variance, 'matrix_variance': matrix,
            'standard_error': math.sqrt(variance), 'interval95': [float(mean - radius), float(mean + radius)]}


def influence(years, differences, block):
    total = math.fsum(differences)
    n = len(differences)
    if n <= block:
        return None
    cases = [{'first_omitted': years[index], 'last_omitted': years[index + block - 1],
              'mean_loss_difference': (total - math.fsum(differences[index:index + block])) / (n - block)}
             for index in range(n - block + 1)]
    return {'minimum': min(cases, key=lambda case: case['mean_loss_difference']),
            'maximum': max(cases, key=lambda case: case['mean_loss_difference']), 'cases': cases}


def compare(method_rows, reference_rows, loss, start, end):
    method = [entry for entry in method_rows if start <= entry['target_year'] <= end]
    reference = [entry for entry in reference_rows if start <= entry['target_year'] <= end]
    years = list(range(start, end + 1))
    for rows in (method, reference):
        if [entry['target_year'] for entry in rows] != years:
            raise RuntimeError('Unmatched paired comparison calendar')
    if [entry['observed'] for entry in method] != [entry['observed'] for entry in reference]:
        raise RuntimeError('Unmatched paired observations')
    method_losses = [entry[loss] for entry in method]
    reference_losses = [entry[loss] for entry in reference]
    differences = [left - right for left, right in zip(method_losses, reference_losses)]
    intervals = [hac(differences, lag) for lag in HAC_LAGS if lag < len(years)]
    return {'period': f'{start}-{end}', 'loss': loss, 'n': len(years),
            'method_score': math.sqrt(avg(method_losses)) if loss == 'squared_error' else avg(method_losses),
            'reference_score': math.sqrt(avg(reference_losses)) if loss == 'squared_error' else avg(reference_losses),
            'mean_loss_difference': avg(differences),
            'method_wins': sum(value < 0 for value in differences),
            'reference_wins': sum(value > 0 for value in differences),
            'ties': sum(value == 0 for value in differences),
            'yearly_differences': list(zip(years, differences)),
            'cumulative_loss_differences': np.cumsum(differences).tolist(), 'hac': intervals,
            'leave_one_target_out': influence(years, differences, 1),
            'leave_five_contiguous_targets_out': influence(years, differences, 5)}


def series_key(model, history, floor=None):
    return model, history, floor, None


def contrasts(groups, last, modern):
    mean = 'response_mean'
    gaussian = 'gaussian_ar'
    recent_mean = series_key(mean, 'recent_30')
    expanding_mean = series_key(mean, 'expanding')
    selected_mean = series_key(mean, 'selected')
    recent_ar = series_key(gaussian, 'recent_30', 1.0)
    point_pairs = (
        (recent_mean, expanding_mean), (recent_ar, series_key(gaussian, 'expanding', 1.0)),
        (recent_ar, recent_mean), (series_key('gaussian_ar_mix', 'recent_30', 1.0), recent_ar),
        (selected_mean, recent_mean), (selected_mean, expanding_mean),
        (series_key(gaussian, 'selected', 1.0), selected_mean),
        (series_key('discounted_mean', 'selected_discount'), recent_mean),
    )
    rows = []
    for start, end in periods(last):
        pairs = [(method, reference, 'squared_error') for method, reference in point_pairs]
        for history in ('expanding', 'recent_30', 'selected'):
            for method, reference in (('iid_nb2', 'iid_poisson'), ('nb2_ar', 'poisson_ar'),
                                      ('poisson_ar', 'iid_poisson'), ('nb2_ar', 'iid_nb2')):
                pairs.extend((series_key(method, history), series_key(reference, history), loss)
                             for loss in ('nll', 'rps'))
        for method, reference, loss in pairs:
            rows.append({'method': group_fields(method), 'reference': group_fields(reference),
                         **compare(groups[method], groups[reference], loss, start, end)})
    if modern:
        for model in CORE_MODELS:
            floor = 1.0 if model == gaussian else None
            for method, reference in (('expanding_from_1964', 'expanding'), ('recent_30', 'expanding'),
                                      ('recent_30', 'expanding_from_1964')):
                left, right = series_key(model, method, floor), series_key(model, reference, floor)
                losses = ('squared_error', 'nll', 'rps') if model in COUNT_MODELS else ('squared_error',)
                for loss in losses:
                    rows.append({'method': group_fields(left), 'reference': group_fields(right),
                                 **compare(groups[left], groups[right], loss, MODERN_FIRST_TARGET, COMMON_END)})
    return rows


def distance_km(first, second):
    lat1, lat2 = map(math.radians, (first['lat'], second['lat']))
    delta_lon = math.radians(second['lon'] - first['lon'])
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(a, 1)))


def audit_event(event):
    return {**{key: event[key] for key in ('id', 'lat', 'lon', 'mag', 'magnitude_type')},
            'time_utc': event['when'].isoformat()}


def match_events(usgs, isc):
    by_id = {event['id']: event for event in isc}
    if len(by_id) != len(isc):
        raise RuntimeError('ISC match scope contains duplicate event identifiers')
    isc_sorted = sorted(isc, key=lambda event: event['when'])
    times = [event['when'] for event in isc_sorted]
    horizon = timedelta(seconds=120)
    exact = {}
    for event in usgs:
        aliases = {alias.removeprefix('iscgem') for alias in event['aliases'] if alias.startswith('iscgem')}
        exact[event['id']] = sorted(aliases & by_id.keys())
    claimed = Counter(identity for candidates in exact.values() for identity in candidates)
    nearby = {}
    for event in usgs:
        if exact[event['id']]:
            continue
        first = bisect_left(times, event['when'] - horizon)
        end = bisect_right(times, event['when'] + horizon)
        nearby[event['id']] = sorted(other['id'] for other in isc_sorted[first:end]
                                     if other['id'] not in claimed and distance_km(event, other) <= 200)
    proximity_claimed = Counter(identity for candidates in nearby.values() for identity in candidates)
    paired, unresolved = [], []
    resolved_isc = set()
    for event in usgs:
        identity = event['id']
        by_alias = bool(exact[identity])
        candidates = exact[identity] if by_alias else nearby[identity]
        reciprocal = claimed if by_alias else proximity_claimed
        if len(candidates) == 1 and reciprocal[candidates[0]] == 1:
            other = by_id[candidates[0]]
            paired.append({'rule': 'exact_isc_identifier' if by_alias else 'mutually_unique_time_distance',
                           'usgs': audit_event(event), 'isc': audit_event(other),
                           'isc_catalogue': other['catalogue'],
                           'isc_minus_usgs_seconds': (other['when'] - event['when']).total_seconds(),
                           'distance_km': distance_km(event, other)})
            resolved_isc.add(other['id'])
        else:
            status = ('ambiguous_exact_identifier' if by_alias else 'ambiguous_time_distance') if candidates else 'unmatched'
            unresolved.append({'usgs': audit_event(event), 'status': status,
                               'usgs_source_ids': event['aliases'],
                               'isc_candidates': [{**audit_event(by_id[candidate]),
                                                   'catalogue': by_id[candidate]['catalogue'],
                                                   'usgs_claimants': reciprocal[candidate]} for candidate in candidates]})
    qualifying = [event for event in isc if event['mag'] is not None and event['mag'] >= 7]
    isc_unmatched = [{**audit_event(event), 'catalogue': event['catalogue'],
                      'status': 'ambiguous_candidate' if event['id'] in claimed or event['id'] in proximity_claimed else 'unmatched'}
                     for event in qualifying if event['id'] not in resolved_isc]
    rules = Counter(pair['rule'] for pair in paired)
    if len(resolved_isc) != len(paired) or len(paired) + len(unresolved) != len(usgs):
        raise RuntimeError('Event match audit failed to account for every USGS record')
    return {'summary': {'usgs_qualifying': len(usgs), 'isc_all_considered': len(isc),
                        'isc_qualifying': len(qualifying), **dict(rules),
                        **dict(Counter(entry['status'] for entry in unresolved)),
                        'matched_isc_missing_mw': sum(pair['isc']['mag'] is None for pair in paired),
                        'matched_isc_mw_below_7': sum(pair['isc']['mag'] is not None and pair['isc']['mag'] < 7 for pair in paired),
                        'isc_qualifying_without_unique_usgs_qualifier': len(isc_unmatched)},
            'scope': 'USGS preferred magnitude >=7 against all ISC main and supplementary records in UTC 1918–2022. '
                     'Unpaired ISC qualifiers lack a unique match in the retained USGS thresholded snapshot.',
            'pairs': paired, 'usgs_without_unique_match': unresolved, 'isc_qualifying_without_unique_match': isc_unmatched}


def load_matching_rows():
    from isc_gem_data import load_study_events

    archive = ROOT / 'isc-gem.zip'
    return [{'id': event['eventid'], 'when': datetime.fromisoformat(event['time_utc']),
             'lat': event['lat'], 'lon': event['lon'], 'mag': event['mw'],
             'magnitude_type': event['mw_source'], 'catalogue': label}
            for label, events in load_study_events(archive).items() for event in events]


def raw_diagnostics(counts, last):
    blocks = [(FIRST, last), (FIRST, MODERN_START - 1), (MODERN_START, COMMON_END), *periods(last)]
    moments = []
    for first, end in blocks:
        values = counts[first - FIRST:end - FIRST + 1]
        moments.append({'period': f'{first}-{end}', 'n': len(values), 'mean': float(values.mean()),
                        'sample_variance': float(values.var(ddof=1))})
    return {'annual_counts': {str(year): int(counts[year - FIRST]) for year in range(FIRST, last + 1)},
            'annual_count_acf': acf(counts, 10), 'period_moments': moments}


def catalogue_metadata(usgs, series, isc_snapshot, matching):
    annual_isc = isc_snapshot['annual']
    main_counts = series['isc_main_ge_7']
    usgs_counts = series['usgs_ge_7']
    differences = {str(year): int(usgs_counts[year - FIRST] - main_counts[year - FIRST])
                   for year in range(FIRST, COMMON_END + 1)}
    return {'usgs_minus_isc_main_ge_7_annual': differences,
            'usgs_threshold_boundaries': {
                name: {str(year): int(value) for year, value in enumerate(
                    annual_counts(usgs, 2025, lambda mag: mag == boundary), FIRST)}
                for name, boundary in (('equal_7', 7), ('equal_7_5', 7.5))},
            'usgs_boundary_exclusions_by_year': {
                name: (usgs_counts - series[name]).tolist() for name in ('usgs_gt_7', 'usgs_ge_7_5')},
            'isc_threshold_annual': annual_isc,
            'isc_unclassified_mw': {label: {key: record[key] for key in ('scope_missing_mw', 'scope_unparseable_mw')}
                                    for label, record in isc_snapshot['catalogues'].items()},
            'isc_archive_sha256': ISC_ARCHIVE_SHA256,
            'isc_usgs_event_audit': matching}


def validate_rows(rows, groups, counts, modern):
    last = FIRST + len(counts) - 1
    histories = (*HISTORIES, 'selected')
    expected = {series_key(model, history, 1.0 if model == 'gaussian_ar' else None)
                for model in CORE_MODELS for history in histories}
    expected.update(series_key(model, 'baseline') for model in ('persistence', 'trailing_10_mean'))
    expected.update(('discounted_mean', 'fixed_discount', None, half) for half in HALF_LIVES)
    expected.add(series_key('discounted_mean', 'selected_discount'))
    expected.update(series_key('gaussian_ar_mix', history, floor) for history in ('expanding', 'recent_30') for floor in FLOORS)
    expected.update(series_key('gaussian_ar', history, floor) for history in ('expanding', 'recent_30') for floor in (0.5, 2.0))
    if modern:
        expected.update(series_key(model, 'expanding_from_1964', 1.0 if model == 'gaussian_ar' else None)
                        for model in CORE_MODELS)
    if groups.keys() != expected:
        raise RuntimeError('Missing or unexpected forecast series')
    for key, entries in groups.items():
        years = range(MODERN_FIRST_TARGET, COMMON_END + 1) if key[1] == 'expanding_from_1964' else range(FIRST_TARGET, last + 1)
        if [entry['target_year'] for entry in entries] != list(years):
            raise RuntimeError('Missing, duplicate or unsorted forecast cases')
        for entry in entries:
            target = entry['target_year']
            if entry['origin'] != target - 1 or entry['observed'] != counts[target - FIRST]:
                raise RuntimeError('Forecast origin or observation differs from the catalogue calendar')
    for history in (*histories, 'expanding_from_1964') if modern else histories:
        matched = [groups[series_key(model, history, 1.0 if model == 'gaussian_ar' else None)] for model in CORE_MODELS]
        for forecasts in zip(*matched):
            dates = {(item['training_start_year'], item['first_response_year'], item['training_responses'], item['applied_history'])
                     for item in forecasts}
            if len(dates) != 1 or forecasts[0]['predicted_mean'] != forecasts[2]['predicted_mean']:
                raise RuntimeError('Core histories or IID Poisson/response mean do not match')
            _, _, iid_poisson, poisson_ar, iid_nb, nb_ar = forecasts
            for left, right, same in ((iid_nb, iid_poisson, iid_nb['dispersion'] == 0),
                                      (nb_ar, poisson_ar, nb_ar['fit_source'] == 'exact_poisson_ar'),
                                      (nb_ar, iid_nb, nb_ar['fit_source'] == 'exact_iid_nb2')):
                if same:
                    np.testing.assert_allclose([left['predicted_mean'], left['nll'], left['rps']],
                                               [right['predicted_mean'], right['nll'], right['rps']], atol=1e-12, rtol=0)
    for exp, infinity in zip(groups[series_key('response_mean', 'expanding')],
                             groups[('discounted_mean', 'fixed_discount', None, 'infinity')]):
        if exp['predicted_mean'] != infinity['predicted_mean']:
            raise RuntimeError('Infinite half-life differs from expanding response mean')
    return {'forecast_rows': len(rows), 'forecast_series': len(groups),
            'negative_means': sum(entry['negative_mean'] for entry in rows),
            'count_forecast_rows': sum('nll' in entry for entry in rows),
            'max_pmf_cap': max(entry['cap'] for entry in rows if 'nll' in entry),
            'max_tail_first_moment_upper': max(entry['tail_first_moment_upper'] for entry in rows if 'nll' in entry),
            'max_mass_discrepancy': max(entry['mass_discrepancy'] for entry in rows if 'nll' in entry),
            'max_mean_discrepancy': max(entry['mean_discrepancy'] for entry in rows if 'nll' in entry)}


def check_calendars_and_selection():
    values = np.arange(108)
    at_origin = 1977
    np.testing.assert_allclose([response_mean(values, at_origin, key) for key in HISTORIES], [50, 45, 40, 30])
    for key, length in HISTORIES.items():
        start, lag, response = training(values, at_origin, key)
        expected_start = at_origin - length + 1 if length else FIRST
        assert start == expected_start
        np.testing.assert_array_equal(lag, np.arange(start, at_origin) - FIRST)
        np.testing.assert_array_equal(response, np.arange(start + 1, at_origin + 1) - FIRST)
    start, lag, response = training(values, 1993, 'expanding_from_1964')
    assert (start, len(lag), len(response)) == (1964, 29, 29)
    np.testing.assert_array_equal(response, training(values, 1993, 'recent_30')[2])
    np.testing.assert_allclose(discounted_mean(np.array([999, 2, 6]), 1920, 1), 14 / 3, atol=1e-12, rtol=0)
    fixed = {origin: {key: response_mean(values, origin, key) for key in HISTORIES} for origin in range(1957, 1977)}
    discount = {origin: {key: discounted_mean(values, origin, half) for key, half in HALF_LIVES.items()}
                for origin in fixed}
    audit = select_history(values, at_origin, fixed, discount)
    assert audit['selected_history'] == 'recent_20'
    assert [row['target_year'] for row in audit['inner_forecasts']] == list(range(1958, 1978))
    np.testing.assert_allclose(list(audit['history_inner_mse'].values()), [100, 225, 400, 620.875])
    assert choose_longer_tie({'short': 3, 'long': 3 + 0.5e-10, 'longest': 3 + 1.5e-10}) == 'long'
    assert choose_longer_tie(dict.fromkeys(HALF_LIVES, 0)) == 'infinity'
    return {'earliest_outer_origin': at_origin, 'inner_target_span': [1958, 1977],
            'hand_calculated_inner_mse': audit['history_inner_mse'], 'modern_calendar': 'pass'}


def check_pmf_and_hac():
    grid = np.arange(65, dtype=float)
    np.testing.assert_allclose(log_count_mass(grid, 12.3, 0.0), poisson.logpmf(grid, 12.3), atol=1e-12, rtol=0)
    np.testing.assert_allclose(log_count_mass(grid, 12.3, 0.3), nbinom.logpmf(grid, 1 / 0.3, 1 / (1 + 12.3 * 0.3)),
                               atol=1e-12, rtol=0)
    log_limit = log_count_mass(grid, 12.3, 1e-15)
    np.testing.assert_allclose(log_limit, log_count_mass(grid, 12.3, 0.0), atol=2e-12, rtol=0)
    for mean, alpha in ((0, 0), (4, 0), (20, 1e-12), (3, 2), (20, 0.7), (40, 3)):
        fit = count_pmf(mean, alpha)
        smaller = pmf_at_cap(mean, alpha, fit['comparison_cap'])
        cap = smaller['cap']
        omitted = fit['mass'][cap + 1:]
        assert omitted.sum() <= smaller['tail_mass_upper'] + 1e-15
        assert omitted @ fit['grid'][cap + 1:] <= smaller['tail_first_moment_upper'] + 1e-15
    assert count_scores(0, 0, 0)['rps'] == 0
    np.testing.assert_allclose(pit_bins(0.05, 0.25), [0.25, 0.50, 0.25, *([0] * 7)], atol=1e-15, rtol=0)
    np.testing.assert_allclose(hac([1, 2, 3], 0)['variance'], 2 / 9, atol=1e-15, rtol=0)
    np.testing.assert_allclose(hac([1, 3, 1, 3], 1)['variance'], 1 / 16, atol=1e-15, rtol=0)
    np.testing.assert_allclose(hac([1, 3, 1, 3], 3)['variance'], 1 / 16, atol=1e-15, rtol=0)
    assert influence([1, 2, 3], [0, 3, 6], 1)['minimum']['mean_loss_difference'] == 1.5
    np.testing.assert_allclose(acf([1, 3, 1, 3], 2)['1'], -0.75, atol=1e-15, rtol=0)
    return {'pmf_reference_and_bounds': 'pass', 'pit_integration': 'pass', 'hac_hand_calculations': 'pass',
            'influence_and_acf': 'pass', 'nb_high_precision': ar.check_nb_numerics()}


def check_leakage_and_candidates():
    # Exactly one outer origin exercises all forecasting entry points and cached reuse.
    counts = 3 + (np.arange(61) * 13 + np.arange(61) // 3) % 11
    perturb = counts.copy()
    perturb[-1] += 23
    cache = {}
    rows, fits, audits = all_forecasts('synthetic', counts, cache)
    updated, updated_fits, updated_audits = all_forecasts('synthetic', perturb, cache)
    assert len(cache) == len(HISTORIES)
    assert (rows, fits, audits) == all_forecasts('synthetic', counts, {})
    assert fits == updated_fits and audits == updated_audits
    for first, second in zip(rows, updated):
        assert first['target_year'] == second['target_year'] == 1978
        assert first['predicted_mean'] == second['predicted_mean']
        assert first['observed'] + 23 == second['observed']
        if 'nll' in first:
            assert first['cap'] == second['cap'] and first['nll'] != second['nll']
    assert len(rows) == 48
    return {'outer_origin': 1977, 'synthetic_rows': len(rows), 'unique_history_fits': len(cache),
            'future_perturbation': 'pass', 'repeated_new_fits': 'pass'}


def check_match_rule():
    base = datetime(2001, 1, 1, tzinfo=timezone.utc)

    def event(identity, seconds, longitude=0, magnitude=7.1, aliases=()):
        return {'id': identity, 'when': base + timedelta(seconds=seconds), 'lat': 0,
                'lon': longitude, 'mag': magnitude, 'magnitude_type': 'test', 'aliases': aliases, 'catalogue': 'main'}

    usgs = [event('a', 0, aliases=('iscgem1',)), event('b', 1000), event('c', 2000), event('d', 2030),
            event('e', 9000, aliases=('iscgem4',)), event('f', 10000, aliases=('iscgem4',)), event('g', 15000)]
    isc = [event('1', 10000, magnitude=6.9), event('2', 1120, 1.0, None), event('3', 2010), event('4', 9000)]
    result = match_events(usgs, isc)
    assert result['summary'] == {'usgs_qualifying': 7, 'isc_all_considered': 4, 'isc_qualifying': 2,
                                 'exact_isc_identifier': 1, 'mutually_unique_time_distance': 1,
                                 'ambiguous_time_distance': 2, 'ambiguous_exact_identifier': 2, 'unmatched': 1,
                                 'matched_isc_missing_mw': 1, 'matched_isc_mw_below_7': 1,
                                 'isc_qualifying_without_unique_usgs_qualifier': 2}
    return {'explicit_conflicts_and_threshold_crossing': 'pass'}


def checks_only():
    return {'calendar_and_selection': check_calendars_and_selection(), 'scoring': check_pmf_and_hac(),
            'leakage_and_candidate_fits': check_leakage_and_candidates(), 'matching': check_match_rule()}


def csv_bytes(rows):
    output = io.StringIO(newline='')
    columns = list(dict.fromkeys(key for entry in rows for key in entry))
    writer = csv.DictWriter(output, columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def scientific_digest(result):
    return ar.scientific_digest(result)


def save_exports(result, forecasts):
    result_path = ROOT / 'time_series_results.json'
    prediction_path = ROOT / 'time_series_predictions.csv'
    if result_path.exists():
        previous = read_json(result_path.name)
        if scientific_digest(previous) != result['scientific_sha256'] or previous['scientific_sha256'] != result['scientific_sha256']:
            raise RuntimeError('Rerun differs from the existing scientific JSON')
    if prediction_path.exists() and prediction_path.read_bytes() != forecasts:
        raise RuntimeError('Rerun differs from the existing prediction CSV')
    if not prediction_path.exists():
        with prediction_path.open('xb') as stream:
            stream.write(forecasts)
    if not result_path.exists():
        with result_path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, separators=(',', ':'), allow_nan=False)
            stream.write('\n')


def settings():
    return {'status': 'Exploratory protocol recorded before these fits with earlier USGS results already known',
            'outer_targets': [FIRST_TARGET, 2025], 'common_end': COMMON_END, 'first_year': FIRST,
            'modern_training_start': MODERN_START, 'modern_target_span': [MODERN_FIRST_TARGET, COMMON_END],
            'fixed_histories_annual_observations': HISTORIES, 'half_lives_annual': HALF_LIVES,
            'inner_realized_forecasts': INNER_LENGTH, 'selector_loss': 'Response mean squared error',
            'tie_rule': 'Longest candidate within absolute tolerance of the minimum', 'tie_tolerance': TIE,
            'gaussian_sd_floors': FLOORS, 'main_gaussian_sd_floor': 1.0, 'gaussian_sd_upper': 100.0,
            'gaussian_affine_bounds': ar.AFFINE_BOUNDS, 'gaussian_mixture_logit_bounds': ar.MIXTURE_LOGIT_BOUNDS,
            'count_bounds': {'poisson_ar': ar.original.BOUNDS['poisson_ar'], 'iid_nb2': ar.original.BOUNDS['iid_nb2'],
                             'nb2_ar': ar.NBAR_BOUNDS},
            'count_nb2_variance': 'mu + alpha*mu^2 with exact Poisson alpha=0',
            'poisson_ar_starts': [-0.8, -0.4, 0.0, 0.2, 0.5, 0.8],
            'iid_nb2_starts': [0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 7.0, 9.0],
            'nb2_ar_starts': {'slope': ar.START_SLOPES, 'dispersion': ar.START_ALPHAS},
            'optimizer': {'method': 'L-BFGS-B', 'options': ar.OPTIMIZER, 'boundary_tolerance': ar.BOUNDARY_TOLERANCE,
                          'selection': 'Minimum training negative log likelihood over successful and exact nested candidates'},
            'prediction': 'Unrounded conditional means and count PMFs with fitted parameters held fixed',
            'count_pmf': {'geometric_omitted_first_moment_tolerance': TAIL_TOL, 'mass_and_scaled_mean_tolerance': PMF_TOL,
                          'integer_cap_doubling_first': 32, 'largest_checked_cap': 8192, 'quantiles': QUANTILES,
                          'scoring': 'Unnormalized direct PMF on integer support. RPS summation uses the larger cap.'},
            'pit': {'bins': 10, 'contribution': 'Uniform over the observation CDF interval'},
            'hac': {'bartlett_lags': HAC_LAGS, 'normal_multiplier': 1.96,
                    'restriction': 'Period-specific lags must be smaller than n',
                    'interpretation': 'Approximate pointwise intervals under weak dependence and stable mean loss difference'},
            'influence': {'one_target': True, 'contiguous_five_targets': True, 'refit': False,
                          'restriction': 'A scored target must remain after omission'},
            'event_match': {'maximum_seconds': 120, 'maximum_km': 200, 'haversine_radius_km': 6371.0088,
                            'priority': 'Exact ISC source identifier, otherwise mutually unique time and distance candidates'},
            'acf': {'annual_count_max_lag': 10, 'forecast_error_max_lag': 4, 'denominator': 'Centered sum of squares'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checks-only', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    checks = checks_only()
    if args.checks_only:
        print(json.dumps(checks, indent=2))
        return
    inputs = frozen_inputs()
    code_sha256 = sha256(Path(__file__).read_bytes())
    usgs_events, usgs = load_usgs()
    isc, isc_snapshot = load_isc()
    series = {**usgs, **isc}
    matching = match_events([event for event in usgs_events if event['when'].year <= COMMON_END], load_matching_rows())
    catalogue_results = {}
    cache = {}
    all_rows = []
    for catalogue, counts in series.items():
        rows, fits, selections = all_forecasts(catalogue, counts, cache)
        groups = defaultdict(list)
        for entry in rows:
            groups[group_key(entry)].append(entry)
        modern = catalogue in MAIN_CATALOGUES
        validation = validate_rows(rows, groups, counts, modern)
        last = FIRST + len(counts) - 1
        catalogue_results[catalogue] = {'raw_series': raw_diagnostics(counts, last),
                                        'validation': validation, 'scores': summarize(groups, last),
                                        'paired_contrasts': contrasts(groups, last, modern),
                                        'selection_audit': selections, 'fits_by_origin_history': fits}
        all_rows.extend(rows)
        print(json.dumps({'catalogue': catalogue, **validation}), flush=True)
    forecasts = csv_bytes(all_rows)
    if frozen_inputs() != inputs or sha256(Path(__file__).read_bytes()) != code_sha256:
        raise RuntimeError('Inputs or source changed during the analysis')
    result = {'schema': 1, 'code_sha256': code_sha256, 'design_section_sha256': PLAN_SHA256,
              'original_private_protocol_sha256': ORIGINAL_PRIVATE_PROTOCOL_SHA256,
              'input_sha256': inputs, 'predictions_csv_sha256': sha256(forecasts), 'settings': settings(),
              'checks': checks, 'distinct_training_fits': len(cache), 'forecast_rows': len(all_rows),
              'data': catalogue_metadata(usgs_events, series, isc_snapshot, matching), 'catalogues': catalogue_results,
              'elapsed_seconds': time.perf_counter() - started,
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}}
    result['scientific_sha256'] = scientific_digest(result)
    save_exports(result, forecasts)
    print(json.dumps({'scientific_sha256': result['scientific_sha256'], 'forecast_rows': len(all_rows),
                      'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
