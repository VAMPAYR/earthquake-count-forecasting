"""Reassess the printed forecasts and the original Gaussian AR families offline.

Run with OPENBLAS_NUM_THREADS=1 and python -B ar_reassessment.py.
Use --checks-only to verify the mathematics without fitting any catalogue data.
On a rerun, the script verifies the existing exports against the new calculation.
"""

import argparse
import csv
from decimal import Decimal, localcontext
import hashlib
import io
import json
import math
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit, gammaln

import analysis as original
from extension import load_snapshot, verify_freeze
from model_extension import BOUNDS as NBAR_BOUNDS
from model_extension import NBAR, START_ALPHAS, START_SLOPES


ROOT = Path(__file__).resolve().parent
FIRST_YEAR = 1918
LAST_YEAR = 2025
ORIGINS = range(1977, 2025)
PERIODS = ((1978, 2025), (1978, 2000), (2001, 2020), (2021, 2025))
WINDOWS = ('expanding', 'recent_30')
FLOORS = (1.0, 0.5, 2.0)
GAUSSIAN_MODELS = ('gaussian_ar', 'gaussian_ar_mix')
AFFINE_BOUNDS = ((-100.0, 100.0), (-0.95, 0.95))
AFFINE_NAMES = ('intercept', 'slope')
GAUSSIAN_NAMES = (*AFFINE_NAMES, 'sd')
MIXTURE_NAMES = ('intercept_1', 'slope_1', 'sd_1', 'intercept_2', 'slope_2', 'sd_2', 'logit_weight_1')
MIXTURE_LOGIT_BOUNDS = (-12.0, 12.0)
LOG_TWO_PI = math.log(2 * math.pi)
OPTIMIZER = {'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-7, 'maxls': 40}
BOUNDARY_TOLERANCE = 1e-5
NB_G_COEFFICIENTS = tuple((-1.0) ** degree * (degree + 1) / (degree + 2) for degree in range(9))
PLAN_HEADING = '## Original AR-family reassessment, 11 November 2026'
PLAN_START = 'First, score '
PLAN_END = 'Countmodel comparisons in this new stage are one-year only.'
PRE_FIX = {
    'reason': 'The previous negative-binomial log mass suffered cancellation near the Poisson limit. The revised local likelihood and derivatives use log1p and an analytic limit. The model, bounds, starts, exact nested candidates and selection rule are retained.',
    'archive': 'before-ar-numerical-fix-2026-11-11.zip',
    'archive_sha256': '99e8d709980e199d84a6825a041f742dff0d124d88ecb560284884c86bed80c9',
    'code_sha256': '7f0c16348e9af156b6df80ce4dd15d9c8d54dd755c23ff3b00b44f38c100b5db',
    'original_full_plan_section_sha256': '2895c56e2fe21da7485026c1993e0ce0ac99fd6670e417958a669ab2877bdaf7',
    'results_json_sha256': 'bfbb516786ec3528a9bb0b9dafaf63b0625635d478c47303478915e23b6bb556',
    'scientific_sha256': '742b8f9fceb138854571d5bfeb29f23e6e575fc84f021e0d2e9247958b96df28',
    'predictions_csv_sha256': 'f942c23d47c1141782b015ce80bb97af0a2311c5f43064b9027cd72f871d9582',
}
PRINTED = {
    'printed_ar': (11.73, 10.36, 11.73, 11.15, 9.39),
    'printed_ar_mix': (9.0, 10.65, 8.98, 8.28, 12.41),
}
PRINTED_SOURCE = {
    'file': 'Time Series Analysis - EQ paper (1).pdf',
    'sha256': 'f151fb117a57358d0793b089120d12c1abbe82d680ab583600b4526ea0fbcb29',
    'document_date': '2024-08-08',
    'printed_ar_table': 2,
    'printed_ar_mix_table': 4,
    'forecast_issuance_verified': False,
    'evaluation_target': 'Reconstructed USGS preferred-magnitude >=7 series',
    'target_caveat': 'The original PDF describes magnitude >7 on the Richter scale. Its original query is unavailable, so target definitions and catalogue vintages may differ.',
}
INPUT_HASHES = {
    'analysis.py': '60751c1f1af0e10f83bedcb6490f4741c59dbd166eb8230b5d338d6b0d17e51f',
    'extension.py': '50312f5a8c597d08779d61d1ac5136f6e577ee4891af32b1070601b7d30dff87',
    'model_extension.py': '81d3c5d6ce58b34f1e10c4a833213e2afd0af88e1e69976b01c1fa3de63e6ed0',
    'usgs_1918_2020_snapshot.json': 'ca7c18d7a69048cf94ce27c4aecf547fb9b8b6b9ce93acc41c774b9388a07623',
    'usgs_2021_2025_snapshot.json': '6a6183f50a414b6e82da1f46799ad4117fd2fab5375074ebf97612f4a3bcfe52',
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def plan_digest():
    plan = (ROOT / 'PLAN.md').read_text(encoding='utf-8-sig')
    section = plan.split(PLAN_HEADING, 1)[1].split('\n## ', 1)[0]
    start = section.index(PLAN_START)
    end = section.index(PLAN_END, start) + len(PLAN_END)
    return sha256(f'{PLAN_HEADING}\n\n{section[start:end]}\n'.encode('utf-8'))


def verify_inputs():
    hashes = {name: sha256((ROOT / name).read_bytes()) for name in INPUT_HASHES}
    if hashes != INPUT_HASHES:
        raise RuntimeError('Frozen model source or catalogue snapshot differs')
    return hashes


def read_counts():
    snapshot, recent = load_snapshot(False, verify_freeze()[1])
    counts = np.concatenate((original.read_counts(), [recent[str(year)] for year in range(2021, 2026)]))
    if counts.size != LAST_YEAR - FIRST_YEAR + 1:
        raise RuntimeError('Incomplete catalogue calendar')
    equal = dict.fromkeys(recent, 0)
    strict = dict.fromkeys(recent, 0)
    for event in snapshot['event_records']:
        year = event['time_utc'][:4]
        if event['magnitude'] == 7.0:
            equal[year] += 1
        elif event['magnitude'] > 7.0:
            strict[year] += 1
    if any(equal[year] + strict[year] != recent[year] for year in recent):
        raise RuntimeError('Magnitude boundary counts do not sum to the benchmark counts')
    return counts, {'preferred_magnitude_equal_7': equal, 'preferred_magnitude_above_7': strict}


def training_slice(counts, origin, window):
    end = origin - FIRST_YEAR + 1
    if window not in WINDOWS or not 30 <= end <= len(counts):
        raise ValueError('Invalid training window or origin')
    start = 0 if window == 'expanding' else end - 30
    return FIRST_YEAR + start, counts[start:end]


def boundaries(params, bounds, names):
    return [f'{name}={side}' for name, value, (lo, hi) in zip(names, params, bounds)
            for side, limit in (('lower', lo), ('upper', hi))
            if abs(value - limit) < BOUNDARY_TOLERANCE]


def bounded_optimizer(objective, initial, args, bounds, names, source, analytic):
    limits = np.asarray(bounds)
    initial = np.clip(initial, limits[:, 0], limits[:, 1])
    result = minimize(objective, initial, args=args, method='L-BFGS-B', jac=analytic,
                      bounds=bounds, options=OPTIMIZER)
    finite = np.isfinite(result.fun) and np.all(np.isfinite(result.x)) and np.all(np.isfinite(result.jac))
    gradient = np.asarray(result.jac)
    projected = result.x - np.clip(result.x - gradient, limits[:, 0], limits[:, 1])
    return {
        'source': source,
        'initial': initial.tolist(),
        'params': [float(value) if math.isfinite(value) else None for value in result.x],
        'objective': float(result.fun) if np.isfinite(result.fun) else None,
        'success': bool(result.success and finite),
        'optimizer_success': bool(result.success),
        'status': int(result.status),
        'message': str(result.message),
        'iterations': int(result.nit),
        'function_evaluations': int(result.nfev),
        'gradient_evaluations': int(result.njev),
        'projected_gradient_infinity_norm': float(np.max(np.abs(projected))) if finite else None,
        'boundary': boundaries(result.x, bounds, names),
    }


def exact_candidate(source, params, objective, boundary):
    return {'source': source, 'params': list(params), 'objective': float(objective),
            'success': True, 'iterations': 0, 'boundary': boundary}


def select_fit(starts):
    chosen = min((start for start in starts if start['success']), key=lambda start: start['objective'])
    return {'params': chosen['params'], 'objective': chosen['objective'],
            'selected_source': chosen['source'], 'selected_boundary': chosen['boundary'],
            'starts': starts, 'failed_starts': sum(not start['success'] for start in starts)}


def gaussian_terms(params, response, lag):
    intercept, slope, sd = params
    residual = (response - intercept - slope * lag) / sd
    log_density = -0.5 * (residual ** 2 + LOG_TWO_PI) - math.log(sd)
    # The derivative columns are for intercept, slope and standard deviation.
    gradient = np.array([residual, residual * lag, residual ** 2 - 1]) / sd
    return log_density, gradient


def single_objective(params, response, lag):
    log_density, gradient = gaussian_terms(params, response, lag)
    return float(-log_density.sum()), -gradient.sum(axis=1)


def mixture_objective(params, response, lag):
    logit = params[-1]
    density_1, score_1 = gaussian_terms(params[:3], response, lag)
    density_2, score_2 = gaussian_terms(params[3:6], response, lag)
    joint_1 = density_1 - np.logaddexp(0, -logit)
    joint_2 = density_2 - np.logaddexp(0, logit)
    log_density = np.logaddexp(joint_1, joint_2)
    responsibility_1 = np.exp(joint_1 - log_density)
    responsibility_2 = np.exp(joint_2 - log_density)
    # The weight score uses both responsibilities to avoid subtracting numbers near one.
    weight_score = (expit(-logit) * responsibility_1 - expit(logit) * responsibility_2).sum()
    gradient = -np.concatenate((score_1 @ responsibility_1, score_2 @ responsibility_2, [weight_score]))
    return float(-log_density.sum()), gradient


def constrained_least_squares(response, lag):
    design = np.column_stack((np.ones_like(lag), lag))
    limits = np.asarray(AFFINE_BOUNDS)
    unconstrained = np.linalg.lstsq(design, response, rcond=None)[0]
    # A constrained minimum outside the unconstrained solution lies on a box edge.
    candidates = []
    if np.all((limits[:, 0] <= unconstrained) & (unconstrained <= limits[:, 1])):
        candidates.append(unconstrained)
    for index, bound in enumerate(limits):
        free = 1 - index
        free_column = design[:, free]
        denominator = free_column @ free_column
        for fixed_value in bound:
            remainder = response - fixed_value * design[:, index]
            free_value = (free_column @ remainder) / denominator if denominator else 0.0
            params = np.empty(2)
            params[index] = fixed_value
            params[free] = np.clip(free_value, *limits[free])
            candidates.append(params)
    return min(candidates, key=lambda params: np.sum((response - design @ params) ** 2))


def fit_single(response, lag, floor):
    intercept, slope = constrained_least_squares(response, lag)
    residual = response - intercept - slope * lag
    sd = np.clip(np.sqrt(np.mean(residual ** 2)), floor, 100.0)
    params = [float(intercept), float(slope), float(sd)]
    bounds = (*AFFINE_BOUNDS, (floor, 100.0))
    objective = single_objective(params, response, lag)[0]
    return select_fit([exact_candidate('constrained_least_squares', params, objective,
                                       boundaries(params, bounds, GAUSSIAN_NAMES))])


def mixture_starts(single, response, lag):
    intercept, slope, sd = single
    center = lag.mean()
    residual = response - intercept - slope * lag
    for index, (shift_1, shift_2, scale_1, scale_2, logit) in enumerate((
        (-0.5, 0.5, 0.8, 1.2, 0.0), (-0.8, 0.4, 0.6, 1.0, 0.7),
        (-0.4, 0.8, 1.0, 0.6, -0.7), (-0.2, 0.2, 0.5, 1.6, 0.0),
    ), 1):
        yield f'perturbed_single_{index}', [intercept + shift_1 * sd, slope, scale_1 * sd,
                                            intercept + shift_2 * sd, slope, scale_2 * sd, logit]
    for index, quantiles in enumerate(((0.25, 0.75), (0.10, 0.90)), 1):
        low, high = np.quantile(residual, quantiles)
        yield f'residual_quantiles_{index}', [intercept + low, slope, sd * 0.75,
                                             intercept + high, slope, sd * 0.75, 0.0]
    for index, threshold in enumerate((0.0, float(np.median(residual))), 1):
        lower = residual <= threshold
        upper = ~lower
        components = []
        for subset in (lower, upper):
            group = residual[subset]
            shift = group.mean() if group.size else 0.0
            group_sd = group.std() if group.size else sd
            components.extend((intercept + shift, slope, group_sd))
        weight = np.clip(lower.mean(), 0.05, 0.95)
        yield f'residual_split_{index}', [*components, math.log(weight / (1.0 - weight))]
    for index, (slope_offset, level_offset, logit) in enumerate((
        (0.2, 0.25, 0.0), (-0.2, 0.25, 0.0), (0.5, 0.5, 0.7), (-0.5, 0.5, -0.7),
    ), 1):
        shift = slope_offset * center - level_offset * sd
        yield f'slope_split_{index}', [intercept + shift, slope - slope_offset, sd,
                                       intercept - shift, slope + slope_offset, sd, logit]


def fit_mixture(response, lag, floor, single):
    component_bounds = (*AFFINE_BOUNDS, (floor, 100.0))
    bounds = (*component_bounds, *component_bounds, MIXTURE_LOGIT_BOUNDS)
    nested = [*single['params'], *single['params'], 0.0]
    nested_objective = mixture_objective(nested, response, lag)[0]
    if not math.isclose(nested_objective, single['objective'], abs_tol=1e-10, rel_tol=0.0):
        raise RuntimeError('Duplicated Gaussian components differ from the single likelihood')
    starts = [exact_candidate('exact_duplicated_single', nested, nested_objective,
                              boundaries(nested, bounds, MIXTURE_NAMES))]
    for source, initial in mixture_starts(single['params'], response, lag):
        starts.append(bounded_optimizer(mixture_objective, initial, (response, lag),
                                        bounds, MIXTURE_NAMES, source, analytic=True))
    if len(starts) != 13:
        raise RuntimeError('Unexpected number of Gaussian mixture candidates')
    return select_fit(starts)


def nb_score_ratio(z):
    # This is [log(1 + z) - z / (1 + z)] / z², with its limit at zero.
    small = z < 0.01
    ratio = np.empty_like(z)
    ratio[small] = np.polynomial.polynomial.polyval(z[small], NB_G_COEFFICIENTS)
    large = z[~small]
    ratio[~small] = (np.log1p(large) - large / (1.0 + large)) / large ** 2
    return ratio


def nb_training(response):
    integer_response = response.astype(int)
    return integer_response, gammaln(response + 1.0), np.arange(integer_response.max(), dtype=float)


def stable_nb_ar_objective(params, log_lag, training):
    intercept, slope, alpha = params
    response, log_factorial, increments = training
    eta = intercept + slope * log_lag
    mu = np.exp(eta)
    if alpha == 0:
        negative_log_mass = mu - response * eta + log_factorial
        mean_score = mu - response
        dispersion_score = ((response - mu) ** 2 - response) / 2.0
    else:
        z = alpha * mu
        denominator = 1.0 + z
        rising_terms = np.concatenate(([0.0], np.cumsum(np.log1p(alpha * increments))))
        rising_scores = np.concatenate(([0.0], np.cumsum(increments / (1.0 + alpha * increments))))
        scaled_log = mu * (np.log1p(z) / z)
        negative_log_mass = (log_factorial - response * eta
                             + (1.0 + alpha * response) * scaled_log - rising_terms[response])
        mean_score = (mu - response) / denominator
        dispersion_score = rising_scores[response] + mu ** 2 * nb_score_ratio(z) - response * mu / denominator
    gradient = np.array([mean_score.sum(), mean_score @ log_lag, -dispersion_score.sum()])
    return float(negative_log_mass.sum()), gradient


def stable_iid_nb_objective(params, zero_lag, training):
    objective, gradient = stable_nb_ar_objective([params[0], 0.0, params[1]], zero_lag, training)
    return objective, gradient[[0, 2]]


def fit_count_models(response, lag):
    # Retain the original count-model bounds, starts and exact nested candidates.
    log_lag = np.log1p(lag)
    log_mean = math.log(response.mean())
    nested_objective = original.objective([log_mean, 0.0], 'poisson_ar', response, log_lag)[0]
    poisson_starts = [exact_candidate('exact_nested', [log_mean, 0.0], nested_objective, [])]
    for slope in (-0.8, -0.4, 0.0, 0.2, 0.5, 0.8):
        initial = [log_mean - slope * log_lag.mean(), slope]
        poisson_starts.append(bounded_optimizer(original.objective, initial, ('poisson_ar', response, log_lag),
                                                original.BOUNDS['poisson_ar'], AFFINE_NAMES,
                                                f'slope={slope}', analytic=True))
    poisson = select_fit(poisson_starts)
    training = nb_training(response)
    zero_lag = np.zeros_like(log_lag)
    nb_starts = [exact_candidate('exact_nested', [log_mean, 0.0], nested_objective, ['alpha=lower'])]
    for alpha in (0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 7.0, 9.0):
        nb_starts.append(bounded_optimizer(stable_iid_nb_objective, [log_mean, alpha], (zero_lag, training),
                                            original.BOUNDS['iid_nb2'], ('intercept', 'alpha'),
                                            f'alpha={alpha}', analytic=True))
    iid_nb = select_fit(nb_starts)
    nbar_starts = []
    names = (*AFFINE_NAMES, 'alpha')
    for source, params, reference in (
        ('exact_poisson_ar', [*poisson['params'], 0.0], poisson['objective']),
        ('exact_iid_nb2', [iid_nb['params'][0], 0.0, iid_nb['params'][1]], iid_nb['objective']),
    ):
        value = stable_nb_ar_objective(params, log_lag, training)[0]
        if not math.isclose(value, reference, rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError('Nested count candidate likelihood mismatch')
        bounds = (*NBAR_BOUNDS[:2], (0.0, NBAR_BOUNDS[2][1])) if params[2] == 0.0 else NBAR_BOUNDS
        nbar_starts.append(exact_candidate(source, params, value, boundaries(params, bounds, names)))
    for slope in START_SLOPES:
        for alpha in START_ALPHAS:
            initial = [log_mean - slope * log_lag.mean(), slope, alpha]
            nbar_starts.append(bounded_optimizer(stable_nb_ar_objective, initial, (log_lag, training),
                                                 NBAR_BOUNDS, names, f'b={slope},alpha={alpha}', analytic=True))
    nbar = select_fit(nbar_starts)
    nbar['training_responses'] = len(response)
    return {'poisson_ar': poisson, 'iid_nb2_nested_candidate': iid_nb, NBAR: nbar}


def affine_mean(model, params):
    if model == 'gaussian_ar':
        return tuple(params[:2])
    weight_1, weight_2 = expit(params[-1]), expit(-params[-1])
    return float(weight_1 * params[0] + weight_2 * params[3]), float(weight_1 * params[1] + weight_2 * params[4])


def mean_path(model, params, last_count, horizons):
    intercept, slope = affine_mean(model, params)
    path = []
    for _ in range(horizons):
        last_count = intercept + slope * last_count
        path.append(last_count)
    return path


def forecast_at_origin(counts, origin):
    lag_count = int(counts[origin - FIRST_YEAR])
    trailing_counts = counts[origin - FIRST_YEAR - 9:origin - FIRST_YEAR + 1]
    common = {'persistence': (float(lag_count), origin), 'trailing_10_mean': (float(trailing_counts.mean()), origin - 9)}
    windows = {}
    for window in WINDOWS:
        start_year, training = training_slice(counts, origin, window)
        lag = training[:-1].astype(float)
        response = training[1:].astype(float)
        count_fits = fit_count_models(response, lag)
        gaussians = {}
        for floor in FLOORS:
            single = fit_single(response, lag, floor)
            mixture = fit_mixture(response, lag, floor, single)
            gaussians[str(floor)] = dict(zip(GAUSSIAN_MODELS, (single, mixture)))
        windows[window] = {
            'training_start_year': start_year,
            'first_response_year': start_year + 1,
            'last_response_year': origin,
            'training_observations': int(training.size),
            'training_responses': int(response.size),
            'response_mean': float(response.mean()),
            'count_fits': count_fits,
            'gaussian_fits_by_sd_floor': gaussians,
        }
    return {'origin': origin, 'lag_count': lag_count, 'common_baselines': common, 'windows': windows}


def prediction_row(protocol, model, window, floor, origin, target_year, prediction, counts,
                   training_start_year=None, first_response_year=None, training_responses=None):
    observed = int(counts[target_year - FIRST_YEAR])
    error = prediction - observed
    return {'protocol': protocol, 'model': model, 'window': window, 'sd_floor': floor,
            'origin': origin, 'target_year': target_year,
            'horizon': target_year - origin if origin is not None else None,
            'training_start_year': training_start_year, 'first_response_year': first_response_year,
            'training_responses': training_responses, 'observed': observed, 'predicted_mean': prediction,
            'signed_error': error, 'squared_error': error ** 2, 'absolute_error': abs(error)}


def rolling_predictions(fits, counts):
    rows = []
    for origin, fit in fits.items():
        for model, (prediction, start) in fit['common_baselines'].items():
            rows.append(prediction_row('rolling', model, 'common', None, origin, origin + 1,
                                       prediction, counts, training_start_year=start))
        for window, fitted_window in fit['windows'].items():
            means = [('response_mean', None, fitted_window['response_mean'])]
            for model in ('poisson_ar', NBAR):
                intercept, slope = fitted_window['count_fits'][model]['params'][:2]
                means.append((model, None, math.exp(intercept + slope * math.log1p(fit['lag_count']))))
            for floor, gaussian_fits in fitted_window['gaussian_fits_by_sd_floor'].items():
                for model, gaussian in gaussian_fits.items():
                    prediction = mean_path(model, gaussian['params'], fit['lag_count'], 1)[0]
                    means.append((model, float(floor), prediction))
            for model, floor, prediction in means:
                rows.append(prediction_row('rolling', model, window, floor, origin, origin + 1, prediction, counts,
                                           fitted_window['training_start_year'], fitted_window['first_response_year'],
                                           fitted_window['training_responses']))
    return rows


def fixed_predictions(fit, counts):
    rows = []
    for window, fitted_window in fit['windows'].items():
        for model, gaussian in fitted_window['gaussian_fits_by_sd_floor']['1.0'].items():
            path = mean_path(model, gaussian['params'], fit['lag_count'], 5)
            for target_year, prediction in enumerate(path, 2021):
                rows.append(prediction_row('fixed_2020', model, window, 1.0, 2020, target_year, prediction, counts,
                                           fitted_window['training_start_year'], fitted_window['first_response_year'],
                                           fitted_window['training_responses']))
    return rows


def printed_predictions(counts):
    return [prediction_row('printed', model, None, None, None, target_year, prediction, counts)
            for model, values in PRINTED.items()
            for target_year, prediction in enumerate(values, 2021)]


def score(rows):
    n = len(rows)
    return {'n': n,
            'rmse': math.sqrt(math.fsum(row['squared_error'] for row in rows) / n),
            'mae': math.fsum(row['absolute_error'] for row in rows) / n,
            'mean_signed_error': math.fsum(row['signed_error'] for row in rows) / n,
            'negative_forecast_means': sum(row['predicted_mean'] < 0 for row in rows)}


def summarize(rows, periods):
    groups = {}
    for row in rows:
        key = (row['window'], row['model'], row['sd_floor'])
        groups.setdefault(key, []).append(row)
    return [
        {'period': f'{start}-{end}', 'window': window, 'model': model, 'sd_floor': floor,
         **score([row for row in group if start <= row['target_year'] <= end])}
        for start, end in periods for (window, model, floor), group in groups.items()
    ]


def optimizer_summary(fits):
    groups = {}
    for fit in fits.values():
        for window, fitted_window in fit['windows'].items():
            for model, count in fitted_window['count_fits'].items():
                groups.setdefault((window, model, None), []).append(count)
            for floor, gaussian_fits in fitted_window['gaussian_fits_by_sd_floor'].items():
                for model, gaussian in gaussian_fits.items():
                    groups.setdefault((window, model, float(floor)), []).append(gaussian)
    return [
        {'window': window, 'model': model, 'sd_floor': floor, 'fits': len(group),
         'optimizer_starts': sum('initial' in start for fit in group for start in fit['starts']),
         'failed_starts': sum(fit['failed_starts'] for fit in group),
         'selected_on_boundary': sum(bool(fit['selected_boundary']) for fit in group),
         'selected_exact_candidate': sum('initial' not in start for fit in group
                                         for start in fit['starts'] if start['source'] == fit['selected_source'])}
        for (window, model, floor), group in groups.items()
    ]


def check_gradient(objective, params, response, lag):
    _, analytic = objective(params, response, lag)
    perturbations = np.eye(len(params)) * 1e-5
    finite_difference = np.array([(objective(params + perturbation, response, lag)[0]
                                   - objective(params - perturbation, response, lag)[0]) / 2e-5
                                  for perturbation in perturbations])
    np.testing.assert_allclose(analytic, finite_difference, atol=1e-7, rtol=1e-7)
    return float(np.max(np.abs(analytic - finite_difference)))


def decimal_nb_objective(params, response, log_lag):
    # An independent high-precision calculation checks cancellation and gradients.
    a, b, alpha = params
    objective = Decimal(0)
    for count, log_x in zip(response, log_lag):
        eta = a + b * log_x
        mu = eta.exp()
        log_factorial = Decimal(math.factorial(count)).ln()
        if alpha:
            scale = (1 + alpha * mu).ln() / alpha
            rising = sum((1 + alpha * j).ln() for j in range(count))
            objective += log_factorial - count * eta + (1 + alpha * count) * scale - rising
        else:
            objective += log_factorial - count * eta + mu
    return objective


def check_nb_numerics():
    response = (0, 11, 24)
    log_lag = np.array([0.0, 2.5, 3.125])
    training = nb_training(np.asarray(response, dtype=float))
    intercept = math.log(12.3)
    # The two cases adjacent to 0.01/12.3 exercise both sides of the series cutoff.
    dispersions = (0.0, 1e-12, 1e-8, 1e-5, 0.01 / 12.3 * (1 - 1e-7),
                   0.01 / 12.3 * (1 + 1e-7), 0.01, 0.3, 10.0)
    objectives, gradients = [], []
    with localcontext() as context:
        context.prec = 70
        decimal_lag = tuple(Decimal(float(value)) for value in log_lag)
        step = Decimal('1e-20')
        for alpha in dispersions:
            params = [intercept, 0.35, alpha]
            actual, analytic = stable_nb_ar_objective(params, log_lag, training)
            base = [Decimal(value) for value in params]
            reference = decimal_nb_objective(base, response, decimal_lag)
            derivative = []
            for index in range(3):
                perturbed = base.copy()
                perturbed[index] += step
                above = decimal_nb_objective(perturbed, response, decimal_lag)
                if index == 2 and alpha == 0:
                    perturbed[index] += step
                    twice_above = decimal_nb_objective(perturbed, response, decimal_lag)
                    derivative.append(float((-3 * reference + 4 * above - twice_above) / (2 * step)))
                else:
                    perturbed[index] -= 2 * step
                    below = decimal_nb_objective(perturbed, response, decimal_lag)
                    derivative.append(float((above - below) / (2 * step)))
            objectives.append(abs(actual - float(reference)))
            gradients.append(float(np.max(np.abs(analytic - derivative))))
    sample = stable_iid_nb_objective([intercept, 1e-8], np.zeros(1), nb_training(np.array([11.0])))[0]
    expected = 2.196716006261794
    if max(objectives) > 2e-12 or max(gradients) > 2e-10 or abs(sample - expected) > 5e-14:
        raise RuntimeError('Stable NB likelihood or gradient differs from high-precision reference')
    return {'dispersions_checked': len(dispersions), 'max_objective_error': max(objectives),
            'max_gradient_error': max(gradients), 'near_poisson_reference_error': abs(sample - expected)}


def check_mean_recursion():
    params = [7.0, 0.25, 1.5, 4.0, -0.5, 2.0, -0.3]
    initial = 12.0
    values = np.array([initial])
    probabilities = np.array([1.0])
    actual = []
    # Two symmetric points integrate each affine conditional mean exactly.
    for _ in range(5):
        children = []
        weights = []
        for (intercept, slope, sd), weight in ((params[:3], expit(params[-1])), (params[3:6], expit(-params[-1]))):
            children.extend([intercept + slope * values - sd, intercept + slope * values + sd])
            weights.extend([weight * probabilities / 2.0] * 2)
        values = np.concatenate(children)
        probabilities = np.concatenate(weights)
        actual.append(float(values @ probabilities))
    expected = mean_path('gaussian_ar_mix', params, initial, 5)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)
    slope = params[1]
    expected_single = [slope ** step * initial + params[0] * (1 - slope ** step) / (1 - slope)
                       for step in range(1, 6)]
    np.testing.assert_allclose(mean_path('gaussian_ar', params[:3], initial, 5), expected_single,
                               atol=1e-12, rtol=0.0)
    return float(np.max(np.abs(np.asarray(actual) - expected)))


def check_calendar():
    years = np.arange(FIRST_YEAR, LAST_YEAR + 1)
    for origin in ORIGINS:
        for window in WINDOWS:
            start, training = training_slice(years, origin, window)
            expected_start = FIRST_YEAR if window == 'expanding' else origin - 29
            if start != expected_start or training[0] != start or training[-1] != origin:
                raise RuntimeError('Training window is off calendar')
            np.testing.assert_array_equal(training[:-1], np.arange(start, origin))
            np.testing.assert_array_equal(training[1:], np.arange(start + 1, origin + 1))
        if years[origin - FIRST_YEAR + 1] != origin + 1:
            raise RuntimeError('One-step target is off calendar')
    return len(ORIGINS) * len(WINDOWS)


def check_leakage_and_repeatability():
    rng = np.random.default_rng(48214)
    counts = rng.poisson(13, size=LAST_YEAR - FIRST_YEAR + 1)
    origin = 2000
    end = origin - FIRST_YEAR + 1
    original_fit = forecast_at_origin(counts, origin)
    perturbed = counts.copy()
    perturbed[end:] += 1000
    if forecast_at_origin(perturbed, origin) != original_fit:
        raise RuntimeError('A future perturbation changed a past-only fit')
    if forecast_at_origin(counts[:end], origin) != original_fit:
        raise RuntimeError('Truncation changed a past-only fit')
    if forecast_at_origin(counts, origin) != original_fit:
        raise RuntimeError('Repeated fits differ')
    predictions = rolling_predictions({origin: original_fit}, counts)
    changed_predictions = rolling_predictions({origin: original_fit}, perturbed)
    if any(row['predicted_mean'] != changed['predicted_mean'] or row['observed'] == changed['observed']
           for row, changed in zip(predictions, changed_predictions)):
        raise RuntimeError('Target values affected forecast means or failed to enter scoring')
    return {'synthetic_origin': origin, 'future_perturbation': 'pass', 'truncation': 'pass', 'repeatability': 'pass'}


def unit_checks():
    lag = np.array([7., 11., 13., 16., 20.])
    response = np.array([9., 16., 12., 21., 14.])
    single = [7.0, 0.4, 2.5]
    mixture = [*single, 5.0, 0.6, 1.5, -0.6]
    collapsed = [*single, *single, 0.3]
    np.testing.assert_allclose(mixture_objective(collapsed, response, lag)[0],
                               single_objective(single, response, lag)[0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(mean_path('gaussian_ar_mix', collapsed, 12.0, 5),
                               mean_path('gaussian_ar', single, 12.0, 5), atol=1e-12, rtol=0.0)
    # Exercise both box faces and the SD floor with simple exact fits.
    np.testing.assert_allclose(constrained_least_squares(150 + 0.2 * lag, lag), [100., 0.95])
    np.testing.assert_allclose(constrained_least_squares(7 + 1.1 * lag, lag),
                               [7 + (1.1 - 0.95) * lag.mean(), 0.95])
    for floor in FLOORS:
        np.testing.assert_allclose(fit_single(7 + 0.2 * lag, lag, floor)['params'], [7., 0.2, floor])
    return {'single_gradient_max_absolute_error': check_gradient(single_objective, np.array(single), response, lag),
            'mixture_gradient_max_absolute_error': check_gradient(mixture_objective, np.array(mixture), response, lag),
            'negative_binomial_high_precision_checks': check_nb_numerics(),
            'collapsed_mixture_likelihood_and_means': 'pass',
            'constrained_least_squares_and_sd_floor': 'pass',
            'mean_recursion_max_absolute_error': check_mean_recursion(),
            'calendar_origin_windows_checked': check_calendar(),
            'synthetic_leakage_and_repeatability': check_leakage_and_repeatability()}


def validate_predictions(rolling, fixed, printed):
    rolling_groups = {(window, model, None)
                      for window in WINDOWS
                      for model in ('response_mean', 'poisson_ar', NBAR)}
    rolling_groups.update((window, model, floor) for window in WINDOWS for model in GAUSSIAN_MODELS for floor in FLOORS)
    rolling_groups.update(('common', model, None) for model in ('persistence', 'trailing_10_mean'))
    rolling_expected = {(window, model, floor, origin + 1) for window, model, floor in rolling_groups for origin in ORIGINS}
    fixed_expected = {(window, model, 1.0, year)
                      for window in WINDOWS for model in GAUSSIAN_MODELS for year in range(2021, 2026)}
    printed_expected = {(None, model, None, year) for model in PRINTED for year in range(2021, 2026)}
    for rows, expected in ((rolling, rolling_expected), (fixed, fixed_expected), (printed, printed_expected)):
        actual = {(row['window'], row['model'], row['sd_floor'], row['target_year']) for row in rows}
        if actual != expected or len(rows) != len(expected):
            raise RuntimeError('Missing, duplicate, or unexpected forecast cases')
        if any(not math.isfinite(row['predicted_mean']) for row in rows):
            raise RuntimeError('Nonfinite point forecasts')
    rolling_2021 = {(row['window'], row['model']): row['predicted_mean'] for row in rolling
                    if row['target_year'] == 2021 and row['sd_floor'] == 1.0}
    for row in fixed:
        if row['target_year'] == 2021 and row['predicted_mean'] != rolling_2021[(row['window'], row['model'])]:
            raise RuntimeError('Fixed and rolling one-step forecasts from 2020 differ')
    return {'rolling_forecasts': len(rolling), 'fixed_2020_forecasts': len(fixed), 'printed_forecasts': len(printed),
            'fixed_2020_first_step_matches_rolling': True}


def csv_bytes(rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8')


def scientific_digest(result):
    deterministic = {key: value for key, value in result.items()
                     if key not in ('elapsed_seconds', 'scientific_sha256', 'runtime')}
    return sha256(json.dumps(deterministic, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8'))


def save_exports(result, prediction_bytes, replace_pre_fix):
    result_path = ROOT / 'ar_reassessment_results.json'
    prediction_path = ROOT / 'ar_reassessment_predictions.csv'
    if replace_pre_fix:
        archive = ROOT / PRE_FIX['archive']
        if sha256(archive.read_bytes()) != PRE_FIX['archive_sha256']:
            raise RuntimeError('Pre-fix archive digest differs')
        with zipfile.ZipFile(archive) as saved:
            for path in (result_path, prediction_path):
                if saved.read(path.name) != path.read_bytes():
                    raise RuntimeError(f'{path.name} differs from its archived pre-fix copy')
        result_bytes = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode('utf-8')
        prediction_path.write_bytes(prediction_bytes)
        result_path.write_bytes(result_bytes)
        return
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding='utf-8'))
        if scientific_digest(previous) != result['scientific_sha256'] or previous['scientific_sha256'] != result['scientific_sha256']:
            raise RuntimeError('Rerun differs from the existing scientific result')
    if prediction_path.exists() and prediction_path.read_bytes() != prediction_bytes:
        raise RuntimeError('Rerun differs from the existing forecast CSV')
    if not prediction_path.exists():
        with prediction_path.open('xb') as output:
            output.write(prediction_bytes)
    if not result_path.exists():
        with result_path.open('x', encoding='utf-8', newline='\n') as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checks-only', action='store_true')
    parser.add_argument('--replace-pre-fix', action='store_true',
                        help='Replace the archived pre-fix exports after verifying them against their archive.')
    args = parser.parse_args()
    started = time.perf_counter()
    checks = unit_checks()
    if args.checks_only:
        print(json.dumps(checks, indent=2))
        return

    inputs = verify_inputs()
    plan_hash = plan_digest()
    code_hash = sha256(Path(__file__).read_bytes())
    counts, magnitude_boundary_counts = read_counts()
    fits = {origin: forecast_at_origin(counts, origin) for origin in ORIGINS}
    rolling = rolling_predictions(fits, counts)
    fixed = fixed_predictions(fits[2020], counts)
    printed = printed_predictions(counts)
    checks['catalogue_cases'] = validate_predictions(rolling, fixed, printed)
    if verify_inputs() != inputs or plan_digest() != plan_hash or sha256(Path(__file__).read_bytes()) != code_hash:
        raise RuntimeError('Inputs, plan, or code changed during the run')
    predictions = csv_bytes(rolling + fixed + printed)
    result = {
        'code_sha256': code_hash, 'design_section_sha256': plan_hash, 'input_sha256': inputs,
        'predictions_csv_sha256': sha256(predictions),
        'negative_binomial_numerical_correction': PRE_FIX,
        'original_printed_forecast_source': {**PRINTED_SOURCE, 'benchmark_magnitude_boundary_counts': magnitude_boundary_counts},
        'settings': {
            'scope': 'Exploratory analysis using previously seen outcomes and constraints fixed before these fits',
            'fit': 'Constrained conditional likelihood with constant-weight iid Gaussian allocations',
            'forecast': 'Conditional mean, with no clipping and no parameter-uncertainty integration',
            'origins': [ORIGINS.start, ORIGINS.stop - 1],
            'training': {'expanding_start_year': FIRST_YEAR, 'recent_observations': 30, 'recent_pairs': 29,
                         'response_years': 'Second through final training year for each window'},
            'sd_floors': list(FLOORS), 'main_sd_floor': FLOORS[0], 'sd_upper': 100.0,
            'intercept_bounds': AFFINE_BOUNDS[0], 'slope_bounds': AFFINE_BOUNDS[1],
            'mixture_logit_bounds': MIXTURE_LOGIT_BOUNDS, 'mixture_optimizer_starts': 12,
            'optimizer': {'method': 'L-BFGS-B', 'options': OPTIMIZER,
                          'boundary_tolerance': BOUNDARY_TOLERANCE,
                          'selection': 'Successful finite optimizer candidates and exact nested candidate, by training likelihood',
                          'global_optimum_claim': False},
            'nb2_numerics': 'Integer log mass using log1p and analytic derivatives, with the exact Poisson candidate at zero dispersion',
            'primary_loss': 'RMSE of means', 'secondary_losses': ['MAE of the same means', 'Mean forecast minus observation'],
        },
        'annual_counts': {str(year): int(count) for year, count in enumerate(counts, FIRST_YEAR)},
        'checks': checks,
        'summaries': {'rolling': summarize(rolling, PERIODS),
                      'fixed_2020': summarize(fixed, (PERIODS[-1],)),
                      'printed': summarize(printed, (PERIODS[-1],))},
        'optimizer_summary': optimizer_summary(fits),
        'fits': {str(origin): fit for origin, fit in fits.items()},
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'elapsed_seconds': time.perf_counter() - started,
    }
    result['scientific_sha256'] = scientific_digest(result)
    save_exports(result, predictions, args.replace_pre_fix)
    print(json.dumps({key: result[key] for key in ('code_sha256', 'design_section_sha256', 'scientific_sha256',
                                                    'predictions_csv_sha256', 'elapsed_seconds', 'checks')}, indent=2))


if __name__ == '__main__':
    main()
