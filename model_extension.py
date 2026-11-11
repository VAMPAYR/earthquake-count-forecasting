"""Exploratory NB autoregression and one-step expert aggregation.
Run only after the design gate: OPENBLAS_NUM_THREADS=1 python model_extension.py.
Run mathematical unit checks without fits or study outcomes: --checks-only.
All original inputs and outputs are read-only. No network access or supplement use.
"""
import argparse
import hashlib
import inspect
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import nbinom, poisson

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    'analysis.py': '60751c1f1af0e10f83bedcb6490f4741c59dbd166eb8230b5d338d6b0d17e51f',
    'usgs_1918_2020_snapshot.json': 'ca7c18d7a69048cf94ce27c4aecf547fb9b8b6b9ce93acc41c774b9388a07623',
    'forecast_2020_frozen.json': 'c8f98250be896080bc9f1ce06d291e49230fad5c34e413f56327352a1dca618c',
}
EXPECTED_RESULTS_SCIENTIFIC_SHA256 = 'ae62a7f25275b53708d382face8057439c4fff67638beafb0c415b7bf8e83be5'
NBAR = 'nb2_ar'
BOUNDS = [(-8, 8), (-.95, .95), (1e-8, 10)]
START_SLOPES = [-.5, 0, .5]
START_ALPHAS = [.02, .1, .3, 1]
TOL = 1e-8


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def results_scientific_digest(result):
    # Runtime alone is excluded; every other original field remains checked.
    scientific = {key: value for key, value in result.items() if key != 'elapsed_seconds'}
    encoded = json.dumps(scientific, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def verify_inputs():
    actual = {name: digest(ROOT / name) for name in EXPECTED}
    if actual != EXPECTED:
        raise RuntimeError('Original frozen input hash differs')
    original = json.loads((ROOT / 'results.json').read_text(encoding='utf-8'))
    scientific = results_scientific_digest(original)
    if scientific != EXPECTED_RESULTS_SCIENTIFIC_SHA256:
        raise RuntimeError('Original results scientific content differs')
    actual['results.json'] = digest(ROOT / 'results.json')
    actual['results_scientific_sha256_excluding_elapsed_seconds'] = scientific
    return actual


# Importing analysis does not execute its guarded main function or refit models.
verify_inputs()
import analysis as base
EXPERTS = base.MODELS + [NBAR]


def nbar_objective(params, response, log_lag):
    a, b, alpha = params
    mu = np.exp(a + b * log_lag)
    if alpha == 0:
        return float(-poisson.logpmf(response, mu).sum())
    r = 1 / alpha
    return float(-nbinom.logpmf(response, r, r / (r + mu)).sum())


def fit_nbar(training, original_fits):
    response = np.asarray(training[1:], dtype=float)
    log_lag = np.log1p(training[:-1])
    ar = original_fits['poisson_ar']
    iid = original_fits['iid_nb2']
    nested = [([*ar['params'], 0.], ar['objective'], 'exact_poisson_ar'),
              ([iid['params'][0], 0., iid['params'][1]], iid['objective'], 'exact_iid_nb2')]
    starts = []
    candidates = []
    for params, objective, source in nested:
        recalculated = nbar_objective(params, response, log_lag)
        if abs(recalculated - objective) > 1e-7:
            raise RuntimeError('Nested candidate training likelihood mismatch')
        record = {'source': source, 'params': params, 'objective': recalculated,
                  'success': True, 'iterations': 0, 'boundary': ['alpha=0'] if params[2] == 0 else []}
        starts.append(record)
        candidates.append(record)
    for slope in START_SLOPES:
        for alpha in START_ALPHAS:
            initial = [math.log(response.mean()) - slope * log_lag.mean(), slope, alpha]
            result = minimize(nbar_objective, initial, args=(response, log_lag), method='L-BFGS-B',
                              bounds=BOUNDS, options={'maxiter': 1500, 'ftol': 1e-12,
                                                     'gtol': 1e-7, 'maxls': 40})
            boundary = [name for name, value, (lo, hi) in zip(['a', 'b', 'alpha'], result.x, BOUNDS)
                        if min(abs(value - lo), abs(value - hi)) < 1e-5]
            record = {'source': f'b={slope},alpha={alpha}', 'initial': initial,
                      'params': result.x.tolist(), 'objective': float(result.fun),
                      'success': bool(result.success), 'status': int(result.status),
                      'message': str(result.message), 'iterations': int(result.nit), 'boundary': boundary}
            starts.append(record)
            if result.success and np.isfinite(result.fun):
                candidates.append(record)
    best = min(candidates, key=lambda item: item['objective'])
    return {'params': best['params'], 'objective': best['objective'],
            'selected_source': best['source'], 'selected_boundary': best['boundary'],
            'starts': starts, 'failed_starts': sum(not item['success'] for item in starts),
            'training_responses': len(response)}


def nbar_kernel(params, lags, cap):
    a, b, alpha = params
    mu = np.exp(a + b * np.log1p(lags))
    grid = np.arange(cap + 1)
    if alpha == 0:
        return (poisson.pmf(grid[None, :], mu[:, None]), poisson.sf(cap, mu),
                mu * poisson.sf(cap - 1, mu))
    r = 1 / alpha
    p = r / (r + mu)
    return (nbinom.pmf(grid[None, :], r, p[:, None]), nbinom.sf(cap, r, p),
            mu * nbinom.sf(cap - 1, r + 1, p))


def nbar_predict_cap(params, lag, horizons, cap):
    matrix, tails, tail_moments = nbar_kernel(params, np.arange(cap + 1), cap)
    first, first_tail, first_moment = nbar_kernel(params, np.array([lag]), cap)
    q = first[0]
    missing = float(first_tail[0])
    upper_moment = float(first_moment[0])
    A = math.exp(params[0])
    forecasts = []
    grid = np.arange(cap + 1)
    for horizon in range(1, horizons + 1):
        if horizon > 1:
            newly_missing = float(q @ tails)
            new_moment = float(q @ tail_moments)
            upper_moment = A * (upper_moment + missing) + new_moment
            missing += newly_missing
            q = q @ matrix
        retained = float(q.sum())
        if not 0 < retained <= 1 + TOL or not np.all(np.isfinite(q)) or np.any(q < 0):
            raise RuntimeError('Invalid retained forecast probabilities')
        pmf = q / retained
        mean = float(pmf @ grid)
        forecasts.append({'pmf': pmf, 'retained_pmf': q.copy(), 'retained_mass': retained,
                          'escaped_mass': missing, 'missing_first_moment_upper': upper_moment,
                          'mean_error_bound': upper_moment + missing * mean,
                          'rps_error_bound': 2 * (cap + 1) * missing + upper_moment,
                          'cdf_error_bound': missing,
                          'roundoff_mass_discrepancy': abs(1 - retained - missing)})
    return forecasts


def nbar_forecast(params, lag, horizons):
    """Numerical cap uses fitted parameters and final training lag only."""
    cap = 64
    while cap <= 2048:
        small = nbar_predict_cap(params, lag, horizons, cap)
        large = nbar_predict_cap(params, lag, horizons, 2 * cap)
        passed = True
        for left, right in zip(small, large):
            lo = base.metrics(left['pmf'], 0)
            hi = base.metrics(right['pmf'], 0)
            check = {'mean': abs(lo['mean'] - hi['mean']), 'rps': abs(lo['rps'] - hi['rps']),
                     'quantiles_equal': all(lo[k] == hi[k] for k in ['median', 'interval80', 'interval95']),
                     'rps_reference_count': 0}
            right['doubling_check'] = check
            right['cap'] = 2 * cap
            if (right['escaped_mass'] > 1e-12 or right['mean_error_bound'] > TOL
                    or right['rps_error_bound'] > TOL or check['mean'] > TOL
                    or check['rps'] > TOL or not check['quantiles_equal']):
                passed = False
        if passed:
            for forecast in large:
                forecast['past_only_cap_checks_passed'] = True
            return large
        cap *= 2
    raise RuntimeError('NB autoregression cap tolerances unmet at cap 4096')


def score_forecast(forecast, target):
    if not 0 <= target < len(forecast['pmf']):
        raise RuntimeError('Observed count outside past-only cap')
    score = base.metrics(forecast['pmf'], target)
    q = float(forecast['retained_pmf'][target])
    if not (np.isfinite(q) and q > 0 and np.isfinite(score['probability'])
            and score['probability'] > 0 and np.isfinite(score['nll'])):
        raise RuntimeError('Invalid observed probability; no numerical floor')
    missing = forecast['escaped_mass']
    bracket = [-math.log(min(1, q + missing)), -math.log(q)]
    bound = abs(math.log(forecast['retained_mass'])) + math.log1p(missing / q)
    if bound > TOL or not bracket[0] - TOL <= score['nll'] <= bracket[1] + TOL:
        raise RuntimeError('Observed NLL numerical guard failed')
    diagnostics = {k: v for k, v in forecast.items() if k not in ['pmf', 'retained_pmf']}
    diagnostics.update({'true_nll_bracket': bracket, 'observed_nll_error_bound_with_mass_roundoff': bound,
                        'observed_probability_guard_passed': True, 'observed_nll_bracket_guard_passed': True,
                        'nll_guard_floating_tolerance': TOL})
    return score, diagnostics


def summarize(rows):
    result = {}
    for model in dict.fromkeys(row['model'] for row in rows):
        subset = [row for row in rows if row['model'] == model]
        result[model] = {key: float(np.mean([row[key] for row in subset]))
                         for key in ['nll', 'rps', 'absolute_error', 'coverage80', 'coverage95', 'width80', 'width95']}
        result[model]['rmse'] = float(np.sqrt(np.mean([row['squared_error'] for row in subset])))
        result[model]['cases'] = len(subset)
    return result


def unit_checks():
    original = json.loads((ROOT / 'results.json').read_text(encoding='utf-8'))
    elapsed_only = dict(original)
    elapsed_only['elapsed_seconds'] = original['elapsed_seconds'] + 1.
    assert results_scientific_digest(elapsed_only) == EXPECTED_RESULTS_SCIENTIFIC_SHA256
    changed_science = json.loads(json.dumps(original))
    changed_science['annual_counts']['1918'] += 1
    assert results_scientific_digest(changed_science) != EXPECTED_RESULTS_SCIENTIFIC_SHA256
    params = [1.2, .3, 0.]
    nb = nbar_forecast(params, 12, 3)
    ar = base.forecast('poisson_ar', params[:2], 12, 3)
    assert max(np.max(abs(a['pmf'] - b['pmf'])) for a, b in zip(nb, ar)) < 1e-12
    params = [math.log(12), 0., .1]
    nb = nbar_forecast(params, 12, 3)
    iid = base.forecast('iid_nb2', [params[0], params[2]], 12, 3)
    assert max(np.max(abs(a['pmf'] - b['pmf'])) for a, b in zip(nb, iid)) < 1e-12
    params = [1.1, .3, .1]
    first = nbar_forecast(params, 12, 1)[0]['pmf']
    mu = math.exp(params[0] + params[1] * math.log(13))
    exact = nbinom.pmf(np.arange(len(first)), 10, 10 / (10 + mu))
    assert np.max(abs(first - exact)) < 1e-12
    assert list(inspect.signature(nbar_forecast).parameters) == ['params', 'lag', 'horizons']
    for origin in range(1977, 2013):
        length = origin - 1918 + 1
        assert length >= 60 and 1918 + length - 1 == origin and origin + 8 <= 2020
    # A synthetic two-step sequence verifies that only a scored outcome changes weights.
    probabilities = np.array([[.1, .2, .3, .4, .5, .6], [.6, .5, .4, .3, .2, .1]])
    log_weights = np.full(6, -math.log(6))
    total = 0.
    for assigned in probabilities:
        normalizer = logsumexp(log_weights + np.log(assigned))
        total -= normalizer
        log_weights += np.log(assigned) - normalizer
    assert abs(total + math.log(np.prod(probabilities, axis=0).mean())) < 1e-12
    return {name: 'passed' for name in ['alpha_zero_equals_poisson_ar', 'slope_zero_equals_iid_nb2',
                                       'one_step_analytic_nb', 'calendar_alignment', 'no_outcome_forecast_argument',
                                       'synthetic_sequential_weight_identity', 'results_elapsed_only_change_accepted',
                                       'results_scientific_change_rejected']}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def aggregate(one_step):
    log_weights = np.full(6, -math.log(6))
    expert_losses = np.zeros(6)
    adaptive_total = 0.
    rows, path = [], []
    for origin in range(1977, 2013):
        forecasts, target = one_step[origin]
        length = max(len(forecasts[m]['pmf']) for m in EXPERTS)
        matrix = np.vstack([np.pad(forecasts[m]['pmf'], (0, length - len(forecasts[m]['pmf']))) for m in EXPERTS])
        before = np.exp(log_weights)
        assigned = matrix[:, target]
        if np.any(~np.isfinite(assigned)) or np.any(assigned <= 0):
            raise RuntimeError('Invalid expert assigned probability')
        period_rows = []
        for name, weights in [('uniform_six', np.full(6, 1 / 6)), ('likelihood_weighted_six', before)]:
            pmf = weights @ matrix
            normalizer = float(pmf.sum())
            if abs(normalizer - 1) > TOL:
                raise RuntimeError('Aggregate probability normalization failed')
            pmf /= normalizer
            score = base.metrics(pmf, target)
            if not np.isfinite(score['nll']) or score['probability'] <= 0:
                raise RuntimeError('Invalid combination observed probability')
            raw_probability = float(sum(w * forecasts[m]['retained_pmf'][target]
                                        for w, m in zip(weights, EXPERTS)))
            missing = float(sum(w * forecasts[m]['escaped_mass'] for w, m in zip(weights, EXPERTS)))
            if not np.isfinite(raw_probability) or raw_probability <= 0:
                raise RuntimeError('Invalid aggregate retained-path probability')
            bracket = [-math.log(min(1, raw_probability + missing)), -math.log(raw_probability)]
            log_error_bound = abs(math.log(score['probability'] / raw_probability)) + math.log1p(missing / raw_probability)
            if log_error_bound > TOL or not bracket[0] - TOL <= score['nll'] <= bracket[1] + TOL:
                raise RuntimeError('Aggregate observed NLL numerical guard failed')
            row = {'model': name, 'origin': origin, 'horizon': 1, 'target_year': origin + 1,
                   'target': target, **score, 'pmf': pmf.tolist(), 'weights': weights.tolist(),
                   'numerical_mixture_normalizer': normalizer,
                   'numerics': {'true_nll_bracket': bracket, 'nll_error_bound_with_mass_roundoff': log_error_bound,
                                'observed_probability_guard_passed': True, 'observed_nll_bracket_guard_passed': True,
                                'floating_tolerance': TOL},
                   'weighted_expert_escaped_mass': float(sum(w * forecasts[m]['escaped_mass'] for w, m in zip(weights, EXPERTS)))}
            rows.append(row)
            period_rows.append(row)
        # The observed outcome is used only after both mixtures have been constructed.
        log_normalizer = float(logsumexp(log_weights + np.log(assigned)))
        if abs(period_rows[1]['nll'] + log_normalizer) > TOL:
            raise RuntimeError('Mixture score differs from log-domain update')
        adaptive_total += period_rows[1]['nll']
        expert_losses -= np.log(assigned)
        log_weights += np.log(assigned) - log_normalizer
        identity = float(math.log(6) - logsumexp(-expert_losses))
        slack = float(expert_losses.min() + math.log(6) - adaptive_total)
        if abs(adaptive_total - identity) > TOL or slack < -TOL:
            raise RuntimeError('Standard expert aggregation identity or bound failed')
        path.append({'origin': origin, 'target_year': origin + 1, 'target': target,
                     'weights_before': before.tolist(), 'weights_after': np.exp(log_weights).tolist(),
                     'expert_assigned_probabilities': assigned.tolist(),
                     'expert_pmfs': {m: matrix[j].tolist() for j, m in enumerate(EXPERTS)},
                     'expert_cumulative_nll': expert_losses.tolist(), 'adaptive_cumulative_nll': adaptive_total,
                     'log_mixture_identity': identity, 'identity_absolute_error': abs(adaptive_total - identity),
                     'best_expert_plus_log6_slack': slack,
                     'paired_nll_difference_adaptive_minus_uniform': period_rows[1]['nll'] - period_rows[0]['nll'],
                     'paired_rps_difference_adaptive_minus_uniform': period_rows[1]['rps'] - period_rows[0]['rps']})
    return {'expert_order': EXPERTS, 'rows': rows, 'weight_path': path, 'summary': summarize(rows),
            'fixed_origin_split': {label: summarize([r for r in rows if condition(r['origin'])])
                                   for label, condition in [('first18', lambda o: o <= 1994), ('last18', lambda o: o >= 1995)]},
            'fixed_expert_summary': {model: {'cumulative_nll': float(expert_losses[j]), 'mean_nll': float(expert_losses[j] / 36)}
                                     for j, model in enumerate(EXPERTS)},
            'checks': {'telescoping_identity': 'passed', 'best_fixed_expert_plus_log6': 'passed',
                       'outcomes_updated_once_in_calendar_order': 'passed'},
            'scope': 'Standard likelihood-weighted expert aggregation at horizon one; normalized numerical expert PMFs. The log(6) bound concerns cumulative log loss, not every expert, equal weighting, longer horizons, or statistical calibration.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checks-only', action='store_true')
    args = parser.parse_args()
    checks = unit_checks()
    if args.checks_only:
        print(json.dumps(checks, indent=2))
        return
    started = time.time()
    inputs = verify_inputs()
    original = json.loads((ROOT / 'results.json').read_text(encoding='utf-8'))
    counts = base.read_counts()
    old_rows = {(r['model'], r['origin'], r['horizon']): r for r in original['rows']}
    fields = ['probability', 'nll', 'rps', 'mean', 'median', 'interval80', 'interval95', 'squared_error',
              'absolute_error', 'coverage80', 'coverage95', 'width80', 'width95']
    reconstructed, max_difference = 0, 0.
    rows, fits, paired, one_step = [], {}, [], {}
    for origin in range(1977, 2013):
        training = counts[:origin - 1918 + 1]
        old_fits = original['fits'][str(origin)]
        forecasts = {model: base.forecast(model, old_fits[model]['params'], int(training[-1]), 8)
                     for model in base.MODELS}
        fit = fit_nbar(training, old_fits)
        fits[str(origin)] = fit
        forecasts[NBAR] = nbar_forecast(fit['params'], int(training[-1]), 8)
        # Every PMF has now been constructed without the held-out outcomes.
        targets = counts[origin - 1918 + 1:origin - 1918 + 9]
        for model in base.MODELS:
            for h, (forecast, target) in enumerate(zip(forecasts[model], targets), 1):
                score, _ = score_forecast(forecast, int(target))
                old = old_rows[(model, origin, h)]
                for field in fields:
                    if isinstance(old[field], list) or isinstance(old[field], bool):
                        if score[field] != old[field]:
                            raise RuntimeError('Original discrete score reconstruction differs')
                    else:
                        difference = abs(score[field] - old[field])
                        max_difference = max(max_difference, difference)
                        if difference > 1e-10:
                            raise RuntimeError('Original score reconstruction differs')
                reconstructed += 1
        for h, (forecast, target) in enumerate(zip(forecasts[NBAR], targets), 1):
            score, diagnostics = score_forecast(forecast, int(target))
            rows.append({'model': NBAR, 'origin': origin, 'horizon': h, 'target_year': origin + h,
                         'target': int(target), **score, 'pmf': forecast['pmf'].tolist(), 'numerics': diagnostics})
            paired.append({'origin': origin, 'horizon': h, 'target_year': origin + h,
                           'differences': {model: {metric: score[metric] - old_rows[(model, origin, h)][metric]
                                                   for metric in ['nll', 'rps', 'squared_error', 'absolute_error']}
                                           for model in ['iid_nb2', 'poisson_ar']}})
        one_step[origin] = ({model: forecasts[model][0] for model in EXPERTS}, int(targets[0]))
        print(f'origin {origin} complete ({time.time() - started:.2f}s)', flush=True)
    if reconstructed != 1440 or len(rows) != 288:
        raise RuntimeError('Incomplete model reconstruction or NB AR forecasts')
    aggregation = aggregate(one_step)
    if verify_inputs() != inputs:
        raise RuntimeError('Frozen inputs changed during run')
    result = {'scope': 'Exploratory extension specified after historical and supplementary outcomes had been seen. No new supplementary analysis, inference, or calibration claims.',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'code_sha256': digest(__file__),
              'input_sha256': inputs, 'plan_sha256': digest(ROOT / 'PLAN.md'), 'unit_checks': checks,
              'settings': {'origins': [1977, 2012], 'horizons': [1, 8], 'bounds': BOUNDS,
                           'exact_alpha_zero_candidate': 'saved original Poisson AR fit',
                           'exact_iid_candidate': 'saved original iid NB2 fit', 'start_slopes': START_SLOPES,
                           'start_alphas': START_ALPHAS, 'number_of_optimizer_starts': 12,
                           'optimizer': 'L-BFGS-B finite-difference derivatives; successful starts plus exact nested candidates; no global maximum guarantee',
                           'training_responses': 'Second through final training year, identical to original fits',
                           'numerical_tolerance': TOL, 'escaped_mass_tolerance': 1e-12,
                           'numerics': 'Normalized retained-path distributions; M/U exact-arithmetic bounds with cap doubling. Floating point mass discrepancy is reported, not a certified rounding bound.',
                           'aggregation': 'Six equal prior weights; one-step likelihood update with learning rate one, no sharing or tuning; each 1978-2013 outcome updates once.'},
              'original_reconstruction': {'records': reconstructed, 'maximum_metric_absolute_difference': max_difference,
                                          'all_original_metric_checks': 'passed', 'refits_of_original_models': 0},
              'fits': fits, 'rows': rows, 'summary': summarize(rows),
              'by_horizon': {str(h): summarize([r for r in rows if r['horizon'] == h]) for h in range(1, 9)},
              'fixed_origin_split': {'first18': summarize([r for r in rows if r['origin'] <= 1994]),
                                     'last18': summarize([r for r in rows if r['origin'] >= 1995])},
              'paired_score_differences': paired, 'aggregation': aggregation, 'elapsed_seconds': time.time() - started}
    (ROOT / 'model_extension_results.json').write_text(json.dumps(jsonable(result), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'nb_ar': result['summary'], 'aggregation': aggregation['summary'],
                      'original_reconstruction': result['original_reconstruction']}, indent=2))


if __name__ == '__main__':
    main()
