"""Generate manuscript tables and the figure from the saved evaluation outputs."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
result = json.loads((ROOT / 'results.json').read_text(encoding='utf-8'))
added = json.loads((ROOT / 'model_extension_results.json').read_text(encoding='utf-8'))
result['summary'].update(added['summary'])
for horizon in result['by_horizon']:
    result['by_horizon'][horizon].update(added['by_horizon'][horizon])
names = {
    'iid_poisson': 'Independent Poisson',
    'iid_nb2': 'Independent NB',
    'iid_poisson_mix': 'Independent mixture',
    'poisson_ar': 'Poisson AR',
    'nb2_ar': 'Negative-binomial AR',
    'poisson_ar_mix': 'Poisson AR mixture',
}
models = list(names)
lines = [r'\begin{table}[htbp]', r'\centering\small',
         r'\caption{Pooled losses and 95\% interval summaries over 288 dependent cases per model. Lower loss is better. Coverage is a percentage; width is the difference between integer endpoints.}',
         r'\label{tab:pooled}', r'\begin{tabular}{lrrrrrr}', r'\toprule',
         r'Model & NLL & RPS & RMSE & MAE & Coverage & Width \\', r'\midrule']
for model in models:
    s = result['summary'][model]
    lines.append(f"{names[model]} & {s['nll']:.4f} & {s['rps']:.4f} & {s['rmse']:.3f} & {s['absolute_error']:.3f} & {100*s['coverage95']:.1f} & {s['width95']:.2f} " + r'\\')
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}',
          r'\begin{table}[htbp]', r'\centering\small',
          r'\caption{Mean NLL at each horizon, with 36 common origins in each row. The first three models are independent. NB denotes negative binomial; mixture denotes a two-Poisson mixture.}',
          r'\label{tab:horizons}', r'\begin{tabular}{rrrrrrr}', r'\toprule',
          r'Horizon & Poisson & NB & Mixture & Poisson AR & NB AR & AR mixture \\', r'\midrule']
for h, scores in result['by_horizon'].items():
    lines.append(h + ' & ' + ' & '.join(f"{scores[m]['nll']:.4f}" for m in models) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
(ROOT / 'tables.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False})
historical_rows = result['rows'] + added['rows']
fig, axes = plt.subplots(3, 2, figsize=(6.5, 5.2), sharex=True, sharey=True,
                         layout='constrained')
diagnostics = {}
for ax, model in zip(axes.flat, models):
    rows = sorted((r for r in historical_rows if r['model'] == model and r['horizon'] == 1),
                  key=lambda r: r['target_year'])
    assert len(rows) == 36 and [r['target_year'] for r in rows] == list(range(1978, 2014))
    years = [r['target_year'] for r in rows]
    ax.fill_between(years, [r['interval95'][0] for r in rows], [r['interval95'][1] for r in rows],
                    color='#0072B2', alpha=.13, label='95% interval')
    ax.plot(years, [r['target'] for r in rows], color='0.15', marker='o', markersize=2.4,
            linewidth=.75, label='Observed')
    ax.plot(years, [r['mean'] for r in rows], color='#D55E00', linewidth=1.5, label='Forecast mean')
    ax.set(title=names[model], xlim=(1978, 2013), ylim=(0, 27), xticks=[1980, 1990, 2000, 2010])
    diagnostics[model] = {
        'cases': len(rows),
        'mean_error_forecast_minus_observed': sum(r['mean'] - r['target'] for r in rows) / len(rows),
        'mean_forecast_min': min(r['mean'] for r in rows),
        'mean_forecast_max': max(r['mean'] for r in rows),
        'observed_min': min(r['target'] for r in rows),
        'observed_max': max(r['target'] for r in rows),
    }
for ax in axes[:, 0]:
    ax.set_ylabel('Annual count')
for ax in axes[-1, :]:
    ax.set_xlabel('Target year')
axes[0, 0].legend(loc='upper left', fontsize=6.5, frameon=False, ncol=1)
fig.savefig(ROOT / 'origin_comparison.png', dpi=400)
plt.close(fig)
print('Generated tables.tex and origin_comparison.png from original and added results')

extension = json.loads((ROOT / 'extension_results.json').read_text(encoding='utf-8'))
supplement_models = list(extension['summary'])
lines = [r'\begin{table}[htbp]', r'\centering\small',
         r'\caption{Observed counts and forecast means for all five completed years after the saved 2020 origin. Each model uses only data through 2020. The first three models are independent; the final two use the previous count. The last two rows give average losses, not counts.}',
         r'\label{tab:extension}', r'\begin{tabular}{lrrrrrr}', r'\toprule',
         r'Year & Observed & Poisson & NB & Mixture & Poisson AR & AR mixture \\', r'\midrule']
for year in range(2021, 2026):
    rows = {r['model']: r for r in extension['rows'] if r['target_year'] == year}
    observed = {r['target'] for r in rows.values()}
    assert len(observed) == 1 and set(rows) == set(supplement_models)
    lines.append(f"{year} & {observed.pop()} & " + ' & '.join(f"{rows[m]['mean']:.2f}" for m in supplement_models) + r' \\')
lines.append(r'\midrule')
for metric, label in [('nll', 'Mean NLL'), ('rps', 'Mean RPS')]:
    lines.append(label + ' & & ' + ' & '.join(f"{extension['summary'][m][metric]:.4f}" for m in supplement_models) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
(ROOT / 'extension_table.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('Generated extension_table.tex from extension_results.json')

aggregation = added['aggregation']
lines = [r'\begin{table}[htbp]', r'\centering\small',
         r'\caption{Six-model combinations for 36 one-year forecasts. Both use the same individual models and observations. Period columns report mean NLL over the first or last 18 origins.}',
         r'\label{tab:combinations}', r'\begin{tabular}{lrrrr}', r'\toprule',
         r'Combination & NLL & RPS & First 18 & Last 18 \\', r'\midrule']
for model, label in [('uniform_six', 'Equal weights'), ('likelihood_weighted_six', 'Updated weights')]:
    s = aggregation['summary'][model]
    first = aggregation['fixed_origin_split']['first18'][model]['nll']
    last = aggregation['fixed_origin_split']['last18'][model]['nll']
    lines.append(f"{label} & {s['nll']:.4f} & {s['rps']:.4f} & {first:.4f} & {last:.4f} " + r'\\')
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
(ROOT / 'combination_table.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('Generated combination_table.tex from model_extension_results.json')

fields = ['evaluation', 'model', 'origin', 'target_year', 'horizon', 'observed_count',
          'forecast_mean', 'forecast_median', 'error_mean_minus_observed',
          'absolute_error_of_mean', 'absolute_error_of_median', 'squared_error_of_mean',
          'lower95', 'upper95', 'covered95', 'observed_count_probability', 'nll', 'rps']
csv_rows = []
for evaluation, source in [('historical', historical_rows), ('combination', aggregation['rows']),
                           ('frozen_2020_supplement', extension['rows'])]:
    for r in source:
        assert r['origin'] + r['horizon'] == r['target_year']
        assert abs((r['mean'] - r['target'])**2 - r['squared_error']) < 1e-10
        csv_rows.append(dict(zip(fields, [evaluation, r['model'], r['origin'], r['target_year'],
            r['horizon'], r['target'], r['mean'], r['median'], r['mean'] - r['target'],
            abs(r['mean'] - r['target']), r['absolute_error'], r['squared_error'],
            r['interval95'][0], r['interval95'][1], r['coverage95'], r['probability'], r['nll'], r['rps']])))
assert len(csv_rows) == 1825
with (ROOT / 'observed_vs_predicted.csv').open('w', newline='', encoding='utf-8-sig') as out:
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(csv_rows)
(ROOT / 'forecast_diagnostics.json').write_text(json.dumps({
    'scope': 'Descriptive one-year forecast errors, targets 1978-2013; no refitting or new observations.',
    'mean_error_sign': 'Forecast mean minus observed count; negative means average underprediction.',
    'one_year': diagnostics, 'csv_rows': len(csv_rows)}, indent=2) + '\n', encoding='utf-8')
print('Generated observed_vs_predicted.csv and forecast_diagnostics.json from saved forecasts')
