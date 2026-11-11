"""Build manuscript displays from the saved, reviewed time-series results."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
MAIN = ('usgs_ge_7', 'isc_main_ge_7')
NAMES = {'usgs_ge_7': 'USGS', 'isc_main_ge_7': 'ISC-GEM',
         'usgs_gt_7': 'USGS, magnitude >7', 'usgs_ge_7_5': 'USGS, magnitude >=7.5',
         'isc_main_plus_supplementary_ge_7': 'ISC-GEM, including supplement'}
MODELS = [('response_mean', 'Response mean'), ('gaussian_ar', 'Gaussian AR'),
          ('gaussian_ar_mix', 'Gaussian AR mixture'), ('poisson_ar', 'Poisson AR'),
          ('iid_nb2', 'Independent negative binomial'), ('nb2_ar', 'Negative-binomial AR')]
HISTORIES = ('expanding', 'recent_30', 'selected')
PERIOD = '1978-2022'


def score(data, catalogue, model, history, period=PERIOD, floor=None):
    if model.startswith('gaussian_') and floor is None:
        floor = 1.0
    matches = [r for r in data['catalogues'][catalogue]['scores']
               if (r['model'], r['history'], r['period'], r['sd_floor'])
               == (model, history, period, floor)]
    if len(matches) != 1:
        raise ValueError(f'Nonunique score: {catalogue, model, history, period, floor}')
    return matches[0]


def contrast(data, catalogue, method, history, reference, ref_history, period=PERIOD):
    matches = [r for r in data['catalogues'][catalogue]['paired_contrasts']
               if (r['method']['model'], r['method']['history'], r['reference']['model'],
                   r['reference']['history'], r['period'], r['loss'])
               == (method, history, reference, ref_history, period, 'squared_error')]
    if len(matches) != 1:
        raise ValueError('Nonunique contrast')
    return matches[0]


def table(filename, columns, title, label, header, body, note):
    text = '\n'.join([
        r'\begin{table}[!htb]', r'\centering', r'\small',
        r'\caption{' + title + '}', r'\label{' + label + '}',
        r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{' + columns + '}',
        r'\toprule', *header, r'\midrule', *body, r'\bottomrule',
        r'\end{tabular}', r'\par\smallskip', r'\begin{minipage}{\linewidth}\footnotesize',
        note, r'\end{minipage}', r'\end{table}', '',
    ])
    (ROOT / filename).write_text(text, encoding='utf-8', newline='\n')


def build_tables(data):
    body = []
    for model, label in MODELS:
        cells = [label]
        for catalogue in MAIN:
            for history in HISTORIES:
                cells.append('--' if model == 'gaussian_ar_mix' and history == 'selected'
                             else f"{score(data, catalogue, model, history)['rmse']:.3f}")
        body.append(' & '.join(cells) + r' \\')
    table('time_series_point_table.tex', 'lrrrrrr',
          'One-year point accuracy on the 45 common target years, 1978--2022.', 'tab:point',
          [r' & \multicolumn{3}{c}{USGS} & \multicolumn{3}{c}{ISC-GEM} \\',
           r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
           r'Model & Expanding & Recent 30 & Selected & Expanding & Recent 30 & Selected \\'],
          body,
          'Entries are RMSE in earthquakes per year. Smaller is better. All models use the same '
          'response years within each history. Selected history minimizes the response mean\'s '
          'mean squared error over 20 earlier forecasts and is shared across models. The Gaussian mixture uses '
          'a standard-deviation floor of one. It has no selected-history arm. Independent Poisson '
          'has exactly the response-mean point forecast.')

    specifications = [
        ('Recent mean minus expanding mean', 'response_mean', 'recent_30', 'response_mean', 'expanding'),
        ('Selected mean minus expanding mean', 'response_mean', 'selected', 'response_mean', 'expanding'),
        ('Selected mean minus recent mean', 'response_mean', 'selected', 'response_mean', 'recent_30'),
        ('Recent AR minus recent mean', 'gaussian_ar', 'recent_30', 'response_mean', 'recent_30'),
        ('Recent mixture minus recent AR', 'gaussian_ar_mix', 'recent_30', 'gaussian_ar', 'recent_30'),
    ]
    body = []
    for label, model, history, ref, ref_history in specifications:
        cells = [label]
        for catalogue in MAIN:
            value = contrast(data, catalogue, model, history, ref, ref_history)
            low, high = next(x for x in value['hac'] if x['lag'] == 3)['interval95']
            cells.extend((f"{value['mean_loss_difference']:.3f}", f'[{low:.3f}, {high:.3f}]'))
        body.append(' & '.join(cells) + r' \\')
    table('time_series_contrast_table.tex', 'lrrrr',
          'Paired squared-error differences on the common target years.', 'tab:contrasts',
          [r' & \multicolumn{2}{c}{USGS} & \multicolumn{2}{c}{ISC-GEM} \\',
           r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
           r'Comparison & Difference & 95\% interval & Difference & 95\% interval \\'],
          body,
          'Negative differences favor the first forecast. Units are squared earthquake counts. '
          'Recent denotes the 30-observation history, and AR denotes Gaussian AR. Intervals use Bartlett covariance weights through lag three '
          'and a normal multiplier of 1.96. They assume a stable mean loss difference and weak '
          'dependence. They do not adjust for model selection or multiple comparisons. '
          'Lag sensitivities and omitted-year/block checks accompany the saved results.')

    body = []
    labels = [('iid_poisson', 'Independent Poisson'), ('iid_nb2', 'Independent negative binomial'),
              ('poisson_ar', 'Poisson AR'), ('nb2_ar', 'Negative-binomial AR')]
    for catalogue in MAIN:
        body.append(r'\multicolumn{7}{l}{\textit{' + NAMES[catalogue] + r'}} \\')
        for model, label in labels:
            value = score(data, catalogue, model, 'recent_30')
            body.append(' & '.join([label, f"{value['nll']:.3f}", f"{value['rps']:.3f}",
                                   f"{100 * value['covered80']:.1f}", f"{value['width80']:.2f}",
                                   f"{100 * value['covered95']:.1f}", f"{value['width95']:.2f}"]) + r' \\')
    table('time_series_count_table.tex', 'lrrrrrr',
          'Count-distribution forecasts with recent-30 training, 1978--2022.', 'tab:count',
          [r' & & & \multicolumn{2}{c}{80\% interval} & \multicolumn{2}{c}{95\% interval} \\',
           r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
           r'Model & NLL & RPS & Covered (\%) & Width & Covered (\%) & Width \\'],
          body,
          'NLL is the negative log probability assigned to the observed count. RPS is the ranked '
          'probability score. Smaller scores are better. Coverage counts both integer endpoints, '
          'and width is the difference between them. These fitted-parameter intervals omit '
          'parameter-estimation uncertainty. Scores for all other histories are in the code release.')

    body = []
    for catalogue in data['catalogues']:
        label = NAMES.get(catalogue, catalogue).replace('>=', r'$\geq$').replace('>7', r'$>7$')
        cells = [label]
        for history in ('expanding', 'recent_20', 'recent_30', 'recent_40', 'selected'):
            cells.append(f"{score(data, catalogue, 'response_mean', history)['rmse']:.3f}")
        body.append(' & '.join(cells) + r' \\')
    table('time_series_sensitivity_table.tex', 'lrrrrr',
          'Response-mean RMSE across histories and target definitions, 1978--2022.', 'tab:sensitivity',
          [r'Target & Expanding & Recent 20 & Recent 30 & Recent 40 & Selected \\'], body,
          'Each row is fitted and scored against its own annual count series. Compare histories '
          'within a row: changing the magnitude threshold changes the outcome and its scale. '
          'The ISC-GEM supplement adds only records with a recorded magnitude at least seven. '
          'Records without a magnitude remain unclassified.')


def build_figures(data, forecasts):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': 0.2, 'savefig.dpi': 300})
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 4.3), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]}, layout='constrained')
    colors = ('#2166ac', '#b35806')
    for catalogue, color in zip(MAIN, colors):
        series = data['catalogues'][catalogue]['raw_series']['annual_counts']
        years, counts = zip(*[(int(y), count) for y, count in series.items()])
        axes[0].plot(years, counts, color=color, lw=1, label=NAMES[catalogue])
    differences = data['data']['usgs_minus_isc_main_ge_7_annual']
    axes[1].bar([int(y) for y in differences], list(differences.values()), color='#555555', width=0.8)
    for ax in axes:
        ax.axvline(1977.5, color='black', linestyle=':', lw=1)
        ax.set_xlim(1917, 2026)
    axes[0].set_ylabel('Annual count')
    axes[0].legend(frameon=False, ncol=2)
    axes[1].set_ylabel('USGS minus\nISC-GEM')
    axes[1].axhline(0, color='black', lw=0.6)
    axes[1].set_xlabel('UTC calendar year')
    fig.savefig(ROOT / 'time_series_catalogues.png', metadata={'Software': 'Matplotlib'})
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(7.1, 7.0), sharex='col', sharey='row', layout='constrained')
    upper = max(float(r['observed']) for r in forecasts
                if r['catalogue'] in MAIN and int(r['target_year']) <= 2022) + 3
    for col, catalogue in enumerate(MAIN):
        def rows(history):
            return [r for r in forecasts if r['catalogue'] == catalogue and r['model'] == 'response_mean'
                    and r['history'] == history and int(r['target_year']) <= 2022]
        values = rows('expanding')
        years = np.array([int(r['target_year']) for r in values])
        axes[0, col].scatter(years, [float(r['observed']) for r in values], color='black', s=12,
                             label='Observed', zorder=3)
        for history, color, label, style in (('expanding', '#2166ac', 'Expanding', '-'),
                                             ('recent_30', '#b35806', 'Recent 30', '-'),
                                             ('selected', '#238b45', 'Selected', '--')):
            axes[0, col].plot(years, [float(r['predicted_mean']) for r in rows(history)],
                              color=color, lw=1.2, label=label, linestyle=style)
        axes[0, col].set_title(NAMES[catalogue])
        axes[0, col].set_ylim(0, upper)
        for history, color, label in (('recent_30', '#b35806', 'Recent 30 minus expanding'),
                                      ('selected', '#238b45', 'Selected minus expanding')):
            c = contrast(data, catalogue, 'response_mean', history, 'response_mean', 'expanding')
            axes[1, col].plot(years, c['cumulative_loss_differences'], color=color, lw=1.3, label=label)
        axes[1, col].axhline(0, color='black', lw=0.7)
        selections = [r for r in data['catalogues'][catalogue]['selection_audit'] if r['origin'] < 2022]
        codes = {'recent_20': 0, 'recent_30': 1, 'recent_40': 2, 'expanding': 3}
        axes[2, col].step([r['origin'] + 1 for r in selections],
                          [codes[r['selected_history']] for r in selections], where='mid',
                          color='#238b45', lw=1.2)
        axes[2, col].set_yticks(range(4), ['20', '30', '40', 'Expanding'])
        axes[2, col].set_ylim(-0.3, 3.3)
        axes[2, col].set_xlabel('Forecast target year')
        for ax in axes[:, col]:
            ax.set_xlim(1977, 2023)
            ax.set_xticks([1980, 1990, 2000, 2010, 2020])
    axes[0, 0].set_ylabel('Count / forecast mean')
    axes[1, 0].set_ylabel('Cumulative squared-error\ndifference')
    axes[2, 0].set_ylabel('Selected history\n(annual observations)')
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=7, loc='upper left')
    axes[1, 0].legend(frameon=False, fontsize=6.5, loc='best')
    fig.savefig(ROOT / 'time_series_forecasts.png', metadata={'Software': 'Matplotlib'})
    plt.close(fig)


def build_recent_table(forecasts):
    body = []
    for year in range(2021, 2026):
        cells = [str(year)]
        for model, history in [('response_mean', 'expanding'), ('response_mean', 'recent_30'),
                               ('response_mean', 'selected'), ('gaussian_ar', 'recent_30')]:
            values = [r for r in forecasts if r['catalogue'] == 'usgs_ge_7'
                      and r['model'] == model and r['history'] == history
                      and int(r['target_year']) == year and r['sd_floor'] in ('', '1.0')]
            if len(values) != 1:
                raise ValueError('Nonunique recent forecast')
            if len(cells) == 1:
                cells.append(values[0]['observed'])
            cells.append(f"{float(values[0]['predicted_mean']):.2f}")
        body.append(' & '.join(cells) + r' \\')
    table('time_series_recent_table.tex', 'rrrrrr',
          'Recent USGS observations and annually updated one-year forecast means.', 'tab:recent',
          [r'Year & Observed & Expanding mean & Recent-30 mean & Selected mean & Recent-30 AR \\'],
          body,
          'Each forecast uses counts through the preceding year. Selected history uses only '
          'earlier realized forecast errors. AR is Gaussian autoregression. These are conditional '
          'means, rather than forecasts issued at a common fixed origin in 2020.')


def main():
    data = json.loads((ROOT / 'time_series_results.json').read_text(encoding='utf-8'))
    with (ROOT / 'time_series_predictions.csv').open(newline='', encoding='utf-8') as stream:
        forecasts = list(csv.DictReader(stream))
    build_tables(data)
    build_figures(data, forecasts)
    build_recent_table(forecasts)
    print('Built five tables and two figures from saved time-series results.')


if __name__ == '__main__':
    main()
