"""Build the two AR comparison tables and figure from the saved reassessment."""

import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from ar_reassessment import scientific_digest


ROOT = Path(__file__).resolve().parent
YEARS = range(1978, 2026)
WINDOWS = ('expanding', 'recent_30')
MODELS = {
    'response_mean': 'Response mean',
    'gaussian_ar': 'Gaussian AR',
    'gaussian_ar_mix': 'Gaussian AR mixture',
    'poisson_ar': 'Poisson AR',
    'nb2_ar': 'NB AR',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_data():
    result_bytes = (ROOT / 'ar_reassessment_results.json').read_bytes()
    prediction_bytes = (ROOT / 'ar_reassessment_predictions.csv').read_bytes()
    result = json.loads(result_bytes)
    hashes = {'ar_reassessment_results.json (scientific digest)': scientific_digest(result),
              'ar_reassessment_predictions.csv': digest(prediction_bytes)}
    if hashes['ar_reassessment_results.json (scientific digest)'] != result['scientific_sha256']:
        raise ValueError('Results JSON differs from its scientific digest')
    if hashes['ar_reassessment_predictions.csv'] != result['predictions_csv_sha256']:
        raise ValueError('Forecast CSV differs from the digest in the results JSON')

    floor = result['settings']['main_sd_floor']
    summary = {}
    for row in result['summaries']['rolling']:
        if row['period'] == '1978-2025' and row['sd_floor'] in (None, floor):
            if row['n'] != len(YEARS):
                raise ValueError('Comparison summary is missing target years')
            summary[row['window'], row['model']] = row

    predictions = {}
    observed = {year: result['annual_counts'][str(year)] for year in YEARS}
    for row in csv.DictReader(io.StringIO(prediction_bytes.decode('utf-8'))):
        if row['protocol'] != 'rolling' or row['sd_floor'] not in ('', str(floor)):
            continue
        year = int(row['target_year'])
        if int(row['observed']) != observed[year] or int(row['origin']) != year - 1:
            raise ValueError('Prediction CSV does not match the observation and origin calendar')
        key = row['window'], row['model'], year
        if key in predictions:
            raise ValueError('Duplicate rolling forecast')
        predictions[key] = float(row['predicted_mean'])
    return summary, predictions, observed, hashes


def write_table(name, lines, hashes):
    # Record the exact source hashes in the generated LaTeX comments.
    comments = [f'% Source SHA-256 {name} {value}' for name, value in hashes.items()]
    (ROOT / name).write_text('\n'.join(comments + lines) + '\n', encoding='utf-8', newline='\n')


def comparison_table(summary):
    lines = [
        r'\begin{table}[htbp]',
        r'\centering\small',
        r'\caption{One-year forecast errors over 48 targets, 1978--2025. The expanding window uses all response years from 1919 through the forecast origin. The recent window uses the last 29 response years from 30 annual observations. Both scores use unrounded forecast means. The Gaussian standard-deviation floor is 1.}',
        r'\label{tab:ar-reassessment}',
        r'\begin{tabular}{@{}lrrrr@{}}',
        r'\toprule',
        r'& \multicolumn{2}{c}{Expanding window} & \multicolumn{2}{c}{Recent 30-year window} \\',
        r'\cmidrule(lr){2-3}\cmidrule(l){4-5}',
        r'Model & RMSE & MAE & RMSE & MAE \\',
        r'\midrule',
    ]
    for model, label in MODELS.items():
        values = [f'{summary[window, model][metric]:.3f}' for window in WINDOWS for metric in ('rmse', 'mae')]
        lines.append(' & '.join([label, *values]) + r' \\')
    persistence = summary['common', 'persistence']
    trailing = summary['common', 'trailing_10_mean']
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\par\smallskip',
        r'\begin{minipage}{0.92\linewidth}\footnotesize',
        r'The response mean averages the same response years used to fit the AR models. NB denotes negative binomial.',
        f"Persistence has RMSE {persistence['rmse']:.3f} and MAE {persistence['mae']:.3f}. "
        f"The trailing 10-year mean has RMSE {trailing['rmse']:.3f} and MAE {trailing['mae']:.3f}.",
        r'\end{minipage}',
        r'\end{table}',
    ])
    return lines


def recent_table(predictions, observed):
    lines = [
        r'\begin{table}[htbp]',
        r'\centering\small',
        r'\caption{Observed counts and one-year forecast means for 2021--2025. Each model is refit annually using the 30 observations through the prior year. The outcomes were already known when this exploratory comparison was designed.}',
        r'\label{tab:ar-reassessment-recent}',
        r'\begin{tabular}{@{}rrrrrr@{}}',
        r'\toprule',
        r'Year & Observed & Response mean & Gaussian AR & \shortstack{Gaussian AR\\mixture} & Poisson AR \\',
        r'\midrule',
    ]
    models = ('response_mean', 'gaussian_ar', 'gaussian_ar_mix', 'poisson_ar')
    for year in range(2021, 2026):
        values = [f'{predictions["recent_30", model, year]:.2f}' for model in models]
        lines.append(' & '.join([str(year), str(observed[year]), *values]) + r' \\')
    return lines + [r'\bottomrule', r'\end{tabular}', r'\end{table}']


def comparison_figure(predictions, observed, hashes):
    style = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
             'axes.spines.right': False, 'axes.edgecolor': '#525252',
             'xtick.color': '#303030', 'ytick.color': '#303030', 'savefig.facecolor': 'white'}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 1, figsize=(6.45, 4.6), sharex=True, sharey=True, layout='constrained')
        observed_counts = [observed[year] for year in YEARS]
        for ax, window, title in zip(axes, WINDOWS, ('Expanding window', 'Recent 30-year window')):
            for boundary in (2000.5, 2020.5):
                ax.axvline(boundary, color='#b5b5b5', linewidth=0.7, linestyle='--', zorder=1)
            ax.grid(axis='y', color='#ebebeb', linewidth=0.65)
            ax.set_axisbelow(True)
            ax.plot(YEARS, observed_counts, color='#1b1b1b', marker='o', markersize=2.9,
                    linewidth=0.8, zorder=3, label='Observed')
            ax.plot(YEARS, [predictions[window, 'gaussian_ar', year] for year in YEARS],
                    color='#006bb0', linewidth=1.65, zorder=2, label='Gaussian AR forecast mean')
            ax.set_title(title, loc='left', fontweight='bold', fontsize=10, pad=6)
            ax.set_ylabel('Annual count')
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.legend(loc='upper left', ncols=2, framealpha=1, facecolor='white',
                      edgecolor='none', fontsize=8.4, handlelength=2.1)
        axes[-1].set(xlim=(1977.35, 2025.65), ylim=(0, max(observed_counts) + 5),
                     xticks=[1980, 1990, 2000, 2010, 2020, 2025], xlabel='Target year')
        provenance = '\n'.join(f'SHA-256 {name} {value}' for name, value in hashes.items())
        fig.savefig(ROOT / 'ar_comparison.png', dpi=300,
                    metadata={'Software': 'build_ar_assets.py', 'Description': provenance})
        plt.close(fig)


def main():
    summary, predictions, observed, hashes = source_data()
    write_table('ar_comparison_table.tex', comparison_table(summary), hashes)
    write_table('ar_recent_table.tex', recent_table(predictions, observed), hashes)
    comparison_figure(predictions, observed, hashes)
    names = ('ar_comparison_table.tex', 'ar_recent_table.tex', 'ar_comparison.png')
    print(json.dumps({'source_sha256': hashes,
                      'output_sha256': {name: digest((ROOT / name).read_bytes()) for name in names}}, indent=2))


if __name__ == '__main__':
    main()
