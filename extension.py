"""Score the frozen 2020 forecasts for 2021-2025, without fitting.
Default: python extension.py (offline, existing snapshot).
Authorized first retrieval only: python extension.py --fetch
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / 'usgs_2021_2025_snapshot.json'
EXPECTED = {
    'analysis.py': '60751c1f1af0e10f83bedcb6490f4741c59dbd166eb8230b5d338d6b0d17e51f',
    'usgs_1918_2020_snapshot.json': 'ca7c18d7a69048cf94ce27c4aecf547fb9b8b6b9ce93acc41c774b9388a07623',
    'forecast_2020_frozen.json': 'c8f98250be896080bc9f1ce06d291e49230fad5c34e413f56327352a1dca618c',
}
QUERY = 'https://earthquake.usgs.gov/fdsnws/event/1/query?' + urlencode({
    'format': 'geojson', 'starttime': '2021-01-01T00:00:00',
    'endtime': '2025-12-31T23:59:59.999', 'minmagnitude': 7,
    'eventtype': 'earthquake', 'orderby': 'time-asc',
})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_freeze():
    actual = {name: digest(ROOT / name) for name in EXPECTED}
    if actual != EXPECTED:
        raise RuntimeError('Frozen source, data, or forecast hash differs')
    frozen = json.loads((ROOT / 'forecast_2020_frozen.json').read_text(encoding='utf-8'))
    if frozen['code_sha256'] != actual['analysis.py'] or frozen['data_sha256'] != actual['usgs_1918_2020_snapshot.json']:
        raise RuntimeError('Frozen internal provenance mismatch')
    if frozen['origin'] != 2020:
        raise RuntimeError('Wrong forecast origin')
    return frozen, actual


def validate_events(geojson):
    features = geojson['features']
    if geojson['type'] != 'FeatureCollection' or geojson['metadata']['count'] != len(features):
        raise RuntimeError('Invalid catalogue count or response type')
    counts = {str(year): 0 for year in range(2021, 2026)}
    records = []
    seen = set()
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    for event in features:
        p = event['properties']
        identity = event['id']
        when = epoch + timedelta(milliseconds=p['time'])
        if identity in seen or p['type'] != 'earthquake' or not 2021 <= when.year <= 2025:
            raise RuntimeError('Duplicate or out-of-scope earthquake')
        if p['mag'] is None or not math.isfinite(p['mag']) or p['mag'] < 7:
            raise RuntimeError('Invalid preferred magnitude')
        if event['geometry']['type'] != 'Point' or len(event['geometry']['coordinates']) != 3:
            raise RuntimeError('Invalid event geometry')
        seen.add(identity)
        counts[str(when.year)] += 1
        records.append({'id': identity, 'time_utc': when.isoformat(), 'magnitude': p['mag'],
                        'magnitude_type': p['magType'], 'coordinates': event['geometry']['coordinates']})
    return counts, records


def load_snapshot(fetch, hashes):
    if fetch:
        if SNAPSHOT.exists():
            raise RuntimeError('Snapshot exists; --fetch never overwrites it')
        before = {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'hashes': hashes,
                  'extension_code_sha256': digest(Path(__file__))}
        print(json.dumps({'before_fetch': before}), flush=True)
        request = Request(QUERY, headers={'User-Agent': 'earthquake-count-reconstruction/1.0'})
        with urlopen(request, timeout=120) as response:
            raw = response.read()
        retrieved = datetime.now(timezone.utc).isoformat()
        geojson = json.loads(raw)
        counts, records = validate_events(geojson)
        snapshot = {'query_url': QUERY, 'before_fetch': before, 'retrieved_at_utc': retrieved,
                    'raw_response_sha256': hashlib.sha256(raw).hexdigest(),
                    'raw_response_bytes': len(raw), 'raw_response_utf8': raw.decode('utf-8'),
                    'geojson': geojson, 'annual_counts': counts, 'event_records': records,
                    'magnitude_type_counts': dict(Counter(r['magnitude_type'] for r in records))}
        with SNAPSHOT.open('x', encoding='utf-8') as out:
            out.write(json.dumps(snapshot, indent=2, allow_nan=False) + '\n')
    if not SNAPSHOT.exists():
        raise RuntimeError('No offline snapshot. Retrieval requires explicit authorization and --fetch.')
    snapshot = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    raw = snapshot['raw_response_utf8'].encode('utf-8')
    if hashlib.sha256(raw).hexdigest() != snapshot['raw_response_sha256'] or len(raw) != snapshot['raw_response_bytes']:
        raise RuntimeError('Raw response hash or length mismatch')
    if snapshot['query_url'] != QUERY or snapshot['before_fetch']['hashes'] != hashes:
        raise RuntimeError('Query or pre-retrieval freeze mismatch')
    if json.loads(raw) != snapshot['geojson']:
        raise RuntimeError('Parsed response differs from raw response')
    counts, records = validate_events(snapshot['geojson'])
    if counts != snapshot['annual_counts'] or records != snapshot['event_records']:
        raise RuntimeError('Snapshot derived records differ')
    return snapshot, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    frozen, hashes = verify_freeze()
    snapshot, counts = load_snapshot(args.fetch, hashes)
    # This import exposes scoring helpers; analysis.main() is never called.
    from analysis import MODELS, metrics, summarize
    rows = []
    for model in MODELS:
        forecasts = frozen['forecasts'][model]
        for h in range(1, 6):
            forecast = forecasts[h - 1]
            if forecast['horizon'] != h or forecast['target_year'] != 2020 + h:
                raise RuntimeError('Frozen horizon calendar mismatch')
            pmf = np.asarray(forecast['pmf'], dtype=float)
            target = counts[str(2020 + h)]
            if not 0 <= target < len(pmf):
                raise RuntimeError('Observed count outside frozen support')
            if np.any(~np.isfinite(pmf)) or np.any(pmf < 0) or abs(pmf.sum() - 1) > 1e-8:
                raise RuntimeError('Invalid frozen normalized PMF')
            score = metrics(pmf, target)
            probability = score['probability']
            retained = forecast['retained_mass']
            escaped = forecast['escaped_mass']
            q = probability * retained
            if not math.isfinite(q) or q <= 0 or not math.isfinite(probability) or probability <= 0 or not math.isfinite(score['nll']):
                raise RuntimeError('Invalid observed probability; no numerical floor applied')
            bracket = [-math.log(min(1, q + escaped)), -math.log(q)]
            bound = abs(math.log(retained)) + math.log1p(escaped / q)
            tolerance = 1e-8
            if bound > tolerance or not bracket[0] - tolerance <= score['nll'] <= bracket[1] + tolerance:
                raise RuntimeError('Frozen observed log-score guard failed')
            rows.append({'origin': 2020, 'horizon': h, 'target_year': 2020 + h,
                         'target': target, 'model': model, **score,
                         'numerics': {'true_nll_bracket': bracket,
                                      'nll_error_bound_with_mass_roundoff': bound,
                                      'floating_tolerance': tolerance,
                                      'observed_probability_guard_passed': True,
                                      'observed_nll_bracket_guard_passed': True}})
    if len(rows) != 25 or len({(r['model'], r['horizon']) for r in rows}) != 25:
        raise RuntimeError('Incomplete extension scoring')
    output = {'scope': 'Five annual outcomes from one frozen 2020 origin. Descriptive supplement; no inference or calibration claims.',
              'frozen_hashes': hashes, 'extension_code_sha256': digest(Path(__file__)),
              'snapshot_sha256': digest(SNAPSHOT), 'before_fetch': snapshot['before_fetch'],
              'retrieved_at_utc': snapshot['retrieved_at_utc'], 'annual_counts': counts,
              'summary': summarize(rows), 'rows': rows}
    (ROOT / 'extension_results.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'annual_counts': counts, 'summary': output['summary']}, indent=2))


if __name__ == '__main__':
    main()
