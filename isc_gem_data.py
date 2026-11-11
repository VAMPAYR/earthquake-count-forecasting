"""Build or verify the offline snapshot of the supplied ISC-GEM 13.0 archive."""

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile


ARCHIVE_SHA256 = "12f255cef062e3ee51345b91dc0431b161d9a9af865cc09bae3a5ad39fdab0db"
MEMBERS = {"main": "isc-gem-cat.csv", "supplementary": "isc-gem-suppl.csv"}
SOURCE_COLUMNS = (
    "date lat lon smajax sminax strike q depth unc q mw unc q s mo fac mo_auth "
    "mpp mpr mrr mrt mtp mtt str1 dip1 rake1 str2 dip2 rake2 type eventid"
).split()
COLUMNS = (
    "date lat lon smajax sminax strike epicentre_quality depth depth_unc depth_quality "
    "mw mw_unc mw_quality mw_source mo fac mo_auth "
    "mpp mpr mrr mrt mtp mtt str1 dip1 rake1 str2 dip2 rake2 type eventid"
).split()
CODE_FIELDS = {
    "epicentre_quality": {"A", "B", "C", "D"},
    "depth_quality": {"A", "B", "C"},
    "mw_quality": {"A", "B", "C", "D"},
    "mw_source": {"d", "p", ""},
    "mo_auth": {"gcmt", "bibliog", ""},
    "type": {"BDC", "BB", "FM", "RA", "G", ""},
}
NUMERIC_FIELDS = [name for name in COLUMNS if name not in (*CODE_FIELDS, "date", "eventid")]
RETAINED_NUMERIC_FIELDS = (
    "lat", "lon", "smajax", "sminax", "strike", "depth", "depth_unc", "mw", "mw_unc",
)
YEARS = range(1918, 2023)
MAG_7_5 = Decimal("7.5")
ANNUAL_KEYS = (
    "all_rows", "mw_missing", "mw_unparseable", "mw_ge_7", "mw_gt_7",
    "mw_ge_7_5", "mw_eq_7", "mw_eq_7_5", "mw_ge_7_direct", "mw_ge_7_proxy",
)
DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{2}")
ID_PATTERN = re.compile(r"[1-9][0-9]*")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_decimal(value):
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def count_annual(row_counts, magnitude, source, missing):
    row_counts["all_rows"] += 1
    if magnitude is None:
        row_counts["mw_missing" if missing else "mw_unparseable"] += 1
    elif magnitude >= 7:
        row_counts["mw_ge_7"] += 1
        row_counts["mw_gt_7"] += magnitude > 7
        row_counts["mw_ge_7_5"] += magnitude >= MAG_7_5
        row_counts["mw_eq_7"] += magnitude == 7
        row_counts["mw_eq_7_5"] += magnitude == MAG_7_5
        row_counts["mw_ge_7_direct"] += source == "d"
        row_counts["mw_ge_7_proxy"] += source == "p"


def parse_catalogue(member, data, supplementary, study_events=None):
    lines = data.decode("utf-8").splitlines()
    header_index = next((index for index, line in enumerate(lines) if not line.startswith("#")), 0)
    preamble = lines[:header_index]
    require("# Version 13.0 - released on 2026-10-21" in preamble, f"{member}: unexpected release")
    require(
        any("Version 13.0" in line and "period covered: 1904-2022" in line for line in preamble),
        f"{member}: unexpected release coverage",
    )
    require(
        any("CC-BY-SA 3.0 (unported)" in line for line in preamble),
        f"{member}: license missing",
    )
    header = next(csv.reader([preamble[-1][1:]], strict=True))
    require([cell.strip() for cell in header] == SOURCE_COLUMNS, f"{member}: unexpected columns")

    missing = dict.fromkeys(COLUMNS, 0)
    unparseable = dict.fromkeys(NUMERIC_FIELDS, 0)
    minimum = {}
    maximum = {}
    codes = {key: Counter() for key in CODE_FIELDS}
    year_counts = {year: dict.fromkeys(ANNUAL_KEYS, 0) for year in YEARS}
    observed_year_counts = Counter()
    ids = set()
    hypocentres = set()
    events = []
    first = None
    last = None

    for line_number, line in enumerate(lines[header_index:], header_index + 1):
        location = f"{member} line {line_number}"
        cells = next(csv.reader([line], strict=True))
        require(len(cells) == len(COLUMNS) and ",".join(cells) == line, f"{location}: malformed row")
        raw = dict(zip(COLUMNS, map(str.strip, cells)))
        date = raw["date"]
        require(DATE_PATTERN.fullmatch(date), f"{location}: invalid timestamp shape")
        time = datetime.fromisoformat(date)
        require(1904 <= time.year <= 2022, f"{location}: timestamp outside release coverage")
        first = date if first is None else min(first, date)
        last = date if last is None else max(last, date)
        observed_year_counts[time.year] += 1

        eventid = raw["eventid"]
        require(ID_PATTERN.fullmatch(eventid), f"{location}: invalid event ID")
        require(eventid not in ids, f"{location}: repeated event ID {eventid}")
        ids.add(eventid)
        for name, value in raw.items():
            missing[name] += value == ""
        for name, allowed in CODE_FIELDS.items():
            value = raw[name]
            require(value in allowed, f"{location}: invalid {name} code {value!r}")
            codes[name][value] += 1
        in_supplement = raw["epicentre_quality"] == "D" or raw["mw_quality"] == "D"
        require(in_supplement == supplementary, f"{location}: unexpected catalogue quality flag")
        require(bool(raw["mw"]) == bool(raw["mw_source"]), f"{location}: inconsistent Mw source flag")

        numbers = {}
        for name in NUMERIC_FIELDS:
            value = raw[name]
            number = finite_decimal(value) if value else None
            numbers[name] = number
            if value and number is None:
                unparseable[name] += 1
                require(name == "mw", f"{location}: invalid numeric {name}")
            if number is not None:
                minimum[name] = min(minimum.get(name, number), number)
                maximum[name] = max(maximum.get(name, number), number)
        lat, lon, depth = (numbers[name] for name in ("lat", "lon", "depth"))
        require(
            lat is not None and lon is not None and depth is not None
            and -90 <= lat <= 90 and -180 <= lon <= 180 and depth >= 0,
            f"{location}: invalid hypocentre",
        )
        hypocentres.add((date, lat, lon))

        if time.year in YEARS:
            magnitude = numbers["mw"]
            count_annual(year_counts[time.year], magnitude, raw["mw_source"], not raw["mw"])
            if study_events is not None:
                study_events.append({
                    "eventid": eventid,
                    "time_utc": date.replace(" ", "T") + "Z",
                    "source_line": line_number,
                    "lat": float(lat),
                    "lon": float(lon),
                    "depth": float(depth),
                    "mw": float(magnitude) if magnitude is not None else None,
                    "mw_unc": float(numbers["mw_unc"]) if numbers["mw_unc"] is not None else None,
                    "mw_source": raw["mw_source"] or None,
                })
            if magnitude is not None and magnitude >= 7:
                event = {
                    "eventid": eventid,
                    "time_utc": date.replace(" ", "T") + "Z",
                    "source_line": line_number,
                    "raw_csv": line,
                }
                for name in RETAINED_NUMERIC_FIELDS:
                    event[name] = float(numbers[name]) if numbers[name] is not None else None
                for name in CODE_FIELDS:
                    event[name] = raw[name] or None
                events.append(event)

    require(len(ids) == len(hypocentres), f"{member}: exact timestamp/epicentre duplicate")
    annual = {key: [year_counts[year][key] for year in YEARS] for key in ANNUAL_KEYS}
    verify_events(member, lines, events, annual)
    result = {
        "source_member": member,
        "version_line": preamble.index("# Version 13.0 - released on 2026-10-21") + 1,
        "csv_header_line": header_index,
        "source_rows": len(ids),
        "rows_by_observed_year": dict(sorted(observed_year_counts.items())),
        "first_time_utc": first.replace(" ", "T") + "Z",
        "last_time_utc": last.replace(" ", "T") + "Z",
        "source_blank_fields": missing,
        "source_unparseable_numeric_fields": unparseable,
        "source_numeric_ranges": {
            key: [float(minimum[key]), float(maximum[key])] if key in minimum else None
            for key in NUMERIC_FIELDS
        },
        "source_code_counts": {key: dict(sorted(counts.items())) for key, counts in codes.items()},
        "scope_source_rows": sum(annual["all_rows"]),
        "scope_missing_mw": sum(annual["mw_missing"]),
        "scope_unparseable_mw": sum(annual["mw_unparseable"]),
        "scope_retained_source_code_counts": {
            key: dict(sorted(Counter(event[key] or "" for event in events).items()))
            for key in CODE_FIELDS
        },
        "events": events,
    }
    return result, annual, ids, hypocentres


def verify_events(member, source_lines, events, annual):
    """Round-trip source positions and independently count serialized records."""
    records = json.loads(json.dumps(events, allow_nan=False))
    for event in records:
        line = source_lines[event["source_line"] - 1]
        values = [cell.strip() for cell in line.split(",")]
        require(
            event["raw_csv"] == line
            and event["eventid"] == values[30]
            and event["time_utc"] == values[0].replace(" ", "T") + "Z",
            f"{member}: retained source record mismatch at line {event['source_line']}",
        )
        for name in RETAINED_NUMERIC_FIELDS:
            value = values[COLUMNS.index(name)]
            require(event[name] == (float(value) if value else None), f"{member}: {name} mismatch")
        for name in CODE_FIELDS:
            require(event[name] == (values[COLUMNS.index(name)] or None), f"{member}: {name} mismatch")

    checks = {
        "mw_ge_7": lambda event: event["mw"] >= 7,
        "mw_gt_7": lambda event: event["mw"] > 7,
        "mw_ge_7_5": lambda event: event["mw"] >= 7.5,
        "mw_eq_7": lambda event: event["mw"] == 7,
        "mw_eq_7_5": lambda event: event["mw"] == 7.5,
        "mw_ge_7_direct": lambda event: event["mw_source"] == "d",
        "mw_ge_7_proxy": lambda event: event["mw_source"] == "p",
    }
    require(len(records) == sum(annual["mw_ge_7"]), f"{member}: retained record count mismatch")
    for key, predicate in checks.items():
        by_year = Counter(int(event["time_utc"][:4]) for event in records if predicate(event))
        require(annual[key] == [by_year[year] for year in YEARS], f"{member}: {key} count mismatch")


def build_snapshot(archive_path, *, study_events=None):
    archive_data = archive_path.read_bytes()
    digest = sha256(archive_data).hexdigest()
    require(digest == ARCHIVE_SHA256, "Archive SHA-256 differs from the supplied release snapshot")
    catalogues = {}
    annual = {"years": list(YEARS)}
    ids = {}
    hypocentres = {}
    metadata = {}
    with ZipFile(BytesIO(archive_data)) as archive:
        names = set(MEMBERS.values()) | {"isc-gem-cat.kmz"}
        require(
            set(archive.namelist()) == names and len(archive.infolist()) == len(names),
            "Unexpected archive membership",
        )
        for name in sorted(names):
            info = archive.getinfo(name)
            payload = archive.read(info)  # ZipFile validates member CRC as it reads the bytes.
            metadata[name] = {"bytes": info.file_size, "sha256": sha256(payload).hexdigest()}
            for key, member in MEMBERS.items():
                if member == name:
                    catalogues[key], annual[key], ids[key], hypocentres[key] = parse_catalogue(
                        member, payload, supplementary=key == "supplementary",
                        study_events=study_events[key] if study_events is not None else None,
                    )

    shared_ids = ids["main"] & ids["supplementary"]
    shared_hypocentres = hypocentres["main"] & hypocentres["supplementary"]
    require(not shared_ids and not shared_hypocentres, "Unresolved cross-catalogue identity overlap")
    require(catalogues["main"]["rows_by_observed_year"].get(2022) == 1755, "2022 source count mismatch")
    annual["main_plus_supplementary"] = {
        key: [main + supplement for main, supplement in zip(
            annual["main"][key], annual["supplementary"][key], strict=True,
        )]
        for key in ANNUAL_KEYS
    }

    return {
        "schema_version": 1,
        "provenance": {
            "catalogue": "ISC-GEM Global Instrumental Earthquake Catalogue",
            "version": "13.0",
            "release_date": "2026-10-21",
            "release_coverage_years_inclusive": [1904, 2022],
            "citation": "International Seismological Centre (2026), ISC-GEM Earthquake Catalogue, "
            "https://doi.org/10.31905/D808B825",
            "copyright_in_csv": "Copyright (C) 2026 International Seismological Centre (ISC)",
            "copyright_on_download_page": "Copyright (C) 2026 International Seismological Centre "
            "and GEM Foundation",
            "license": "CC BY-SA 3.0 (unported)",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "doi_url": "https://doi.org/10.31905/D808B825",
            "download_and_legal_url": "https://ftp.isc.ac.uk/iscgem/download.php",
            "request_form_url": "https://ftp.isc.ac.uk/iscgem/request_catalogue.php",
            "release_overview_url": "https://isc.ac.uk/iscgem/overview.php",
            "update_log_url": "https://www.isc.ac.uk/iscgem/update_log/",
            "moment_bibliography_url": "https://www.isc.ac.uk/iscgem/mw_bibliography.php",
            "acquisition": {
                "status": "User-supplied archive, processed locally",
                "received_date": "2026-11-11",
                "retrieval_timestamp": None,
            },
            "archive": {
                "name": "isc-gem.zip",
                "bytes": len(archive_data),
                "sha256": digest,
                "members": metadata,
            },
        },
        "schema": {
            "source_columns": SOURCE_COLUMNS,
            "disambiguated_columns": COLUMNS,
            "column_documentation_lines": {"main": [54, 65], "supplementary": [58, 73]},
            "mw_source_codes": {"d": "direct computation", "p": "proxy"},
            "quality_codes": {
                "A": "Highest quality per source CSV documentation",
                "B": "Source quality code B",
                "C": "Source quality code C",
                "D": "Poorly determined epicentre or Mw, as documented by the supplement",
            },
            "time_encoding": "Offset-free source origin dates interpreted as UTC. The output "
            "retains hundredths of a second and adds the ISO 8601 UTC suffix Z.",
            "time_zone_cross_check": {
                "isc_eventid": "625480155",
                "isc_gem_raw_time": "2022-12-31 23:04:12.46",
                "isc_bulletin_time_utc": "2022-12-31T23:04:12.25Z",
                "isc_bulletin_url": "https://www.isc.ac.uk/fdsnws/event/1/query?eventid=625480155&format=xml",
                "usgs_eventid": "us7000j1b1",
                "usgs_time_utc": "2022-12-31T23:04:12.381Z",
                "usgs_event_url": "https://earthquake.usgs.gov/earthquakes/eventpage/us7000j1b1",
            },
            "retained_events": "Inclusive UTC calendar years 1918–2022 and published Mw >= 7. "
            "Includes direct and proxy Mw at their published precision. Blank and unparseable "
            "Mw are counted separately and excluded from threshold totals. Source lines use "
            "one-based numbering. The original row text excludes the terminating newline.",
            "annual_values": "All arrays align with annual.years and include zero-count years.",
        },
        "identity_checks": {
            "key": "ISC eventid",
            "cross_catalogue_shared_eventids": sorted(shared_ids),
            "cross_catalogue_exact_time_lat_lon_matches": len(shared_hypocentres),
            "within_catalogue_duplicates": dict.fromkeys(MEMBERS, 0),
            "combined_series_rule": "Sum separate counts after checking all source IDs are unique "
            "within each file and disjoint across files. Exact time/latitude/longitude triples "
            "are also unique within and disjoint across files.",
        },
        "annual": annual,
        "catalogues": catalogues,
    }


def load_study_events(archive_path):
    """Return separate validated 1918–2022 catalogue records for event matching."""
    study_events = {key: [] for key in MEMBERS}
    build_snapshot(archive_path, study_events=study_events)
    return study_events


def write_snapshot(path, payload):
    """Publish a complete snapshot after writing the bytes on the same volume."""
    temporary = NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False)
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            temporary.write(payload)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=directory / "isc-gem.zip")
    parser.add_argument("--output", type=Path, default=directory / "isc_gem_snapshot.json")
    parser.add_argument("--check", action="store_true", help="compare with an existing snapshot")
    args = parser.parse_args()
    if args.archive.resolve() == args.output.resolve():
        parser.error("Archive and output must be separate paths")
    try:
        snapshot = build_snapshot(args.archive)
        payload = (json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
        if args.check:
            require(args.output.read_bytes() == payload, "Snapshot differs from the deterministic build")
        else:
            write_snapshot(args.output, payload)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"{args.output.name} SHA-256 {sha256(payload).hexdigest()}")


if __name__ == "__main__":
    main()
