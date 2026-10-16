"""Import one simulation into a versioned Excel tracker. Python 3.10+."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from datetime import datetime, timezone

from openpyxl import Workbook, load_workbook
from openpyxl.chart import ScatterChart, Reference, Series
from openpyxl.styles import Font, PatternFill

LABELS = {
    'time': 'Time [s]', 'altitude': 'Altitude [ft]', 'thrust': 'Thrust [N]',
    'velocity': 'Velocity [m/s]', 'acceleration': 'Acceleration [m/s²]',
    'dynamic_pressure': 'Dynamic pressure [psi]', 'stability': 'Stability [cal]',
}
UNITS = {
    'time': {'s': 1, 'ms': .001},
    'altitude': {'ft': 1, 'm': 1 / .3048},
    'thrust': {'N': 1, 'lbf': 4.4482216152605},
    'velocity': {'m/s': 1, 'ft/s': .3048},
    'acceleration': {'m/s2': 1, 'm/s²': 1, 'ft/s2': .3048, 'g': 9.80665},
    'dynamic_pressure': {'Pa': 1, 'kPa': 1000, 'bar': 100000, 'psi': 6894.757293},
    'stability': {'cal': 1},
}

def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None

def read_csv(path, cfg):
    lines = path.read_text(encoding=cfg.get('encoding', 'utf-8-sig')).splitlines()
    delimiter = cfg.get('delimiter', ',')
    start = cfg.get('header_line')
    if start is None:
        # A commented OpenRocket header may follow metadata comments.
        candidates = []
        for i, line in enumerate(lines):
            fields = next(csv.reader([line.lstrip('# ').strip()], delimiter=delimiter))
            if any(re.fullmatch(r'time\s*(?:[\[(].*?[\])])?', x.strip(), re.I) for x in fields):
                candidates.append(i)
        if len(candidates) != 1:
            raise ValueError('Cannot identify a unique time-series header. Set header_line (zero-based) in config.')
        start = candidates[0]
    header = next(csv.reader([lines[start].lstrip('# ').strip()], delimiter=delimiter))
    header = [x.strip() for x in header]
    if len(set(header)) != len(header):
        raise ValueError('Duplicate column names; export unique column names.')
    rows = []
    for line in lines[start + 1:]:
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        row = next(csv.reader([line], delimiter=delimiter))
        if len(row) != len(header):
            raise ValueError('CSV row width differs from header; use one time-series export per file.')
        rows.append(row)
    return header, rows

def convert(header, rows, cfg):
    mapping = cfg.get('columns', {})
    series = {}
    resolved = {}
    for key in LABELS:
        spec = mapping.get(key)
        if spec is None:
            # Only exact variable names with explicit supported units are inferred.
            base = key.replace('_', ' ')
            matches = []
            for name in header:
                match = re.fullmatch(re.escape(base) + r'\s*[\[(](.*?)[\])]', name, re.I)
                if match and match[1] in UNITS[key]:
                    matches.append({'name': name, 'unit': match[1]})
            if len(matches) > 1:
                raise ValueError(f'Ambiguous {key}; explicitly map its column.')
            spec = matches[0] if matches else None
        if spec is None:
            continue
        name, unit = spec['name'], spec['unit']
        if name not in header or unit not in UNITS[key]:
            raise ValueError(f'Invalid mapping/unit for {key}: {spec}')
        idx = header.index(name)
        factor = UNITS[key][unit]
        if key == 'dynamic_pressure':
            factor /= 6894.757293  # Convert to psi for internal storage.
        series[key] = [None if number(r[idx]) is None else number(r[idx]) * factor for r in rows]
        resolved[key] = spec
    if 'time' not in series:
        raise ValueError('Time column is required. Set columns.time name and unit.')
    times = series['time']
    if len(times) < 2 or any(t is None for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Need at least two strictly increasing numeric times; split multiple simulations first.')
    for p in cfg.get('pressures', []):
        idx = header.index(p['name'])
        factor = UNITS['dynamic_pressure'][p['unit']] / 6894.757293
        key = 'pressure: ' + p.get('label', p['name'])
        if key in series:
            raise ValueError('Duplicate pressure label.')
        series[key] = [None if number(r[idx]) is None else number(r[idx]) * factor for r in rows]
    return series, resolved

def metrics(series, cfg):
    result = {'Propellant pair': cfg.get('propellant_pair', 'Not supplied'),
              'Altitude reference': cfg.get('altitude_reference', 'Unspecified'),
              'Final recorded time [s]': series['time'][-1]}
    for key, label in [('altitude', 'Apogee [ft]'), ('thrust', 'Peak thrust [N]'),
                       ('velocity', 'Maximum velocity [m/s]'), ('acceleration', 'Maximum acceleration [m/s²]'),
                       ('dynamic_pressure', 'Maximum dynamic pressure [psi]')]:
        values = [x for x in series.get(key, []) if x is not None]
        result[label] = max(values) if values else None
    thrust = series.get('thrust')
    impulse = None
    if thrust and all(x is not None for x in thrust):
        impulse = sum((b - a) * (fa + fb) / 2 for a, b, fa, fb in
                      zip(series['time'], series['time'][1:], thrust, thrust[1:]))
    result['Integrated impulse over recorded interval [N s]'] = impulse
    consumed = cfg.get('consumed_propellant_kg')
    loaded, dry = cfg.get('loaded_propellant_kg'), cfg.get('dry_mass_kg')
    for name, value in [('consumed_propellant_kg', consumed), ('loaded_propellant_kg', loaded), ('dry_mass_kg', dry)]:
        if value is not None and (number(value) is None or value <= 0):
            raise ValueError(f'{name} must be a positive number or null.')
    result['Isp [s]'] = impulse / (consumed * 9.80665) if impulse is not None and consumed else None
    result['Loaded propellant [kg]'] = loaded
    result['Dry mass [kg]'] = dry
    result['Dry mass scope'] = cfg.get('dry_mass_scope', 'Unspecified')
    result['m_prop / m_dry'] = loaded / dry if loaded and dry else None
    result['Notes'] = cfg.get('notes', '')
    return result

def put(sheet, row):
    sheet.append(row)
    # CSVs and user labels must remain literal text, including leading '='.
    for cell in sheet[sheet.max_row]:
        if isinstance(cell.value, str):
            cell.data_type = 's'

def style(sheet):
    sheet.freeze_panes = 'A2'
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(color='FFFFFF', bold=True)
        cell.fill = PatternFill('solid', fgColor='17365D')
    for column in sheet.columns:
        letter = column[0].column_letter
        sheet.column_dimensions[letter].width = min(55, max(16, max(len(str(c.value or '')) for c in column[:100]) + 2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path, nargs='?', help='CSV file to import')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, default=Path('simulation_tracker.xlsx'))
    parser.add_argument('--name', default='Simulation')
    parser.add_argument('--source-url', help='Shared web URL for this CSV, if available')
    args = parser.parse_args()

    folder = Path(__file__).resolve().parent

    if args.csv is None: 
        csv_files = list(folder.glob('*.csv'))

        if not csv_files:
            parser.error(f'No CSV files found in {folder}. Please provide a CSV file to import.')

        args.csv = max(csv_files, key=lambda f: f.stat().st_mtime)

    if not args.output.is_absolute():
        args.output = folder / args.output

    print(f'Reading CSV: {args.csv.name}')

    print(f'Reading CSV: {args.csv.name}')

    cfg = json.loads(args.config.read_text()) if args.config else {}
    header, rows = read_csv(args.csv, cfg)
    series, resolved = convert(header, rows, cfg)
    values = metrics(series, cfg)
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wb = load_workbook(args.output) if args.output.exists() else Workbook()
    if 'History' not in wb.sheetnames:
        if args.output.exists():
            raise ValueError('Existing workbook is not a tracker; choose a new output path.')
        wb.remove(wb.active)
        put(wb.create_sheet('History'), ['Run ID', 'Imported UTC', 'Run name', 'Latest', 'CSV source', 'SHA256', *values])
    history = wb['History']
    expected = ['Run ID', 'Imported UTC', 'Run name', 'Latest', 'CSV source', 'SHA256', *values]
    if [c.value for c in history[1]] != expected:
        raise ValueError('Tracker schema differs; choose a new workbook or use the same script version.')
    run = f'R{history.max_row:04d}'
    digest = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    archive = args.output.parent / 'sources' / f'{run}_{digest[:12]}.csv'
    archive.parent.mkdir(exist_ok=True)
    shutil.copy2(args.csv, archive)
    link = args.source_url or archive.relative_to(args.output.parent).as_posix()
    for row in history.iter_rows(min_row=2):
        row[3].value = ''
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    put(history, [run, stamp, args.name, 'LATEST', link, digest, *values.values()])
    history.cell(history.max_row, 5).hyperlink = link
    history.cell(history.max_row, 5).style = 'Hyperlink'
    raw = wb.create_sheet(run + '_Raw')
    put(raw, header)
    for row in rows:
        put(raw, [number(x) if number(x) is not None else x for x in row])
    data = wb.create_sheet(run + '_Data')
    keys = list(series)
    put(data, [LABELS.get(k, k + ' [Pa]') for k in keys])
    for row in zip(*(series[k] for k in keys)):
        put(data, list(row))
    summary = wb.create_sheet(run)
    put(summary, ['Metric', 'Value'])
    for k, v in {'Run ID': run, 'Imported UTC': stamp, 'Run name': args.name,
                 'CSV source': link, 'SHA256': digest, **values,
                 'Column mappings': json.dumps(resolved),
                 'Config used': json.dumps(cfg),
                 'Calculation note': 'Impulse and Isp require the complete thrust interval and matching consumed mass. Missing values remain blank.'}.items():
        put(summary, [k, v])
        if k == 'CSV source':
            summary.cell(summary.max_row, 2).hyperlink = link
    for index, key in enumerate(k for k in keys if k != 'time'):
        col = keys.index(key) + 1
        chart = ScatterChart()
        chart.title = LABELS.get(key, key + ' [psi]')
        chart.x_axis.title = 'Time [s]'
        chart.y_axis.title = LABELS.get(key, key + ' [psi]')
        chart.scatterStyle = 'line'
        chart.series.append(Series(Reference(data, min_col=col, min_row=2, max_row=data.max_row),
                                   Reference(data, min_col=keys.index('time') + 1, min_row=2, max_row=data.max_row), title=key))
        chart.width, chart.height = 22, 11
        summary.add_chart(chart, f'D{1 + index * 23}')
    if 'Latest' in wb.sheetnames:
        del wb['Latest']
    latest = wb.create_sheet('Latest', 0)
    put(latest, ['Latest imported simulation', 'Value'])
    for row in summary.iter_rows(min_row=2, values_only=True):
        put(latest, list(row))
    latest.cell(5, 2).hyperlink = link
    latest.cell(2, 2).hyperlink = f"#'{run}'!A1"
    for sheet in [history, raw, data, summary, latest]:
        style(sheet)
    fd, tmp = tempfile.mkstemp(suffix='.xlsx', dir=args.output.parent)
    os.close(fd)
    try:
        wb.save(tmp)
        os.replace(tmp, args.output)
    finally:
        Path(tmp).unlink(missing_ok=True)
    print(f'Imported {run}: {args.output}')
    print('Detected mappings:', json.dumps(resolved))
    print('Missing metrics:', ', '.join(k for k, v in values.items() if v is None))

if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(f'Import failed: {exc}')
