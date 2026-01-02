"""
Config Parser for CHAOS Calibration Framework.

Parses pipe-delimited tables from YAML config files.

Example config:
```yaml
info: |
  MS file        | Fields
  -------------- | ------------------------
  flux_cal.ms    | 3C147,3C286

solve_jones: |
  Jones types | Fields    | Scans | Spw  | Freq interval | Time interval | ...
  ----------- | --------- | ----- | ---- | ------------- | ------------- | ...
  K,B,G       | 3C147,... | 1~10  | 0~15 | full          | inf           | ...

apply_jones: |
  MS file   | Jones terms | From fields | Scans | Spw | Freq interpolation | ...
  --------- | ----------- | ----------- | ----- | --- | ------------------ | ...
  target.ms | K,B,G       | 3C147,...   | *     | *   | linear             | ...
```
"""

import yaml
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class SolveEntry:
    """Single row from solve_jones table."""
    jones_types: List[str]          # K, B, G, P, D, X, etc.
    fields: List[str]               # Field per Jones type (: for multi-field)
    scans: str                      # e.g., "1~10" or "*"
    spw: str                        # e.g., "0~15" or "*"
    freq_interval: str              # "full", "per_spw", "per_channel", or Hz
    time_interval: str              # "inf" or seconds like "30s"
    pre_apply_jones: List[str]      # Previously solved Jones to apply
    pre_apply_tables: List[str]     # Tables containing pre-solved Jones
    output_table: str               # Output HDF5 file
    model_column: str               # MODEL_DATA
    parallactic: bool               # Apply parallactic angle correction


@dataclass
class ApplyEntry:
    """Single row from apply_jones table."""
    ms_file: str                    # Target MS file
    jones_terms: List[str]          # K, B, G, P, etc.
    from_fields: List[str]          # Source field per Jones term
    scans: str                      # "*" or range
    spw: str                        # "*" or range
    freq_interp: str                # linear, nearest, cubic
    time_interp: str                # linear, nearest
    cal_tables: List[str]           # Calibration tables to use
    output_column: str              # CORRECTED_DATA
    parallactic: bool               # Apply parallactic angle correction


@dataclass
class CalConfig:
    """Complete calibration configuration."""
    info: Dict[str, List[str]]      # MS files and their fields
    solve_entries: List[SolveEntry]
    apply_entries: List[ApplyEntry]
    raw_yaml: dict                  # Original YAML for reference


def parse_pipe_table(table_str: str) -> List[Dict[str, str]]:
    """
    Parse a pipe-delimited table string into list of dicts.
    
    Parameters
    ----------
    table_str : str
        Multi-line string with pipe-delimited table
    
    Returns
    -------
    rows : list of dict
        Each row as a dict with column names as keys
    """
    lines = [line.strip() for line in table_str.strip().split('\n')]
    lines = [line for line in lines if line and not line.startswith('---')]
    
    if len(lines) < 2:
        return []
    
    # Parse header
    header_line = lines[0]
    headers = [h.strip() for h in header_line.split('|')]
    headers = [h for h in headers if h]  # Remove empty strings
    
    # Skip separator line (---)
    data_start = 1
    if lines[1].replace('-', '').replace('|', '').strip() == '':
        data_start = 2
    
    # Parse data rows
    rows = []
    for line in lines[data_start:]:
        if not line or line.replace('-', '').replace('|', '').strip() == '':
            continue
        
        values = [v.strip() for v in line.split('|')]
        values = [v for v in values if v != '']
        
        if len(values) != len(headers):
            # Try to handle missing trailing values
            while len(values) < len(headers):
                values.append('')
        
        row = {headers[i]: values[i] if i < len(values) else '' 
               for i in range(len(headers))}
        rows.append(row)
    
    return rows


def parse_list(value: str, separator: str = ',') -> List[str]:
    """Parse comma-separated list."""
    if not value or value.strip() == '':
        return []
    return [v.strip() for v in value.split(separator)]


def parse_bool(value: str) -> bool:
    """Parse boolean value."""
    return value.lower() in ('true', 'yes', '1', 'on')


def parse_info_table(info_str: str) -> Dict[str, List[str]]:
    """Parse info table into dict of MS -> fields."""
    rows = parse_pipe_table(info_str)
    info = {}
    for row in rows:
        ms = row.get('MS file', '').strip()
        fields_str = row.get('Fields', '').strip()
        if ms:
            info[ms] = parse_list(fields_str)
    return info


def parse_solve_table(solve_str: str) -> List[SolveEntry]:
    """Parse solve_jones table into list of SolveEntry."""
    rows = parse_pipe_table(solve_str)
    entries = []
    
    for row in rows:
        entry = SolveEntry(
            jones_types=parse_list(row.get('Jones types', '')),
            fields=parse_list(row.get('Fields', '')),
            scans=row.get('Scans', '*').strip(),
            spw=row.get('Spw', '*').strip(),
            freq_interval=row.get('Freq interval', 'full').strip(),
            time_interval=row.get('Time interval', 'inf').strip(),
            pre_apply_jones=parse_list(row.get('Pre-solved Jones types', '')),
            pre_apply_tables=parse_list(row.get('Pre-solved calibration tables', '')),
            output_table=row.get('Output table', '').strip(),
            model_column=row.get('Model column', 'MODEL_DATA').strip(),
            parallactic=parse_bool(row.get('Parallactic', 'false'))
        )
        entries.append(entry)
    
    return entries


def parse_apply_table(apply_str: str) -> List[ApplyEntry]:
    """Parse apply_jones table into list of ApplyEntry."""
    rows = parse_pipe_table(apply_str)
    entries = []
    
    for row in rows:
        entry = ApplyEntry(
            ms_file=row.get('MS file', '').strip(),
            jones_terms=parse_list(row.get('Jones terms', '')),
            from_fields=parse_list(row.get('From fields', '')),
            scans=row.get('Scans', '*').strip(),
            spw=row.get('Spw', '*').strip(),
            freq_interp=row.get('Freq interpolation', 'linear').strip(),
            time_interp=row.get('Time interpolation', 'linear').strip(),
            cal_tables=parse_list(row.get('Calibration tables', '')),
            output_column=row.get('Output column', 'CORRECTED_DATA').strip(),
            parallactic=parse_bool(row.get('Parallactic', 'false'))
        )
        entries.append(entry)
    
    return entries


def load_config(config_path: str) -> CalConfig:
    """
    Load calibration configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
    
    Returns
    -------
    config : CalConfig
        Parsed configuration
    """
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    
    # Parse each section
    info = parse_info_table(raw.get('info', ''))
    solve_entries = parse_solve_table(raw.get('solve_jones', ''))
    apply_entries = parse_apply_table(raw.get('apply_jones', ''))
    
    return CalConfig(
        info=info,
        solve_entries=solve_entries,
        apply_entries=apply_entries,
        raw_yaml=raw
    )


def parse_scan_range(scan_str: str) -> Optional[List[int]]:
    """
    Parse scan range string.
    
    Examples:
        "*" -> None (all scans)
        "1~10" -> [1, 2, 3, ..., 10]
        "1,3,5" -> [1, 3, 5]
        "1~5,10~15" -> [1,2,3,4,5,10,11,12,13,14,15]
    
    Returns
    -------
    scans : list of int or None
        None means all scans
    """
    if scan_str.strip() == '*':
        return None
    
    scans = []
    for part in scan_str.split(','):
        part = part.strip()
        if '~' in part:
            start, end = part.split('~')
            scans.extend(range(int(start), int(end) + 1))
        else:
            scans.append(int(part))
    
    return scans


def parse_spw_range(spw_str: str) -> Optional[List[int]]:
    """Parse SPW range (same format as scan)."""
    return parse_scan_range(spw_str)


def parse_time_interval(time_str: str) -> Optional[float]:
    """
    Parse time interval string.
    
    Examples:
        "inf" -> None (average all)
        "30s" -> 30.0
        "1m" -> 60.0
        "0.5h" -> 1800.0
    
    Returns
    -------
    interval : float or None
        Interval in seconds, None for infinite
    """
    time_str = time_str.strip().lower()
    
    if time_str == 'inf':
        return None
    
    # Parse with unit
    match = re.match(r'^([\d.]+)\s*([smh]?)$', time_str)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        
        if unit == 'm':
            value *= 60
        elif unit == 'h':
            value *= 3600
        
        return value
    
    return float(time_str)


def parse_freq_interval(freq_str: str) -> str:
    """
    Parse frequency interval string.
    
    Returns one of:
        "full" - average all frequencies
        "per_spw" - one solution per SPW
        "per_channel" - one solution per channel
        or Hz value as string
    """
    freq_str = freq_str.strip().lower()
    
    if freq_str in ('full', 'per_spw', 'per_channel'):
        return freq_str
    
    # Could be Hz value
    return freq_str


__all__ = [
    'CalConfig', 'SolveEntry', 'ApplyEntry',
    'load_config',
    'parse_scan_range', 'parse_spw_range',
    'parse_time_interval', 'parse_freq_interval'
]
