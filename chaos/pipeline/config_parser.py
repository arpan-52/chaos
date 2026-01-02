"""
Configuration Parser.

Parse pipe-delimited YAML configuration files.
"""

import yaml
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


@dataclass
class SolveEntry:
    """Single solve_jones entry."""
    jones_types: List[str]           # e.g., ['K', 'B', 'G']
    fields: List[str]                # e.g., ['3C147', '3C147', '3C147:3C286']
    scans: str                       # e.g., '*' or '1~10'
    spw: str                         # e.g., '*' or '0~15'
    freq_interval: str               # 'full', 'per_channel', 'per_spw', '4MHz'
    time_interval: str               # 'inf', '30s', '1m', 'int', 'scan'
    pre_apply_jones: List[str]       # Jones types to pre-apply
    pre_apply_tables: List[str]      # Tables for pre-apply
    output_table: str                # Output HDF5 file
    model_column: str                # MODEL_DATA
    parallactic: bool                # Correct for parallactic angle


@dataclass
class ApplyEntry:
    """Single apply_jones entry."""
    ms_file: str
    jones_types: List[str]
    from_fields: List[str]
    scans: str
    spw: str
    freq_interp: str                 # 'nearest', 'linear', 'cubic'
    time_interp: str
    cal_tables: List[str]
    output_column: str
    parallactic: bool


@dataclass
class CalConfig:
    """Complete calibration configuration."""
    info: Dict[str, List[str]]       # MS -> fields mapping
    solve_entries: List[SolveEntry]
    apply_entries: List[ApplyEntry]


def load_config(filepath: str) -> CalConfig:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    filepath : str
        Path to YAML configuration file
    
    Returns
    -------
    config : CalConfig
    """
    with open(filepath, "r") as f:
        raw = yaml.safe_load(f)
    
    # Parse info table
    info = {}
    if "info" in raw:
        info = _parse_info_table(raw["info"])
    
    # Parse solve_jones table
    solve_entries = []
    if "solve_jones" in raw:
        solve_entries = _parse_solve_table(raw["solve_jones"])
    
    # Parse apply_jones table
    apply_entries = []
    if "apply_jones" in raw:
        apply_entries = _parse_apply_table(raw["apply_jones"])
    
    return CalConfig(
        info=info,
        solve_entries=solve_entries,
        apply_entries=apply_entries,
    )


def _parse_pipe_table(table_str: str) -> List[Dict[str, str]]:
    """
    Parse pipe-delimited table string.
    
    Format:
        Header1 | Header2 | Header3
        ------- | ------- | -------
        value1  | value2  | value3
    
    Returns list of dicts mapping header -> value.
    """
    lines = [l.strip() for l in table_str.strip().split("\n") if l.strip()]
    
    if len(lines) < 2:
        return []
    
    # Parse header
    header_line = lines[0]
    headers = [h.strip() for h in header_line.split("|")]
    
    # Skip separator line (dashes)
    data_start = 1
    if len(lines) > 1 and re.match(r"^[-|\s]+$", lines[1]):
        data_start = 2
    
    # Parse data rows
    rows = []
    for line in lines[data_start:]:
        values = [v.strip() for v in line.split("|")]
        
        # Pad values if needed
        while len(values) < len(headers):
            values.append("")
        
        row = {h: v for h, v in zip(headers, values)}
        rows.append(row)
    
    return rows


def _parse_info_table(table_str: str) -> Dict[str, List[str]]:
    """Parse info table: MS -> fields mapping."""
    rows = _parse_pipe_table(table_str)
    
    info = {}
    for row in rows:
        ms_file = row.get("MS file", "")
        fields_str = row.get("Fields", "")
        
        if ms_file:
            fields = [f.strip() for f in fields_str.split(",")]
            info[ms_file] = fields
    
    return info


def _parse_solve_table(table_str: str) -> List[SolveEntry]:
    """Parse solve_jones table."""
    rows = _parse_pipe_table(table_str)
    
    entries = []
    for row in rows:
        # Parse Jones types
        jones_str = row.get("Jones", row.get("Jones types", ""))
        jones_types = [j.strip() for j in jones_str.split(",") if j.strip()]
        
        # Parse fields
        fields_str = row.get("Fields", "")
        fields = [f.strip() for f in fields_str.split(",") if f.strip()]
        
        # Ensure same number of fields as jones types
        while len(fields) < len(jones_types):
            fields.append(fields[-1] if fields else "")
        
        # Parse pre-apply
        pre_apply_str = row.get("Pre-apply", row.get("Pre-solved Jones types", ""))
        pre_apply_jones = [j.strip() for j in pre_apply_str.split(",") if j.strip()]
        
        pre_tables_str = row.get("Pre-tables", row.get("Pre-solved calibration tables", ""))
        pre_apply_tables = [t.strip() for t in pre_tables_str.split(",") if t.strip()]
        
        # Parse booleans
        parang_str = row.get("Parang", row.get("Parallactic", "false"))
        parallactic = parang_str.lower() in ("true", "yes", "1")
        
        entry = SolveEntry(
            jones_types=jones_types,
            fields=fields,
            scans=row.get("Scans", "*"),
            spw=row.get("Spw", "*"),
            freq_interval=row.get("Freq int", row.get("Freq interval", "full")),
            time_interval=row.get("Time int", row.get("Time interval", "inf")),
            pre_apply_jones=pre_apply_jones,
            pre_apply_tables=pre_apply_tables,
            output_table=row.get("Output", row.get("Output table", "cal.h5")),
            model_column=row.get("Model col", row.get("Model column", "MODEL_DATA")),
            parallactic=parallactic,
        )
        entries.append(entry)
    
    return entries


def _parse_apply_table(table_str: str) -> List[ApplyEntry]:
    """Parse apply_jones table."""
    rows = _parse_pipe_table(table_str)
    
    entries = []
    for row in rows:
        # Parse Jones types
        jones_str = row.get("Jones", row.get("Jones terms", ""))
        jones_types = [j.strip() for j in jones_str.split(",") if j.strip()]
        
        # Parse from fields
        from_str = row.get("From fields", row.get("Fields", ""))
        from_fields = [f.strip() for f in from_str.split(",") if f.strip()]
        
        # Parse tables
        tables_str = row.get("Tables", row.get("Calibration tables", ""))
        cal_tables = [t.strip() for t in tables_str.split(",") if t.strip()]
        
        # Parse booleans
        parang_str = row.get("Parang", row.get("Parallactic", "false"))
        parallactic = parang_str.lower() in ("true", "yes", "1")
        
        entry = ApplyEntry(
            ms_file=row.get("MS file", row.get("MS", "")),
            jones_types=jones_types,
            from_fields=from_fields,
            scans=row.get("Scans", "*"),
            spw=row.get("Spw", "*"),
            freq_interp=row.get("Freq interp", row.get("Freq interpolation", "linear")),
            time_interp=row.get("Time interp", row.get("Time interpolation", "linear")),
            cal_tables=cal_tables,
            output_column=row.get("Output col", row.get("Output column", "CORRECTED_DATA")),
            parallactic=parallactic,
        )
        entries.append(entry)
    
    return entries


def parse_scan_range(scan_str: str) -> Optional[List[int]]:
    """
    Parse scan range string.
    
    Examples:
        '*' -> None (all scans)
        '1~10' -> [1, 2, ..., 10]
        '1,3,5' -> [1, 3, 5]
    """
    if scan_str == "*" or not scan_str:
        return None
    
    scans = []
    for part in scan_str.split(","):
        part = part.strip()
        if "~" in part:
            start, end = part.split("~")
            scans.extend(range(int(start), int(end) + 1))
        else:
            scans.append(int(part))
    
    return scans


def parse_spw_range(spw_str: str) -> Optional[List[int]]:
    """Parse SPW range (same format as scan range)."""
    return parse_scan_range(spw_str)


def parse_time_interval(interval_str: str) -> Optional[float]:
    """
    Parse time interval string to seconds.
    
    Examples:
        'inf' -> None (single solution)
        '30s' -> 30.0
        '1m' -> 60.0
        'int' -> 0.0 (per integration)
        'scan' -> -1.0 (per scan)
    """
    interval_str = interval_str.lower().strip()
    
    if interval_str in ("inf", "all"):
        return None
    if interval_str == "int":
        return 0.0
    if interval_str == "scan":
        return -1.0
    
    # Parse with units
    match = re.match(r"(\d+(?:\.\d+)?)\s*([smh]?)", interval_str)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        
        if unit == "m":
            value *= 60
        elif unit == "h":
            value *= 3600
        
        return value
    
    return None


def parse_freq_interval(interval_str: str) -> str:
    """
    Parse frequency interval string.
    
    Returns normalized string: 'full', 'per_channel', 'per_spw', or frequency in Hz.
    """
    interval_str = interval_str.lower().strip()
    
    if interval_str in ("full", "all"):
        return "full"
    if interval_str in ("per_channel", "channel"):
        return "per_channel"
    if interval_str in ("per_spw", "spw"):
        return "per_spw"
    
    # Parse frequency with units
    match = re.match(r"(\d+(?:\.\d+)?)\s*(hz|khz|mhz|ghz)?", interval_str, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = (match.group(2) or "hz").lower()
        
        multipliers = {"hz": 1, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
        value *= multipliers.get(unit, 1)
        
        return str(value)
    
    return "full"
