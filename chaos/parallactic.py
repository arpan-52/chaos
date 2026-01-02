"""
Parallactic Angle Computation.

Computes parallactic angle for each antenna at each time for a given source.

The parallactic angle is the angle between:
- The great circle through the source and celestial north pole
- The great circle through the source and local zenith

For alt-az mounts, this angle changes as the source is tracked.
For equatorial mounts, the parallactic angle is constant (typically 0).
"""

import numpy as np
from typing import Optional, Tuple


def compute_parallactic_angle(
    time_mjd: np.ndarray,
    ra_rad: float,
    dec_rad: float,
    longitude_rad: float,
    latitude_rad: float
) -> np.ndarray:
    """
    Compute parallactic angle for alt-az mounted antenna.
    
    Parameters
    ----------
    time_mjd : ndarray
        Time in MJD (days)
    ra_rad : float
        Source right ascension in radians
    dec_rad : float
        Source declination in radians
    longitude_rad : float
        Antenna longitude in radians (East positive)
    latitude_rad : float
        Antenna latitude in radians
    
    Returns
    -------
    psi : ndarray
        Parallactic angle in radians
    """
    # Convert MJD to LST (approximate)
    # LST = GMST + longitude
    # GMST ≈ 18.697374558 + 24.06570982441908 * D (hours)
    # where D = JD - 2451545.0
    
    jd = time_mjd + 2400000.5
    D = jd - 2451545.0
    
    # GMST in hours
    gmst_hours = 18.697374558 + 24.06570982441908 * D
    gmst_hours = gmst_hours % 24.0
    
    # GMST in radians
    gmst_rad = gmst_hours * (2 * np.pi / 24.0)
    
    # LST in radians
    lst_rad = gmst_rad + longitude_rad
    
    # Hour angle
    ha = lst_rad - ra_rad
    
    # Parallactic angle
    # tan(psi) = sin(HA) / (cos(dec)*tan(lat) - sin(dec)*cos(HA))
    sin_ha = np.sin(ha)
    cos_ha = np.cos(ha)
    sin_dec = np.sin(dec_rad)
    cos_dec = np.cos(dec_rad)
    tan_lat = np.tan(latitude_rad)
    
    psi = np.arctan2(sin_ha, cos_dec * tan_lat - sin_dec * cos_ha)
    
    return psi


def compute_parallactic_angles_from_ms(
    ms_path: str,
    field_id: int = 0,
    antenna_ids: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute parallactic angles for all times and antennas in an MS.
    
    Parameters
    ----------
    ms_path : str
        Path to MeasurementSet
    field_id : int
        Field ID to use
    antenna_ids : ndarray, optional
        Specific antenna IDs (default: all)
    
    Returns
    -------
    times : ndarray
        Unique times (MJD seconds)
    psi : ndarray (n_time, n_ant)
        Parallactic angles in radians
    """
    from casacore.tables import table
    
    # Get source position from FIELD table
    with table(f"{ms_path}/FIELD", ack=False) as tb:
        phase_dir = tb.getcol('PHASE_DIR')  # (n_field, n_poly, 2)
        ra_rad = phase_dir[field_id, 0, 0]
        dec_rad = phase_dir[field_id, 0, 1]
    
    # Get antenna positions from ANTENNA table
    with table(f"{ms_path}/ANTENNA", ack=False) as tb:
        positions = tb.getcol('POSITION')  # ITRF (n_ant, 3)
        n_ant = len(positions)
        
        if antenna_ids is None:
            antenna_ids = np.arange(n_ant)
        
        # Convert ITRF to geodetic (approximate)
        # Using simple spherical approximation
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        latitude_rad = np.arcsin(z / r)
        longitude_rad = np.arctan2(y, x)
    
    # Get unique times from main table
    with table(ms_path, ack=False) as tb:
        times = np.unique(tb.getcol('TIME'))  # MJD seconds
    
    times_mjd = times / 86400.0  # Convert to MJD days
    
    # Compute parallactic angle for each antenna at each time
    n_time = len(times)
    n_ant_sel = len(antenna_ids)
    psi = np.zeros((n_time, n_ant_sel))
    
    for i, ant in enumerate(antenna_ids):
        psi[:, i] = compute_parallactic_angle(
            times_mjd,
            ra_rad,
            dec_rad,
            longitude_rad[ant],
            latitude_rad[ant]
        )
    
    return times, psi


def get_mount_type(ms_path: str) -> str:
    """
    Detect antenna mount type from MS.
    
    Returns
    -------
    mount_type : str
        'alt-az', 'equatorial', 'x-y', or 'unknown'
    """
    from casacore.tables import table
    
    with table(f"{ms_path}/ANTENNA", ack=False) as tb:
        if 'MOUNT' in tb.colnames():
            mounts = tb.getcol('MOUNT')
            # Most common mount
            mount = mounts[0].lower()
            if 'alt' in mount or 'az' in mount:
                return 'alt-az'
            elif 'eq' in mount:
                return 'equatorial'
            elif 'x-y' in mount or 'xy' in mount:
                return 'x-y'
    
    return 'unknown'


__all__ = [
    'compute_parallactic_angle',
    'compute_parallactic_angles_from_ms',
    'get_mount_type'
]
