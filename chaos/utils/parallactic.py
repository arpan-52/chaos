"""
Parallactic Angle Computation.

Compute parallactic angle from antenna position, source position, and time.
"""

import numpy as np
from typing import Optional, Tuple


def compute_parallactic_angle(
    hour_angle: float,
    declination: float,
    latitude: float,
) -> float:
    """
    Compute parallactic angle.
    
    Formula:
        tan(ψ) = sin(HA) / (cos(δ) * tan(φ) - sin(δ) * cos(HA))
    
    where:
        HA = hour angle
        δ = declination
        φ = observatory latitude
    
    Parameters
    ----------
    hour_angle : float
        Hour angle in radians
    declination : float  
        Source declination in radians
    latitude : float
        Observatory latitude in radians
    
    Returns
    -------
    psi : float
        Parallactic angle in radians
    """
    sin_ha = np.sin(hour_angle)
    cos_ha = np.cos(hour_angle)
    sin_dec = np.sin(declination)
    cos_dec = np.cos(declination)
    tan_lat = np.tan(latitude)
    
    numerator = sin_ha
    denominator = cos_dec * tan_lat - sin_dec * cos_ha
    
    # Handle edge cases
    if np.abs(denominator) < 1e-10:
        if numerator >= 0:
            return np.pi / 2
        else:
            return -np.pi / 2
    
    return np.arctan2(numerator, denominator)


def compute_parallactic_angles(
    times: np.ndarray,
    ra: float,
    dec: float,
    ant_positions: np.ndarray,
    array_position: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute parallactic angles for multiple times and antennas.
    
    Parameters
    ----------
    times : ndarray (n_time,)
        Times in MJD seconds
    ra : float
        Source right ascension in radians
    dec : float
        Source declination in radians
    ant_positions : ndarray (n_ant, 3)
        Antenna positions in ITRF (meters)
    array_position : ndarray (3,), optional
        Array center position. If None, use mean of antennas.
    
    Returns
    -------
    psi : ndarray (n_time, n_ant)
        Parallactic angles in radians
    """
    n_time = len(times)
    n_ant = ant_positions.shape[0]
    
    # Get array center
    if array_position is None:
        array_position = ant_positions.mean(axis=0)
    
    # Convert array position to geodetic coordinates
    latitude = _itrf_to_latitude(array_position)
    longitude = _itrf_to_longitude(array_position)
    
    # Compute parallactic angle for each time
    psi = np.zeros((n_time, n_ant))
    
    for t_idx, t in enumerate(times):
        # Compute hour angle
        # LST = GMST + longitude
        gmst = _mjd_to_gmst(t / 86400.0)  # Convert seconds to days
        lst = gmst + longitude
        ha = lst - ra
        
        # Compute parallactic angle (same for all antennas at same site)
        psi_t = compute_parallactic_angle(ha, dec, latitude)
        
        psi[t_idx, :] = psi_t
    
    return psi


def compute_parallactic_angles_from_ms(
    ms_path: str,
    field_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute parallactic angles from MS metadata.
    
    Parameters
    ----------
    ms_path : str
        Path to MeasurementSet
    field_id : int
        Field ID for source position
    
    Returns
    -------
    times : ndarray (n_time,)
        Unique timestamps
    psi : ndarray (n_time, n_ant)
        Parallactic angles in radians
    """
    from casacore.tables import table
    
    # Get source position from FIELD table
    with table(f"{ms_path}/FIELD", ack=False) as tb:
        phase_dir = tb.getcol("PHASE_DIR")  # (n_field, n_poly, 2)
        ra = phase_dir[field_id, 0, 0]
        dec = phase_dir[field_id, 0, 1]
    
    # Get antenna positions
    with table(f"{ms_path}/ANTENNA", ack=False) as tb:
        ant_positions = tb.getcol("POSITION")  # (n_ant, 3)
    
    # Get unique times
    with table(ms_path, ack=False) as tb:
        times = np.unique(tb.getcol("TIME"))
    
    psi = compute_parallactic_angles(times, ra, dec, ant_positions)
    
    return times, psi


def _itrf_to_latitude(position: np.ndarray) -> float:
    """Convert ITRF position to geodetic latitude."""
    x, y, z = position
    
    # WGS84 parameters
    a = 6378137.0  # semi-major axis
    b = 6356752.314245  # semi-minor axis
    e2 = (a**2 - b**2) / a**2  # first eccentricity squared
    
    # Iterative solution
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    
    for _ in range(10):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        lat_new = np.arctan2(z + e2 * N * np.sin(lat), p)
        if np.abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new
    
    return lat


def _itrf_to_longitude(position: np.ndarray) -> float:
    """Convert ITRF position to geodetic longitude."""
    x, y, _ = position
    return np.arctan2(y, x)


def _mjd_to_gmst(mjd: float) -> float:
    """
    Convert Modified Julian Date to Greenwich Mean Sidereal Time.
    
    Parameters
    ----------
    mjd : float
        Modified Julian Date
    
    Returns
    -------
    gmst : float
        GMST in radians
    """
    # Julian centuries from J2000.0
    T = (mjd - 51544.5) / 36525.0
    
    # GMST at 0h UT (in seconds)
    gmst_0h = 24110.54841 + 8640184.812866 * T + 0.093104 * T**2 - 6.2e-6 * T**3
    
    # Fraction of day
    frac = mjd % 1.0
    
    # Total GMST in seconds
    gmst_sec = gmst_0h + 86400.0 * 1.00273790935 * frac
    
    # Convert to radians
    gmst_rad = (gmst_sec % 86400.0) * 2 * np.pi / 86400.0
    
    return gmst_rad
