"""
Apply calibration solutions to visibilities and write corrected data.

The measurement equation is:
    V_obs = J_i * V_sky * J_j^†

To recover the sky visibilities, we invert:
    V_sky = J_i^(-1) * V_obs * J_j^(-†)
"""

import numpy as np
from casacore.tables import table
from .array_ops import xp, to_cpu, to_gpu


def apply_jones_to_vis(vis, jones, antenna1, antenna2):
    """
    Apply Jones matrices to correct visibilities.

    V_obs = J_i × V_sky × J_j†
    V_sky = J_i^(-1) × V_obs × (J_j^(-1))†

    Parameters
    ----------
    vis : array, shape (n_baselines, 2, 2)
        Observed visibilities (complex)
    jones : array, shape (n_antennas, 2, 2)
        Jones matrices (complex)
    antenna1 : array, shape (n_baselines,)
        First antenna indices
    antenna2 : array, shape (n_baselines,)
        Second antenna indices

    Returns
    -------
    vis_corrected : array, shape (n_baselines, 2, 2)
        Corrected visibilities
    """
    n_bl = len(antenna1)
    vis_corrected = xp.zeros_like(vis)

    for i in range(n_bl):
        a1 = antenna1[i]
        a2 = antenna2[i]

        J_i_inv = xp.linalg.inv(jones[a1])
        J_j_inv = xp.linalg.inv(jones[a2])
        J_j_inv_H = J_j_inv.conj().T

        vis_corrected[i] = J_i_inv @ vis[i] @ J_j_inv_H

    return vis_corrected


def applycal(ms_path, jones_path,
             data_column='DATA',
             output_column='CORRECTED_DATA',
             overwrite=False):
    """
    Apply Jones matrix solutions to MS data and write corrected visibilities.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    jones_path : str
        Path to Jones matrices (.npy file from chaos calibration)
    data_column : str
        Input data column to correct
    output_column : str
        Output column name for corrected data
    overwrite : bool
        If True, overwrite existing output column
    """
    print(f"[CHAOS] Applying calibration solutions")
    print(f"[CHAOS]   MS: {ms_path}")
    print(f"[CHAOS]   Jones: {jones_path}")
    print(f"[CHAOS]   Input column: {data_column}")
    print(f"[CHAOS]   Output column: {output_column}")

    # Load Jones matrices
    jones = np.load(jones_path)
    print(f"[CHAOS]   Jones shape: {jones.shape}")
    jones_device = to_gpu(jones)

    # Open MS for reading and writing
    with table(ms_path, readonly=False, ack=False) as tb:
        colnames = tb.colnames()

        # Check output column
        if output_column in colnames:
            if not overwrite:
                print(f"[CHAOS] ERROR: Column {output_column} already exists. Use --overwrite to replace.")
                return
            else:
                print(f"[CHAOS]   Overwriting existing column {output_column}")
        else:
            print(f"[CHAOS]   Creating new column {output_column}")
            data_desc = tb.getcoldesc(data_column)
            tb.addcols({output_column: data_desc})

        # Read data
        print(f"[CHAOS] Reading {data_column}...")
        vis_obs = tb.getcol(data_column)
        flags = tb.getcol('FLAG')
        antenna1 = tb.getcol('ANTENNA1')
        antenna2 = tb.getcol('ANTENNA2')

        n_rows, n_chan, n_corr = vis_obs.shape
        print(f"[CHAOS]   Shape: {vis_obs.shape} (rows={n_rows}, chan={n_chan}, corr={n_corr})")

        # Determine correlation structure
        if n_corr == 4:
            corr_to_jones = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
            print("[CHAOS]   Polarization: Full (XX, XY, YX, YY)")
        elif n_corr == 2:
            corr_to_jones = {0: (0, 0), 1: (1, 1)}
            print("[CHAOS]   Polarization: Dual (XX, YY)")
        elif n_corr == 1:
            corr_to_jones = {0: (0, 0)}
            print("[CHAOS]   Polarization: Single")
        else:
            raise ValueError(f"Unsupported number of correlations: {n_corr}")

        # Apply corrections channel by channel
        vis_corrected = np.zeros_like(vis_obs)

        print("[CHAOS] Applying Jones matrix corrections...")
        for chan in range(n_chan):
            if chan % 100 == 0 or chan == n_chan - 1:
                print(f"[CHAOS]   Channel {chan+1}/{n_chan}")

            # Build 2x2 visibility matrices
            vis_matrices = xp.zeros((n_rows, 2, 2), dtype=complex)
            for corr_idx, (i, j) in corr_to_jones.items():
                vis_matrices[:, i, j] = to_gpu(vis_obs[:, chan, corr_idx])

            # Apply corrections
            vis_corr_matrices = apply_jones_to_vis(vis_matrices, jones_device, antenna1, antenna2)

            # Convert back
            vis_corr_matrices = to_cpu(vis_corr_matrices)
            for corr_idx, (i, j) in corr_to_jones.items():
                vis_corrected[:, chan, corr_idx] = vis_corr_matrices[:, i, j]

        # Write corrected data
        print(f"[CHAOS] Writing corrected data to {output_column}...")
        tb.putcol(output_column, vis_corrected)

        # Statistics
        unflagged = ~flags
        if unflagged.any():
            rms_before = np.sqrt(np.mean(np.abs(vis_obs[unflagged])**2))
            rms_after = np.sqrt(np.mean(np.abs(vis_corrected[unflagged])**2))
            print(f"[CHAOS]   RMS before: {rms_before:.6e}")
            print(f"[CHAOS]   RMS after:  {rms_after:.6e}")

        print(f"[CHAOS] Done.")


__all__ = ['applycal', 'apply_jones_to_vis']
