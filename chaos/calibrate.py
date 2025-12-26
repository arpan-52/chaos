"""
CHAOS Calibration Pipeline.

Main entry point for Jones calibration.
"""

import numpy as np
from .ms_loader import MSLoader
from .quality import compute_quality_matrix, compute_baseline_quality
from .weighted_combiner import multi_ref_solve_and_combine
from .utils import compute_residuals


def detect_bad_antennas(vis_obs, antenna1, antenna2, flags, threshold=0.05):
    """
    Detect bad antennas using amplitude-based method.

    Parameters
    ----------
    vis_obs : ndarray (n_bl, 2, 2)
    antenna1, antenna2 : ndarray (n_bl,)
    flags : ndarray (n_bl, 2, 2)
    threshold : float
        Fraction of reference amplitude below which antenna is flagged

    Returns
    -------
    bad_antennas : set
    """
    n_ant = max(antenna1.max(), antenna2.max()) + 1
    antenna_amps = {i: [] for i in range(n_ant)}

    # Collect amplitudes per antenna
    for idx, (a1, a2) in enumerate(zip(antenna1, antenna2)):
        if flags[idx].any():
            continue
        amp = np.abs(vis_obs[idx]).mean()
        antenna_amps[a1].append(amp)
        antenna_amps[a2].append(amp)

    # Compute mean per antenna
    antenna_means = {}
    for ant, amps in antenna_amps.items():
        if len(amps) > 0:
            antenna_means[ant] = np.mean(amps)
        else:
            antenna_means[ant] = 0.0

    # Reference: 75th percentile
    all_means = [v for v in antenna_means.values() if v > 0]
    if len(all_means) == 0:
        return set()

    ref_amp = np.percentile(all_means, 75)
    cutoff = threshold * ref_amp

    # Find bad antennas
    bad_antennas = set()
    for ant, mean_amp in antenna_means.items():
        if mean_amp < cutoff and mean_amp > 0:
            bad_antennas.add(ant)
            print(f"[CHAOS] Bad antenna: {ant} (amp={mean_amp:.3f}, cutoff={cutoff:.3f})")

    return bad_antennas


def flag_rfi_median(vis_obs, sigma=5.0):
    """
    Simple median-based RFI flagging.

    Parameters
    ----------
    vis_obs : ndarray (n_bl, 2, 2)
    sigma : float

    Returns
    -------
    flags : ndarray (n_bl, 2, 2) bool
    """
    n_bl = vis_obs.shape[0]
    flags = np.zeros((n_bl, 2, 2), dtype=bool)

    for i in range(2):
        for j in range(2):
            amp = np.abs(vis_obs[:, i, j])
            median = np.median(amp)
            mad = np.median(np.abs(amp - median))

            if mad == 0:
                continue

            sigma_robust = 1.4826 * mad
            flags[:, i, j] = np.abs(amp - median) > sigma * sigma_robust

    n_flagged = flags.sum()
    if n_flagged > 0:
        print(f"[CHAOS] RFI flagged: {n_flagged}/{flags.size} ({100*n_flagged/flags.size:.1f}%)")

    return flags


def calibrate_ms(
    ms_path,
    ref_antenna=0,
    mode='diagonal',
    solver='single_chain',
    field_id=0,
    spw=0,
    model_column='MODEL_DATA',
    rfi_sigma=5.0,
    bad_ant_threshold=0.05,
    max_iter=100,
    output_prefix='chaos_cal'
):
    """
    Main calibration pipeline.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    ref_antenna : int
        Reference antenna for final alignment
    mode : str
        'phase_only', 'diagonal', or 'full'
    solver : str
        'single_chain' or 'ratio_chain'
    field_id : int
    spw : int
    model_column : str
    rfi_sigma : float
    bad_ant_threshold : float
    max_iter : int
    output_prefix : str

    Returns
    -------
    jones : ndarray (n_ant, 2, 2)
    diagnostics : dict
    """
    print("="*70)
    print("CHAOS - Chain-based Algebraic Optimal Solver")
    print("Multi-reference weighted Jones calibration")
    print("="*70)

    # Load data
    print(f"\n[CHAOS] Loading MS: {ms_path}")
    loader = MSLoader(ms_path)
    data = loader.load_data(field_id=field_id, spw=spw, model_column=model_column)

    vis_obs = data['vis_obs']
    vis_model = data['vis_model']
    flags = data['flags']
    antenna1 = data['antenna1']
    antenna2 = data['antenna2']
    n_ant = data['n_ant']

    print(f"[CHAOS] Data shape: {vis_obs.shape}, Antennas: {n_ant}")

    # Bad antenna detection
    print(f"\n[CHAOS] Detecting bad antennas (threshold={bad_ant_threshold})...")
    bad_antennas = detect_bad_antennas(vis_obs, antenna1, antenna2, flags, bad_ant_threshold)
    print(f"[CHAOS] Bad antennas: {sorted(bad_antennas) if bad_antennas else 'None'}")

    # RFI flagging
    print(f"\n[CHAOS] RFI flagging (sigma={rfi_sigma})...")
    rfi_flags = flag_rfi_median(vis_obs, sigma=rfi_sigma)
    flags_combined = flags | rfi_flags

    # Compute quality matrix
    print(f"\n[CHAOS] Computing baseline quality matrix...")
    quality_matrix = compute_quality_matrix(vis_obs, flags_combined, antenna1, antenna2, n_ant)

    # Multi-reference solving
    jones_final, diagnostics = multi_ref_solve_and_combine(
        vis_obs, vis_model, antenna1, antenna2,
        quality_matrix, bad_antennas, ref_antenna,
        mode, solver, max_iter
    )

    # Compute final residuals
    print(f"\n[CHAOS] Computing residuals...")
    res_stats = compute_residuals(jones_final, vis_obs, vis_model, antenna1, antenna2)
    print(f"[CHAOS] Final RMS: {res_stats['rms']:.6e}")
    print(f"[CHAOS]   XX: {res_stats['XX_rms']:.6e}, YY: {res_stats['YY_rms']:.6e}")
    print(f"[CHAOS]   XY: {res_stats['XY_rms']:.6e}, YX: {res_stats['YX_rms']:.6e}")

    # Save results
    print(f"\n[CHAOS] Saving results...")

    # Main Jones file
    output_jones = f"{output_prefix}.npy"
    np.save(output_jones, jones_final)
    print(f"[CHAOS]   Jones: {output_jones} {jones_final.shape}")

    # Diagnostics
    output_diag = f"{output_prefix}_diagnostics.npz"
    np.savez(
        output_diag,
        jones=jones_final,
        ref_antenna=ref_antenna,
        mode=mode,
        solver=solver,
        bad_antennas=np.array(list(bad_antennas)),
        quality_matrix=quality_matrix,
        weights=diagnostics['weights'],
        residual_rms=res_stats['rms'],
        n_ant=n_ant
    )
    print(f"[CHAOS]   Diagnostics: {output_diag}")

    print(f"\n{'='*70}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*70}")

    # Add to diagnostics
    diagnostics['residuals'] = res_stats
    diagnostics['bad_antennas'] = bad_antennas
    diagnostics['quality_matrix'] = quality_matrix

    return jones_final, diagnostics


__all__ = ['calibrate_ms']
