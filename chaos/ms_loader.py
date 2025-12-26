"""
MS Loader for CHAOS calibration.

Loads visibility data from CASA Measurement Sets.
"""

import numpy as np
from casacore.tables import table
import os
from .utils import corr_to_jones


class MSLoader:
    """
    Load data from Measurement Set.
    """

    def __init__(self, ms_path):
        """Initialize MS loader."""
        if not os.path.exists(ms_path):
            raise FileNotFoundError(f"MS not found: {ms_path}")
        self.ms_path = ms_path
        self.metadata = None

    def load_metadata(self, field_id=0, spw=0, scan_id=None):
        """
        Load MS metadata only.

        Returns
        -------
        metadata : dict
        """
        print(f"[MS] Loading metadata: {self.ms_path}")

        with table(self.ms_path) as t:
            query_str = f"FIELD_ID=={field_id} && DATA_DESC_ID=={spw}"
            if scan_id is not None:
                query_str += f" && SCAN_NUMBER=={scan_id}"

            t_sel = t.query(query_str)

            if t_sel.nrows() == 0:
                raise ValueError(f"No data for field {field_id}, spw {spw}")

            times = np.unique(t_sel.getcol('TIME'))
            time_step = np.median(np.diff(times)) if len(times) > 1 else 1.0

            # Get antenna count
            with table(os.path.join(self.ms_path, 'ANTENNA')) as t_ant:
                n_ant = t_ant.nrows()

            # Get frequencies
            with table(os.path.join(self.ms_path, 'SPECTRAL_WINDOW')) as t_spw:
                freqs = t_spw.getcol('CHAN_FREQ')[spw]
                chan_width = t_spw.getcol('CHAN_WIDTH')[spw][0]

            # Count baselines
            ant1 = t_sel.getcol('ANTENNA1')
            ant2 = t_sel.getcol('ANTENNA2')
            baselines = list(set(zip(ant1, ant2)))

            metadata = {
                'n_ant': n_ant,
                'times': times,
                'freqs': freqs,
                'channel_width': float(chan_width),
                'time_step': float(time_step),
                'n_baselines': len(baselines),
                'time_min': float(times.min()),
                'time_max': float(times.max()),
                'n_chan': len(freqs)
            }

        print(f"[MS]   Antennas: {n_ant}, Baselines: {metadata['n_baselines']}")
        print(f"[MS]   Times: {len(times)} samples, step: {time_step:.2f}s")
        print(f"[MS]   Channels: {len(freqs)}, width: {chan_width/1e6:.3f} MHz")

        self.metadata = metadata
        return metadata

    def load_data(self, field_id=0, spw=0, scan_id=None, model_column='MODEL_DATA'):
        """
        Load full data for calibration.

        Returns
        -------
        data : dict
            - vis_obs: ndarray (n_bl, 2, 2) complex - averaged over time/freq
            - vis_model: ndarray (n_bl, 2, 2) complex
            - flags: ndarray (n_bl, 2, 2) bool
            - antenna1, antenna2: ndarray (n_bl,)
            - n_ant: int
        """
        print(f"[MS] Loading data for field={field_id}, spw={spw}")

        with table(self.ms_path) as t:
            query_str = f"FIELD_ID=={field_id} && DATA_DESC_ID=={spw}"
            if scan_id is not None:
                query_str += f" && SCAN_NUMBER=={scan_id}"

            t_sel = t.query(query_str)

            if t_sel.nrows() == 0:
                raise ValueError(f"No data for field {field_id}, spw {spw}")

            # Read all data
            antenna1 = t_sel.getcol('ANTENNA1')
            antenna2 = t_sel.getcol('ANTENNA2')
            data = t_sel.getcol('DATA')  # (n_rows, n_chan, n_corr)
            model = t_sel.getcol(model_column)

            # Flags
            if 'FLAG' in t_sel.colnames():
                flags = t_sel.getcol('FLAG')
            else:
                flags = np.zeros_like(data, dtype=bool)

            # Weights
            if 'WEIGHT_SPECTRUM' in t_sel.colnames():
                weights = t_sel.getcol('WEIGHT_SPECTRUM')
            elif 'WEIGHT' in t_sel.colnames():
                w = t_sel.getcol('WEIGHT')
                weights = np.repeat(w[:, np.newaxis, :], data.shape[1], axis=1)
            else:
                weights = np.ones_like(data.real)

            # Zero weight where flagged
            weights[flags] = 0.0

        # Get n_ant
        with table(os.path.join(self.ms_path, 'ANTENNA')) as t_ant:
            n_ant = t_ant.nrows()

        # Average per baseline
        baselines = antenna1 * 10000 + antenna2
        unique_bl = np.unique(baselines)

        vis_obs_list = []
        vis_model_list = []
        flags_list = []
        ant1_list = []
        ant2_list = []

        for bl in unique_bl:
            mask = baselines == bl
            d = data[mask]
            m = model[mask]
            w = weights[mask]
            f = flags[mask]

            # Per correlation averaging
            vis_obs_corr = np.zeros(4, dtype=complex)
            vis_model_corr = np.zeros(4, dtype=complex)
            flags_corr = np.zeros(4, dtype=bool)

            for corr in range(4):
                d_corr = d[:, :, corr]
                m_corr = m[:, :, corr]
                w_corr = w[:, :, corr]
                f_corr = f[:, :, corr]

                w_sum = w_corr.sum()
                if w_sum > 0:
                    vis_obs_corr[corr] = (d_corr * w_corr).sum() / w_sum
                    vis_model_corr[corr] = (m_corr * w_corr).sum() / w_sum
                    flags_corr[corr] = f_corr.all()
                else:
                    flags_corr[corr] = True

            vis_obs_list.append(vis_obs_corr)
            vis_model_list.append(vis_model_corr)
            flags_list.append(flags_corr)
            ant1_list.append(antenna1[mask][0])
            ant2_list.append(antenna2[mask][0])

        vis_obs = np.array(vis_obs_list)
        vis_model = np.array(vis_model_list)
        flags_arr = np.array(flags_list)

        # Convert to Jones format
        vis_obs_jones = corr_to_jones(vis_obs)
        vis_model_jones = corr_to_jones(vis_model)
        flags_jones = corr_to_jones(flags_arr.astype(float)).astype(bool)

        print(f"[MS]   Loaded: {len(vis_obs_jones)} baselines, {n_ant} antennas")

        return {
            'vis_obs': vis_obs_jones,
            'vis_model': vis_model_jones,
            'flags': flags_jones,
            'antenna1': np.array(ant1_list),
            'antenna2': np.array(ant2_list),
            'n_ant': n_ant
        }


__all__ = ['MSLoader']
