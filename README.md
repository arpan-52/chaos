# CHAOS - Chain-based Antenna-gain Optimization Solver

**Multi-reference weighted Jones calibration for radio interferometry**

## Core Innovation

Traditional calibration: solve from ONE reference antenna → noise propagates along chain

**CHAOS approach:**
1. Solve from **EVERY** antenna as reference
2. Align phases to user-specified reference
3. Weighted average using geometric mean of chain quality

Result: Robust solutions through redundancy!

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Basic usage
chaos mydata.ms

# Specify reference and mode
chaos mydata.ms --ref-ant 5 --mode diagonal

# Use ratio chain solver
chaos mydata.ms --solver ratio_chain

# Full Jones calibration
chaos mydata.ms --mode full
```

## Modes

| Mode | J_ref Parameters | Description |
|------|------------------|-------------|
| `phase_only` | I (identity) | Phases only |
| `diagonal` | diag(g_X, g_Y) real | Separate X/Y amplitudes |
| `full` | 2×2 complex | Full Jones (8 real params) |

## Algorithm

### Step 1: Solve from every antenna

For each antenna r = 0 to N-1:
- Build optimal chain from r using baseline quality
- Optimize J_ref (for diagonal: g_X, g_Y)
- Chain algebraically to all other antennas

### Step 2: Phase alignment

Align all solutions to user's reference antenna:
- Subtract phase of ref_ant from all antennas
- Amplitude unchanged

### Step 3: Weighted combination

For each antenna i:
```
J_i^(final) = Σ_r w_ir × J_i^(r,aligned) / Σ_r w_ir
```

Where `w_ir` = geometric mean of baseline qualities along chain from r to i.

## Output

- `chaos_cal.npy`: Jones matrices (n_ant, 2, 2)
- `chaos_cal_diagnostics.npz`: Full diagnostics including weights, chains, quality matrix

## Requirements

- numpy
- scipy
- python-casacore

Optional:
- cupy (GPU acceleration)

## Citation

```bibtex
@article{chaos2025,
  title={CHAOS: Multi-reference Weighted Jones Calibration},
  author={Pal, Arpan},
  year={2025}
}
```

## Author

Arpan Pal

December 2025
