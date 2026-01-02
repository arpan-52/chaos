# CHAOS - Chain-based Algebraic Optimal Solver

Multi-reference weighted Jones calibration for radio interferometry.

## Features

- **Algebraic Chaining**: Reduces N-antenna optimization to single reference antenna
- **Multi-Reference Weighted Averaging**: Solves from every antenna as reference, weighted combination
- **Least Squares Polish**: Refines algebraic solution for CASA-level accuracy
- **Config-Based Pipeline**: YAML configuration for complex calibration workflows
- **All Jones Terms**: K (delay), B (bandpass), G (gain), P (parallactic), D (leakage), X (cross-hand)
- **Linear & Circular Feeds**: Full support for both X/Y and R/L polarizations
- **Memory Efficient**: Chunked I/O for large datasets
- **GPU Acceleration**: Optional CuPy backend

## Installation

```bash
pip install chaos-cal

# With GPU support
pip install chaos-cal[gpu]
```

## Quick Start

### Simple MS Calibration

```bash
# Basic solve
chaos mydata.ms

# With polish for maximum accuracy
chaos mydata.ms --polish --tol 1e-12

# Single reference mode (faster)
chaos mydata.ms --single_ao --ref-ant 0

# Apply solutions
chaos applycal mydata.ms --jones chaos_cal.npy
```

### Config-Based Pipeline

Create `calibration.yaml`:

```yaml
info: |
  MS file        | Fields
  -------------- | ------------------------
  flux_cal.ms    | 3C147,3C286
  phase_cal.ms   | J1234+567
  target.ms      | Target1

solve_jones: |
  Jones types | Fields                  | Scans | Spw  | Freq interval | Time interval | Pre-solved Jones types | Pre-solved calibration tables | Output table | Model column | Parallactic
  ----------- | ----------------------- | ----- | ---- | ------------- | ------------- | ---------------------- | ----------------------------- | ------------ | ------------ | -----------
  K,B,G       | 3C147,3C147,3C147:3C286 | *     | 0~15 | full          | inf           |                        |                               | flux_cal.h5  | MODEL_DATA   | false
  G           | J1234+567               | *     | 0~15 | full          | 30s           | K,B                    | flux_cal.h5                   | phase_cal.h5 | MODEL_DATA   | true

apply_jones: |
  MS file    | Jones terms | From fields             | Scans | Spw | Freq interpolation | Time interpolation | Calibration tables | Output column  | Parallactic
  ---------- | ----------- | ----------------------- | ----- | --- | ------------------ | ------------------ | ------------------ | -------------- | -----------
  target.ms  | K,B,G       | 3C147,3C147,3C147:3C286 | *     | *   | linear             | linear             | flux_cal.h5        | CORRECTED_DATA | false
  target.ms  | G           | J1234+567               | *     | *   | linear             | nearest            | phase_cal.h5       | CORRECTED_DATA | true
```

Run:

```bash
chaos run calibration.yaml --ref-ant 0 --polish
```

## Algorithm

### Measurement Equation

For baseline (i,j):

$$\mathbf{V}_{ij} = \mathbf{J}_i \mathbf{M}_{ij} \mathbf{J}_j^\dagger$$

Where the composite Jones is:

$$\mathbf{J} = \mathbf{G} \cdot \mathbf{D} \cdot \mathbf{P} \cdot \mathbf{B} \cdot \mathbf{K}$$

### CHAOS Algorithm

1. **Chain from Reference**: Algebraically solve for all antennas starting from reference
2. **Multi-Reference**: Solve from every antenna as reference
3. **Phase Align**: Align all solutions to common gauge
4. **Weighted Average**: Combine using geometric mean of chain quality
5. **Polish**: Refine with least squares over all baselines

### Jones Matrices

| Term | Linear Feeds (X,Y) | Circular Feeds (R,L) |
|------|-------------------|---------------------|
| **P** | Rotation matrix | Diagonal phase |
| **G** | Diagonal complex | Diagonal complex |
| **B** | Diagonal complex | Diagonal complex |
| **K** | Diagonal phase (freq-dep) | Diagonal phase (freq-dep) |
| **D** | Off-diagonal leakage | Off-diagonal leakage |

## Python API

```python
from chaos import calibrate_ms, applycal

# Calibrate
jones, diagnostics = calibrate_ms(
    'mydata.ms',
    ref_antenna=0,
    mode='diagonal',
    polish=True
)

# Apply
applycal('mydata.ms', 'chaos_cal.npy')
```

## Requirements

- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7
- python-casacore >= 3.4
- h5py >= 3.0
- PyYAML >= 5.0
- CuPy >= 10.0 (optional, for GPU)

## Citation

If you use CHAOS in your research, please cite:

```bibtex
@article{chaos2025,
  title={CHAOS: Chain-based Algebraic Optimal Solver for Radio Interferometry},
  author={Pal, Arpan},
  journal={TBD},
  year={2025}
}
```

## License

MIT License
