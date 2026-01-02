# CHAOS - Chain-based Algebraic Optimal Solver

**Version 2.0** - Complete Radio Interferometry Calibration Framework

## Overview

CHAOS is a calibration framework for radio interferometry that uses algebraic chaining to provide excellent initial guesses, followed by least-squares polishing to achieve CASA-level accuracy.

## Algorithm

1. **RFI Flagging**: Robust MAD-based outlier detection
2. **Initial Guess**: Single-chain algebraic solution from reference antenna
3. **Polish**: Least-squares refinement over all baselines (ALWAYS runs)
4. Every solve = RFI flag + initial guess + polish

## Installation

```bash
pip install -e .
```

## Usage

### Simple Calibration
```bash
chaos mydata.ms --ref-ant 0
```

### Config-based Pipeline
```bash
chaos run calibration.yaml --ref-ant 0
```

---

## Measurement Equation

The relationship between observed and ideal visibilities:

```
V_obs(i,j) = J_i · M_ij · J_j†
```

Where the composite Jones matrix is (in signal propagation order, from sky to correlator):

```
J = T · P · E · D · G · B · K
```

| Term | Description |
|------|-------------|
| T | Tropospheric effects (opacity, path-length) |
| P | Parallactic angle rotation |
| E | Primary beam / elevation effects |
| D | Instrumental polarization (leakage) |
| G | Electronic gain |
| B | Bandpass |
| K | Delay |

---

## Jones Terms - Complete Reference

### K-Jones: Delay

Frequency-dependent phase due to signal path delay.

**Linear Feeds (X,Y):**
```
K = | exp(2πi·ν·τ_X)        0           |
    |       0         exp(2πi·ν·τ_Y)    |
```

**Circular Feeds (R,L):**
```
K = | exp(2πi·ν·τ_R)        0           |
    |       0         exp(2πi·ν·τ_L)    |
```

- **Parameters**: 2 real values (delays τ in seconds)
- **Solution**: Per antenna, typically one per observation
- **Reference constraint**: τ_X = τ_Y = 0 for reference antenna

---

### KCROSS-Jones: Cross-hand Delay

Residual delay difference between polarizations on reference antenna.
**GLOBAL** - single value applied to all antennas.

**Linear Feeds (X,Y):**
```
K_cross = | 1                  0              |
          | 0          exp(2πi·ν·Δτ_XY)       |
```

**Circular Feeds (R,L):**
```
K_cross = | 1                  0              |
          | 0          exp(2πi·ν·Δτ_RL)       |
```

- **Parameters**: 1 real value (cross-delay Δτ in seconds)
- **Solution**: Global (one value for entire array)
- **Reference constraint**: None (it IS the reference antenna's X-Y delay)

---

### B-Jones: Bandpass

Frequency-dependent complex gain.

**Linear Feeds (X,Y):**
```
B = | b_X(ν)      0     |
    |   0      b_Y(ν)   |
```

**Circular Feeds (R,L):**
```
B = | b_R(ν)      0     |
    |   0      b_L(ν)   |
```

- **Parameters**: 2 complex values per channel
- **Solution**: Per antenna, per channel
- **Reference constraint**: arg(b_X) = arg(b_Y) = 0 for reference antenna

---

### G-Jones: Gain (Full Complex)

Time-dependent complex gain (amplitude + phase).

**Linear Feeds (X,Y):**
```
G = | g_X    0   |
    |  0    g_Y  |
```

**Circular Feeds (R,L):**
```
G = | g_R    0   |
    |  0    g_L  |
```

- **Parameters**: 2 complex values (amplitude and phase per pol)
- **Solution**: Per antenna, per solution interval
- **Reference constraint**: arg(g_X) = arg(g_Y) = 0 for reference antenna

---

### G[p]-Jones: Phase-only Gain

Phase-only gain (amplitude fixed at 1).

**Linear Feeds (X,Y):**
```
G[p] = | exp(i·φ_X)      0        |
       |     0       exp(i·φ_Y)   |
```

**Circular Feeds (R,L):**
```
G[p] = | exp(i·φ_R)      0        |
       |     0       exp(i·φ_L)   |
```

- **Parameters**: 2 real values (phases in radians)
- **Solution**: Per antenna, per solution interval
- **Reference constraint**: φ_X = φ_Y = 0 for reference antenna

---

### P-Jones: Parallactic Angle

Rotation of polarization frame as antenna tracks source.
**COMPUTED** from antenna position, source position, and time.

**Linear Feeds (X,Y):** (Rotation matrix)
```
P = | cos(ψ)   -sin(ψ) |
    | sin(ψ)    cos(ψ) |
```

**Circular Feeds (R,L):** (Diagonal phase)
```
P = | exp(-i·ψ)      0       |
    |     0      exp(+i·ψ)   |
```

- **Parameters**: Computed from geometry
- **Note**: For circular feeds, P is diagonal so it commutes with other diagonal terms

---

### D-Jones: Leakage (D-terms)

Instrumental polarization - leakage between feeds.

**Linear Feeds (X,Y):**
```
D = |  1    d_X  |
    | d_Y    1   |
```
Where:
- d_X = leakage from Y into X
- d_Y = leakage from X into Y

**Circular Feeds (R,L):**
```
D = |  1    d_R  |
    | d_L    1   |
```
Where:
- d_R = leakage from L into R
- d_L = leakage from R into L

- **Parameters**: 2 complex values per antenna
- **Solution**: Per antenna, typically per channel (Df)
- **Reference constraint**: d_X = 0 (or d_Y = 0) for reference antenna when using unpolarized calibrator

---

### Xf-Jones: Cross-hand Phase

Residual X-Y (or R-L) phase on reference antenna.
**GLOBAL** - single value applied to all antennas (per channel for Xf).

**Linear Feeds (X,Y):**
```
X = | 1           0          |
    | 0      exp(i·φ_XY)     |
```

**Circular Feeds (R,L):**
```
X = | 1           0          |
    | 0      exp(i·φ_RL)     |
```

- **Parameters**: 1 real value per channel (or 1 total for X)
- **Solution**: Global (one value for entire array per channel)
- **Reference constraint**: None (it IS the reference antenna's cross-pol phase)

---

## Reference Antenna Constraints Summary

| Jones Term | Scope | Reference Antenna Constraint |
|------------|-------|------------------------------|
| K | Per antenna | τ_X = τ_Y = 0 |
| KCROSS | Global | N/A (defines ref ant cross-delay) |
| B | Per antenna | arg(b_X) = arg(b_Y) = 0 |
| G | Per antenna | arg(g_X) = arg(g_Y) = 0 |
| G[p] | Per antenna | φ_X = φ_Y = 0 |
| D | Per antenna | d_X = 0 (for unpolarized cal) |
| Xf | Global | N/A (defines ref ant cross-phase) |
| P | Computed | N/A |

---

## Configuration File Format

CHAOS uses pipe-delimited tables in YAML:

```yaml
info: |
  MS file        | Fields
  -------------- | ------------------------
  flux_cal.ms    | 3C147,3C286
  phase_cal.ms   | J1234+567
  target.ms      | Target1

solve_jones: |
  Jones | Fields    | Scans | Spw  | Freq int    | Time int | Pre-apply | Pre-tables  | Output      | Model col  | Parang
  ----- | --------- | ----- | ---- | ----------- | -------- | --------- | ----------- | ----------- | ---------- | ------
  K     | 3C147     | *     | *    | full        | inf      |           |             | flux_cal.h5 | MODEL_DATA | false
  B     | 3C147     | *     | *    | per_channel | inf      | K         | flux_cal.h5 | flux_cal.h5 | MODEL_DATA | false
  G     | 3C147     | *     | *    | full        | inf      | K,B       | flux_cal.h5 | flux_cal.h5 | MODEL_DATA | false
  G[p]  | J1234+567 | *     | *    | full        | 30s      | K,B,G     | flux_cal.h5 | phase_cal.h5| MODEL_DATA | true

apply_jones: |
  MS file   | Jones   | From fields | Scans | Spw | Freq interp | Time interp | Tables       | Output col     | Parang
  --------- | ------- | ----------- | ----- | --- | ----------- | ----------- | ------------ | -------------- | ------
  target.ms | K,B,G   | 3C147       | *     | *   | linear      | linear      | flux_cal.h5  | CORRECTED_DATA | false
  target.ms | G[p]    | J1234+567   | *     | *   | linear      | nearest     | phase_cal.h5 | CORRECTED_DATA | true
```

---

## Bad Antenna Handling

- If an antenna has > 80% flagged data (configurable via `--flag-threshold`), it is excluded from solving
- Bad antennas get identity Jones matrices
- User can manually specify bad antennas with `--bad-ant`

---

## RFI Flagging

CHAOS uses robust Median Absolute Deviation (MAD) based RFI detection.

### Why MAD?

- **Robust to outliers**: Unlike mean/std, median is not affected by extreme values
- **Works with high RFI**: Can handle up to ~50% contaminated data
- **No tuning required**: Works well with default 5-sigma threshold

### Algorithm

```
MAD = median(|x - median(x)|)
σ_robust = 1.4826 × MAD
threshold = n_sigma × σ_robust
flag if |x - median(x)| > threshold
```

### Options

- `--rfi-sigma`: Threshold in sigma (default: 5.0)
- `--no-rfi-flag`: Disable RFI flagging

### Iterative Flagging

By default, flagging runs iteratively (up to 3 passes) until convergence.
This handles cases where initial RFI affects the statistics.

---

## Solution Intervals

- `inf` - Single solution for all selected data
- `30s` - Solution every 30 seconds
- `int` - Solution per integration
- `scan` - Solution per scan

---

## Frequency Intervals

- `full` - Single solution across all channels
- `per_spw` - One solution per spectral window
- `per_channel` - One solution per channel
- `4MHz` - Solution per 4 MHz (example)

---

## Interpolation Methods

- `nearest` - Nearest neighbor
- `linear` - Linear interpolation
- `cubic` - Cubic spline (for smooth variations)

---

## File Structure

```
chaos/
├── __init__.py
├── __main__.py
├── cli/                 # Command line interface
├── core/                # Solvers and algorithms
├── pipeline/            # Config-based pipeline
├── io/                  # MS and table I/O
├── jones/               # Jones matrix definitions
├── utils/               # Utilities
└── tests/               # Unit tests
```

---

## Author

Arpan Pal

## License

MIT
