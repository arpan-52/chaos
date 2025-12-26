"""
CHAOS Command Line Interface.
"""

import argparse
import sys
import os
from .calibrate import calibrate_ms


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='CHAOS - Chain-based Algebraic Optimal Solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CHAOS performs multi-reference weighted Jones calibration:
1. Solves from every antenna as reference
2. Aligns phases to user-specified reference
3. Weighted average using geometric mean of chain quality

Examples:
  # Basic usage
  chaos mydata.ms

  # Specify reference antenna and mode
  chaos mydata.ms --ref-ant 5 --mode diagonal

  # Use ratio chain solver
  chaos mydata.ms --solver ratio_chain

  # Full Jones calibration
  chaos mydata.ms --mode full

Requires MODEL_DATA column (use CASA tclean with savemodel='modelcolumn').
        """
    )

    parser.add_argument('ms', help='Measurement Set path')
    parser.add_argument('--ref-ant', '-r', type=int, default=0,
                       help='Reference antenna for final alignment (default: 0)')
    parser.add_argument('--mode', '-m', default='diagonal',
                       choices=['phase_only', 'diagonal', 'full'],
                       help='Calibration mode (default: diagonal)')
    parser.add_argument('--solver', '-s', default='single_chain',
                       choices=['single_chain', 'ratio_chain'],
                       help='Solver method (default: single_chain)')
    parser.add_argument('--field', type=int, default=0,
                       help='Field ID (default: 0)')
    parser.add_argument('--spw', type=int, default=0,
                       help='Spectral window (default: 0)')
    parser.add_argument('--model-column', default='MODEL_DATA',
                       help='Model data column (default: MODEL_DATA)')
    parser.add_argument('--rfi-sigma', type=float, default=5.0,
                       help='RFI flagging threshold in sigma (default: 5.0)')
    parser.add_argument('--bad-ant-threshold', type=float, default=0.05,
                       help='Bad antenna threshold as fraction of median (default: 0.05)')
    parser.add_argument('--max-iter', type=int, default=100,
                       help='Max optimization iterations (default: 100)')
    parser.add_argument('--output', '-o', default='chaos_cal',
                       help='Output file prefix (default: chaos_cal)')

    args = parser.parse_args()

    # Check MS exists
    if not os.path.exists(args.ms):
        print(f"ERROR: MS not found: {args.ms}", file=sys.stderr)
        sys.exit(1)

    try:
        jones, diagnostics = calibrate_ms(
            ms_path=args.ms,
            ref_antenna=args.ref_ant,
            mode=args.mode,
            solver=args.solver,
            field_id=args.field,
            spw=args.spw,
            model_column=args.model_column,
            rfi_sigma=args.rfi_sigma,
            bad_ant_threshold=args.bad_ant_threshold,
            max_iter=args.max_iter,
            output_prefix=args.output
        )

        print(f"\nFinal Jones shape: {jones.shape}")
        print(f"Reference antenna: {args.ref_ant}")
        print(f"Bad antennas: {sorted(diagnostics['bad_antennas'])}")

        sys.exit(0)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
