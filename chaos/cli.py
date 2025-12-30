"""
CHAOS Command Line Interface.
"""

import argparse
import sys
import os


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='CHAOS - Chain-based Algebraic Optimal Solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  solve         Solve for Jones matrices (default if no command given)
  applycal      Apply calibration solutions to MS

Examples:
  chaos solve mydata.ms
  chaos solve mydata.ms --ref-ant 5 --mode diagonal
  chaos solve mydata.ms --single_ao --ref-ant 0
  chaos applycal mydata.ms --jones chaos_cal.npy
  chaos applycal mydata.ms --jones chaos_cal.npy --output-column CORRECTED_DATA
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Solve subcommand
    solve_parser = subparsers.add_parser('solve', help='Solve for Jones matrices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algorithm Modes:
  --single_ao    Single reference only (original CATCAL-style)
  (default)      Multi-reference weighted averaging

Calibration Modes:
  phase_only    J_ref = I (identity), phases only
  diagonal      J_ref = diag(g_X, g_Y), real amplitudes optimized
  full          J_ref = 2x2 complex (8 real parameters)

Solver Methods:
  single_chain  Direct algebraic chaining (1 baseline per antenna)
  ratio_chain   Ratio method (2 baselines per antenna, pivot cancels)
        """
    )
    solve_parser.add_argument('ms', help='Measurement Set path')
    solve_parser.add_argument('--ref-ant', '-r', type=int, default=0,
                       help='Reference antenna (default: 0)')
    solve_parser.add_argument('--mode', '-m', default='diagonal',
                       choices=['phase_only', 'diagonal', 'full'],
                       help='Calibration mode (default: diagonal)')
    solve_parser.add_argument('--solver', '-s', default='single_chain',
                       choices=['single_chain', 'ratio_chain'],
                       help='Solver method (default: single_chain)')
    solve_parser.add_argument('--single_ao', action='store_true',
                       help='Single reference only (no multi-ref weighted averaging)')
    solve_parser.add_argument('--field', type=int, default=0,
                       help='Field ID (default: 0)')
    solve_parser.add_argument('--spw', type=int, default=0,
                       help='Spectral window (default: 0)')
    solve_parser.add_argument('--model-column', default='MODEL_DATA',
                       help='Model data column (default: MODEL_DATA)')
    solve_parser.add_argument('--rfi-sigma', type=float, default=5.0,
                       help='RFI flagging threshold in sigma (default: 5.0)')
    solve_parser.add_argument('--bad-ant-threshold', type=float, default=0.05,
                       help='Bad antenna threshold as fraction of median (default: 0.05)')
    solve_parser.add_argument('--max-iter', type=int, default=100,
                       help='Max optimization iterations (default: 100)')
    solve_parser.add_argument('--output', '-o', default='chaos_cal',
                       help='Output file prefix (default: chaos_cal)')

    # Applycal subcommand
    applycal_parser = subparsers.add_parser('applycal', help='Apply calibration solutions',
        epilog="""
Examples:
  chaos applycal mydata.ms --jones chaos_cal.npy
  chaos applycal mydata.ms --jones chaos_cal.npy --data-column DATA
  chaos applycal mydata.ms --jones chaos_cal.npy --output-column MY_CORRECTED --overwrite
        """
    )
    applycal_parser.add_argument('ms', help='Measurement Set path')
    applycal_parser.add_argument('--jones', '-j', required=True,
                       help='Jones matrices file (.npy from chaos solve)')
    applycal_parser.add_argument('--data-column', default='DATA',
                       help='Input data column (default: DATA)')
    applycal_parser.add_argument('--output-column', default='CORRECTED_DATA',
                       help='Output column for corrected data (default: CORRECTED_DATA)')
    applycal_parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output column')

    args = parser.parse_args()

    # Default to solve if no command given but MS provided
    if args.command is None:
        # Check if first positional arg looks like an MS
        if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
            args.command = 'solve'
            # Re-parse with solve as default
            args = solve_parser.parse_args(sys.argv[1:])
        else:
            parser.print_help()
            sys.exit(0)

    if args.command == 'solve':
        run_solve(args)
    elif args.command == 'applycal':
        run_applycal(args)


def run_solve(args):
    """Run calibration solver."""
    from .calibrate import calibrate_ms

    if not os.path.exists(args.ms):
        print(f"ERROR: MS not found: {args.ms}", file=sys.stderr)
        sys.exit(1)

    try:
        jones, diagnostics = calibrate_ms(
            ms_path=args.ms,
            ref_antenna=args.ref_ant,
            mode=args.mode,
            solver=args.solver,
            single_ao=args.single_ao,
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
        print(f"Mode: {'single_ao' if args.single_ao else 'multi-ref weighted'}")
        print(f"Bad antennas: {sorted(diagnostics['bad_antennas'])}")

        sys.exit(0)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_applycal(args):
    """Run applycal to correct visibilities."""
    from .applycal import applycal

    if not os.path.exists(args.ms):
        print(f"ERROR: MS not found: {args.ms}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.jones):
        print(f"ERROR: Jones file not found: {args.jones}", file=sys.stderr)
        sys.exit(1)

    try:
        applycal(
            ms_path=args.ms,
            jones_path=args.jones,
            data_column=args.data_column,
            output_column=args.output_column,
            overwrite=args.overwrite
        )
        sys.exit(0)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
