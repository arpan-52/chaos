"""
CHAOS Calibration Framework CLI.

Commands:
    chaos run config.yaml          Run full calibration pipeline
    chaos solve config.yaml        Run solve only
    chaos apply config.yaml        Run apply only
    chaos applycal ms --jones file Apply Jones to MS (standalone)
"""

import argparse
import sys
import os


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='CHAOS - Chain-based Algebraic Optimal Solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
    run         Run calibration pipeline from config
    solve       Run solve only from config
    apply       Run apply only from config  
    applycal    Apply Jones matrices to MS (standalone)

Examples:
    chaos run calibration.yaml --ref-ant 0 --polish
    chaos solve calibration.yaml --solver ratio_chain
    chaos apply calibration.yaml
    chaos applycal mydata.ms --jones chaos_cal.npy

Config format uses pipe-delimited tables in YAML.
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # =========================================================================
    # RUN command
    # =========================================================================
    run_parser = subparsers.add_parser('run', help='Run full calibration pipeline',
        epilog="""
Example config:
  info: |
    MS file      | Fields
    ------------ | --------
    flux_cal.ms  | 3C147

  solve_jones: |
    Jones types | Fields | Scans | ...
    ----------- | ------ | ----- | ...
    K,B,G       | 3C147  | *     | ...
        """
    )
    run_parser.add_argument('config', help='YAML configuration file')
    _add_common_args(run_parser)
    
    # =========================================================================
    # SOLVE command
    # =========================================================================
    solve_parser = subparsers.add_parser('solve', help='Run solve only')
    solve_parser.add_argument('config', help='YAML configuration file')
    _add_common_args(solve_parser)
    
    # =========================================================================
    # APPLY command
    # =========================================================================
    apply_parser = subparsers.add_parser('apply', help='Run apply only')
    apply_parser.add_argument('config', help='YAML configuration file')
    apply_parser.add_argument('--verbose', '-v', action='store_true',
                              help='Verbose output')
    
    # =========================================================================
    # APPLYCAL command (standalone)
    # =========================================================================
    applycal_parser = subparsers.add_parser('applycal', 
        help='Apply Jones matrices to MS (standalone)')
    applycal_parser.add_argument('ms', help='Measurement Set path')
    applycal_parser.add_argument('--jones', '-j', required=True,
                                 help='Jones matrices file (.npy)')
    applycal_parser.add_argument('--data-column', default='DATA',
                                 help='Input data column (default: DATA)')
    applycal_parser.add_argument('--output-column', default='CORRECTED_DATA',
                                 help='Output column (default: CORRECTED_DATA)')
    applycal_parser.add_argument('--overwrite', action='store_true',
                                 help='Overwrite existing column')
    
    # =========================================================================
    # Legacy: direct MS solve (backwards compatibility)
    # =========================================================================
    # If first arg is an MS path (not a subcommand), treat as legacy solve
    args = parser.parse_args()
    
    if args.command is None:
        # Check if first positional arg looks like an MS
        if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
            path = sys.argv[1]
            if os.path.isdir(path) and (
                os.path.exists(f"{path}/table.dat") or 
                path.endswith('.ms') or 
                path.endswith('.MS')
            ):
                # Legacy MS solve
                _run_legacy_solve(sys.argv[1:])
                return
        
        parser.print_help()
        sys.exit(0)
    
    # Dispatch to appropriate handler
    if args.command == 'run':
        _run_pipeline(args, solve_only=False, apply_only=False)
    elif args.command == 'solve':
        _run_pipeline(args, solve_only=True, apply_only=False)
    elif args.command == 'apply':
        _run_pipeline(args, solve_only=False, apply_only=True)
    elif args.command == 'applycal':
        _run_applycal(args)


def _add_common_args(parser):
    """Add common arguments for solve commands."""
    parser.add_argument('--ref-ant', '-r', type=int, default=0,
                        help='Reference antenna (default: 0)')
    parser.add_argument('--solver', '-s', default='single_chain',
                        choices=['single_chain', 'ratio_chain'],
                        help='Solver method (default: single_chain)')
    parser.add_argument('--single_ao', action='store_true',
                        help='Single reference mode (no multi-ref averaging)')
    parser.add_argument('--polish', action='store_true', default=True,
                        help='Enable least squares polish (default: True)')
    parser.add_argument('--no-polish', action='store_false', dest='polish',
                        help='Disable least squares polish')
    parser.add_argument('--tol', type=float, default=1e-10,
                        help='Polish tolerance (default: 1e-10)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')


def _run_pipeline(args, solve_only: bool, apply_only: bool):
    """Run calibration pipeline from config."""
    from .runner import run_from_config
    
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    try:
        run_from_config(
            config_path=args.config,
            ref_antenna=args.ref_ant if hasattr(args, 'ref_ant') else 0,
            solver=args.solver if hasattr(args, 'solver') else 'single_chain',
            single_ao=args.single_ao if hasattr(args, 'single_ao') else False,
            polish=args.polish if hasattr(args, 'polish') else True,
            polish_tol=args.tol if hasattr(args, 'tol') else 1e-10,
            solve_only=solve_only,
            apply_only=apply_only,
            verbose=args.verbose if hasattr(args, 'verbose') else True
        )
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _run_applycal(args):
    """Run standalone applycal."""
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


def _run_legacy_solve(argv):
    """Run legacy single-MS solve (backwards compatibility)."""
    parser = argparse.ArgumentParser(description='CHAOS legacy solve')
    parser.add_argument('ms', help='Measurement Set path')
    parser.add_argument('--ref-ant', '-r', type=int, default=0)
    parser.add_argument('--mode', '-m', default='diagonal',
                        choices=['phase_only', 'diagonal', 'full'])
    parser.add_argument('--solver', '-s', default='single_chain',
                        choices=['single_chain', 'ratio_chain'])
    parser.add_argument('--single_ao', action='store_true')
    parser.add_argument('--polish', action='store_true')
    parser.add_argument('--tol', type=float, default=1e-10)
    parser.add_argument('--field', type=int, default=0)
    parser.add_argument('--spw', type=int, default=0)
    parser.add_argument('--model-column', default='MODEL_DATA')
    parser.add_argument('--rfi-sigma', type=float, default=5.0)
    parser.add_argument('--bad-ant-threshold', type=float, default=0.05)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--output', '-o', default='chaos_cal')
    
    args = parser.parse_args(argv)
    
    # Import legacy calibrate
    from .calibrate import calibrate_ms
    
    try:
        jones, diagnostics = calibrate_ms(
            ms_path=args.ms,
            ref_antenna=args.ref_ant,
            mode=args.mode,
            solver=args.solver,
            single_ao=args.single_ao,
            polish=args.polish,
            polish_tol=args.tol,
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
        if args.polish:
            print(f"Polish: enabled (tol={args.tol})")
        print(f"Bad antennas: {sorted(diagnostics['bad_antennas'])}")
        
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
