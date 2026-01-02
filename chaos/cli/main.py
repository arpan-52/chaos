"""
CHAOS Command Line Interface.

Commands:
    chaos <ms>                Run calibration on MS
    chaos run <config.yaml>   Run pipeline from config
    chaos solve <config>      Run solve only
    chaos apply <config>      Run apply only
    chaos applycal <ms>       Apply Jones to MS (standalone)
"""

import argparse
import sys
import os


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CHAOS - Chain-based Algebraic Optimal Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Calibrate MS directly
    chaos mydata.ms --ref-ant 0
    
    # Run config-based pipeline
    chaos run calibration.yaml --ref-ant 0
    
    # Apply solutions
    chaos applycal mydata.ms --table cal.h5

Configuration format uses pipe-delimited tables in YAML.
See documentation for details.
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # =========================================================================
    # RUN command - config-based pipeline
    # =========================================================================
    run_parser = subparsers.add_parser("run", help="Run pipeline from config")
    run_parser.add_argument("config", help="YAML configuration file")
    _add_common_args(run_parser)
    
    # =========================================================================
    # SOLVE command - solve only
    # =========================================================================
    solve_parser = subparsers.add_parser("solve", help="Run solve only")
    solve_parser.add_argument("config", help="YAML configuration file")
    _add_common_args(solve_parser)
    
    # =========================================================================
    # APPLY command - apply only
    # =========================================================================
    apply_parser = subparsers.add_parser("apply", help="Run apply only")
    apply_parser.add_argument("config", help="YAML configuration file")
    apply_parser.add_argument("-v", "--verbose", action="store_true")
    
    # =========================================================================
    # APPLYCAL command - standalone apply
    # =========================================================================
    applycal_parser = subparsers.add_parser("applycal", help="Apply Jones to MS")
    applycal_parser.add_argument("ms", help="MeasurementSet path")
    applycal_parser.add_argument(
        "-t", "--table", required=True, 
        help="Calibration table (HDF5)"
    )
    applycal_parser.add_argument(
        "-j", "--jones", nargs="+",
        help="Jones types to apply (default: all)"
    )
    applycal_parser.add_argument(
        "-o", "--output-column", default="CORRECTED_DATA",
        help="Output column (default: CORRECTED_DATA)"
    )
    applycal_parser.add_argument(
        "--time-interp", default="linear",
        choices=["nearest", "linear", "cubic"],
        help="Time interpolation method"
    )
    applycal_parser.add_argument(
        "--freq-interp", default="linear",
        choices=["nearest", "linear", "cubic"],
        help="Frequency interpolation method"
    )
    applycal_parser.add_argument("-v", "--verbose", action="store_true")
    
    # =========================================================================
    # Parse arguments
    # =========================================================================
    args = parser.parse_args()
    
    # If no command given, check if first arg is an MS
    if args.command is None:
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            path = sys.argv[1]
            if _is_ms(path):
                _run_legacy_solve(sys.argv[1:])
                return
        
        parser.print_help()
        sys.exit(0)
    
    # Dispatch to handler
    if args.command == "run":
        _run_pipeline(args, solve_only=False, apply_only=False)
    elif args.command == "solve":
        _run_pipeline(args, solve_only=True, apply_only=False)
    elif args.command == "apply":
        _run_pipeline(args, solve_only=False, apply_only=True)
    elif args.command == "applycal":
        _run_applycal(args)


def _add_common_args(parser):
    """Add common arguments for solve commands."""
    parser.add_argument(
        "-r", "--ref-ant", type=int, default=0,
        help="Reference antenna (default: 0)"
    )
    parser.add_argument(
        "--flag-threshold", type=float, default=0.8,
        help="Fraction flagged to mark antenna bad (default: 0.8)"
    )
    parser.add_argument(
        "--rfi-sigma", type=float, default=5.0,
        help="RFI flagging threshold in sigma (default: 5.0)"
    )
    parser.add_argument(
        "--no-rfi-flag", action="store_true",
        help="Disable RFI flagging"
    )
    parser.add_argument(
        "--tol", type=float, default=1e-10,
        help="Polish tolerance (default: 1e-10)"
    )
    parser.add_argument(
        "--max-iter", type=int, default=100,
        help="Max polish iterations (default: 100)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")


def _is_ms(path: str) -> bool:
    """Check if path is a MeasurementSet."""
    if not os.path.isdir(path):
        return False
    return (
        os.path.exists(f"{path}/table.dat") or
        path.endswith(".ms") or
        path.endswith(".MS")
    )


def _run_pipeline(args, solve_only: bool, apply_only: bool):
    """Run config-based pipeline."""
    from chaos.pipeline.runner import run_pipeline
    
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    try:
        run_pipeline(
            config_path=args.config,
            ref_antenna=getattr(args, "ref_ant", 0),
            flag_threshold=getattr(args, "flag_threshold", 0.8),
            polish_tol=getattr(args, "tol", 1e-10),
            max_iter=getattr(args, "max_iter", 100),
            solve_only=solve_only,
            apply_only=apply_only,
            verbose=getattr(args, "verbose", True),
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _run_applycal(args):
    """Run standalone applycal."""
    from chaos.io.applycal import applycal
    
    if not os.path.exists(args.ms):
        print(f"ERROR: MS not found: {args.ms}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.table):
        print(f"ERROR: Table not found: {args.table}", file=sys.stderr)
        sys.exit(1)
    
    try:
        applycal(
            ms_path=args.ms,
            cal_tables=args.table,
            jones_types=args.jones,
            output_column=args.output_column,
            time_interp=args.time_interp,
            freq_interp=args.freq_interp,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _run_legacy_solve(argv):
    """Run direct MS calibration (legacy mode)."""
    parser = argparse.ArgumentParser(description="CHAOS direct MS calibration")
    parser.add_argument("ms", help="MeasurementSet path")
    parser.add_argument("-r", "--ref-ant", type=int, default=0)
    parser.add_argument(
        "-t", "--jones-type", default="G",
        choices=["K", "B", "G", "G[p]", "D"],
        help="Jones type to solve"
    )
    parser.add_argument("--field", type=int, default=0)
    parser.add_argument("--spw", type=int, default=0)
    parser.add_argument("--model-column", default="MODEL_DATA")
    parser.add_argument("--flag-threshold", type=float, default=0.8)
    parser.add_argument("--rfi-sigma", type=float, default=5.0,
                        help="RFI flagging threshold (default: 5.0)")
    parser.add_argument("--no-rfi-flag", action="store_true",
                        help="Disable RFI flagging")
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("-o", "--output", default="chaos_cal")
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    
    args = parser.parse_args(argv)
    
    from chaos.io.ms_reader import MSReader
    from chaos.core.solver import solve_jones
    from chaos.io.table_io import save_jones_table
    import numpy as np
    
    try:
        print("="*60)
        print("CHAOS - Chain-based Algebraic Optimal Solver")
        print("="*60)
        
        # Read data
        reader = MSReader(args.ms)
        data = reader.read_data(
            field_id=args.field,
            spw=args.spw,
            model_column=args.model_column,
            average_channels=True,
        )
        
        print(f"Data: {len(data.antenna1)} baselines, {data.n_ant} antennas")
        print(f"Feed type: {data.feed_type}")
        
        # Solve
        jones, diagnostics = solve_jones(
            vis_obs=data.vis_obs,
            vis_model=data.vis_model,
            antenna1=data.antenna1,
            antenna2=data.antenna2,
            n_ant=data.n_ant,
            ref_antenna=args.ref_ant,
            jones_type=args.jones_type,
            flag_threshold=args.flag_threshold,
            flags=data.flags,
            rfi_flag=not args.no_rfi_flag,
            rfi_sigma=args.rfi_sigma,
            max_iter=args.max_iter,
            polish_tol=args.tol,
            verbose=args.verbose,
        )
        
        # Save
        output_file = f"{args.output}.h5"
        save_jones_table(
            output_file,
            args.jones_type,
            jones,
            np.array([data.time.mean()]),
            data.freq,
            ref_antenna=args.ref_ant,
            mode=diagnostics["mode"],
            overwrite=True,
        )
        
        print(f"\nSaved to: {output_file}")
        print("="*60)
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
