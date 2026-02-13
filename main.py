#!/usr/bin/env python3
"""
Stock Prediction Engine - Main Entry Point

Usage:
    python main.py run                  # Run full analysis (7-model ensemble)
    python main.py run --basic          # Run with basic scoring only
    python main.py run --symbols TCS,INFY  # Run for specific symbols
    python main.py analyze RELIANCE     # Analyze single stock (7-model)
    python main.py analyze RELIANCE --basic  # Analyze with basic scoring
    python main.py ui                   # Start Streamlit UI

Enhanced Mode (default) - 7-Model Ensemble:
    - Base Technical/Momentum/Volume/News (25%)
    - Physics Engine (18%)
    - Math Engine (14%)
    - HMM Regime Detection (13%)
    - Macro Engine - commodities, currencies, bonds (10%) [NEW]
    - Alternative Data - earnings, options, institutional flow (10%) [NEW]
    - Advanced Math - Kalman, Wavelet, PCA, Markov (10%) [NEW]

Features:
    - Multi-model agreement required for signals (3/7 moderate, 5/7 strong)
    - SELL signals disabled (historically 28.6% accurate = inverted)
    - Calibrated confidence from backtest data
    - Target accuracy: 65-68% (improved from 57.6%)
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


def setup_logging():
    """Configure logging."""
    from src.utils.logger import setup_logger
    setup_logger()


def cmd_run(args):
    """Run full prediction pipeline."""
    # Choose generator based on mode
    if args.basic:
        from src.signals.generator import SignalGenerator
        logger.info("Starting Stock Prediction Engine (BASIC mode)...")
        generator = SignalGenerator()
    else:
        from src.signals.enhanced_generator import EnhancedSignalGenerator
        logger.info("Starting Stock Prediction Engine (ENHANCED mode - 7-model ensemble)...")
        generator = EnhancedSignalGenerator()

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        logger.info(f"Running for {len(symbols)} symbols: {symbols}")

    # Run with appropriate parameters
    run_kwargs = {
        'symbols': symbols,
        'include_intraday': not args.no_intraday,
        'include_swing': not args.no_swing
    }

    # Enhanced mode has additional parameters
    if not args.basic:
        run_kwargs['min_model_agreement'] = args.min_agreement

    results = generator.run(**run_kwargs)

    # Output results
    if args.output == "json":
        print(json.dumps(results, indent=2, default=str))
    else:
        print(results.get('summary', 'No summary available'))

    # Save to file if requested
    if args.save:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

    return results


def cmd_analyze(args):
    """Analyze a single stock."""
    from src.storage.models import TradeType

    # Choose generator based on mode
    if args.basic:
        from src.signals.generator import SignalGenerator
        logger.info(f"Analyzing {args.symbol} (BASIC mode)...")
        generator = SignalGenerator()
    else:
        from src.signals.enhanced_generator import EnhancedSignalGenerator
        logger.info(f"Analyzing {args.symbol} (ENHANCED mode - 7-model ensemble)...")
        generator = EnhancedSignalGenerator()

    trade_type = TradeType.INTRADAY if args.type == "intraday" else TradeType.SWING

    result = generator.get_single_prediction(args.symbol.upper(), trade_type)

    if result:
        print("\n" + "=" * 60)
        mode_label = "BASIC" if args.basic else "ENHANCED"
        print(f"ANALYSIS: {args.symbol.upper()} ({args.type.upper()}) [{mode_label}]")
        print("=" * 60)
        print(f"\nSignal: {result['signal']} ({result['confidence']:.0%} confidence)")

        # Show enhanced metadata if available
        if 'enhanced_data' in result:
            enh = result['enhanced_data']
            total_models = enh.get('total_models', 4)
            print(f"\nModel Agreement: {enh['model_agreement']}/{total_models} models")
            print(f"Signal Strength: {enh['signal_strength']}")
            print(f"Market Regime: {enh['regime']} (stability: {enh['regime_stability']:.0%})")
            print(f"Predictability: {enh['market_predictability']:.0%}")
            print(f"Strategy: {enh['recommended_strategy']}")
            # Show sector for macro analysis
            if enh.get('sector') and enh['sector'] != 'Unknown':
                print(f"Sector: {enh['sector']}")

            print("\nModel Votes:")
            for model, vote in enh['model_votes'].items():
                emoji = "+" if vote == "BUY" else ("-" if vote == "SELL" else " ")
                print(f"  [{emoji}] {model.title()}: {vote}")

        print(f"\nTrade Plan:")
        print(f"  Entry:  ₹{result['entry_price']}")
        print(f"  Stop:   ₹{result['stop_loss']}")
        print(f"  Target: ₹{result['target_price']}")
        print(f"\n{result['summary']}")
        print("\nReasons:")
        for reason in result['reasons'][:10]:
            print(f"  - {reason}")

        # Show warnings if enhanced mode
        if 'enhanced_data' in result and result['enhanced_data'].get('warnings'):
            print("\nWarnings:")
            for warning in result['enhanced_data']['warnings']:
                print(f"  ! {warning}")

        print("=" * 60)
    else:
        print(f"Could not analyze {args.symbol}. Check if the symbol is valid.")


def cmd_ui(args):
    """Launch Streamlit UI."""
    import subprocess
    ui_path = Path(__file__).parent / "ui" / "app.py"
    subprocess.run(["streamlit", "run", str(ui_path)])


def cmd_backtest(args):
    """Run backtesting."""
    logger.info("Backtesting not yet implemented. Coming soon!")


def main():
    parser = argparse.ArgumentParser(
        description="Stock Prediction Engine - Deterministic stock analysis for Indian markets"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run full prediction pipeline")
    run_parser.add_argument(
        "--symbols", "-s",
        type=str,
        help="Comma-separated list of symbols (default: NIFTY 50)"
    )
    run_parser.add_argument(
        "--basic",
        action="store_true",
        help="Use basic scoring (single model) instead of enhanced 7-model ensemble"
    )
    run_parser.add_argument(
        "--min-agreement",
        type=int,
        default=3,
        dest="min_agreement",
        help="Minimum models that must agree for a signal (1-7, default: 3)"
    )
    run_parser.add_argument(
        "--no-intraday",
        action="store_true",
        help="Skip intraday predictions"
    )
    run_parser.add_argument(
        "--no-swing",
        action="store_true",
        help="Skip swing predictions"
    )
    run_parser.add_argument(
        "--output", "-o",
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    run_parser.add_argument(
        "--save",
        type=str,
        help="Save results to file"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single stock")
    analyze_parser.add_argument("symbol", type=str, help="Stock symbol (e.g., RELIANCE)")
    analyze_parser.add_argument(
        "--type", "-t",
        choices=["intraday", "swing"],
        default="intraday",
        help="Trade type"
    )
    analyze_parser.add_argument(
        "--basic",
        action="store_true",
        help="Use basic scoring instead of enhanced 7-model ensemble"
    )

    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch Streamlit dashboard")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtesting")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "ui":
        cmd_ui(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
