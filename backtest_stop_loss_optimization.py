#!/usr/bin/env python3
"""
===========================================================================
BACKTESTING SCRIPT - STOP LOSS OPTIMIZATION
===========================================================================

Script per eseguire backtesting comparativi tra:
1. Strategia originale (10% stop loss)
2. Strategia ottimizzata (3% stop loss + ATR + multi-tier)

Uso: python backtest_stop_loss_optimization.py
"""

import subprocess
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BacktestManager:
    """Gestore per backtesting comparativi delle ottimizzazioni di stop loss"""

    def __init__(self, config_dir="user_data", results_dir="backtest_results"):
        self.config_dir = config_dir
        self.results_dir = results_dir
        self.ensure_results_directory()

    def ensure_results_directory(self):
        """Crea la directory dei risultati se non esiste"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def run_backtest(self, strategy_name: str, timerange: str, config_file: str = None,
                    result_filename: str = None) -> str:
        """Esegue un backtest e ritorna il path del file dei risultati"""

        if not result_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"backtest_{strategy_name}_{timestamp}.json"

        result_path = os.path.join(self.results_dir, result_filename)

        # Build freqtrade command
        cmd = [
            "freqtrade", "backtesting",
            "--strategy", strategy_name,
            "--timerange", timerange,
            "--userdir", self.config_dir,
            "--export", "trades",
            "--export-filename", result_filename.replace('.json', ''),
            "--breakdown", "month"
        ]

        if config_file:
            cmd.extend(["--config", config_file])

        logger.info(f"Running backtest for {strategy_name}...")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            # Run backtest
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Backtest completed successfully for {strategy_name}")
            logger.info(f"Results saved to: {result_path}")

            return result_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Backtest failed for {strategy_name}: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            raise

    def run_comparative_backtests(self, timerange: str = "20241001-20241107"):
        """Esegue backtesting comparativi tra strategie"""

        logger.info("🚀 Starting comparative backtests...")

        backtest_results = {}

        # Test 1: Original strategy (10% stop loss)
        try:
            logger.info("Testing original strategy (10% stop loss)...")
            original_result = self.run_backtest(
                strategy_name="IchimokuEnhancedV8092",
                timerange=timerange,
                config_file="user_data/config_enhanced_8092.json",
                result_filename="backtest_original_10pct_stoploss.json"
            )
            backtest_results['original'] = original_result
        except Exception as e:
            logger.error(f"Failed to backtest original strategy: {e}")

        # Test 2: Optimized strategy (3% stop loss + ATR)
        try:
            logger.info("Testing optimized strategy (3% stop loss + ATR)...")
            optimized_result = self.run_backtest(
                strategy_name="IchimokuEnhancedV8092Optimized",
                timerange=timerange,
                config_file="user_data/config_enhanced_8092.json",
                result_filename="backtest_optimized_3pct_stoploss.json"
            )
            backtest_results['optimized'] = optimized_result
        except Exception as e:
            logger.error(f"Failed to backtest optimized strategy: {e}")

        return backtest_results

    def analyze_backtest_results(self, result_files: dict) -> dict:
        """Analizza e compara i risultati dei backtest"""

        logger.info("📊 Analyzing backtest results...")

        analyses = {}

        for strategy_type, result_file in result_files.items():
            try:
                # Convert JSON to DB for analysis
                db_file = result_file.replace('.json', '.sqlite')
                if os.path.exists(db_file):
                    analysis = self.analyze_backtest_database(db_file)
                    analyses[strategy_type] = analysis
                else:
                    logger.warning(f"Database file not found: {db_file}")

            except Exception as e:
                logger.error(f"Error analyzing {strategy_type} results: {e}")

        # Generate comparison
        if len(analyses) >= 2:
            self.generate_comparison_report(analyses)

        return analyses

    def analyze_backtest_database(self, db_file: str) -> dict:
        """Analizza il database del backtest"""

        conn = sqlite3.connect(db_file)

        # Load trades
        trades_query = """
        SELECT
            pair,
            profit_ratio,
            profit_abs,
            close_rate,
            open_rate,
            is_open,
            exit_reason,
            open_date,
            close_date,
            stake_amount
        FROM trades
        WHERE is_open = 0
        ORDER BY close_date
        """

        trades = pd.read_sql_query(trades_query, conn)

        if trades.empty:
            return {"error": "No trades found"}

        # Calculate metrics
        total_trades = len(trades)
        winning_trades = trades[trades['profit_ratio'] > 0]
        losing_trades = trades[trades['profit_ratio'] <= 0]

        win_rate = len(winning_trades) / total_trades
        total_profit = trades['profit_ratio'].sum()
        total_profit_abs = trades['profit_abs'].sum()

        avg_profit = trades['profit_ratio'].mean()
        avg_win = winning_trades['profit_ratio'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['profit_ratio'].mean() if len(losing_trades) > 0 else 0

        max_profit = trades['profit_ratio'].max()
        max_loss = trades['profit_ratio'].min()

        # Risk metrics
        total_loss = abs(losing_trades['profit_ratio'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Drawdown calculation
        cumulative_profit = trades['profit_ratio'].cumsum()
        running_max = cumulative_profit.expanding().max()
        drawdown = (cumulative_profit - running_max) / running_max
        max_drawdown = drawdown.min()

        # Exit reason analysis
        exit_reason_counts = trades['exit_reason'].value_counts().to_dict()

        # Loss analysis
        catastrophic_losses = trades[trades['profit_ratio'] < -0.05]
        severe_losses = trades[(trades['profit_ratio'] < -0.02) & (trades['profit_ratio'] >= -0.05)]

        analysis = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_profit_pct': total_profit,
            'total_profit_abs': total_profit_abs,
            'avg_profit_pct': avg_profit,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'max_profit_pct': max_profit,
            'max_loss_pct': max_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown,
            'catastrophic_losses': len(catastrophic_losses),
            'severe_losses': len(severe_losses),
            'exit_reasons': exit_reason_counts
        }

        conn.close()
        return analysis

    def generate_comparison_report(self, analyses: dict):
        """Genera un report di comparazione tra strategie"""

        print("\n" + "="*80)
        print("🔄 BACKTEST COMPARISON REPORT")
        print("="*80)

        if 'original' not in analyses or 'optimized' not in analyses:
            print("❌ Cannot generate comparison - missing data")
            return

        original = analyses['original']
        optimized = analyses['optimized']

        # Key metrics comparison
        metrics_to_compare = [
            ('Total Trades', 'total_trades', ''),
            ('Win Rate', 'win_rate', '%'),
            ('Total Profit (%)', 'total_profit_pct', '%'),
            ('Average Profit (%)', 'avg_profit_pct', '%'),
            ('Profit Factor', 'profit_factor', ''),
            ('Max Drawdown (%)', 'max_drawdown_pct', '%'),
            ('Max Loss (%)', 'max_loss_pct', '%'),
            ('Catastrophic Losses', 'catastrophic_losses', '')
        ]

        print(f"\n{'Metric':<20} {'Original':<15} {'Optimized':<15} {'Improvement':<15}")
        print("-" * 65)

        total_improvement = 0
        improvements_count = 0

        for metric_name, key, suffix in metrics_to_compare:
            original_val = original.get(key, 0)
            optimized_val = optimized.get(key, 0)

            # Format values
            if suffix == '%':
                original_str = f"{original_val*100:.2f}%"
                optimized_str = f"{optimized_val*100:.2f}%"
                improvement = (optimized_val - original_val) * 100
                improvement_str = f"{improvement:+.2f}%"
            else:
                original_str = f"{original_val:.2f}"
                optimized_str = f"{optimized_val:.2f}"
                improvement = optimized_val - original_val
                improvement_str = f"{improvement:+.2f}"

            # Track improvement
            if key in ['total_profit_pct', 'profit_factor', 'win_rate']:
                total_improvement += improvement
                improvements_count += 1

            print(f"{metric_name:<20} {original_str:<15} {optimized_str:<15} {improvement_str:<15}")

        # Summary
        print(f"\n📊 SUMMARY:")
        if improvements_count > 0:
            avg_improvement = total_improvement / improvements_count
            print(f"   Average key improvement: {avg_improvement:+.2f}%")

        profit_improvement = (optimized['total_profit_pct'] - original['total_profit_pct']) * 100
        print(f"   Profit improvement: {profit_improvement:+.2f}%")

        # Risk analysis
        print(f"\n⚠️ RISK ANALYSIS:")
        original_max_loss = original.get('max_loss_pct', 0) * 100
        optimized_max_loss = optimized.get('max_loss_pct', 0) * 100
        print(f"   Max loss reduction: {original_max_loss:.2f}% → {optimized_max_loss:.2f}% ({original_max_loss - optimized_max_loss:.2f}% improvement)")

        original_catastrophic = original.get('catastrophic_losses', 0)
        optimized_catastrophic = optimized.get('catastrophic_losses', 0)
        print(f"   Catastrophic losses: {original_catastrophic} → {optimized_catastrophic} ({original_catastrophic - optimized_catastrophic} reduction)")

        # Recommendation
        print(f"\n💡 RECOMMENDATION:")
        if profit_improvement > 0 and optimized_catastrophic < original_catastrophic:
            print("   ✅ OPTIMIZED STRATEGY RECOMMENDED")
            print("   ✅ Significant improvement in both profitability and risk management")
        elif profit_improvement > 0:
            print("   ⚠️ CONDITIONAL RECOMMENDATION")
            print("   ✅ Profit improvement achieved")
            print("   ⚠️ Monitor risk metrics closely")
        else:
            print("   ❌ ORIGINAL STRATEGY RECOMMENDED")
            print("   ❌ Optimization did not improve performance")

        print("="*80)

def main():
    """Main execution function"""

    backtest_manager = BacktestManager()

    try:
        logger.info("🚀 Starting stop loss optimization backtesting...")

        # Define timerange (last ~5 weeks)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=35)
        timerange = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"

        print(f"📅 Backtesting period: {timerange}")

        # Run comparative backtests
        result_files = backtest_manager.run_comparative_backtests(timerange)

        # Analyze results
        if result_files:
            analyses = backtest_manager.analyze_backtest_results(result_files)

            # Save analysis to file
            analysis_file = os.path.join(backtest_manager.results_dir, f"backtest_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(analysis_file, 'w') as f:
                json.dump(analyses, f, indent=2)

            logger.info(f"📄 Analysis saved to: {analysis_file}")

        else:
            logger.error("❌ No backtest results to analyze")

    except Exception as e:
        logger.error(f"❌ Error during backtesting: {e}")
        raise

if __name__ == "__main__":
    main()