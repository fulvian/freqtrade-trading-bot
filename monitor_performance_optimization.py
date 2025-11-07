#!/usr/bin/env python3
"""
===========================================================================
PERFORMANCE MONITOR - STOP LOSS OPTIMIZATION TRACKER
===========================================================================

Script per monitorare l'efficacia delle ottimizzazioni di stop loss.
Traccia le metriche chiave prima e dopo l'implementazione.

Uso: python monitor_performance_optimization.py --db tradesv3_enhanced_8092.sqlite
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import json
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitoraggio delle performance per le ottimizzazioni di stop loss"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def get_trades_data(self, days_back: int = 30) -> pd.DataFrame:
        """Estrai i trade degli ultimi N giorni"""
        query = f"""
        SELECT
            pair,
            close_profit,
            realized_profit,
            close_rate,
            open_rate,
            is_open,
            exit_reason,
            open_date,
            close_date,
            stop_loss_pct,
            amount,
            stake_amount,
            strategy,
            leverage
        FROM trades
        WHERE is_open = 0
        AND close_date >= datetime('now', '-{days_back} days')
        ORDER BY close_date DESC
        """
        return pd.read_sql_query(query, self.conn)

    def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict:
        """Calcola le metriche di performance principali"""

        if df.empty:
            return {"error": "No trades found"}

        total_trades = len(df)
        winning_trades = df[df['close_profit'] > 0]
        losing_trades = df[df['close_profit'] <= 0]

        # Basic metrics
        win_rate = len(winning_trades) / total_trades
        total_profit = df['close_profit'].sum()
        avg_profit = df['close_profit'].mean()

        # Loss analysis
        avg_loss = losing_trades['close_profit'].mean() if len(losing_trades) > 0 else 0
        max_loss = df['close_profit'].min()
        total_loss = losing_trades['close_profit'].sum() if len(losing_trades) > 0 else 0

        # Win analysis
        avg_win = winning_trades['close_profit'].mean() if len(winning_trades) > 0 else 0
        max_win = df['close_profit'].max()

        # Risk metrics
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        sharpe_ratio = self.calculate_sharpe_ratio(df['close_profit'])

        # Stop loss analysis
        stop_loss_trades = df[df['exit_reason'] == 'stop_loss']
        catastrophic_losses = df[df['close_profit'] < -0.05]  # >5% losses

        # Time analysis
        df['duration'] = pd.to_datetime(df['close_date']) - pd.to_datetime(df['open_date'])
        avg_duration = df['duration'].mean()

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_profit_pct': total_profit,
            'avg_profit_pct': avg_profit,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'max_profit_pct': max_win,
            'max_loss_pct': max_loss,
            'total_loss_pct': total_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'stop_loss_trades': len(stop_loss_trades),
            'catastrophic_losses': len(catastrophic_losses),
            'avg_duration': str(avg_duration),
            'largest_loss_trades': self.get_largest_losses(df, 5)
        }

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calcola Sharpe ratio annualizzato"""
        if len(returns) < 2:
            return 0.0

        # Assumendo ritorni giornalieri e annualizzazione
        excess_returns = returns - risk_free_rate/252  # risk free rate giornaliero
        if excess_returns.std() == 0:
            return 0.0

        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return sharpe

    def get_largest_losses(self, df: pd.DataFrame, n: int = 5) -> List[Dict]:
        """Ottiene le N perdite più grandi"""
        worst_trades = df.nsmallest(n, 'close_profit')
        return [
            {
                'pair': row['pair'],
                'loss_pct': row['close_profit'] * 100,
                'exit_reason': row['exit_reason'],
                'date': row['close_date'][:10]  # Solo data
            }
            for _, row in worst_trades.iterrows()
        ]

    def analyze_stop_loss_effectiveness(self, df: pd.DataFrame) -> Dict:
        """Analisi specifica dell'efficacia degli stop loss"""

        stop_loss_analysis = {}

        # Analisi per exit reason
        exit_reasons = df.groupby('exit_reason').agg({
            'close_profit': ['count', 'mean', 'sum'],
            'pair': 'count'
        }).round(4)

        stop_loss_analysis['by_exit_reason'] = exit_reasons.to_dict()

        # Analisi delle perdite per range
        loss_ranges = [
            (-1.0, -0.1, 'Catastrophic (-10% to -100%)'),
            (-0.1, -0.05, 'Severe (-5% to -10%)'),
            (-0.05, -0.02, 'Significant (-2% to -5%)'),
            (-0.02, 0, 'Minor (0% to -2%)')
        ]

        loss_range_analysis = {}
        for min_loss, max_loss, label in loss_ranges:
            range_trades = df[(df['close_profit'] >= min_loss) & (df['close_profit'] < max_loss)]
            loss_range_analysis[label] = {
                'count': len(range_trades),
                'total_loss': range_trades['close_profit'].sum() if len(range_trades) > 0 else 0,
                'avg_loss': range_trades['close_profit'].mean() if len(range_trades) > 0 else 0
            }

        stop_loss_analysis['loss_ranges'] = loss_range_analysis

        return stop_loss_analysis

    def generate_alerts(self, metrics: Dict) -> List[str]:
        """Genera alert basati sulle metriche"""
        alerts = []

        if metrics['max_loss_pct'] < -0.05:  # Perdita >5%
            alerts.append(f"🚨 CRITICAL: Maximum loss {metrics['max_loss_pct']*100:.2f}% exceeds 5% threshold!")

        if metrics['catastrophic_losses'] > 0:
            alerts.append(f"⚠️ WARNING: {metrics['catastrophic_losses']} catastrophic losses (>5%) detected!")

        if metrics['avg_loss_pct'] < -0.025:  # Perdita media >2.5%
            alerts.append(f"⚠️ WARNING: Average loss {metrics['avg_loss_pct']*100:.2f}% exceeds 2.5% target!")

        if metrics['profit_factor'] < 1.5:
            alerts.append(f"📊 PERFORMANCE: Profit factor {metrics['profit_factor']:.2f} below 1.5 target!")

        if metrics['win_rate'] < 0.85:  # Win rate <85%
            alerts.append(f"📈 WIN RATE: Win rate {metrics['win_rate']*100:.1f}% below 85% minimum!")

        if len(metrics['largest_loss_trades']) > 0 and metrics['largest_loss_trades'][0]['loss_pct'] < -8:
            alerts.append(f"💀 EXTREME LOSS: Trade with {metrics['largest_loss_trades'][0]['loss_pct']:.1f}% loss detected!")

        return alerts

    def print_dashboard(self, metrics: Dict, stop_loss_analysis: Dict = None):
        """Stampa una dashboard completa delle performance"""

        print("\n" + "="*80)
        print("📊 PERFORMANCE DASHBOARD - STOP LOSS OPTIMIZATION")
        print("="*80)

        # Performance Overview
        print(f"\n📈 PERFORMANCE OVERVIEW:")
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Win Rate: {metrics['win_rate']*100:.1f}% ({metrics['winning_trades']}/{metrics['total_trades']})")
        print(f"   Total Profit: {metrics['total_profit_pct']*100:.2f}%")
        print(f"   Avg Profit/Trade: {metrics['avg_profit_pct']*100:.3f}%")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

        # Win/Loss Analysis
        print(f"\n🎯 WIN/LOSS ANALYSIS:")
        print(f"   Average Win: {metrics['avg_win_pct']*100:.2f}%")
        print(f"   Average Loss: {metrics['avg_loss_pct']*100:.2f}%")
        print(f"   Largest Win: {metrics['max_profit_pct']*100:.2f}%")
        print(f"   Largest Loss: {metrics['max_loss_pct']*100:.2f}%")
        print(f"   Stop Loss Trades: {metrics['stop_loss_trades']}")
        print(f"   Catastrophic Losses: {metrics['catastrophic_losses']}")

        # Worst Trades
        if metrics['largest_loss_trades']:
            print(f"\n💀 WORST TRADES (Top 5):")
            for i, trade in enumerate(metrics['largest_loss_trades'], 1):
                print(f"   {i}. {trade['pair']}: {trade['loss_pct']:.2f}% ({trade['exit_reason']}) on {trade['date']}")

        # Alerts
        alerts = self.generate_alerts(metrics)
        if alerts:
            print(f"\n🚨 ALERTS:")
            for alert in alerts:
                print(f"   {alert}")
        else:
            print(f"\n✅ NO ALERTS - All metrics within acceptable ranges!")

        print(f"\n⏱️ Average Trade Duration: {metrics['avg_duration']}")
        print("="*80)

    def compare_before_after(self, before_df: pd.DataFrame, after_df: pd.DataFrame):
        """Compara performance prima e dopo l'ottimizzazione"""

        before_metrics = self.calculate_performance_metrics(before_df)
        after_metrics = self.calculate_performance_metrics(after_df)

        print("\n" + "="*80)
        print("🔄 BEFORE vs AFTER OPTIMIZATION COMPARISON")
        print("="*80)

        comparison_metrics = [
            ('Win Rate', 'win_rate', '%'),
            ('Total Profit', 'total_profit_pct', '%'),
            ('Average Profit', 'avg_profit_pct', '%'),
            ('Profit Factor', 'profit_factor', ''),
            ('Sharpe Ratio', 'sharpe_ratio', ''),
            ('Max Loss', 'max_loss_pct', '%'),
            ('Avg Loss', 'avg_loss_pct', '%'),
            ('Catastrophic Losses', 'catastrophic_losses', '')
        ]

        print(f"\n{'Metric':<20} {'Before':<15} {'After':<15} {'Improvement':<15}")
        print("-" * 65)

        for metric_name, key, suffix in comparison_metrics:
            before_val = before_metrics.get(key, 0)
            after_val = after_metrics.get(key, 0)

            # Format values
            if suffix == '%':
                before_str = f"{before_val*100:.2f}%"
                after_str = f"{after_val*100:.2f}%"
                improvement = (after_val - before_val) * 100
                improvement_str = f"{improvement:+.2f}%"
            else:
                before_str = f"{before_val:.2f}"
                after_str = f"{after_val:.2f}"
                improvement = after_val - before_val
                improvement_str = f"{improvement:+.2f}"

            print(f"{metric_name:<20} {before_str:<15} {after_str:<15} {improvement_str:<15}")

        print("\n" + "="*80)

    def close(self):
        """Chiudi la connessione al database"""
        self.conn.close()

def main():
    parser = argparse.ArgumentParser(description='Monitor performance of stop loss optimization')
    parser.add_argument('--db', required=True, help='Database file path')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    parser.add_argument('--compare', action='store_true', help='Compare before/after optimization')
    parser.add_argument('--cutoff-date', help='Date for before/after comparison (YYYY-MM-DD)')

    args = parser.parse_args()

    monitor = PerformanceMonitor(args.db)

    try:
        if args.compare and args.cutoff_date:
            # Compare before/after optimization
            before_df = monitor.get_trades_data(365)  # Last year
            before_df = before_df[before_df['close_date'] < args.cutoff_date]

            after_df = monitor.get_trades_data(args.days)

            monitor.compare_before_after(before_df, after_df)
        else:
            # Single period analysis
            df = monitor.get_trades_data(args.days)
            metrics = monitor.calculate_performance_metrics(df)
            stop_loss_analysis = monitor.analyze_stop_loss_effectiveness(df)

            monitor.print_dashboard(metrics, stop_loss_analysis)

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
    finally:
        monitor.close()

if __name__ == "__main__":
    main()