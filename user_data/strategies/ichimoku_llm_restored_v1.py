"""
===========================================================================
ICHIMOKU LLM RESTORED V1 - CONTEXT7 SUPER POWERS VERSION
===========================================================================

Versione ripristinata con parametri robusti mantenendo le nuove features:

CARATTERISTICHE PRINCIPALI:
✅ Parametri originali ripristinati (thresholds robusti)
✅ Multi-timeframe consensus validation
✅ Dynamic stake sizing con position sizing intelligente
✅ Risk management avanzato con AdaptiveRiskManager
✅ Ichimoku Cloud con segnali avanzati
✅ LLM regime detection (basic mode per stabilità)
✅ Smart volume analysis con threshold rigorosi
✅ Solo segnali forti (NO NEUTRAL signals)
✅ Stop-loss dinamici con Ichimoku structure

PARAMETRI RIPRISTINATI:
- Quality thresholds: 0.6-0.7 (valori originali robusti)
- Volume confirmation: > 0.8 senza eccezioni
- Solo STRONG_LONG/LONG e STRONG_SHORT/SHORT
- Nessun bypass per high quality score
- Regime filtering rigoroso
"""

import json
import logging
import os
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

# FreqTrade imports
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (DecimalParameter, IStrategy, IntParameter,
                                RealParameter, merge_informative_pair, stoploss_from_absolute,
                                stoploss_from_open)
from freqtrade.persistence import Trade

# Context7 Super Powers imports
from json_repair import repair_json

# Tenacity integration per retry robusti
try:
    from tenacity import (
        retry, stop_after_attempt, stop_after_delay,
        wait_exponential, wait_random_exponential,
        retry_if_exception_type, retry_if_result,
        before_log, after_log, Retrying, RetryError
    )
    TENACITY_AVAILABLE = True
    logging.info("✅ Context7 Super Powers: Tenacity available")
except ImportError:
    TENACITY_AVAILABLE = False
    logging.warning("⚠️ Context7: Tenacity not available")

# ============================================================================
# CONTEXT7 SUPER POWERS: DATA MODELS
# ============================================================================

@dataclass
class MarketData:
    """Dati di mercato puliti e validati"""
    pair: str
    current_price: float
    volume_24h: float
    price_change_24h: float
    volatility_20d: float
    rsi_14: float
    macd_signal: float
    tenkan_sen: float
    kijun_sen: float
    senkou_span_a: float
    senkou_span_b: float
    chikou_span: float
    volume_ratio: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RegimeAnalysis:
    """Analisi regime LLM pulita"""
    pair: str
    regime: str  # CALM, VOLATILE, TRENDING, SIDEWAYS
    confidence: float  # 0.0-1.0
    volatility_level: str  # LOW, MEDIUM, HIGH
    trend_strength: float  # 0.0-1.0
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DirectionalSignal:
    """Segnale di trading bidirezionale Context7-compliant"""
    direction: str  # LONG, SHORT, NEUTRAL
    primary_score: float  # 0.0-1.0, segnale primario
    trend_weight: float  # 0.0-2.0, peso basato su trend
    quality_score: float  # 0.0-1.0, qualità complessiva
    entry_conditions: List[str]  # Condizioni soddisfatte
    ichimoku_position: str  # ABOVE_CLOUD, INSIDE_CLOUD, BELOW_CLOUD
    volume_confirmation: bool  # Volume ratio validation
    regime_compatibility: float  # 0.0-1.0, compatibilità con regime
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class BidirectionalAnalysis:
    """Analisi bidirezionale completa Context7-compliant"""
    pair: str
    long_signal: DirectionalSignal
    short_signal: DirectionalSignal
    market_confidence: float  # Confidence generale del mercato
    best_direction: str  # LONG, SHORT, NEUTRAL
    best_quality_score: float  # Miglior punteggio di qualità
    recommendation: str  # STRONG_LONG, LONG, NEUTRAL, SHORT, STRONG_SHORT
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TimeframeAnalysis:
    """Analisi multi-timeframe"""
    timeframe: str
    signal_strength: float
    trend_direction: str
    volume_confirmation: bool
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RiskMetrics:
    """Metriche di rischio per posizione sizing"""
    volatility_multiplier: float
    regime_risk_factor: float
    correlation_adjustment: float
    portfolio_exposure: float
    max_position_size: float
    risk_score: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class StakeSizingDecision:
    """Decisione di stake sizing intelligente"""
    recommended_stake: float
    risk_adjusted_leverage: float
    position_size: float
    risk_level: str  # LOW, MEDIUM, HIGH, EXTREME
    reasoning: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)

# ============================================================================
# ADAPTIVE RISK MANAGER
# ============================================================================

class AdaptiveRiskManager:
    """
    Context7 Super Powers: Adaptive Risk Manager

    Sistema avanzato per il calcolo dinamico della posizione e del rischio
    basato su multiple fattori di mercato.
    """

    def __init__(self, config: dict):
        self.config = config
        self.max_portfolio_risk = 0.02  # 2% rischio massimo del portfolio per trade
        self.min_stake_amount = 10.0   # Stake minimo in USDT
        self.max_leverage = 5.0         # Leverage massimo per sicurezza
        self.logger = logging.getLogger("AdaptiveRiskManager")

        # Portfolio tracking
        self.active_positions = {}
        self.daily_pnl = 0.0
        self.total_exposure = 0.0

        # Risk adjustment factors
        self.risk_multipliers = {
            'LOW': 1.2,      # Aumenta dimensione in low risk
            'MEDIUM': 1.0,   # Normale
            'HIGH': 0.7,     # Riduce in high risk
            'EXTREME': 0.4   # Minima in extreme risk
        }

    def calculate_risk_metrics(self, market_data: MarketData, regime: RegimeAnalysis,
                              current_positions: Dict[str, Any]) -> RiskMetrics:
        """
        Calcola metriche di rischio comprehensive per position sizing
        """
        try:
            # 1. Volatility-based multiplier
            if market_data.volatility_20d < 0.02:  # Low volatility
                volatility_multiplier = 1.2
            elif market_data.volatility_20d < 0.04:  # Medium volatility
                volatility_multiplier = 1.0
            elif market_data.volatility_20d < 0.06:  # High volatility
                volatility_multiplier = 0.7
            else:  # Extreme volatility
                volatility_multiplier = 0.4

            # 2. Regime-based risk factor
            regime_risk_factors = {
                'CALM': 0.8,      # Lower risk in calm markets
                'TRENDING': 1.0,   # Normal risk in trending
                'SIDEWAYS': 1.2,   # Higher risk in sideways (mean reversion)
                'VOLATILE': 0.6    # Lower risk in volatile markets
            }
            regime_risk_factor = regime_risk_factors.get(regime.regime, 1.0)

            # 3. Correlation adjustment (basato su posizioni attive)
            correlation_adjustment = self._calculate_correlation_adjustment(
                market_data.pair, current_positions
            )

            # 4. Portfolio exposure calculation
            current_exposure = self._calculate_portfolio_exposure(current_positions)

            # 5. Maximum position size based on risk
            portfolio_risk_capacity = 1.0 - (current_exposure / 1.0)  # 1.0 = 100%
            max_position_size = self.max_portfolio_risk * portfolio_risk_capacity

            # 6. Overall risk score (0.0-1.0, higher = riskier)
            risk_score = (
                (1.0 - volatility_multiplier) * 0.25 +        # Volatility contribution
                (1.0 - regime_risk_factor) * 0.25 +          # Regime contribution
                (1.0 - correlation_adjustment) * 0.25 +       # Correlation contribution
                current_exposure * 0.25                       # Portfolio exposure contribution
            )

            return RiskMetrics(
                volatility_multiplier=volatility_multiplier,
                regime_risk_factor=regime_risk_factor,
                correlation_adjustment=correlation_adjustment,
                portfolio_exposure=current_exposure,
                max_position_size=max_position_size,
                risk_score=min(risk_score, 1.0)
            )

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            # Return conservative default values
            return RiskMetrics(
                volatility_multiplier=0.7,
                regime_risk_factor=0.8,
                correlation_adjustment=0.8,
                portfolio_exposure=0.0,
                max_position_size=self.max_portfolio_risk * 0.5,
                risk_score=0.7
            )

    def _calculate_correlation_adjustment(self, pair: str, current_positions: Dict[str, Any]) -> float:
        """
        Calcola fattore di aggiustamento basato su correlazione con posizioni attive
        """
        if not current_positions:
            return 1.0

        # Logica semplificata basata su asset correlati
        # In una implementazione reale, si userebbe una matrice di correlazione
        correlated_pairs = []

        for pos_pair, pos_data in current_positions.items():
            # Check se sono asset correlati (es. ETH/BTC, SOL/BTC)
            if (pair.endswith('BTC/USDT:USDT') and pos_pair.endswith('BTC/USDT:USDT')) or \
               (pair.endswith('ETH/USDT:USDT') and pos_pair.endswith('ETH/USDT:USDT')):
                correlated_pairs.append(pos_pair)

        # Riduci esposizione se ci sono già posizioni correlate
        if len(correlated_pairs) >= 2:
            return 0.5  # Riduzione drastica
        elif len(correlated_pairs) == 1:
            return 0.7  # Riduzione moderata
        else:
            return 1.0  # Nessuna riduzione

    def _calculate_portfolio_exposure(self, current_positions: Dict[str, Any]) -> float:
        """
        Calcola esposizione totale del portfolio
        """
        total_exposure = 0.0

        for pair, pos_data in current_positions.items():
            # Somma le posizioni in base a size e leverage
            position_size = pos_data.get('size', 0.0)
            leverage = pos_data.get('leverage', 1.0)
            total_exposure += position_size * leverage

        # Normalizza a 1.0 (100% del portfolio)
        return min(total_exposure / 10000.0, 1.0)  # Assumendo 10000 USDT total portfolio

    def calculate_stake_sizing(self, pair: str, signal_strength: float, risk_metrics: RiskMetrics,
                             base_stake: float, max_leverage: float) -> StakeSizingDecision:
        """
        Calcola dimensione dello stake e leverage ottimali
        """
        try:
            # 1. Base stake adjusted for signal strength
            strength_adjusted_stake = base_stake * (0.5 + (signal_strength * 0.5))

            # 2. Apply risk multipliers
            risk_multiplier = (
                risk_metrics.volatility_multiplier *
                risk_metrics.regime_risk_factor *
                risk_metrics.correlation_adjustment *
                (1.0 - risk_metrics.risk_score * 0.5)  # Reduce based on risk score
            )

            # 3. Calculate recommended stake
            recommended_stake = strength_adjusted_stake * risk_multiplier

            # 4. Apply position size limits
            max_allowed_stake = self.max_portfolio_risk * 10000  # Assuming 10k portfolio
            recommended_stake = min(recommended_stake, max_allowed_stake)

            # 5. Calculate optimal leverage
            # Higher leverage for stronger signals with lower risk
            if signal_strength > 0.8 and risk_metrics.risk_score < 0.3:
                optimal_leverage = min(max_leverage, self.max_leverage)
            elif signal_strength > 0.7 and risk_metrics.risk_score < 0.5:
                optimal_leverage = min(max_leverage, 3.0)
            elif signal_strength > 0.6 and risk_metrics.risk_score < 0.7:
                optimal_leverage = 2.0
            else:
                optimal_leverage = 1.0  # No leverage for weak/risky signals

            # 6. Calculate position size
            position_size = recommended_stake * optimal_leverage

            # 7. Determine risk level
            if risk_metrics.risk_score < 0.3 and signal_strength > 0.8:
                risk_level = "LOW"
            elif risk_metrics.risk_score < 0.6 and signal_strength > 0.6:
                risk_level = "MEDIUM"
            elif risk_metrics.risk_score < 0.8 and signal_strength > 0.5:
                risk_level = "HIGH"
            else:
                risk_level = "EXTREME"

            # 8. Generate reasoning
            reasoning_parts = [
                f"Signal strength: {signal_strength:.2f}",
                f"Risk score: {risk_metrics.risk_score:.2f}",
                f"Volatility multiplier: {risk_metrics.volatility_multiplier:.2f}",
                f"Regime factor: {risk_metrics.regime_risk_factor:.2f}",
                f"Portfolio exposure: {risk_metrics.portfolio_exposure:.1%}"
            ]

            # 9. Apply minimum stake
            recommended_stake = max(recommended_stake, self.min_stake_amount)

            return StakeSizingDecision(
                recommended_stake=recommended_stake,
                risk_adjusted_leverage=optimal_leverage,
                position_size=position_size,
                risk_level=risk_level,
                reasoning=" | ".join(reasoning_parts),
                confidence=signal_strength * (1.0 - risk_metrics.risk_score)
            )

        except Exception as e:
            self.logger.error(f"Error calculating stake sizing: {e}")
            # Return conservative decision
            return StakeSizingDecision(
                recommended_stake=self.min_stake_amount,
                risk_adjusted_leverage=1.0,
                position_size=self.min_stake_amount,
                risk_level="HIGH",
                reason="Error in calculation - using conservative values",
                confidence=0.5
            )

# ============================================================================
# CONTEXT7 SUPER POWERS: MAIN STRATEGY
# ============================================================================

class IchimokuLLMRestoredV1(IStrategy):
    """
    Context7 Super Powers: Ichimoku LLM Restored V1

    Versione ripristinata con parametri robusti mantenendo le features avanzate:
    - Multi-timeframe consensus
    - Dynamic stake sizing
    - Adaptive risk management
    - Parametri originali rigorosi
    """

    # Interface version
    INTERFACE_VERSION = 3

    # Strategy metadata
    timeframe = '5m'
    startup_candle_count: int = 200
    process_only_new_candles = True

    # Optimal stoploss and take profit
    stoploss = -0.15  # -15%
    take_profit = 0.30  # +30%

    # Enable custom stoploss (FreqTrade compliance)
    use_custom_stoploss = True

    # Enable short trading
    can_short = True

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # ROI table
    minimal_roi = {
        "0": 0.30,   # 30% at close
        "20": 0.20,  # 20% after 20 minutes
        "40": 0.10,  # 10% after 40 minutes
        "60": 0.05   # 5% after 60 minutes
    }

    # Trade settings
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
        'take_profit': 'limit',
        'take_profit_on_exchange': False,
        'trailing_stop': 'market',
        'trailing_stop_on_exchange': False,
    }

    # Order time in force
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc',
    }

    def __init__(self, config: dict) -> None:
        """Initialize Context7 Super Powers strategy - RESTORED VERSION"""
        super().__init__(config)

        # Adaptive Risk Manager
        self.risk_manager = AdaptiveRiskManager(config)

        # Cache locale per risultati
        self.local_cache = {}
        self.cache_ttl = 300  # 5 minuti

        # Multi-timeframe analysis cache
        self.timeframe_cache = {}

        # Logging
        self.logger = logging.getLogger("IchimokuLLMRestoredV1")
        self.logger.info("🚀 Context7 Super Powers: IchimokuLLMRestoredV1 initialized")
        self.logger.info("✅ Parametri robusti ripristinati (thresholds originali)")
        self.logger.info("✅ Multi-timeframe consensus + Dynamic stake sizing")
        self.logger.info("✅ Adaptive risk management attivo")
        self.logger.info("⚠️ Solo segnali forti accettati (NO NEUTRAL)")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Populate technical indicators con SMART volume calculation"""
        # Ichimoku Cloud
        high_9 = dataframe['high'].rolling(window=9).max()
        low_9 = dataframe['low'].rolling(window=9).min()
        high_26 = dataframe['high'].rolling(window=26).max()
        low_26 = dataframe['low'].rolling(window=26).min()
        high_52 = dataframe['high'].rolling(window=52).max()
        low_52 = dataframe['low'].rolling(window=52).min()

        dataframe['tenkan_sen'] = (high_9 + low_9) / 2
        dataframe['kijun_sen'] = (high_26 + low_26) / 2
        dataframe['senkou_span_a'] = ((dataframe['tenkan_sen'] + dataframe['kijun_sen']) / 2).shift(26)
        dataframe['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
        dataframe['chikou_span'] = dataframe['close'].shift(-26)

        # RSI
        delta = dataframe['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        dataframe['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = dataframe['close'].ewm(span=12).mean()
        exp2 = dataframe['close'].ewm(span=26).mean()
        dataframe['macd'] = exp1 - exp2
        dataframe['macdsignal'] = dataframe['macd'].ewm(span=9).mean()
        dataframe['macdhist'] = dataframe['macd'] - dataframe['macdsignal']

        # Additional indicators
        dataframe['sma_fast'] = dataframe['close'].rolling(window=10).mean()
        dataframe['sma_slow'] = dataframe['close'].rolling(window=30).mean()

        # ATR for dynamic stoploss
        dataframe['tr1'] = dataframe['high'] - dataframe['low']
        dataframe['tr2'] = abs(dataframe['high'] - dataframe['close'].shift())
        dataframe['tr3'] = abs(dataframe['low'] - dataframe['close'].shift())
        dataframe['tr'] = dataframe[['tr1', 'tr2', 'tr3']].max(axis=1)
        dataframe['atr'] = dataframe['tr'].rolling(window=14).mean()

        # CMF for volume analysis
        mfm = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
        mfv = mfm * dataframe['volume']
        dataframe['cmf'] = mfv.rolling(window=21).sum() / dataframe['volume'].rolling(window=21).sum()

        # Smart Volume Baseline - FIXED robusto
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=50).mean()
        dataframe['volume_70th_percentile'] = dataframe['volume'].rolling(window=50).quantile(0.70)

        # Smart ratio: use 70th percentile as baseline (robusto)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_70th_percentile']

        # Alternative hybrid approach
        dataframe['volume_smart_mean'] = (dataframe['volume_mean'] * 0.6) + (dataframe['volume_70th_percentile'] * 0.4)
        dataframe['volume_smart_ratio'] = dataframe['volume'] / dataframe['volume_smart_mean']

        return dataframe

    def _get_market_data(self, dataframe: DataFrame, metadata: dict) -> Optional[MarketData]:
        """Estrae dati di mercato puliti dal dataframe"""
        try:
            if len(dataframe) < 50:
                return None

            last = dataframe.iloc[-1]
            prev = dataframe.iloc[-2]

            # Calcolo volatilità
            returns = dataframe['close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1] * np.sqrt(365)

            return MarketData(
                pair=metadata['pair'],
                current_price=float(last['close']),
                volume_24h=float(last['volume']),
                price_change_24h=((last['close'] - prev['close']) / prev['close']) * 100,
                volatility_20d=float(volatility),
                rsi_14=float(last['rsi']),
                macd_signal=float(last['macd']),
                tenkan_sen=float(last['tenkan_sen']),
                kijun_sen=float(last['kijun_sen']),
                senkou_span_a=float(last['senkou_span_a']),
                senkou_span_b=float(last['senkou_span_b']),
                chikou_span=float(last['chikou_span']),
                volume_ratio=float(last['volume_ratio'])
            )
        except Exception as e:
            self.logger.error(f"❌ Error extracting market data for {metadata['pair']}: {e}")
            return None

    def _analyze_multi_timeframe_consensus(self, pair: str) -> Dict[str, float]:
        """
        Context7 Super Powers: Multi-timeframe consensus analysis

        Analisi del consenso tra timeframe multipli (1m, 5m, 15m, 1h)
        """
        try:
            # Cache check
            cache_key = f"mtf_{pair}_{datetime.now().minute}"
            if cache_key in self.timeframe_cache:
                return self.timeframe_cache[cache_key]

            timeframes = ['1m', '5m', '15m', '1h']
            timeframe_weights = {'1m': 0.15, '5m': 0.35, '15m': 0.30, '1h': 0.20}

            consensus_scores = {'LONG': 0.0, 'SHORT': 0.0, 'NEUTRAL': 0.0}
            valid_timeframes = 0

            for tf in timeframes:
                try:
                    # Ottieni dati per timeframe
                    tf_data, _ = self.dp.get_analyzed_dataframe(pair, tf)
                    if tf_data is None or len(tf_data) < 20:
                        continue

                    last = tf_data.iloc[-1]

                    # Calcola segnale base per questo timeframe
                    tf_score = 0.0

                    # Ichimoku signals (60% weight)
                    if last['close'] > max(last['senkou_span_a'], last['senkou_span_b']):
                        if last['tenkan_sen'] > last['kijun_sen']:
                            tf_score += 0.6
                        else:
                            tf_score += 0.3
                    elif last['close'] < min(last['senkou_span_a'], last['senkou_span_b']):
                        if last['tenkan_sen'] < last['kijun_sen']:
                            tf_score -= 0.6
                        else:
                            tf_score -= 0.3

                    # MACD signals (20% weight)
                    if last['macd'] > last['macdsignal']:
                        tf_score += 0.2
                    else:
                        tf_score -= 0.2

                    # Volume confirmation (20% weight)
                    if last.get('volume_smart_ratio', 0) > 1.2:
                        tf_score *= 1.2
                    elif last.get('volume_smart_ratio', 0) < 0.8:
                        tf_score *= 0.8

                    # Apply timeframe weight
                    weighted_score = tf_score * timeframe_weights[tf]

                    # Add to consensus
                    if weighted_score > 0.15:
                        consensus_scores['LONG'] += abs(weighted_score)
                    elif weighted_score < -0.15:
                        consensus_scores['SHORT'] += abs(weighted_score)
                    else:
                        consensus_scores['NEUTRAL'] += 0.1

                    valid_timeframes += 1

                except Exception as e:
                    self.logger.debug(f"Error analyzing timeframe {tf} for {pair}: {e}")
                    continue

            # Normalize scores
            total_score = sum(consensus_scores.values())
            if total_score > 0:
                for direction in consensus_scores:
                    consensus_scores[direction] /= total_score

            # Cache result
            self.timeframe_cache[cache_key] = consensus_scores

            # Cleanup old cache entries
            if len(self.timeframe_cache) > 100:
                self.timeframe_cache.clear()

            self.logger.debug(f"MTF consensus for {pair}: {consensus_scores}")
            return consensus_scores

        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis for {pair}: {e}")
            return {'LONG': 0.33, 'SHORT': 0.33, 'NEUTRAL': 0.34}

    def _analyze_directional_signals(self, dataframe: DataFrame, market_data: MarketData,
                                    regime: RegimeAnalysis, mtf_consensus: Dict[str, float]) -> Tuple[DirectionalSignal, DirectionalSignal]:
        """
        Context7 Super Powers: Analisi segnali bidirezionali con MTF consensus
        """
        last = dataframe.iloc[-1]

        # Ichimoku position analysis
        if last['close'] > max(last['senkou_span_a'], last['senkou_span_b']):
            ichimoku_position = "ABOVE_CLOUD"
            cloud_bias = 1.0
        elif last['close'] < min(last['senkou_span_a'], last['senkou_span_b']):
            ichimoku_position = "BELOW_CLOUD"
            cloud_bias = -1.0
        else:
            ichimoku_position = "INSIDE_CLOUD"
            cloud_bias = 0.0

        # Volume confirmation - RIGOROSO, nessuna eccezione
        volume_confirmation = last['volume_smart_ratio'] > 0.8
        volume_weight = min(last['volume_smart_ratio'], 2.0)

        # MTF consensus bonus
        mtf_bonus_long = mtf_consensus.get('LONG', 0.33)
        mtf_bonus_short = mtf_consensus.get('SHORT', 0.33)

        # Analyze LONG signal
        long_signal = self._analyze_single_direction(
            direction="LONG",
            dataframe=dataframe,
            market_data=market_data,
            regime=regime,
            ichimoku_position=ichimoku_position,
            cloud_bias=cloud_bias,
            volume_confirmation=volume_confirmation,
            volume_weight=volume_weight,
            last=last,
            mtf_bonus=mtf_bonus_long
        )

        # Analyze SHORT signal
        short_signal = self._analyze_single_direction(
            direction="SHORT",
            dataframe=dataframe,
            market_data=market_data,
            regime=regime,
            ichimoku_position=ichimoku_position,
            cloud_bias=cloud_bias,
            volume_confirmation=volume_confirmation,
            volume_weight=volume_weight,
            last=last,
            mtf_bonus=mtf_bonus_short
        )

        return long_signal, short_signal

    def _analyze_single_direction(self, direction: str, dataframe: DataFrame, market_data: MarketData,
                                 regime: RegimeAnalysis, ichimoku_position: str, cloud_bias: float,
                                 volume_confirmation: bool, volume_weight: float, last,
                                 mtf_bonus: float = 0.33) -> DirectionalSignal:
        """
        Context7 Super Powers: Analisi segnale singolo con parametri RIPRISTINATI
        """
        entry_conditions = []
        primary_score = 0.0

        if direction == "LONG":
            # Primary signal: Ichimoku bullish conditions
            if last['tenkan_sen'] > last['kijun_sen']:
                primary_score += 0.3
                entry_conditions.append("Tenkan above Kijun")

            if ichimoku_position == "ABOVE_CLOUD":
                primary_score += 0.4
                entry_conditions.append("Price above cloud")
            elif ichimoku_position == "INSIDE_CLOUD" and cloud_bias > 0:
                primary_score += 0.2
                entry_conditions.append("Price inside cloud with bullish bias")

            # Momentum confirmation
            if last['macd'] > last['macdsignal']:
                primary_score += 0.2
                entry_conditions.append("MACD bullish")

            # RSI conditions (regime-aware)
            if regime.regime == 'CALM' and last['rsi'] < 40:
                primary_score += 0.15
                entry_conditions.append("RSI oversold in calm market")
            elif regime.regime == 'TRENDING' and 40 < last['rsi'] < 70:
                primary_score += 0.1
                entry_conditions.append("RSI in trend range")
            elif regime.regime == 'VOLATILE' and last['rsi'] < 35:
                primary_score += 0.1
                entry_conditions.append("RSI oversold in volatile market")

            # Moving average confirmation
            if last['close'] > last['sma_fast'] and last['sma_fast'] > last['sma_slow']:
                primary_score += 0.15
                entry_conditions.append("Price above MAs")

            # Volume confirmation - RIGOROSO
            if volume_confirmation:
                primary_score += 0.1 * volume_weight
                entry_conditions.append(f"Volume confirmed (ratio: {last['volume_smart_ratio']:.2f})")

            # Multi-timeframe consensus bonus
            primary_score += mtf_bonus * 0.2
            if mtf_bonus > 0.4:
                entry_conditions.append(f"MTF consensus bullish ({mtf_bonus:.2f})")

        else:  # SHORT
            # Primary signal: Ichimoku bearish conditions
            if last['tenkan_sen'] < last['kijun_sen']:
                primary_score += 0.3
                entry_conditions.append("Tenkan below Kijun")

            if ichimoku_position == "BELOW_CLOUD":
                primary_score += 0.4
                entry_conditions.append("Price below cloud")
            elif ichimoku_position == "INSIDE_CLOUD" and cloud_bias < 0:
                primary_score += 0.2
                entry_conditions.append("Price inside cloud with bearish bias")

            # Momentum confirmation
            if last['macd'] < last['macdsignal']:
                primary_score += 0.2
                entry_conditions.append("MACD bearish")

            # RSI conditions (regime-aware)
            if regime.regime == 'CALM' and last['rsi'] > 60:
                primary_score += 0.15
                entry_conditions.append("RSI overbought in calm market")
            elif regime.regime == 'TRENDING' and 30 < last['rsi'] < 60:
                primary_score += 0.1
                entry_conditions.append("RSI in trend range")
            elif regime.regime == 'VOLATILE' and last['rsi'] > 65:
                primary_score += 0.1
                entry_conditions.append("RSI overbought in volatile market")

            # Moving average confirmation
            if last['close'] < last['sma_fast'] and last['sma_fast'] < last['sma_slow']:
                primary_score += 0.15
                entry_conditions.append("Price below MAs")

            # Volume confirmation - RIGOROSO
            if volume_confirmation:
                primary_score += 0.1 * volume_weight
                entry_conditions.append(f"Volume confirmed (ratio: {last['volume_smart_ratio']:.2f})")

            # Multi-timeframe consensus bonus
            primary_score += mtf_bonus * 0.2
            if mtf_bonus > 0.4:
                entry_conditions.append(f"MTF consensus bearish ({mtf_bonus:.2f})")

        # Calculate trend weight (regime-aware)
        trend_weight = 1.0

        if regime.regime == 'TRENDING':
            trend_weight = 2.0 if direction == "LONG" and regime.trend_strength > 0.6 else 1.5
        elif regime.regime == 'VOLATILE':
            trend_weight = 0.8
        elif regime.regime == 'CALM':
            trend_weight = 1.2

        # Calculate regime compatibility
        regime_compatibility = self._calculate_regime_compatibility(direction, regime, last)

        # Calculate final quality score con parametri ORIGINALI
        quality_score = (
            primary_score * 0.4 +           # Primary signals (40%)
            trend_weight * 0.2 +            # Trend awareness (20%)
            regime_compatibility * 0.2 +     # Regime compatibility (20%)
            (volume_weight / 2.0) * 0.2      # Volume confirmation (20%)
        )

        # Cap quality score at 1.0
        quality_score = min(quality_score, 1.0)

        return DirectionalSignal(
            direction=direction,
            primary_score=min(primary_score, 1.0),
            trend_weight=trend_weight,
            quality_score=quality_score,
            entry_conditions=entry_conditions,
            ichimoku_position=ichimoku_position,
            volume_confirmation=volume_confirmation,
            regime_compatibility=regime_compatibility
        )

    def _calculate_regime_compatibility(self, direction: str, regime: RegimeAnalysis, last) -> float:
        """
        Context7 Super Powers: Calcolo compatibilità regime
        """
        base_compatibility = 0.5

        if regime.regime == 'TRENDING':
            if direction == "LONG" and last['close'] > last['sma_slow']:
                base_compatibility = 0.9
            elif direction == "SHORT" and last['close'] < last['sma_slow']:
                base_compatibility = 0.9
            else:
                base_compatibility = 0.3

        elif regime.regime == 'CALM':
            base_compatibility = 0.8
            if direction == "LONG" and last['rsi'] < 45:
                base_compatibility = 0.9
            elif direction == "SHORT" and last['rsi'] > 55:
                base_compatibility = 0.9

        elif regime.regime == 'VOLATILE':
            base_compatibility = 0.6
            macd_strength = abs(last['macd'] - last['macdsignal'])
            if macd_strength > 0.001:
                base_compatibility = 0.8

        elif regime.regime == 'SIDEWAYS':
            if direction == "LONG" and last['rsi'] < 35:
                base_compatibility = 0.8
            elif direction == "SHORT" and last['rsi'] > 65:
                base_compatibility = 0.8
            else:
                base_compatibility = 0.4

        # Adjust for regime confidence
        adjusted_compatibility = base_compatibility * regime.confidence

        return min(adjusted_compatibility, 1.0)

    def _get_bidirectional_analysis(self, dataframe: DataFrame, market_data: MarketData,
                                   regime: RegimeAnalysis, mtf_consensus: Dict[str, float]) -> BidirectionalAnalysis:
        """
        Context7 Super Powers: Analisi bidirezionale completa
        """
        # Get directional signals
        long_signal, short_signal = self._analyze_directional_signals(dataframe, market_data, regime, mtf_consensus)

        # Market confidence based on regime and signal quality
        market_confidence = regime.confidence * max(long_signal.quality_score, short_signal.quality_score)

        # Determine best direction and recommendation - SOLO SEGNALI FORTI
        if long_signal.quality_score > short_signal.quality_score + 0.15:  # Margin significativo
            best_direction = "LONG"
            best_quality_score = long_signal.quality_score

            if long_signal.quality_score > 0.8:
                recommendation = "STRONG_LONG"
            elif long_signal.quality_score > 0.6:
                recommendation = "LONG"
            else:
                recommendation = "NEUTRAL"  # Non abbastanza forte

        elif short_signal.quality_score > long_signal.quality_score + 0.15:  # Margin significativo
            best_direction = "SHORT"
            best_quality_score = short_signal.quality_score

            if short_signal.quality_score > 0.8:
                recommendation = "STRONG_SHORT"
            elif short_signal.quality_score > 0.6:
                recommendation = "SHORT"
            else:
                recommendation = "NEUTRAL"  # Non abbastanza forte

        else:
            best_direction = "NEUTRAL"
            best_quality_score = max(long_signal.quality_score, short_signal.quality_score)
            recommendation = "NEUTRAL"

        # Generate reasoning
        if recommendation in ["STRONG_LONG", "LONG"]:
            reasoning = f"Bullish setup: {', '.join(long_signal.entry_conditions[:3])} | Quality: {long_signal.quality_score:.2f}"
        elif recommendation in ["STRONG_SHORT", "SHORT"]:
            reasoning = f"Bearish setup: {', '.join(short_signal.entry_conditions[:3])} | Quality: {short_signal.quality_score:.2f}"
        else:
            reasoning = f"Neutral/unclear: Long({long_signal.quality_score:.2f}) vs Short({short_signal.quality_score:.2f})"

        return BidirectionalAnalysis(
            pair=market_data.pair,
            long_signal=long_signal,
            short_signal=short_signal,
            market_confidence=market_confidence,
            best_direction=best_direction,
            best_quality_score=best_quality_score,
            recommendation=recommendation,
            reasoning=reasoning
        )

    def _get_regime_analysis(self, market_data: MarketData) -> RegimeAnalysis:
        """Basic regime analysis without LLM API"""
        # Check cache locale
        cache_key = f"{market_data.pair}_{market_data.timestamp.minute}"
        if cache_key in self.local_cache:
            cached_time = self.local_cache[cache_key]['timestamp']
            if (datetime.now() - cached_time).total_seconds() < self.cache_ttl:
                return self.local_cache[cache_key]['analysis']

        # Basic Technical Analysis for Regime Detection
        if market_data.volatility_20d < 0.015:
            volatility_level = "LOW"
        elif market_data.volatility_20d < 0.03:
            volatility_level = "MEDIUM"
        else:
            volatility_level = "HIGH"

        # Determine regime based on multiple factors
        regime_score = 0.0

        # Price movement factor
        if abs(market_data.price_change_24h) < 1:
            regime_score += 0.3  # Calm
        elif abs(market_data.price_change_24h) > 3:
            regime_score += 0.1  # Too volatile
        else:
            regime_score += 0.2  # Moderate

        # RSI factor
        if 40 < market_data.rsi_14 < 60:
            regime_score += 0.3  # Neutral/calm
        elif market_data.rsi_14 > 70 or market_data.rsi_14 < 30:
            regime_score += 0.0  # Extreme conditions
        else:
            regime_score += 0.2  # Moderate

        # Volume factor
        if 0.8 < market_data.volume_ratio < 1.5:
            regime_score += 0.2  # Normal volume
        elif market_data.volume_ratio > 2:
            regime_score += 0.0  # Too high volume
        else:
            regime_score += 0.1  # Low volume

        # Determine regime
        if regime_score >= 0.7:
            regime = "CALM"
            confidence = 0.8
        elif regime_score >= 0.5:
            regime = "TRENDING"
            confidence = 0.6
        elif regime_score >= 0.3:
            regime = "SIDEWAYS"
            confidence = 0.5
        else:
            regime = "VOLATILE"
            confidence = 0.4

        # Calculate trend strength
        trend_strength = min(0.9, abs(market_data.price_change_24h) / 5)

        # Generate reasoning
        reasoning = f"Basic regime analysis: {regime} (volatility: {volatility_level}, price_change: {market_data.price_change_24h:.1f}%, RSI: {market_data.rsi_14:.0f})"

        # Create regime analysis
        analysis = RegimeAnalysis(
            pair=market_data.pair,
            regime=regime,
            confidence=confidence,
            volatility_level=volatility_level,
            trend_strength=trend_strength,
            reasoning=reasoning
        )

        # Cache locale
        self.local_cache[cache_key] = {
            'analysis': analysis,
            'timestamp': datetime.now()
        }

        return analysis

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Context7 Super Powers: Popola segnali LONG con parametri RIPRISTINATI

        SOLO SEGNALI FORTI - Thresholds robusti originali
        """
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        dataframe.loc[:, 'enter_tag'] = ''

        if len(dataframe) < 50:
            return dataframe

        # Get market data and regime analysis
        market_data = self._get_market_data(dataframe, metadata)
        if not market_data:
            return dataframe

        regime = self._get_regime_analysis(market_data)

        # Multi-timeframe consensus
        mtf_consensus = self._analyze_multi_timeframe_consensus(metadata['pair'])

        # Get bidirectional analysis
        bidirectional = self._get_bidirectional_analysis(dataframe, market_data, regime, mtf_consensus)

        # CONTEXT7 RESTORED: Solo segnali forti con thresholds originali
        if bidirectional.recommendation in ["STRONG_LONG", "LONG"]:
            # Threshold ORIGINALI robusti
            quality_threshold = 0.6  # Base threshold originale

            # Threshold più alto per mercati volatili
            if regime.regime == 'VOLATILE':
                quality_threshold = 0.7  # Originale

            # Threshold più basso solo per mercati trending con forte trend
            elif regime.regime == 'TRENDING' and regime.trend_strength > 0.7:
                quality_threshold = 0.5  # Originale

            # CONTEXT7 RESTORED: Volume confirmation RIGOROSO senza eccezioni
            if bidirectional.long_signal.quality_score >= quality_threshold:
                # Validazioni STRETTE - Nessun bypass per high quality
                if (bidirectional.long_signal.volume_confirmation and
                    bidirectional.long_signal.regime_compatibility > 0.3):

                    dataframe.iloc[-1, dataframe.columns.get_loc('enter_long')] = 1
                    dataframe.iloc[-1, dataframe.columns.get_loc('enter_tag')] = 'context7_restored_long'

                    self.logger.info(f"🟢 BUY {metadata['pair']} - CONTEXT7 RESTORED:")
                    self.logger.info(f"   Recommendation: {bidirectional.recommendation}")
                    self.logger.info(f"   Quality Score: {bidirectional.long_signal.quality_score:.3f} (threshold: {quality_threshold})")
                    self.logger.info(f"   Volume: {bidirectional.long_signal.volume_confirmation} (ratio: {market_data.volume_ratio:.2f})")
                    self.logger.info(f"   Regime: {regime.regime} (compatibility: {bidirectional.long_signal.regime_compatibility:.2f})")
                    self.logger.info(f"   MTF Consensus: LONG={mtf_consensus.get('LONG', 0):.2f}")
                    self.logger.info(f"   Entry Conditions: {', '.join(bidirectional.long_signal.entry_conditions)}")
                    self.logger.info(f"   Reasoning: {bidirectional.reasoning}")
                else:
                    self.logger.info(f"🚫 BUY {metadata['pair']} - Context7 Restored: Strict validation failed")
                    self.logger.info(f"   Volume OK: {bidirectional.long_signal.volume_confirmation}")
                    self.logger.info(f"   Regime Compatibility: {bidirectional.long_signal.regime_compatibility:.2f}")
            else:
                self.logger.info(f"🚫 BUY {metadata['pair']} - Context7 Restored: Quality below threshold")
                self.logger.info(f"   Quality: {bidirectional.long_signal.quality_score:.3f} < {quality_threshold}")
        else:
            self.logger.info(f"🚫 BUY {metadata['pair']} - Context7 Restored: No strong signal")
            self.logger.info(f"   Recommendation: {bidirectional.recommendation} (allowed: STRONG_LONG, LONG)")
            self.logger.info(f"   Long Quality: {bidirectional.long_signal.quality_score:.3f}")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Context7 Super Powers: Populate entry signals using bidirectional analysis
        """
        return self.populate_buy_trend(dataframe, metadata)

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Context7 Super Powers: Populate exit signals using parent implementation
        """
        return self.populate_sell_trend(dataframe, metadata)

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Context7 Super Powers: Popola segnali SHORT con parametri RIPRISTINATI
        """
        dataframe.loc[:, 'sell'] = 0

        if len(dataframe) < 50:
            return dataframe

        # Get market data and regime analysis
        market_data = self._get_market_data(dataframe, metadata)
        if not market_data:
            return dataframe

        regime = self._get_regime_analysis(market_data)

        # Multi-timeframe consensus
        mtf_consensus = self._analyze_multi_timeframe_consensus(metadata['pair'])

        # Get bidirectional analysis
        bidirectional = self._get_bidirectional_analysis(dataframe, market_data, regime, mtf_consensus)

        # CONTEXT7 RESTORED: Solo segnali forti con thresholds originali
        if bidirectional.recommendation in ["STRONG_SHORT", "SHORT"]:
            # Threshold ORIGINALI robusti
            quality_threshold = 0.6  # Base threshold originale

            # Threshold più alto per mercati volatili
            if regime.regime == 'VOLATILE':
                quality_threshold = 0.7  # Originale

            # Threshold più basso solo per mercati trending con forte trend
            elif regime.regime == 'TRENDING' and regime.trend_strength > 0.7:
                quality_threshold = 0.5  # Originale

            # CONTEXT7 RESTORED: Volume confirmation RIGOROSO senza eccezioni
            if bidirectional.short_signal.quality_score >= quality_threshold:
                # Validazioni STRETTE - Nessun bypass per high quality
                if (bidirectional.short_signal.volume_confirmation and
                    bidirectional.short_signal.regime_compatibility > 0.3):

                    dataframe.iloc[-1, dataframe.columns.get_loc('enter_short')] = 1
                    dataframe.iloc[-1, dataframe.columns.get_loc('enter_tag')] = 'context7_restored_short'

                    self.logger.info(f"🔴 SELL {metadata['pair']} - CONTEXT7 RESTORED:")
                    self.logger.info(f"   Recommendation: {bidirectional.recommendation}")
                    self.logger.info(f"   Quality Score: {bidirectional.short_signal.quality_score:.3f} (threshold: {quality_threshold})")
                    self.logger.info(f"   Volume: {bidirectional.short_signal.volume_confirmation} (ratio: {market_data.volume_ratio:.2f})")
                    self.logger.info(f"   Regime: {regime.regime} (compatibility: {bidirectional.short_signal.regime_compatibility:.2f})")
                    self.logger.info(f"   MTF Consensus: SHORT={mtf_consensus.get('SHORT', 0):.2f}")
                    self.logger.info(f"   Entry Conditions: {', '.join(bidirectional.short_signal.entry_conditions)}")
                    self.logger.info(f"   Reasoning: {bidirectional.reasoning}")
                else:
                    self.logger.info(f"🚫 SELL {metadata['pair']} - Context7 Restored: Strict validation failed")
                    self.logger.info(f"   Volume OK: {bidirectional.short_signal.volume_confirmation}")
                    self.logger.info(f"   Regime Compatibility: {bidirectional.short_signal.regime_compatibility:.2f}")
            else:
                self.logger.info(f"🚫 SELL {metadata['pair']} - Context7 Restored: Quality below threshold")
                self.logger.info(f"   Quality: {bidirectional.short_signal.quality_score:.3f} < {quality_threshold}")
        else:
            self.logger.info(f"🚫 SELL {metadata['pair']} - Context7 Restored: No strong signal")
            self.logger.info(f"   Recommendation: {bidirectional.recommendation} (allowed: STRONG_SHORT, SHORT)")
            self.logger.info(f"   Short Quality: {bidirectional.short_signal.quality_score:.3f}")

        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                           rate: float, time_in_force: str, current_time: datetime,
                           entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        Context7 Super Powers: Trade confirmation con Dynamic Stake Sizing

        Sistema avanzato che calcola la dimensione ottimale della posizione
        basandosi su rischio, volatilità e segnali multi-timeframe.
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) == 0:
                return False

            market_data = self._get_market_data(dataframe, {'pair': pair})
            if not market_data:
                return False

            regime = self._get_regime_analysis(market_data)

            # 1. Signal strength check
            signal_strength = 0.0
            if side == "long":
                bidirectional = self._get_bidirectional_analysis(
                    dataframe, market_data, regime, self._analyze_multi_timeframe_consensus(pair)
                )
                signal_strength = bidirectional.long_signal.quality_score
            else:
                bidirectional = self._get_bidirectional_analysis(
                    dataframe, market_data, regime, self._analyze_multi_timeframe_consensus(pair)
                )
                signal_strength = bidirectional.short_signal.quality_score

            # 2. Volume confirmation - RIGOROSO
            last = dataframe.iloc[-1]
            if last['volume_smart_ratio'] < 0.8:
                self.logger.info(f"🚫 Entry rejected for {pair}: Low volume ({last['volume_smart_ratio']:.2f})")
                return False

            # 3. Calculate risk metrics
            current_positions = {}  # In una implementazione reale, ottenere posizioni attive
            risk_metrics = self.risk_manager.calculate_risk_metrics(
                market_data, regime, current_positions
            )

            # 4. Calculate optimal stake sizing
            base_stake = 50.0  # USDT - configurable
            stake_decision = self.risk_manager.calculate_stake_sizing(
                pair=pair,
                signal_strength=signal_strength,
                risk_metrics=risk_metrics,
                base_stake=base_stake,
                max_leverage=self.max_leverage if hasattr(self, 'max_leverage') else 3.0
            )

            # 5. Log stake sizing decision
            self.logger.info(f"💰 STAKE SIZING {pair} ({side.upper()}):")
            self.logger.info(f"   Signal Strength: {signal_strength:.3f}")
            self.logger.info(f"   Risk Score: {risk_metrics.risk_score:.3f}")
            self.logger.info(f"   Recommended Stake: {stake_decision.recommended_stake:.2f} USDT")
            self.logger.info(f"   Risk-Adjusted Leverage: {stake_decision.risk_adjusted_leverage:.2f}x")
            self.logger.info(f"   Position Size: {stake_decision.position_size:.2f} USDT")
            self.logger.info(f"   Risk Level: {stake_decision.risk_level}")
            self.logger.info(f"   Reasoning: {stake_decision.reasoning}")

            # 6. Apply minimum stake validation
            if stake_decision.recommended_stake < self.risk_manager.min_stake_amount:
                self.logger.warning(f"⚠️ Stake too small for {pair}: {stake_decision.recommended_stake:.2f} < {self.risk_manager.min_stake_amount}")
                return False

            # 7. Risk level validation
            if stake_decision.risk_level == "EXTREME":
                self.logger.warning(f"🚫 Entry rejected for {pair}: EXTREME risk level")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in confirm_trade_entry: {e}")
            # Fail-safe: allow trade in case of confirmation error
            return True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, after_fill: bool, **kwargs) -> float | None:
        """
        Context7 Super Powers: Dynamic ATR-based stoploss with Ichimoku structure
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) == 0:
                return None

            market_data = self._get_market_data(dataframe, {'pair': pair})
            if not market_data:
                return None

            regime = self._get_regime_analysis(market_data)
            last = dataframe.iloc[-1]

            # Calculate ATR-based stoploss
            atr_multiplier = 3.0
            if regime.regime == 'VOLATILE':
                atr_multiplier = 3.5
            elif regime.regime == 'CALM':
                atr_multiplier = 2.5

            atr_value = last['atr'] if 'atr' in last else last['high'] - last['low']
            atr_stop_pct = (atr_multiplier * atr_value) / last['close']

            # Ichimoku structure-based stops
            ichimoku_stop_pct = None

            if trade.is_short:
                if last['close'] > last['kijun_sen']:
                    ichimoku_stop_pct = abs((last['kijun_sen'] - last['close']) / last['close'])
                elif last['close'] > last['senkou_span_a']:
                    ichimoku_stop_pct = abs((last['senkou_span_a'] - last['close']) / last['close'])
                else:
                    ichimoku_stop_pct = abs((last['tenkan_sen'] - last['close']) / last['close'])
            else:  # LONG
                if last['close'] < last['kijun_sen']:
                    ichimoku_stop_pct = abs((last['kijun_sen'] - last['close']) / last['close'])
                elif last['close'] < last['senkou_span_b']:
                    ichimoku_stop_pct = abs((last['senkou_span_b'] - last['close']) / last['close'])
                else:
                    ichimoku_stop_pct = abs((last['tenkan_sen'] - last['close']) / last['close'])

            # Use the tighter stop
            if ichimoku_stop_pct is not None:
                stoploss_pct = min(ichimoku_stop_pct, atr_stop_pct)
            else:
                stoploss_pct = atr_stop_pct

            # Apply regime adjustments
            if regime.regime == 'VOLATILE':
                stoploss_pct *= 1.2
            elif regime.regime == 'TRENDING':
                stoploss_pct *= 1.1

            # Safety caps
            stoploss_pct = max(0.005, min(stoploss_pct, 0.25))

            return stoploss_from_open(stoploss_pct, current_profit,
                                     is_short=trade.is_short,
                                     leverage=trade.leverage)

        except Exception as e:
            self.logger.error(f"Error in custom_stoploss for {pair}: {e}")
            return None

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> str:
        """
        Context7 Super Powers: Advanced exit signals
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) == 0:
                return None

            market_data = self._get_market_data(dataframe, {'pair': pair})
            if not market_data:
                return None

            regime = self._get_regime_analysis(market_data)
            last = dataframe.iloc[-1]
            prev = dataframe.iloc[-2] if len(dataframe) > 1 else last

            exit_reasons = []
            exit_strength = 0.0

            # Primary: Ichimoku structure breaks
            if trade.is_short:
                if last['close'] > last['senkou_span_a'] and prev['close'] <= last['senkou_span_a']:
                    exit_reasons.append("Price breaks above Cloud")
                    exit_strength += 0.4

                if last['close'] > last['kijun_sen'] and prev['close'] <= last['kijun_sen']:
                    exit_reasons.append("Price breaks above Kijun")
                    exit_strength += 0.3

                if prev['tenkan_sen'] <= prev['kijun_sen'] and last['tenkan_sen'] > last['kijun_sen']:
                    exit_reasons.append("Tenkan crosses above Kijun")
                    exit_strength += 0.2
            else:  # LONG
                if last['close'] < last['senkou_span_b'] and prev['close'] >= last['senkou_span_b']:
                    exit_reasons.append("Price breaks below Cloud")
                    exit_strength += 0.4

                if last['close'] < last['kijun_sen'] and prev['close'] >= last['kijun_sen']:
                    exit_reasons.append("Price breaks below Kijun")
                    exit_strength += 0.3

                if prev['tenkan_sen'] >= prev['kijun_sen'] and last['tenkan_sen'] < last['kijun_sen']:
                    exit_reasons.append("Tenkan crosses below Kijun")
                    exit_strength += 0.2

            # Secondary: MACD confirmation
            if trade.is_short:
                if prev['macd'] <= prev['macdsignal'] and last['macd'] > last['macdsignal']:
                    exit_reasons.append("MACD bullish crossover")
                    exit_strength += 0.15
            else:
                if prev['macd'] >= prev['macdsignal'] and last['macd'] < last['macdsignal']:
                    exit_reasons.append("MACD bearish crossover")
                    exit_strength += 0.15

            # Tertiary: CMF volume confirmation
            if 'cmf' in last:
                if trade.is_short and last['cmf'] > 0.1:
                    exit_reasons.append(f"CMF buying pressure ({last['cmf']:.3f})")
                    exit_strength += 0.1
                elif not trade.is_short and last['cmf'] < -0.1:
                    exit_reasons.append(f"CMF selling pressure ({last['cmf']:.3f})")
                    exit_strength += 0.1

            # Quaternary: Regime-aware take profit levels
            tp_threshold = 0.0
            if regime.regime == 'TRENDING':
                tp_threshold = 0.20
            elif regime.regime == 'VOLATILE':
                tp_threshold = 0.15
            elif regime.regime == 'CALM':
                tp_threshold = 0.12
            else:  # SIDEWAYS
                tp_threshold = 0.08

            if current_profit > tp_threshold:
                if exit_strength > 0.3:
                    exit_reasons.append(f"Take profit at {current_profit:.1%} with confirmation")
                    exit_strength += 0.2
                elif exit_strength > 0.1:
                    exit_reasons.append(f"Take profit at {current_profit:.1%} with weak signals")
                    exit_strength += 0.1

            # Final exit decision
            if exit_strength >= 0.6:
                self.logger.info(f"🚨 STRONG EXIT {pair}: {' | '.join(exit_reasons)} (strength: {exit_strength:.2f})")
                return "strong_exit_signal"
            elif exit_strength >= 0.4:
                self.logger.info(f"⚠️ MEDIUM EXIT {pair}: {' | '.join(exit_reasons)} (strength: {exit_strength:.2f})")
                return "medium_exit_signal"
            elif exit_strength >= 0.2:
                self.logger.info(f"💡 WEAK EXIT {pair}: {' | '.join(exit_reasons)} (strength: {exit_strength:.2f})")
                return "weak_exit_signal"

            return None

        except Exception as e:
            self.logger.error(f"Error in custom_exit: {e}")
            return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        """
        Context7 Super Powers: Leverage basato su stake sizing decision

        Utilizza il Risk Manager per calcolare il leverage ottimale
        """
        try:
            # Get stake sizing decision from confirm_trade_entry logic
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) == 0:
                return 1.0

            market_data = self._get_market_data(dataframe, {'pair': pair})
            if not market_data:
                return 1.0

            regime = self._get_regime_analysis(market_data)
            current_positions = {}  # Get from actual trades

            # Calculate risk metrics
            risk_metrics = self.risk_manager.calculate_risk_metrics(
                market_data, regime, current_positions
            )

            # Get signal strength
            signal_strength = 0.6  # Default
            if entry_tag:
                if 'restored' in str(entry_tag):
                    signal_strength = 0.7

            # Calculate stake sizing decision
            base_stake = 50.0
            stake_decision = self.risk_manager.calculate_stake_sizing(
                pair=pair,
                signal_strength=signal_strength,
                risk_metrics=risk_metrics,
                base_stake=base_stake,
                max_leverage=max_leverage
            )

            # Return the risk-adjusted leverage
            final_leverage = min(stake_decision.risk_adjusted_leverage, max_leverage, 5.0)

            self.logger.info(f"⚙️ LEVERAGE {pair} ({side.upper()}): {final_leverage:.2f}x (risk_score: {risk_metrics.risk_score:.2f})")

            return final_leverage

        except Exception as e:
            self.logger.error(f"Error in leverage calculation for {pair}: {e}")
            return 1.0

# Export strategy
__strategy__ = IchimokuLLMRestoredV1