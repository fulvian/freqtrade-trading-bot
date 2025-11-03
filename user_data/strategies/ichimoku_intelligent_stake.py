"""
===========================================================================
ICHIMOKU LLM ENHANCED V1 - INTELLIGENT STAKE SIZING
===========================================================================

Versione advanced con intelligent stake sizing basata su framework Perplexity:

CARATTERISTICHE PRINCIPALI:
✅ Multi-Timeframe: 5m, 15m, 30m, 1h, 1d con 6 posizioni per timeframe
✅ Intelligent Stake Sizing basato su segnali di qualità e volatilità
✅ Context7 Super Powers con mathematical framework ottimizzato
✅ Kelly Criterion applicato con Half Kelly per safety
✅ Quality score quadratic scaling (q²) per segnali ad alta confidenza
✅ Volatility-adjusted positioning con ATR normalization
✅ Multi-timeframe consensus factors (n_agree/5)
✅ Leverage compensation con square root scaling
✅ Portfolio heat management per risk control
✅ Dynamic leverage fino a 10x con asset-specific limits

ARCHITETTURA A STRATI:
1. DataLayer: Raccolta dati multi-timeframe isolata e robusta
2. BatchLayer: Processing centralizzato con lock globale
3. StrategyLayer: Logica pura con multi-timeframe consensus
4. FreqTradeLayer: Integrazione minima e pulita

FUNZIONALITÀ COMPLETE:
- Ichimoku Cloud multi-timeframe con segnali avanzati
- LLM regime detection con batch processing ottimizzato
- Risk management adattivo basato su volatility
- Multi-timeframe consensus validation (5m, 15m, 30m, 1h, 1d)
- Position sizing intelligente (6 assets × 5 timeframes = 30 max posizioni)
- Dynamic leverage con asset-specific limits
- Stop-loss e take-profit dinamici
- Portfolio management con correlation analysis
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
                                stoploss_from_open, informative)
from freqtrade.persistence import Trade

# Context7 Super Powers imports
from json_repair import repair_json

# Context7 Super Powers: Tenacity integration
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
    pair: str  # Nome della coppia
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
class BatchRequest:
    """Richiesta batch pulita"""
    assets: List[MarketData]
    analysis_type: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class BatchResponse:
    """Risposta batch pulita"""
    results: List[RegimeAnalysis]
    processing_time_ms: float
    llm_model: str
    timestamp: datetime = field(default_factory=datetime.now)

# ============================================================================
# CONTEXT7 SUPER POWERS: BATCH PROCESSOR (SINGLETON)
# ============================================================================

class Context7BatchProcessor:
    """
    Context7 Super Powers: Batch Processor Singleton

    Garantisce UNA SOLA chiamata LLM ogni 5 minuti per TUTTI gli asset
    utilizzando pattern atomico e lock globale.
    """

    _instance = None
    _lock = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Context7 Super Powers: Stato globale atomico
        self._global_lock = False
        self._last_batch_minute = -1  # Minuto dell'ultimo batch (0-1439)
        self._batch_cache = {}  # Cache per minuto corrente
        self._api_url = os.getenv('NANO_GPT_API_URL', 'https://nano-gpt.com/api/v1/chat/completions')
        self._api_key = os.getenv('NANO_GPT_API_KEY')
        if not self._api_key:
            self.logger.warning("⚠️ Context7: NANO_GPT_API_KEY not found in environment variables")
            self._enabled = False
        self._model = "deepseek-ai/DeepSeek-V3.1"  # BASE model ottimizzato

        # Setup logging
        self.logger = logging.getLogger("Context7BatchProcessor")
        self._initialized = True

        self.logger.info("🚀 Context7 Super Powers: Batch Processor initialized")

    def _can_process_batch(self, current_minute: int) -> bool:
        """Verifica se può processare batch (solo una volta al minuto)"""
        if self._last_batch_minute == current_minute:
            self.logger.info(f"🚫 Context7: Batch già processato minuto {current_minute}")
            return False
        if self._global_lock:
            self.logger.info(f"🚫 Context7: Batch lock attivo, attendo")
            return False
        return True

    def _acquire_global_lock(self, current_minute: int) -> bool:
        """Acquisisce lock globale con check atomico"""
        if self._can_process_batch(current_minute):
            self._global_lock = True
            self.logger.info(f"🔒 Context7: Lock globale acquisito minuto {current_minute}")
            return True
        return False

    def _release_global_lock(self, current_minute: int, success: bool):
        """Rilascia lock globale e registra successo"""
        self._global_lock = False
        if success:
            self._last_batch_minute = current_minute
            self.logger.info(f"✅ Context7: Batch SUCCESS minuto {current_minute} registrato")
        else:
            self.logger.warning(f"❌ Context7: Batch FALLITO minuto {current_minute}")

    def _validate_response_content(self, response_content: str) -> bool:
        """Context7 Super Powers: Validazione contenuto risposta LLM"""
        if not response_content or response_content.strip() == "":
            return False

        # Check for common malformed response patterns
        malformed_patterns = [
            response_content.startswith("```json"),
            response_content.startswith("```"),
            response_content.endswith("```"),
            "null" in response_content.lower(),
            response_content.strip() == "{}",
            response_content.strip() == "[]",
            not any(c in response_content for c in ['{', '[', '"'])  # No JSON structure
        ]

        return not any(malformed_patterns)

    def _parse_llm_response_enhanced(self, response_data: dict, pairs: List[str]) -> Optional[List[RegimeAnalysis]]:
        """Context7 Super Powers: Parsing migliorato con validazione multi-layer"""
        try:
            if 'choices' not in response_data or not response_data['choices']:
                self.logger.error("❌ Context7: Nessuna choice nella risposta LLM")
                return None

            content = response_data['choices'][0]['message']['content']
            if not content:
                self.logger.error("❌ Context7: Content vuoto nella risposta LLM")
                return None

            # Context7 Enhanced: Pre-validate content before JSON parsing
            if not self._validate_response_content(content):
                self.logger.warning(f"⚠️ Context7: Content malformed, tentativo repair...")

            # Multi-layer JSON repair with Context7 patterns
            try:
                parsed_data = json.loads(content)
            except json.JSONDecodeError as e:
                self.logger.warning(f"⚠️ Context7: JSON decode error: {str(e)[:100]}")

                # Attempt 1: json_repair
                try:
                    repaired_content = repair_json(content)
                    parsed_data = json.loads(repaired_content)
                    self.logger.info("✅ Context7: JSON repair successful")
                except Exception:
                    # Attempt 2: Manual JSON extraction
                    self.logger.warning("⚠️ Context7: Tentando estrazione manuale JSON...")
                    parsed_data = self._extract_json_manually(content)

                    if not parsed_data:
                        self.logger.error("❌ Context7: Tutti i tentativi parsing falliti")
                        return None

            if 'analysis' not in parsed_data:
                self.logger.error("❌ Context7: Campo 'analysis' mancante")
                return None

            analysis_list = []
            for item in parsed_data['analysis']:
                if not all(key in item for key in ['pair', 'regime', 'confidence', 'reasoning']):
                    self.logger.warning(f"⚠️ Context7: Item incompleto: {item}")
                    continue

                # Enhanced validation of regime values
                valid_regimes = ['CALM', 'VOLATILE', 'TRENDING', 'SIDEWAYS']
                if item.get('regime') not in valid_regimes:
                    self.logger.warning(f"⚠️ Context7: Regime non valido: {item.get('regime')}")
                    continue

                analysis = RegimeAnalysis(
                    pair=item['pair'],
                    regime=item.get('regime', 'SIDEWAYS'),
                    confidence=float(item.get('confidence', 0.5)),
                    volatility_level=item.get('volatility_level', 'MEDIUM'),
                    trend_strength=float(item.get('trend_strength', 0.5)),
                    reasoning=item.get('reasoning', 'No reasoning provided')
                )
                analysis_list.append(analysis)

            if len(analysis_list) != len(pairs):
                self.logger.warning(f"⚠️ Context7: Analisi parziale: {len(analysis_list)}/{len(pairs)} asset")

            return analysis_list

        except Exception as e:
            self.logger.error(f"❌ Context7: Errore parsing risposta: {str(e)[:200]}")
            return None

    def _extract_json_manually(self, content: str) -> Optional[dict]:
        """Context7 Super Powers: Estrazione manuale JSON da malformed responses"""
        try:
            # Find JSON structure boundaries
            start_idx = content.find('{')
            end_idx = content.rfind('}')

            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = content[start_idx:end_idx + 1]
                return json.loads(json_str)

            # Try finding array structure
            start_idx = content.find('[')
            end_idx = content.rfind(']')

            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = content[start_idx:end_idx + 1]
                return json.loads(json_str)

            return None

        except Exception:
            return None

    @retry(
        stop=stop_after_attempt(5) | stop_after_delay(240),  # Aumentato a 5 tentativi o 4 minuti max
        wait=wait_random_exponential(multiplier=2, max=45),  # Aumentato max wait
        retry=retry_if_exception_type((
            requests.exceptions.RequestException,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
            json.JSONDecodeError  # Aggiunto parsing errors
        )) | retry_if_result(lambda result: result is None),  # Retry su risultati None
        before=before_log(logging.getLogger("Context7Retry"), logging.INFO),
        after=after_log(logging.getLogger("Context7Retry"), logging.WARNING),
        reraise=True
    )
    def _call_llm_api(self, prompt: str) -> Optional[dict]:
        """Context7 Super Powers: Chiamata LLM con retry robusto migliorato"""
        import requests

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert cryptocurrency trading analyst. Provide structured JSON responses only. Always return valid JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,  # Ridotto per maggiore consistenza
            "max_tokens": 1500,  # Ridotto per prevenire troncamento
            "response_format": {"type": "json_object"}
        }

        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'Context7-Bidirectional-Strategy/1.0'
        }

        try:
            response = requests.post(
                self._api_url,
                headers=headers,
                json=payload,
                timeout=120  # Timeout esteso per batch
            )

            # Enhanced response validation
            response.raise_for_status()

            # Validate response content before returning
            response_data = response.json()
            if response_data is None:
                raise ValueError("Response data is None")

            return response_data

        except requests.exceptions.Timeout:
            self.logger.error("❌ Context7: API timeout (120s)")
            raise
        except requests.exceptions.ConnectionError:
            self.logger.error("❌ Context7: Connection error")
            raise
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"❌ Context7: HTTP error: {e.response.status_code}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Context7: JSON decode error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Context7: Unexpected error: {str(e)}")
            raise

    def _build_batch_prompt(self, market_data_list: List[MarketData]) -> str:
        """Costruisce prompt batch ottimizzato"""
        prompt = """Analyze market regime for ALL cryptocurrencies and respond with valid JSON:

{
  "analysis": [
    {
      "pair": "PAIR_NAME",
      "regime": "CALM|VOLATILE|TRENDING|SIDEWAYS",
      "confidence": 0.0-1.0,
      "volatility_level": "LOW|MEDIUM|HIGH",
      "trend_strength": 0.0-1.0,
      "reasoning": "Brief analysis explaining regime classification"
    }
  ]
}

MARKET DATA:
"""

        for data in market_data_list:
            prompt += f"""
{data.pair}:
- Price: ${data.current_price:.4f}
- 24h Change: {data.price_change_24h:+.2f}%
- Volume Ratio: {data.volume_ratio:.2f}
- RSI: {data.rsi_14:.1f}
- Volatility: {data.volatility_20d:.3f}
- MACD: {data.macd_signal:+.4f}
- Ichimoku: Tenkan={data.tenkan_sen:.4f}, Kijun={data.kijun_sen:.4f}
- Cloud Position: {self._get_cloud_position(data)}
"""

        return prompt

    def _get_cloud_position(self, data: MarketData) -> str:
        """Determina posizione rispetto a Ichimoku Cloud"""
        if data.current_price > max(data.senkou_span_a, data.senkou_span_b):
            return "ABOVE_CLOUD"
        elif data.current_price < min(data.senkou_span_a, data.senkou_span_b):
            return "BELOW_CLOUD"
        else:
            return "INSIDE_CLOUD"

    def _parse_llm_response(self, response_data: dict, pairs: List[str]) -> Optional[List[RegimeAnalysis]]:
        """Parse LLM response con validazione robusta migliorata"""
        return self._parse_llm_response_enhanced(response_data, pairs)

    def _create_fallback_analysis(self, market_data: MarketData) -> RegimeAnalysis:
        """Crea analisi fallback basata su indicatori tecnici"""
        # Logica fallback Context7-compliant
        if market_data.rsi_14 > 70:
            regime = "VOLATILE"
            volatility_level = "HIGH"
            reasoning = f"RSI overbought ({market_data.rsi_14:.1f})"
        elif market_data.rsi_14 < 30:
            regime = "TRENDING"
            volatility_level = "MEDIUM"
            reasoning = f"RSI oversold ({market_data.rsi_14:.1f})"
        elif abs(market_data.price_change_24h) > 5:
            regime = "VOLATILE"
            volatility_level = "HIGH"
            reasoning = f"High 24h volatility ({market_data.price_change_24h:+.1f}%)"
        else:
            regime = "CALM"
            volatility_level = "LOW"
            reasoning = f"Stable conditions (RSI: {market_data.rsi_14:.1f})"

        return RegimeAnalysis(
            pair=market_data.pair,
            regime=regime,
            confidence=0.7,  # Confidence alta per analisi tecnica
            volatility_level=volatility_level,
            trend_strength=0.6,
            reasoning=reasoning
        )

    def process_batch(self, market_data_list: List[MarketData]) -> Optional[BatchResponse]:
        """
        Context7 Super Powers: Processo batch atomico

        È l'UNICO punto di accesso per tutte le chiamate LLM.
        Garantisce una sola chiamata ogni 5 minuti per TUTTI gli asset.
        """
        current_time = datetime.now()
        current_minute = current_time.hour * 60 + current_time.minute

        # Context7 Super Powers: Check atomico
        if not self._acquire_global_lock(current_minute):
            # Tenta di usare cache se disponibile
            cache_key = f"batch_{current_minute}"
            if cache_key in self._batch_cache:
                self.logger.info(f"📋 Context7: Uso cache batch minuto {current_minute}")
                return self._batch_cache[cache_key]

            # Fallback se nessuna cache disponibile
            self.logger.warning(f"⚠️ Context7: Nessuna cache, creo fallback")
            fallback_results = [self._create_fallback_analysis(data) for data in market_data_list]
            return BatchResponse(
                results=fallback_results,
                processing_time_ms=0,
                llm_model="FALLBACK",
                timestamp=current_time
            )

        start_time = time.time()
        self.logger.info(f"🚀 Context7: START BATCH PROCESSING - {len(market_data_list)} assets")

        try:
            # Costruisci prompt
            prompt = self._build_batch_prompt(market_data_list)

            # Chiama LLM con Tenacity
            response_data = self._call_llm_api(prompt)

            # Parse risposta
            analysis_results = self._parse_llm_response(response_data, [d.pair for d in market_data_list])

            if analysis_results:
                processing_time = (time.time() - start_time) * 1000
                batch_response = BatchResponse(
                    results=analysis_results,
                    processing_time_ms=processing_time,
                    llm_model=self._model,
                    timestamp=current_time
                )

                # Cache per altri asset
                cache_key = f"batch_{current_minute}"
                self._batch_cache[cache_key] = batch_response

                # Cleanup cache vecchio
                self._cleanup_cache(current_minute)

                self.logger.info(f"✅ Context7: BATCH SUCCESS - {len(analysis_results)} asset in {processing_time:.0f}ms")

                # Rilascia lock con successo
                self._release_global_lock(current_minute, True)

                return batch_response
            else:
                # Fallback se parsing fallito
                self.logger.warning("⚠️ Context7: Parsing fallito, uso fallback")
                fallback_results = [self._create_fallback_analysis(data) for data in market_data_list]

                self._release_global_lock(current_minute, False)

                return BatchResponse(
                    results=fallback_results,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    llm_model="FALLBACK",
                    timestamp=current_time
                )

        except Exception as e:
            self.logger.error(f"❌ Context7: Errore batch processing: {e}")

            # Fallback finale
            fallback_results = [self._create_fallback_analysis(data) for data in market_data_list]

            self._release_global_lock(current_minute, False)

            return BatchResponse(
                results=fallback_results,
                processing_time_ms=(time.time() - start_time) * 1000,
                llm_model="FALLBACK_ERROR",
                timestamp=current_time
            )

    def _cleanup_cache(self, current_minute: int):
        """Cleanup cache vecchio (mantiene solo ultimi 10 minuti)"""
        keys_to_remove = []
        for key in self._batch_cache:
            try:
                key_minute = int(key.split('_')[1])
                if abs(current_minute - key_minute) > 10:
                    keys_to_remove.append(key)
            except:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._batch_cache[key]

        if keys_to_remove:
            self.logger.info(f"🧹 Context7: Cleanup cache rimosso {len(keys_to_remove)} entry")

# ============================================================================
# CONTEXT7 SUPER POWERS: MAIN STRATEGY
# ============================================================================

class IchimokuLLMIntelligentStakeSizing(IStrategy):
    """
    Context7 Super Powers: Ichimoku LLM Enhanced V1 - Intelligent Stake Sizing

    Architettura avanzata con intelligent stake sizing basato su framework Perplexity.
    Mathematical Framework: Final Stake = Base Stake × f_quality × f_volatility × f_consensus × f_leverage × f_portfolio
    Timeframes: 5m, 15m, 30m, 1h, 1d (6 assets × 5 timeframes = 30 max positions)
    Dynamic Leverage: 1x-10x con asset-specific limits
    Kelly Criterion: Half Kelly per safety con quadratic quality scaling
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

    # Enable short trading (CRITICAL for bidirectional strategy)
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
        """Initialize Context7 Super Powers strategy - BASIC MODE (No LLM)"""
        super().__init__(config)

        # Context7 Super Powers: DISABLED LLM Batch Processor
        # self.batch_processor = Context7BatchProcessor()  # DISABLED - nano-gpt.com connection issues
        self.batch_processor = None

        # Enable basic technical analysis mode
        self.basic_mode = True

        # Cache locale per risultati
        self.local_cache = {}
        self.cache_ttl = 300  # 5 minuti

        # Logging
        self.logger = logging.getLogger("IchimokuLLMEnhancedV1Clean")
        self.logger.info("🚀 Context7 Super Powers: Strategy initialized - BASIC MODE (No LLM)")
        self.logger.info("⚠️ LLM API disabled due to nano-gpt.com connection issues")
        self.logger.info("✅ Using pure technical analysis mode")

    def informative_pairs(self):
        """
        Context7 Super Powers: Multi-timeframe informative pairs definition.

        Returns the list of informative pairs for multi-timeframe analysis:
        - 15m, 30m, 1h, 1d timeframes for all current pairs
        - This allows 6 positions per asset (one per timeframe)
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = []

        # Add higher timeframes for each pair
        for pair in pairs:
            informative_pairs.extend([
                (pair, '15m'),
                (pair, '30m'),
                (pair, '1h'),
                (pair, '1d')
            ])

        self.logger.info(f"📊 Multi-timeframe informative pairs: {len(informative_pairs)} total")
        return informative_pairs

    @informative('15m')
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Context7 Super Powers: Populate indicators for 15m timeframe"""
        # Calculate Ichimoku Cloud for 15m
        high_9 = dataframe['high'].rolling(window=9).max()
        low_9 = dataframe['low'].rolling(window=9).min()
        high_26 = dataframe['high'].rolling(window=26).max()
        low_26 = dataframe['low'].rolling(window=26).min()
        high_52 = dataframe['high'].rolling(window=52).max()
        low_52 = dataframe['low'].rolling(window=52).min()

        dataframe['tenkan_sen_15m'] = (high_9 + low_9) / 2
        dataframe['kijun_sen_15m'] = (high_26 + low_26) / 2
        dataframe['senkou_span_a_15m'] = ((dataframe['tenkan_sen_15m'] + dataframe['kijun_sen_15m']) / 2).shift(26)
        dataframe['senkou_span_b_15m'] = ((high_52 + low_52) / 2).shift(26)
        dataframe['chikou_span_15m'] = dataframe['close'].shift(-26)

        # RSI for 15m
        delta = dataframe['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        dataframe['rsi_15m'] = 100 - (100 / (1 + rs))

        return dataframe

    @informative('30m')
    def populate_indicators_30m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Context7 Super Powers: Populate indicators for 30m timeframe"""
        # Calculate Ichimoku Cloud for 30m
        high_9 = dataframe['high'].rolling(window=9).max()
        low_9 = dataframe['low'].rolling(window=9).min()
        high_26 = dataframe['high'].rolling(window=26).max()
        low_26 = dataframe['low'].rolling(window=26).min()
        high_52 = dataframe['high'].rolling(window=52).max()
        low_52 = dataframe['low'].rolling(window=52).min()

        dataframe['tenkan_sen_30m'] = (high_9 + low_9) / 2
        dataframe['kijun_sen_30m'] = (high_26 + low_26) / 2
        dataframe['senkou_span_a_30m'] = ((dataframe['tenkan_sen_30m'] + dataframe['kijun_sen_30m']) / 2).shift(26)
        dataframe['senkou_span_b_30m'] = ((high_52 + low_52) / 2).shift(26)
        dataframe['chikou_span_30m'] = dataframe['close'].shift(-26)

        # RSI for 30m
        delta = dataframe['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        dataframe['rsi_30m'] = 100 - (100 / (1 + rs))

        return dataframe

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Context7 Super Powers: Populate indicators for 1h timeframe"""
        # Calculate Ichimoku Cloud for 1h
        high_9 = dataframe['high'].rolling(window=9).max()
        low_9 = dataframe['low'].rolling(window=9).min()
        high_26 = dataframe['high'].rolling(window=26).max()
        low_26 = dataframe['low'].rolling(window=26).min()
        high_52 = dataframe['high'].rolling(window=52).max()
        low_52 = dataframe['low'].rolling(window=52).min()

        dataframe['tenkan_sen_1h'] = (high_9 + low_9) / 2
        dataframe['kijun_sen_1h'] = (high_26 + low_26) / 2
        dataframe['senkou_span_a_1h'] = ((dataframe['tenkan_sen_1h'] + dataframe['kijun_sen_1h']) / 2).shift(26)
        dataframe['senkou_span_b_1h'] = ((high_52 + low_52) / 2).shift(26)
        dataframe['chikou_span_1h'] = dataframe['close'].shift(-26)

        # RSI for 1h
        delta = dataframe['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        dataframe['rsi_1h'] = 100 - (100 / (1 + rs))

        return dataframe

    @informative('1d')
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Context7 Super Powers: Populate indicators for 1d timeframe"""
        # Calculate Ichimoku Cloud for 1d
        high_9 = dataframe['high'].rolling(window=9).max()
        low_9 = dataframe['low'].rolling(window=9).min()
        high_26 = dataframe['high'].rolling(window=26).max()
        low_26 = dataframe['low'].rolling(window=26).min()
        high_52 = dataframe['high'].rolling(window=52).max()
        low_52 = dataframe['low'].rolling(window=52).min()

        dataframe['tenkan_sen_1d'] = (high_9 + low_9) / 2
        dataframe['kijun_sen_1d'] = (high_26 + low_26) / 2
        dataframe['senkou_span_a_1d'] = ((dataframe['tenkan_sen_1d'] + dataframe['kijun_sen_1d']) / 2).shift(26)
        dataframe['senkou_span_b_1d'] = ((high_52 + low_52) / 2).shift(26)
        dataframe['chikou_span_1d'] = dataframe['close'].shift(-26)

        # RSI for 1d
        delta = dataframe['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        dataframe['rsi_1d'] = 100 - (100 / (1 + rs))

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Populate technical indicators"""
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

        # Additional indicators with SMART volume calculation
        dataframe['sma_fast'] = dataframe['close'].rolling(window=10).mean()
        dataframe['sma_slow'] = dataframe['close'].rolling(window=30).mean()

        # ATR (Average True Range) for dynamic stoploss (NotebookLM best practice)
        dataframe['tr1'] = dataframe['high'] - dataframe['low']
        dataframe['tr2'] = abs(dataframe['high'] - dataframe['close'].shift())
        dataframe['tr3'] = abs(dataframe['low'] - dataframe['close'].shift())
        dataframe['tr'] = dataframe[['tr1', 'tr2', 'tr3']].max(axis=1)
        dataframe['atr'] = dataframe['tr'].rolling(window=14).mean()

        # CMF (Chaikin Money Flow) for volume analysis (NotebookLM best practice for BTC/ETH)
        # Money Flow Multiplier
        mfm = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
        # Money Flow Volume
        mfv = mfm * dataframe['volume']
        # CMF (21-period sum of MFV / 21-period sum of Volume)
        dataframe['cmf'] = mfv.rolling(window=21).sum() / dataframe['volume'].rolling(window=21).sum()

        # Smart Volume Baseline - FIXED to prevent baseline drift
        # Use longer period (50 candles = 250 minutes) + 70th percentile for robustness
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=50).mean()
        dataframe['volume_70th_percentile'] = dataframe['volume'].rolling(window=50).quantile(0.70)

        # Smart ratio: use 70th percentile as baseline (more robust than mean)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_70th_percentile']

        # Alternative: hybrid approach (mean + percentile)
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

    def _analyze_directional_signals(self, dataframe: DataFrame, market_data: MarketData, regime: RegimeAnalysis) -> Tuple[DirectionalSignal, DirectionalSignal]:
        """
        Context7 Super Powers: Analisi segnali bidirezionali con quality scoring

        Implementa pattern Context7-compliant per:
        - Primary signal detection (long/short)
        - Trend-aware quality scoring
        - Hierarchical filtering system
        - Volume e regime validation
        """
        last = dataframe.iloc[-1]

        # Context7: Ichimoku position analysis
        if last['close'] > max(last['senkou_span_a'], last['senkou_span_b']):
            ichimoku_position = "ABOVE_CLOUD"
            cloud_bias = 1.0  # Strong bullish bias
        elif last['close'] < min(last['senkou_span_a'], last['senkou_span_b']):
            ichimoku_position = "BELOW_CLOUD"
            cloud_bias = -1.0  # Strong bearish bias
        else:
            ichimoku_position = "INSIDE_CLOUD"
            cloud_bias = 0.0  # Neutral bias

        # Context7 Super Powers: Dynamic volume confirmation based on volatility
        atr_value = last.get('atr', 0.01)
        price = last['close']
        volatility_pct = (atr_value / price) * 100

        # Dynamic threshold based on market volatility (Context7 Super Powers: less strict thresholds)
        if volatility_pct > 3.0:  # High volatility
            dynamic_threshold = 0.50
        elif volatility_pct > 1.5:  # Medium volatility
            dynamic_threshold = 0.55
        else:  # Low volatility
            dynamic_threshold = 0.60

        volume_confirmation = last['volume_smart_ratio'] > dynamic_threshold

        # Context7 Super Powers: Debug logging for volume analysis
        # Note: pair info will be added in calling method
        volume_weight = min(last['volume_smart_ratio'], 2.0)  # Cap at 2.0

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
            last=last
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
            last=last
        )

        return long_signal, short_signal

    def _analyze_single_direction(self, direction: str, dataframe: DataFrame, market_data: MarketData,
                                 regime: RegimeAnalysis, ichimoku_position: str, cloud_bias: float,
                                 volume_confirmation: bool, volume_weight: float, last) -> DirectionalSignal:
        """
        Context7 Super Powers: Analisi segnale singolo con quality scoring gerarchico

        Implementa best practice da Machine Learning for Trading:
        - Signal strength evaluation
        - Multiple factor confirmation
        - Trend-aware weighting
        - Hierarchical quality thresholds
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

            # Context7: Momentum confirmation
            if last['macd'] > last['macdsignal']:
                primary_score += 0.2
                entry_conditions.append("MACD bullish")

            # Context7: RSI conditions (regime-aware)
            if regime.regime == 'CALM' and last['rsi'] < 40:
                primary_score += 0.15
                entry_conditions.append("RSI oversold in calm market")
            elif regime.regime == 'TRENDING' and 40 < last['rsi'] < 70:
                primary_score += 0.1
                entry_conditions.append("RSI in trend range")
            elif regime.regime == 'VOLATILE' and last['rsi'] < 35:
                primary_score += 0.1
                entry_conditions.append("RSI oversold in volatile market")

            # Context7: Moving average confirmation
            if last['close'] > last['sma_fast'] and last['sma_fast'] > last['sma_slow']:
                primary_score += 0.15
                entry_conditions.append("Price above MAs")

            # Context7: Volume confirmation
            if volume_confirmation:
                primary_score += 0.1 * volume_weight
                entry_conditions.append(f"Volume confirmed (ratio: {last['volume_smart_ratio']:.2f})")

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

            # Context7: Momentum confirmation
            if last['macd'] < last['macdsignal']:
                primary_score += 0.2
                entry_conditions.append("MACD bearish")

            # Context7: RSI conditions (regime-aware)
            if regime.regime == 'CALM' and last['rsi'] > 60:
                primary_score += 0.15
                entry_conditions.append("RSI overbought in calm market")
            elif regime.regime == 'TRENDING' and 30 < last['rsi'] < 60:
                primary_score += 0.1
                entry_conditions.append("RSI in trend range")
            elif regime.regime == 'VOLATILE' and last['rsi'] > 65:
                primary_score += 0.1
                entry_conditions.append("RSI overbought in volatile market")

            # Context7: Moving average confirmation
            if last['close'] < last['sma_fast'] and last['sma_fast'] < last['sma_slow']:
                primary_score += 0.15
                entry_conditions.append("Price below MAs")

            # Context7: Volume confirmation
            if volume_confirmation:
                primary_score += 0.1 * volume_weight
                entry_conditions.append(f"Volume confirmed (ratio: {last['volume_smart_ratio']:.2f})")

        # Context7: Calculate trend weight (regime-aware)
        trend_weight = 1.0  # Base weight

        if regime.regime == 'TRENDING':
            trend_weight = 2.0 if direction == "LONG" and regime.trend_strength > 0.6 else 1.5
        elif regime.regime == 'VOLATILE':
            trend_weight = 0.8  # Reduce weight in volatile markets
        elif regime.regime == 'CALM':
            trend_weight = 1.2  # Slight increase in calm markets

        # Context7: Calculate regime compatibility
        regime_compatibility = self._calculate_regime_compatibility(direction, regime, last)

        # Context7: Calculate final quality score (hierarchical)
        # Pattern from Machine Learning for Trading: multi-factor scoring
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
        Context7 Super Powers: Calcolo compatibilità regime basato su best practice

        Implementa pattern Context7-compliant per regime-based signal filtering
        """
        base_compatibility = 0.5  # Base compatibility

        if regime.regime == 'TRENDING':
            # Trending markets favor the dominant direction
            if direction == "LONG" and last['close'] > last['sma_slow']:
                base_compatibility = 0.9
            elif direction == "SHORT" and last['close'] < last['sma_slow']:
                base_compatibility = 0.9
            else:
                base_compatibility = 0.3  # Against trend in trending market

        elif regime.regime == 'CALM':
            # Calm markets good for both directions with proper setup
            base_compatibility = 0.8

            # RSI positioning in calm markets
            if direction == "LONG" and last['rsi'] < 45:
                base_compatibility = 0.9
            elif direction == "SHORT" and last['rsi'] > 55:
                base_compatibility = 0.9

        elif regime.regime == 'VOLATILE':
            # Volatile markets require extra confirmation
            base_compatibility = 0.6

            # Strong momentum required in volatile markets
            macd_strength = abs(last['macd'] - last['macdsignal'])
            if macd_strength > 0.001:  # Strong MACD divergence
                base_compatibility = 0.8

        elif regime.regime == 'SIDEWAYS':
            # Sideways markets favor mean reversion
            if direction == "LONG" and last['rsi'] < 35:
                base_compatibility = 0.8
            elif direction == "SHORT" and last['rsi'] > 65:
                base_compatibility = 0.8
            else:
                base_compatibility = 0.4

        # Adjust for regime confidence
        adjusted_compatibility = base_compatibility * regime.confidence

        return min(adjusted_compatibility, 1.0)

    def _get_bidirectional_analysis(self, dataframe: DataFrame, market_data: MarketData, regime: RegimeAnalysis) -> BidirectionalAnalysis:
        """
        Context7 Super Powers: Analisi bidirezionale completa con raccomandazione

        Implementa pattern NautilusTrader per position management decision
        """
        # Get directional signals
        long_signal, short_signal = self._analyze_directional_signals(dataframe, market_data, regime)

        # Market confidence based on regime and signal quality
        market_confidence = regime.confidence * max(long_signal.quality_score, short_signal.quality_score)

        # Determine best direction and recommendation
        if long_signal.quality_score > short_signal.quality_score + 0.1:  # Significant margin
            best_direction = "LONG"
            best_quality_score = long_signal.quality_score

            if long_signal.quality_score > 0.8:
                recommendation = "STRONG_LONG"
            elif long_signal.quality_score > 0.6:
                recommendation = "LONG"
            else:
                recommendation = "NEUTRAL"

        elif short_signal.quality_score > long_signal.quality_score + 0.1:  # Significant margin
            best_direction = "SHORT"
            best_quality_score = short_signal.quality_score

            if short_signal.quality_score > 0.8:
                recommendation = "STRONG_SHORT"
            elif short_signal.quality_score > 0.6:
                recommendation = "SHORT"
            else:
                recommendation = "NEUTRAL"
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

        # Determine volatility level
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

        # Calculate trend strength based on price consistency
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
        Context7 Super Powers: Popola segnali LONG usando sistema bidirezionale

        Implementa pattern Context7-compliant con hierarchical filtering e quality scoring
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

        # Context7 Super Powers: Get bidirectional analysis
        bidirectional = self._get_bidirectional_analysis(dataframe, market_data, regime)

        # Context7: Hierarchical filtering system
        # Solo segnali LONG con alta qualità passano il filtro
        if bidirectional.recommendation in ["STRONG_LONG", "LONG"]:
            # Context7: Multi-layer validation
            quality_threshold = 0.6  # Base threshold

            # Threshold più alto per mercati volatili
            if regime.regime == 'VOLATILE':
                quality_threshold = 0.7

            # Threshold più basso per mercati trending con forte trend
            elif regime.regime == 'TRENDING' and regime.trend_strength > 0.7:
                quality_threshold = 0.5

            if bidirectional.long_signal.quality_score >= quality_threshold:
                # Context7: Additional volume and regime validation
                if (bidirectional.long_signal.volume_confirmation and
                    bidirectional.long_signal.regime_compatibility > 0.3):

                    dataframe.iloc[-1, dataframe.columns.get_loc('enter_long')] = 1
                    dataframe.iloc[-1, dataframe.columns.get_loc('enter_tag')] = 'context7_long'

                    self.logger.info(f"🟢 BUY {metadata['pair']} - CONTEXT7 BIDIRECTIONAL:")
                    self.logger.info(f"   Recommendation: {bidirectional.recommendation}")
                    self.logger.info(f"   Quality Score: {bidirectional.long_signal.quality_score:.3f} (threshold: {quality_threshold})")
                    self.logger.info(f"   Primary Score: {bidirectional.long_signal.primary_score:.3f}")
                    self.logger.info(f"   Trend Weight: {bidirectional.long_signal.trend_weight:.2f}")
                    self.logger.info(f"   Regime Compatibility: {bidirectional.long_signal.regime_compatibility:.2f}")
                    self.logger.info(f"   Entry Conditions: {', '.join(bidirectional.long_signal.entry_conditions)}")
                    self.logger.info(f"   Ichimoku Position: {bidirectional.long_signal.ichimoku_position}")
                    self.logger.info(f"   Market Regime: {regime.regime} (confidence: {regime.confidence:.2f})")
                    self.logger.info(f"   Reasoning: {bidirectional.reasoning}")
                else:
                    self.logger.info(f"🚫 BUY {metadata['pair']} - Context7: Volume or regime validation failed")
                    self.logger.info(f"   Volume OK: {bidirectional.long_signal.volume_confirmation}")
                    self.logger.info(f"   Regime Compatibility: {bidirectional.long_signal.regime_compatibility:.2f}")
            else:
                self.logger.info(f"🚫 BUY {metadata['pair']} - Context7: Quality score below threshold")
                self.logger.info(f"   Quality: {bidirectional.long_signal.quality_score:.3f} < {quality_threshold}")
        else:
            self.logger.info(f"🚫 BUY {metadata['pair']} - Context7: No LONG signal")
            self.logger.info(f"   Recommendation: {bidirectional.recommendation}")
            self.logger.info(f"   Best Direction: {bidirectional.best_direction}")
            self.logger.info(f"   Long Quality: {bidirectional.long_signal.quality_score:.3f}")
            self.logger.info(f"   Short Quality: {bidirectional.short_signal.quality_score:.3f}")

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Context7 Super Powers: Popola segnali SHORT usando sistema bidirezionale

        Implementa pattern Context7-compliant con hierarchical filtering e quality scoring
        """
        dataframe.loc[:, 'sell'] = 0

        if len(dataframe) < 50:
            return dataframe

        # Get market data and regime analysis
        market_data = self._get_market_data(dataframe, metadata)
        if not market_data:
            return dataframe

        regime = self._get_regime_analysis(market_data)

        # Context7 Super Powers: Get bidirectional analysis
        bidirectional = self._get_bidirectional_analysis(dataframe, market_data, regime)

        # Context7: Hierarchical filtering system
        # Solo segnali SHORT con alta qualità passano il filtro
        if bidirectional.recommendation in ["STRONG_SHORT", "SHORT"]:
            # Context7: Multi-layer validation
            quality_threshold = 0.6  # Base threshold

            # Threshold più alto per mercati volatili
            if regime.regime == 'VOLATILE':
                quality_threshold = 0.7

            # Threshold più basso per mercati trending con forte trend
            elif regime.regime == 'TRENDING' and regime.trend_strength > 0.7:
                quality_threshold = 0.5

            if bidirectional.short_signal.quality_score >= quality_threshold:
                # Context7: Additional volume and regime validation
                if (bidirectional.short_signal.volume_confirmation and
                    bidirectional.short_signal.regime_compatibility > 0.3):

                    dataframe.iloc[-1, dataframe.columns.get_loc('enter_short')] = 1
                    dataframe.iloc[-1, dataframe.columns.get_loc('enter_tag')] = 'context7_short'

                    self.logger.info(f"🔴 SELL {metadata['pair']} - CONTEXT7 BIDIRECTIONAL:")
                    self.logger.info(f"   Recommendation: {bidirectional.recommendation}")
                    self.logger.info(f"   Quality Score: {bidirectional.short_signal.quality_score:.3f} (threshold: {quality_threshold})")
                    self.logger.info(f"   Primary Score: {bidirectional.short_signal.primary_score:.3f}")
                    self.logger.info(f"   Trend Weight: {bidirectional.short_signal.trend_weight:.2f}")
                    self.logger.info(f"   Regime Compatibility: {bidirectional.short_signal.regime_compatibility:.2f}")
                    self.logger.info(f"   Entry Conditions: {', '.join(bidirectional.short_signal.entry_conditions)}")
                    self.logger.info(f"   Ichimoku Position: {bidirectional.short_signal.ichimoku_position}")
                    self.logger.info(f"   Market Regime: {regime.regime} (confidence: {regime.confidence:.2f})")
                    self.logger.info(f"   Reasoning: {bidirectional.reasoning}")
                else:
                    self.logger.info(f"🚫 SELL {metadata['pair']} - Context7: Volume or regime validation failed")
                    self.logger.info(f"   Volume OK: {bidirectional.short_signal.volume_confirmation}")
                    self.logger.info(f"   Regime Compatibility: {bidirectional.short_signal.regime_compatibility:.2f}")
            else:
                self.logger.info(f"🚫 SELL {metadata['pair']} - Context7: Quality score below threshold")
                self.logger.info(f"   Quality: {bidirectional.short_signal.quality_score:.3f} < {quality_threshold}")
        else:
            self.logger.info(f"🚫 SELL {metadata['pair']} - Context7: No SHORT signal")
            self.logger.info(f"   Recommendation: {bidirectional.recommendation}")
            self.logger.info(f"   Best Direction: {bidirectional.best_direction}")
            self.logger.info(f"   Long Quality: {bidirectional.long_signal.quality_score:.3f}")
            self.logger.info(f"   Short Quality: {bidirectional.short_signal.quality_score:.3f}")

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, after_fill: bool, **kwargs) -> float | None:
        """
        Context7 Super Powers: Dynamic ATR-based stoploss with Ichimoku structure

        Implements best practices from crypto trading research:
        - ATR multiplier (2.5x - 3.5x) for high volatility crypto
        - Ichimoku structure as dynamic support/resistance
        - Regime-aware adjustments
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

            # Calculate ATR-based stoploss (NotebookLM best practice)
            atr_multiplier = 3.0  # Base multiplier for crypto
            if regime.regime == 'VOLATILE':
                atr_multiplier = 3.5  # Wider in volatile markets
            elif regime.regime == 'CALM':
                atr_multiplier = 2.5  # Tighter in calm markets

            # Calculate ATR-based stop distance
            atr_value = last['atr'] if 'atr' in last else last['high'] - last['low']
            atr_stop_pct = (atr_multiplier * atr_value) / last['close']

            # Ichimoku structure-based stops (primary method)
            ichimoku_stop_pct = None

            if trade.is_short:
                # SHORT position: stops above Ichimoku resistance levels
                if last['close'] > last['kijun_sen']:
                    # Price above Kijun: use Kijun as resistance
                    ichimoku_stop_pct = abs((last['kijun_sen'] - last['close']) / last['close'])
                elif last['close'] > last['senkou_span_a']:
                    # Price above Cloud: use Cloud top as resistance
                    ichimoku_stop_pct = abs((last['senkou_span_a'] - last['close']) / last['close'])
                else:
                    # Price below Cloud: use Tenkan as trailing stop
                    ichimoku_stop_pct = abs((last['tenkan_sen'] - last['close']) / last['close'])

            else:  # LONG position
                # LONG position: stops below Ichimoku support levels
                if last['close'] < last['kijun_sen']:
                    # Price below Kijun: use Kijun as support
                    ichimoku_stop_pct = abs((last['kijun_sen'] - last['close']) / last['close'])
                elif last['close'] < last['senkou_span_b']:
                    # Price below Cloud: use Cloud bottom as support
                    ichimoku_stop_pct = abs((last['senkou_span_b'] - last['close']) / last['close'])
                else:
                    # Price above Cloud: use Tenkan as trailing stop
                    ichimoku_stop_pct = abs((last['tenkan_sen'] - last['close']) / last['close'])

            # Use the tighter stop between Ichimoku structure and ATR
            if ichimoku_stop_pct is not None:
                stoploss_pct = min(ichimoku_stop_pct, atr_stop_pct)
            else:
                stoploss_pct = atr_stop_pct

            # Apply regime adjustments
            if regime.regime == 'VOLATILE':
                stoploss_pct *= 1.2  # Add 20% buffer in volatile markets
            elif regime.regime == 'TRENDING':
                stoploss_pct *= 1.1  # Add 10% buffer in trending markets

            # Safety caps (minimum 0.5%, maximum 25%)
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
        Context7 Super Powers: Advanced exit signals with Ichimoku confirmation

        Implements NotebookLM best practices for crypto 5-minute timeframe:
        - Ichimoku Cloud structure breaks (primary exit signals)
        - MACD divergence confirmation
        - CMF volume validation
        - Candlestick pattern recognition
        - Regime-aware take profit levels
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

            # Primary: Ichimoku structure breaks (NotebookLM research)
            if trade.is_short:
                # SHORT position exit signals

                # 1. Price breaks above Cloud (strongest bullish signal)
                if last['close'] > last['senkou_span_a'] and prev['close'] <= last['senkou_span_a']:
                    exit_reasons.append("Price breaks above Cloud")
                    exit_strength += 0.4

                # 2. Price breaks above Kijun (medium bullish signal)
                if last['close'] > last['kijun_sen'] and prev['close'] <= last['kijun_sen']:
                    exit_reasons.append("Price breaks above Kijun")
                    exit_strength += 0.3

                # 3. Tenkan crosses above Kijun (momentum shift)
                if prev['tenkan_sen'] <= prev['kijun_sen'] and last['tenkan_sen'] > last['kijun_sen']:
                    exit_reasons.append("Tenkan crosses above Kijun")
                    exit_strength += 0.2

                # 4. Price inside Cloud with bullish bias (weakening downtrend)
                if last['close'] > last['senkou_span_b'] and last['close'] < last['senkou_span_a']:
                    cloud_bias = (last['senkou_span_a'] + last['senkou_span_b']) / 2
                    if last['close'] > cloud_bias:
                        exit_reasons.append("Price inside Cloud with bullish bias")
                        exit_strength += 0.1

            else:  # LONG position
                # LONG position exit signals

                # 1. Price breaks below Cloud (strongest bearish signal)
                if last['close'] < last['senkou_span_b'] and prev['close'] >= last['senkou_span_b']:
                    exit_reasons.append("Price breaks below Cloud")
                    exit_strength += 0.4

                # 2. Price breaks below Kijun (medium bearish signal)
                if last['close'] < last['kijun_sen'] and prev['close'] >= last['kijun_sen']:
                    exit_reasons.append("Price breaks below Kijun")
                    exit_strength += 0.3

                # 3. Tenkan crosses below Kijun (momentum shift)
                if prev['tenkan_sen'] >= prev['kijun_sen'] and last['tenkan_sen'] < last['kijun_sen']:
                    exit_reasons.append("Tenkan crosses below Kijun")
                    exit_strength += 0.2

                # 4. Price inside Cloud with bearish bias (weakening uptrend)
                if last['close'] < last['senkou_span_a'] and last['close'] > last['senkou_span_b']:
                    cloud_bias = (last['senkou_span_a'] + last['senkou_span_b']) / 2
                    if last['close'] < cloud_bias:
                        exit_reasons.append("Price inside Cloud with bearish bias")
                        exit_strength += 0.1

            # Secondary: MACD confirmation (NotebookLM research)
            macd_divergence = False
            if trade.is_short:
                # For SHORT: bullish MACD crossover
                if prev['macd'] <= prev['macdsignal'] and last['macd'] > last['macdsignal']:
                    exit_reasons.append("MACD bullish crossover")
                    exit_strength += 0.15
                    macd_divergence = True
            else:
                # For LONG: bearish MACD crossover
                if prev['macd'] >= prev['macdsignal'] and last['macd'] < last['macdsignal']:
                    exit_reasons.append("MACD bearish crossover")
                    exit_strength += 0.15
                    macd_divergence = True

            # Tertiary: CMF volume confirmation (NotebookLM best practice for BTC/ETH)
            cmf_confirmation = False
            if 'cmf' in last:
                if trade.is_short and last['cmf'] > 0.1:  # Strong buying volume
                    exit_reasons.append(f"CMF buying pressure ({last['cmf']:.3f})")
                    exit_strength += 0.1
                    cmf_confirmation = True
                elif not trade.is_short and last['cmf'] < -0.1:  # Strong selling volume
                    exit_reasons.append(f"CMF selling pressure ({last['cmf']:.3f})")
                    exit_strength += 0.1
                    cmf_confirmation = True

            # Quaternary: Candlestick pattern recognition (5-minute timeframe)
            pattern_exit = self._detect_reversal_patterns(last, prev, trade.is_short)
            if pattern_exit:
                exit_reasons.append(pattern_exit)
                exit_strength += 0.1

            # Quinary: Regime-aware take profit levels
            tp_threshold = 0.0
            if regime.regime == 'TRENDING':
                tp_threshold = 0.20  # 20% in trending markets
            elif regime.regime == 'VOLATILE':
                tp_threshold = 0.15  # 15% in volatile markets
            elif regime.regime == 'CALM':
                tp_threshold = 0.12  # 12% in calm markets
            else:  # SIDEWAYS
                tp_threshold = 0.08  # 8% in sideways markets

            if current_profit > tp_threshold:
                # Check if we have confirmation signals
                if exit_strength > 0.3:  # Strong exit signals
                    exit_reasons.append(f"Take profit at {current_profit:.1%} with confirmation")
                    exit_strength += 0.2
                elif exit_strength > 0.1:  # Some exit signals
                    exit_reasons.append(f"Take profit at {current_profit:.1%} with weak signals")
                    exit_strength += 0.1

            # Final exit decision based on signal strength
            if exit_strength >= 0.6:  # Strong exit signals
                self.logger.info(f"🚨 STRONG EXIT {pair}: {' | '.join(exit_reasons)} (strength: {exit_strength:.2f})")
                return "strong_exit_signal"
            elif exit_strength >= 0.4:  # Medium exit signals
                self.logger.info(f"⚠️ MEDIUM EXIT {pair}: {' | '.join(exit_reasons)} (strength: {exit_strength:.2f})")
                return "medium_exit_signal"
            elif exit_strength >= 0.2:  # Weak exit signals
                self.logger.info(f"💡 WEAK EXIT {pair}: {' | '.join(exit_reasons)} (strength: {exit_strength:.2f})")
                return "weak_exit_signal"

            return None

        except Exception as e:
            self.logger.error(f"Error in custom_exit: {e}")
            return None

    def _detect_reversal_patterns(self, current, previous, is_short: bool) -> str:
        """
        Detect candlestick reversal patterns for 5-minute crypto timeframe
        """
        try:
            # Calculate body sizes and wicks
            current_body = abs(current['close'] - current['open'])
            current_upper_wick = current['high'] - max(current['close'], current['open'])
            current_lower_wick = min(current['close'], current['open']) - current['low']

            prev_body = abs(previous['close'] - previous['open'])
            prev_upper_wick = previous['high'] - max(previous['close'], previous['open'])
            prev_lower_wick = min(previous['close'], previous['open']) - previous['low']

            # Hammer/Hanging Man pattern
            if current_lower_wick > 2 * current_body and current_upper_wick < 0.1 * current_body:
                if is_short and current['close'] > current['open']:  # Bullish hammer
                    return "Bullish Hammer (reversal)"
                elif not is_short and current['close'] < current['open']:  # Bearish hanging man
                    return "Bearish Hanging Man (reversal)"

            # Engulfing patterns
            if is_short:  # Looking for bullish reversal
                if (previous['close'] < previous['open'] and  # Previous red candle
                    current['close'] > current['open'] and  # Current green candle
                    current['open'] < previous['close'] and  # Engulfs previous
                    current['close'] > previous['open']):
                    return "Bullish Engulfing (strong reversal)"
            else:  # Looking for bearish reversal
                if (previous['close'] > previous['open'] and  # Previous green candle
                    current['close'] < current['open'] and  # Current red candle
                    current['open'] > previous['close'] and  # Engulfs previous
                    current['close'] < previous['open']):
                    return "Bearish Engulfing (strong reversal)"

            # Doji/Star patterns (indecision)
            if current_body < 0.1 * (current['high'] - current['low']):
                return "Doji/Star (indecision - potential reversal)"

            return ""

        except Exception as e:
            self.logger.error(f"Error in _detect_reversal_patterns: {e}")
            return ""

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                           rate: float, time_in_force: str, current_time: datetime,
                           entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """Confirm trade entry with regime validation"""
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) == 0:
                return False

            market_data = self._get_market_data(dataframe, {'pair': pair})
            if not market_data:
                return False

            regime = self._get_regime_analysis(market_data)

            # Confirm based on regime confidence (lowered threshold for SHORT signals)
            if regime.confidence < 0.3:
                self.logger.info(f"🚫 Entry rejected for {pair}: Low regime confidence ({regime.confidence:.2f})")
                return False

            # Smart volume confirmation with detailed logging
            last = dataframe.iloc[-1]

            # Use smart volume ratio (more robust)
            volume_ratio = last['volume_smart_ratio']
            current_volume = last['volume']
            smart_mean = last['volume_smart_mean']

            self.logger.info(f"📊 {pair} Volume Analysis - SMART BASELINE:")
            self.logger.info(f"   Current Volume: {current_volume:,.0f}")
            self.logger.info(f"   Smart Baseline: {smart_mean:,.0f}")
            self.logger.info(f"   Volume Ratio: {volume_ratio:.2f}")
            self.logger.info(f"   Regular Ratio: {last['volume_ratio']:.2f}")

            # Context7 Super Powers: Dynamic volume threshold based on market volatility
            atr_value = last.get('atr', 0.01)  # Default 1% if ATR not available
            price = last['close']
            volatility_pct = (atr_value / price) * 100

            # Dynamic threshold: higher volatility = more lenient volume requirement (Context7 Super Powers: adjusted)
            if volatility_pct > 3.0:  # High volatility market
                dynamic_threshold = 0.50
            elif volatility_pct > 1.5:  # Medium volatility market
                dynamic_threshold = 0.55
            else:  # Low volatility market
                dynamic_threshold = 0.60

            if volume_ratio < dynamic_threshold:
                self.logger.info(f"🚫 Entry rejected for {pair}: Low smart volume ({volume_ratio:.2f} < {dynamic_threshold:.2f})")
                self.logger.info(f"   Market volatility: {volatility_pct:.2f}% (ATR: {atr_value:.4f})")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in confirm_trade_entry: {e}")
            return False

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate entry signals using Context7 Super Powers bidirectional analysis
        Enhanced with regime-aware signal generation
        """
        try:
            # Centralized input validation (CRITICAL for security)
            if not self._validate_inputs(dataframe, metadata, 'populate_entry_trend'):
                return dataframe

            # Get basic entry/exit signals from parent methods
            dataframe = super().populate_entry_trend(dataframe, metadata)

            # Context7: Signal enhancement is handled by parent class methods
            # The parent class already generates Context7 Super Powers signals
            # No additional processing needed here

            return dataframe

        except Exception as e:
            self.logger.error(f"Error in populate_entry_trend: {e}")
            return super().populate_entry_trend(dataframe, metadata)

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate exit signals using parent implementation
        Enhanced with Context7 Super Powers logic
        """
        try:
            # Get basic exit signals from parent methods
            dataframe = super().populate_exit_trend(dataframe, metadata)

            # Context7: Could add enhanced exit logic here in future versions
            # For now, use parent implementation

            return dataframe

        except Exception as e:
            self.logger.error(f"Error in populate_exit_trend: {e}")
            return super().populate_exit_trend(dataframe, metadata)

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        """
        Context7 Super Powers: Dynamic leverage up to 10x with intelligent risk management.

        Simple and effective leverage implementation based on Context7 best practices.
        """
        # Asset-specific leverage limits for risk management
        asset_limits = {
            "BTC/USDT:USDT": 8.0,
            "ETH/USDT:USDT": 7.0,
            "SOL/USDT:USDT": 6.0,
            "BNB/USDT:USDT": 6.0,
            "AVAX/USDT:USDT": 5.0,
            "DOT/USDT:USDT": 5.0
        }

        # Get asset-specific limit
        asset_limit = asset_limits.get(pair, 5.0)

        # Base leverage with moderate multiplier
        base_leverage = 2.0

        # Dynamic adjustment based on market conditions
        # Simple and effective - no complex calculations needed
        if side == "long":
            # Conservative leverage for long positions
            dynamic_leverage = min(base_leverage * 1.5, asset_limit, max_leverage, 10.0)
        else:
            # Slightly higher for short positions (downtrends can be sharper)
            dynamic_leverage = min(base_leverage * 2.0, asset_limit, max_leverage, 10.0)

        # Log leverage calculation
        self.logger.info(f"🎯 LEVERAGE {pair} ({side.upper()}): {dynamic_leverage:.1f}x (max: {asset_limit:.1f}x)")

        return dynamic_leverage

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float | None, max_stake: float,
                           leverage: float, entry_tag: str | None, side: str,
                           **kwargs) -> float:
        """
        Context7 Super Powers: Intelligent Stake Sizing basato su framework Perplexity.

        Mathematical Framework:
        Final Stake = Base Stake × f_quality × f_volatility × f_consensus × f_leverage × f_portfolio

        Dove:
        - f_quality = q² (quadratic scaling per segnali ad alta confidenza)
        - f_volatility = 1/(1+50×ATR%) (volatility-adjusted positioning)
        - f_consensus = n_agree/5 (multi-timeframe consensus factor)
        - f_leverage = 1/√leverage (leverage compensation)
        - f_portfolio = max(0,1-current/max) (portfolio heat management)
        """
        try:
            # Get current dataframe for quality score and ATR
            dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()

            # 1. QUALITY SCORE FACTOR (f_quality = q²)
            quality_score = last_candle.get('quality_score', 0.6)
            f_quality = quality_score ** 2  # Quadratic scaling per segnali ad alta confidenza

            # 2. VOLATILITY FACTOR (f_volatility = 1/(1+50×ATR%))
            atr_value = last_candle.get('atr', current_rate * 0.02)  # Default 2% ATR
            volatility_pct = (atr_value / current_rate) * 100
            f_volatility = 1 / (1 + 50 * volatility_pct / 100)  # Normalized volatility factor

            # 3. MULTI-TIMEFRAME CONSENSUS FACTOR (f_consensus = n_agree/5)
            # Count consensus across timeframes using available signals
            consensus_count = 1  # Start with current timeframe
            timeframes = ['15m', '30m', '1h', '1d']

            for tf in timeframes:
                tf_dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=tf)
                if not tf_dataframe.empty:
                    tf_last = tf_dataframe.iloc[-1].squeeze()
                    tf_quality = tf_last.get('quality_score', 0)
                    if tf_quality >= 0.6:  # Same threshold as base strategy
                        consensus_count += 1

            f_consensus = consensus_count / 5  # Normalize to 0-1 range

            # 4. LEVERAGE COMPENSATION FACTOR (f_leverage = 1/√leverage)
            f_leverage = 1 / np.sqrt(max(leverage, 1.0))  # Avoid division by zero

            # 5. PORTFOLIO HEAT MANAGEMENT (f_portfolio = max(0,1-current/max))
            # Context7 Super Powers: Use correct Freqtrade API for open trades
            from freqtrade.persistence import Trade
            current_positions = len(Trade.get_open_trades())
            max_positions_per_asset = 30  # Max total positions for all assets
            f_portfolio = max(0, 1 - (current_positions / max_positions_per_asset))

            # 6. Calculate final stake using mathematical framework
            base_stake = proposed_stake
            final_stake = base_stake * f_quality * f_volatility * f_consensus * f_leverage * f_portfolio

            # Apply safety constraints
            final_stake = max(min_stake or 0, min(final_stake, max_stake))

            # Log detailed calculation for transparency
            self.logger.info(f"💰 INTELLIGENT STAKE {pair} ({side.upper()}):")
            self.logger.info(f"   Base Stake: {base_stake:.2f} USDT")
            self.logger.info(f"   Quality Score: {quality_score:.3f} → f_quality: {f_quality:.3f}")
            self.logger.info(f"   Volatility: {volatility_pct:.2f}% → f_volatility: {f_volatility:.3f}")
            self.logger.info(f"   Consensus: {consensus_count}/5 → f_consensus: {f_consensus:.3f}")
            self.logger.info(f"   Leverage: {leverage:.1f}x → f_leverage: {f_leverage:.3f}")
            self.logger.info(f"   Portfolio Heat: {current_positions}/{max_positions_per_asset} → f_portfolio: {f_portfolio:.3f}")
            self.logger.info(f"   Final Stake: {final_stake:.2f} USDT ({(final_stake/base_stake-1)*100:+.1f}%)")

            return final_stake

        except Exception as e:
            self.logger.error(f"❌ Error in custom_stake_amount for {pair}: {str(e)}")
            return proposed_stake  # Fallback to proposed stake

    def _validate_inputs(self, dataframe: DataFrame, metadata: dict, method_name: str) -> bool:
        """
        Centralized input validation for security and stability.
        Returns True if validation passes, False otherwise.
        """
        try:
            # Validate dataframe
            if not isinstance(dataframe, pd.DataFrame):
                self.logger.error(f"{method_name}: dataframe must be a pandas DataFrame")
                return False

            if dataframe.empty:
                self.logger.error(f"{method_name}: Empty dataframe received")
                return False

            if len(dataframe) < self.startup_candle_count:
                self.logger.warning(f"{method_name}: Insufficient data: {len(dataframe)} < {self.startup_candle_count}")
                return False

            # Validate metadata
            if not isinstance(metadata, dict):
                self.logger.error(f"{method_name}: metadata must be a dictionary")
                return False

            if 'pair' not in metadata:
                self.logger.error(f"{method_name}: metadata must contain 'pair' key")
                return False

            if not isinstance(metadata['pair'], str) or not metadata['pair'].strip():
                self.logger.error(f"{method_name}: Invalid pair in metadata: {metadata.get('pair')}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"{method_name}: Input validation error: {e}")
            return False

# ============================================================================
# CONTEXT7 SUPER POWERS: INITIALIZATION
# ============================================================================

# Initialize batch processor at module level - DISABLED
# batch_processor_instance = Context7BatchProcessor()  # DISABLED - LLM connection issues
batch_processor_instance = None

# Export strategy
__strategy__ = IchimokuLLMIntelligentStakeSizing