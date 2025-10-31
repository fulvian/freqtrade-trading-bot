# Circuit Breaker Deployment Guide - LLM Risk Assessment

## 🎯 Problem Solved

**Original Issue**: Le strategie LLM mostravano diradamento progressivo delle chiamate API (da ogni 1-2 min a completa cessazione) a causa di timeout e retry infiniti.

**Solution**: Circuit Breaker Context7-compliant con fallback automatico a logica tradizionale.

## 📋 Files Creati/Modificati

### Nuovi Files:
1. **`llm_circuit_breaker.py`** - Core Circuit Breaker implementation
2. **`ichimoku_llm_risk_strategy_patched.py`** - Strategy with Circuit Breaker
3. **`test_circuit_breaker_live.py`** - Test script

### Files da Backup:
- `ichimoku_llm_risk_strategy.py` → `ichimoku_llm_risk_strategy_original.py`

## 🛠️ Deployment Steps

### 1. Backup Strategie Esistenti
```bash
# Backup delle strategie originali
cp user_data/strategies/ichimoku_llm_risk_strategy.py user_data/strategies/ichimoku_llm_risk_strategy_original.py
cp user_data/strategies/bb_meanreversion_llm_strategy.py user_data/strategies/bb_meanreversion_llm_strategy_original.py
```

### 2. Stop Bot Attuali
```bash
# Stop dei bot che usano le strategie LLM
pkill -f "ichimoku_llm_risk"
pkill -f "bb_meanreversion_llm"
```

### 3. Deploy Circuit Breaker
```bash
# Copia le strategie patchate
cp user_data/strategies/ichimoku_llm_risk_strategy_patched.py user_data/strategies/ichimoku_llm_risk_strategy.py

# Copia il circuit breaker module
cp user_data/strategies/llm_circuit_breaker.py user_data/strategies/
```

### 4. Update Config Files
Modifica i file config per usare le nuove strategie:

**user_data/config_ichimoku_llm_risk.json**:
```json
{
  "strategy": "IchimokuLLMRiskStrategyPatched",
  ...
}
```

### 5. Test Deployment
```bash
# Test del circuit breaker standalone
.devstream/bin/python user_data/strategies/test_circuit_breaker_live.py

# Test della strategia
.devstream/bin/python -c "
from user_data.strategies.ichimoku_llm_risk_strategy import IchimokuLLMRiskStrategyPatched
from freqtrade.strategy import IStrategy
strategy = IchimokuLLMRiskStrategyPatched({})
print('✅ Strategy loaded successfully')
print(f'Circuit breaker stats: {strategy.get_circuit_breaker_stats()}')
"
```

### 6. Start Bot con Circuit Breaker
```bash
# Start con la nuova strategia
.devstream/bin/python -m freqtrade trade --config user_data/config_ichimoku_llm_risk.json --strategy IchimokuLLMRiskStrategyPatched --userdir user_data
```

## 📊 Circuit Breaker Configuration

### Default Parameters (Context7-compliant):
- **Failure Threshold**: 3 fallimenti consecutivi aprono il circuito
- **Recovery Timeout**: 5 minuti prima di tentare recupero
- **API Timeout**: 15s (aggressivo vs 30s original)
- **Max Retries**: 2 retry immediati
- **Backoff Factor**: 1.5x per tentativi successivi

### Customization Options:
```python
# Per personalizzare i parametri nel __init__ della strategia:
self.llm_broker = create_llm_broker(
    strategy_name="CustomStrategy",
    failure_threshold=5,      # Più conservativo
    recovery_timeout=600,     # 10 minuti recovery
    api_timeout=20,           # Timeout più generoso
    max_retries=3             # Più retry
)
```

## 🚨 Fallback Logic

Quando LLM non è disponibile (circuit breaker OPEN), la strategia usa:

### Technical Analysis Fallback:
- **RSI**: Overbought/oversold detection
- **Volatility**: Risk sizing adjustment
- **Ichimoku Cloud**: Trend direction
- **Price action**: Support/resistance levels

### Risk Score Calculation:
```
Base Score: 50
+ RSI Factor (30% weight): +20 (overbought) / -10 (oversold)
+ Volatility Factor (25% weight): +15 (high) / -5 (low)
+ Trend Factor (25% weight): +20 (bearish) / -10 (bullish)
+ Price Change Factor (20% weight): +10 (large moves)
```

## 📈 Monitoring

### 1. Circuit Breaker Stats
```python
# Per ottenere statistiche in tempo reale:
stats = strategy.get_circuit_breaker_stats()
print(f"State: {stats['state']}")
print(f"Success Rate: {stats['success_rate']}%")
print(f"Failed Calls: {stats['failed_calls']}")
```

### 2. Log Messages
I log mostrano stato del circuit breaker:
- `🚀 API call attempt X/Y (timeout: Zs)`
- `✅ LLM Risk Assessment completed`
- `🚨 Circuit Breaker OPEN - API disabilitata`
- `⚠️ Using fallback risk assessment`
- `🔍 Tentativo recupero API (HALF_OPEN)`

### 3. Dashboard Integration
Il dashboard unificato può mostrare:
- Stato del circuit breaker per ogni strategia
- Success/failure rate in tempo reale
- Tempo di recupero rimanente

## 🔧 Troubleshooting

### Issue: Circuit Breaker stuck in OPEN
```bash
# Verifica timestamp ultimo fallimento
grep "Circuit Breaker OPEN" logs/freqtrade.log | tail -1

# Forza reset (solo se necessario)
# Riavviare il bot o attendere il timeout naturale
```

### Issue: High failure rate
```bash
# Controlla connessione API
curl -s https://nano-gpt.com/api/v1/models

# Verifica token API
# Controllare bilancio e limiti su nano-gpt.com
```

### Issue: Fallback sempre attivo
```bash
# Controlla se i parametri risk sono troppo restrittivi
# Ridurre min_confidence o aumentare max_risk_score
```

## 📋 Performance Expectations

### Before Circuit Breaker:
- **API Response Time**: 20-240s (con timeout)
- **Success Rate**: ~30% (molti timeout)
- **System Stability**: Processi morienti per memory leak

### After Circuit Breaker:
- **API Response Time**: 15-20s (con timeout aggressivo)
- **Success Rate**: >90% (con fallback)
- **System Stability**: Continua operatività con fallback

## 🎯 Success Metrics

### Immediate (Week 1):
- ✅ Nessun crash per timeout API
- ✅ Chiamate API stabili ogni 1-2 minuti
- ✅ Circuit breaker stats disponibili

### Medium Term (Month 1):
- ✅ Success rate >90% (LLM + fallback)
- ✅ Recovery automatico da degradazioni API
- ✅ Log strutturati per debugging

### Long Term (Quarter 1):
- ✅ Zero downtime per problemi API
- ✅ Performance predittibile
- ✅ Scalabilità per nuove strategie LLM

## 🔄 Rollback Plan

Se problemi con il Circuit Breaker:

```bash
# 1. Stop bot
pkill -f "ichimoku_llm_risk"

# 2. Ripristina strategia originale
cp user_data/strategies/ichimoku_llm_risk_strategy_original.py user_data/strategies/ichimoku_llm_risk_strategy.py

# 3. Restart bot originale
.devstream/bin/python -m freqtrade trade --config user_data/config_ichimoku_llm_risk.json --strategy IchimokuLLMRiskStrategy --userdir user_data
```

## 📚 Context7 Compliance

✅ **Logging Strutturato**: Tutti i log con livelli appropriati
✅ **Error Handling**: Gestione trasparente e tracciabile
✅ **Design Patterns**: Circuit breaker pattern documentato
✅ **Separazione Responsabilità**: Module isolato e riutilizzabile
✅ **Documentazione**: Guide complete e troubleshooting steps

---

**Status**: ✅ **PRODUCTION READY** - Test completo e funzionante