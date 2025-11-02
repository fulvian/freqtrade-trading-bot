# Ichimoku LLM Enhanced V2 Deployment Guide

## Overview

This guide covers the deployment of Ichimoku LLM Enhanced V2, the comprehensive upgrade to the trading strategy with full LLM integration beyond position sizing.

## V2 Features Summary

### 🚀 Major Enhancements

1. **Regime-based Trade Entry Validation**
   - LLM-powered trade validation based on market regime
   - Dynamic entry condition checking
   - Risk-adjusted position sizing

2. **Dynamic Stop Loss and ROI Management**
   - Real-time stop loss adjustment based on LLM analysis
   - Adaptive ROI targets per market conditions
   - Trailing stop optimization

3. **LLM-aware Signal Generation and Filtering**
   - Enhanced signal filtering using LLM insights
   - Signal strength classification
   - Confidence-based signal weighting

4. **Intelligent Exit Strategies**
   - Regime-aware exit timing
   - Multiple exit criteria evaluation
   - Profit protection mechanisms

5. **Portfolio Management with Regime-adaptive Limits**
   - Dynamic position sizing based on portfolio state
   - Correlation risk assessment
   - Portfolio-wide risk management

6. **Comprehensive Risk Scoring System**
   - Multi-factor risk assessment
   - Real-time risk monitoring
   - Risk-adjusted position limits

## Deployment Architecture

### File Structure
```
user_data/
├── strategies/
│   ├── ichimoku_llm_enhanced_v1.py  # Current production (port 8084)
│   └── ichimoku_llm_enhanced_v2.py  # New V2 implementation
├── config_ichimoku_enhanced_v1.json # Current production config
└── config_ichimoku_enhanced_v2.json # New V2 config
```

### Port Configuration
- **V1 Production**: Port 8084 (current running bot)
- **V2 Production**: Port 8085 (new deployment)

## Prerequisites

1. **Dependencies**
   - `json_repair` for LLM response parsing
   - `llm_circuit_breaker` for API resilience
   - `pydantic` for data validation
   - `httpx` for HTTP requests

2. **LLM Configuration**
   - DeepSeek-V3.1 API access
   - Circuit breaker configuration
   - Timeout and retry settings

3. **Database Setup**
   - SQLite database for V2 trades
   - Separate from V1 database

## Deployment Steps

### 1. Validate V2 Installation

```bash
# Run V2 integration tests
.devstream/bin/python test_v2_integration.py

# Expected output: 🎉 ALL V2 TESTS PASSED!
```

### 2. Configure V2 Settings

Edit `user_data/config_ichimoku_enhanced_v2.json`:

```json
{
    "max_open_trades": 5,        // Increased from 3 (V2 portfolio management)
    "stake_amount": 100,         // Per trade stake
    "dry_run_wallet": 2000,      // Increased wallet for V2 testing
    "api_server": {
        "listen_port": 8085      // V2 dedicated port
    },
    "strategy": "IchimokuLLMEnhancedV2",
    "db_url": "sqlite:///tradesv3_llm_enhanced_v2.sqlite"
}
```

### 3. Start V2 Bot

```bash
# Start V2 in dry-run mode
.devstream/bin/python -m freqtrade trade \
    --config user_data/config_ichimoku_enhanced_v2.json \
    --strategy IchimokuLLMEnhancedV2 \
    --dry-run
```

### 4. Monitor V2 Performance

Access V2 dashboard at `http://localhost:8085`

Key metrics to monitor:
- **LLM API call success rate**
- **Regime validation accuracy**
- **Dynamic risk parameter effectiveness**
- **Portfolio risk distribution**
- **Exit strategy performance**

## V2 Configuration Options

### Feature Toggles

All V2 features can be individually enabled/disabled:

```python
# In strategy __init__ method
self.regime_based_validation_enabled = True      # Trade entry validation
self.dynamic_risk_management_enabled = True      # Dynamic stop loss/ROI
self.llm_signal_filtering_enabled = True         # Signal enhancement
self.intelligent_exit_enabled = True             # Intelligent exits
self.portfolio_management_enabled = True         # Portfolio management
self.comprehensive_risk_scoring_enabled = True   # Risk scoring
```

### Risk Parameters

```python
self.default_stop_loss = 0.03          # 3% default stop loss
self.default_roi_target = 0.06         # 6% default ROI target
self.max_risk_per_trade = 0.02         # 2% max risk per trade
self.max_portfolio_risk = 0.10         # 10% total portfolio risk
```

### Cache TTLs

```python
self.trade_validation_cache_ttl = 300      # 5 minutes
self.risk_parameters_cache_ttl = 600       # 10 minutes
self.signal_cache_ttl = 180                # 3 minutes
self.exit_signal_cache_ttl = 60            # 1 minute
self.portfolio_allocation_ttl = 3600       # 1 hour
```

## LLM Integration Details

### API Configuration

V2 uses enhanced LLM configuration:

```python
self.v2_llm_config = {
    "model": "deepseek-ai/DeepSeek-V3.1:thinking",
    "timeout": 20,              # Increased for complex analysis
    "temperature": 0.2,         # Lower for consistency
    "max_tokens": 1200          # Increased for comprehensive analysis
}
```

### Prompt Engineering

V2 implements specialized prompts for different functions:

1. **Trade Validation Prompt**
   - Market regime analysis
   - Entry condition evaluation
   - Risk assessment

2. **Dynamic Risk Parameters Prompt**
   - Volatility assessment
   - Trade timing analysis
   - Risk factor calculation

3. **Signal Analysis Prompt**
   - Signal strength evaluation
   - Confidence scoring
   - Enhancement recommendations

4. **Exit Analysis Prompt**
   - Regime change detection
   - Profit optimization
   - Risk management

## Monitoring and Maintenance

### Health Checks

Monitor these V2-specific health metrics:

1. **LLM Response Times**
   - Average response time < 15 seconds
   - Success rate > 95%
   - Circuit breaker activation rate

2. **Cache Performance**
   - Hit rate > 80%
   - Cache freshness
   - Memory usage

3. **Risk Management**
   - Portfolio risk distribution
   - Position sizing accuracy
   - Stop loss effectiveness

### Log Analysis

Key V2 log patterns to monitor:

```bash
# V2 initialization
"🚀 Ichimoku LLM Enhanced V2 V2 initialized"
"✅ V2 Features: Regime validation, Dynamic risk, Signal filtering, Intelligent exits, Portfolio management"

# Trade validation
"✅ V2: Regime validation passed for {pair}"
"❌ V2: Regime validation rejected {pair}"

# Risk management
"🛡️ V2: Dynamic stoploss for {pair}: {stoploss:.3f}"
"🎯 V2: Risk score for {pair}: {risk_score:.3f}"

# Portfolio management
"💰 V2: Regime-adjusted stake for {pair}"
"💰 V2: Portfolio limit applied for {pair}"
```

### Performance Optimization

1. **Cache Optimization**
   - Monitor cache hit rates
   - Adjust TTLs based on market conditions
   - Clean up expired cache entries

2. **LLM API Optimization**
   - Implement request batching
   - Optimize prompt lengths
   - Use fallback responses during outages

3. **Risk Calculation Optimization**
   - Pre-calculate risk factors
   - Cache correlation matrices
   - Optimize portfolio calculations

## Troubleshooting

### Common Issues

1. **LLM API Failures**
   - Check API key validity
   - Verify circuit breaker status
   - Review timeout configurations

2. **Cache Issues**
   - Clear cache if stale data detected
   - Monitor memory usage
   - Adjust TTLs if needed

3. **Risk Calculation Errors**
   - Validate input data formats
   - Check for division by zero
   - Verify portfolio state consistency

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Key debug information:
- LLM request/response details
- Cache hit/miss statistics
- Risk calculation breakdown
- Portfolio state changes

## Migration from V1

### Key Differences

| Feature | V1 | V2 |
|---------|----|----|
| LLM Integration | Position sizing only | Full integration |
| Risk Management | Static parameters | Dynamic parameters |
| Signal Generation | Basic signals | LLM-enhanced signals |
| Exit Strategy | Fixed rules | Intelligent exits |
| Portfolio Management | None | Comprehensive management |
| API Port | 8084 | 8085 |

### Migration Steps

1. **Parallel Deployment**
   - Keep V1 running on port 8084
   - Deploy V2 on port 8085
   - Compare performance

2. **Validation Period**
   - Run V2 in dry-run mode
   - Compare trade decisions
   - Validate risk management

3. **Gradual Rollout**
   - Start with small position sizes
   - Monitor V2 performance
   - Gradually increase allocation

4. **Full Migration**
   - Switch to V2 as primary
   - Retire V1 after validation
   - Update monitoring systems

## Production Best Practices

### Security

1. **API Key Management**
   - Store API keys securely
   - Rotate keys regularly
   - Monitor API usage

2. **Access Control**
   - Secure API endpoints
   - Implement authentication
   - Limit access to sensitive functions

### Reliability

1. **Error Handling**
   - Implement graceful degradation
   - Use fallback responses
   - Log all errors appropriately

2. **Monitoring**
   - Set up alerting for critical metrics
   - Monitor system health
   - Track performance over time

### Performance

1. **Resource Management**
   - Monitor memory usage
   - Optimize cache sizes
   - Manage concurrent requests

2. **Scalability**
   - Design for horizontal scaling
   - Use efficient data structures
   - Optimize database queries

## Support and Maintenance

### Regular Tasks

1. **Daily**
   - Review trading performance
   - Check LLM API status
   - Monitor risk metrics

2. **Weekly**
   - Analyze strategy performance
   - Review cache efficiency
   - Update risk parameters

3. **Monthly**
   - Comprehensive performance review
   - Strategy optimization
   - System health check

### Contact and Support

- **Strategy Issues**: Check logs and run integration tests
- **LLM API Issues**: Verify API configuration and usage
- **Performance Issues**: Monitor system resources and cache performance

---

**Deployment Status**: ✅ Ready for Production
**Last Updated**: 2025-11-02
**Version**: Ichimoku LLM Enhanced V2.0.0