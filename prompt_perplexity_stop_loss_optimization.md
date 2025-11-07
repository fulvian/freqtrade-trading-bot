# PROMPT DI RICERCA - PERPLEXITY

## Critical Stop Loss Analysis Required - Trading Algorithm Optimization

### Current System Performance Crisis
My FreqTrade Ichimoku strategy has a **93.9% win rate** but is **losing money overall (-6.36%)** due to a critical stop loss design flaw.

**The Problem:**
- **3 catastrophic losses** at exactly -10% (AVAX: -10.17%, BNB: -10.30%, DOT: -10.13%)
- These 3 losses **erased all profits** from the other 79 winning trades
- Current stop loss acts as "extreme loss" trigger, not risk management

### Research Questions

**1. Industry Standards Analysis:**
What are professional stop loss best practices for cryptocurrency futures trading with similar characteristics:
- High win rate strategies (>90%)
- Ichimoku-based technical analysis
- 5-minute timeframe trading
- Multi-asset portfolio (BTC, ETH, SOL, BNB, AVAX, DOT)

**2. Stop Loss System Design:**
Research and compare:
- Fixed percentage stop losses vs dynamic volatility-based systems
- ATR (Average True Range) based stop losses for crypto futures
- Ichimoku-based stop loss levels (Kijun-sen, Senkou Spans)
- Trailing stop loss optimization strategies
- Multi-tier exit systems (breakeven, partial profits, final exit)

**3. Quantitative Risk Management:**
Calculate optimal parameters for:
- Maximum acceptable loss per trade
- Risk/reward ratios for 93.9% win rate systems
- Portfolio heat management across 6 concurrent positions
- Maximum drawdown optimization

**4. FreqTrade Implementation:**
Technical research needed on:
- `custom_stoploss` function optimization vs native stoploss
- Dynamic stop loss adjustment based on market volatility
- Combining technical indicators for adaptive exit signals
- Implementation of time-based exits alongside price-based stops

**5. Market-Specific Considerations:**
Analyze stop loss optimization for:
- Cryptocurrency futures market structure
- 24/7 market volatility patterns
- Correlation risks in crypto asset baskets
- Leverage considerations in futures trading

### Expected Deliverables

1. **Quantitative Analysis:** Specific stop loss percentage recommendations with mathematical justification
2. **Implementation Code:** FreqTrade-compatible code examples for the proposed stop loss system
3. **Risk Management Framework:** Complete parameter set for portfolio-level risk control
4. **Performance Projections:** Expected improvements in risk-adjusted returns

### Critical Success Criteria
The solution must:
- Eliminate catastrophic -10% losses
- Preserve the high win rate advantage
- Improve overall profitability from -6.36% to positive territory
- Be implementable within existing FreqTrade/Ichimoku framework

### Context Notes
- Strategy uses "IchimokuEnhancedV8092" with custom_stoploss enabled
- Current configuration: stoploss = -0.10 (as hard limit only)
- Trading mode: futures with cross-margin
- Assets: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, AVAX/USDT, DOT/USDT

**Research Focus:** Practical, implementable solutions backed by quantitative analysis and crypto trading best practices. Avoid theoretical approaches - provide actionable recommendations for immediate implementation.

---
*This research will directly impact real trading performance. Please prioritize evidence-based, data-driven recommendations.*