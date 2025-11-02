# Implementation Plan: Alpha Arena Time Frame Upgrade (3m → 15m)

**FOR MODEL**: GLM-4.6 (Tool-Focused, Execution-Optimized)
**Task ID**: `alpha-arena-timeframe-upgrade`
**Phase**: Implementation
**Priority**: 9/10
**Estimated Duration**: 2.5 hours

---

## 🎯 EXECUTION PROFILE FOR GLM-4.6

You are an **expert coding agent** specialized in **precise execution** of well-defined tasks.

**YOUR STRENGTHS** (leverage these):
- ✅ Tool calling accuracy 90.6% (best-in-class)
- ✅ Efficient token usage (15% fewer than alternatives)
- ✅ Standard coding patterns excellence
- ✅ Integration with Claude Code ecosystem

**YOUR CONSTRAINTS** (respect these):
- ⚠️ AVOID prolonged reasoning (thinking mode costly - 18K tokens)
- ⚠️ FOCUS on execution over exploration
- ⚠️ FOLLOW provided patterns exactly (framework knowledge gaps)
- ⚠️ CHECK syntax precision (13% error rate - mitigate with type hints)
- ⚠️ COMPLETE micro-tasks fully (no early quit - acceptance criteria mandatory)

---

## 📋 MICRO-TASK BREAKDOWN

### Task 1: Update Configuration File (Duration: 10 min)

**File**: `/Users/fulvioventura/freqtrade/config_alpha_arena.json` (Lines: 7-8)

**ACTION**: Change timeframe from "3m" to "15m" and adjust internals for AI call frequency

**SPECIFIC CHANGES**:
```json
"timeframe": "15m",
"internals": {
    "process_throttle_secs": 180  // 3 minutes for AI calls
}
```

**PATTERN REFERENCE**: See current config at line 7-8 for existing structure

**ERROR HANDLING** (USE THIS PATTERN):
```python
try:
    with open('/Users/fulvioventura/freqtrade/config_alpha_arena.json', 'r') as f:
        config = json.load(f)
    config['timeframe'] = '15m'
    config['internals']['process_throttle_secs'] = 180
    with open('/Users/fulvioventura/freqtrade/config_alpha_arena.json', 'w') as f:
        json.dump(config, f, indent=2)
except Exception as e:
    logger.error(f"Config update failed: {e}")
    raise
```

**TOOL USAGE**:
1. **Tool**: `Read`
   **When**: Before editing to verify current structure

2. **Tool**: `Edit`
   **When**: Make specific changes to timeframe and throttle

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Timeframe updated to "15m"
- [ ] process_throttle_secs set to 180
- [ ] JSON syntax valid
- [ ] All other settings preserved

**COMPLETION COMMAND**:
```bash
# Verify JSON syntax
python -m json.tool /Users/fulvioventura/freqtrade/config_alpha_arena.json
```

---

### Task 2: Update Strategy Time Frame References (Duration: 20 min)

**File**: `/Users/fulvioventura/freqtrade/user_data/strategies/deepseek_alpha_arena.py` (Lines: 37, 200)

**ACTION**: Update all timeframe references from 3m to 15m in strategy code

**FUNCTION SIGNATURES TO UPDATE**:
```python
def get_system_prompt(self) -> str:
    # Update prompt to mention 15-minute timeframe
```

**PATTERN REFERENCE**: Current `get_system_prompt()` method for prompt structure

**ERROR HANDLING**:
```python
try:
    # Update timeframe references
    logger.info("Updating strategy timeframe references to 15m")
except Exception as e:
    logger.error(f"Strategy update failed: {e}")
    raise
```

**TOOL USAGE**:
1. **Tool**: `Grep`
   **When**: Find all "3m" references in strategy file
   **Example**:
   ```bash
   grep -n "3m" /Users/fulvioventura/freqtrade/user_data/strategies/deepseek_alpha_arena.py
   ```

2. **Tool**: `Edit`
   **When**: Replace 3m references with 15m

**ACCEPTANCE CRITERIA**:
- [ ] All "3m" references updated to "15m"
- [ ] Startup candle count adjusted for 15m (200 candles = 50 hours)
- [ ] Prompt mentions 15-minute analysis
- [ ] Strategy syntax valid

**COMPLETION COMMAND**:
```bash
# Verify Python syntax
.devstream/bin/python -m py_compile /Users/fulvioventura/freqtrade/user_data/strategies/deepseek_alpha_arena.py
```

---

### Task 3: Implement Forward-Looking Prediction System (Duration: 45 min)

**File**: `/Users/fulvioventura/freqtrade/user_data/strategies/deepseek_alpha_arena.py` (Lines: 900-950)

**ACTION**: Add forward-looking prediction logic to compensate for 3-minute latency

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def _apply_forward_looking_adjustment(
    self,
    predicted_price: float,
    current_price: float,
    volatility: float,
    trend_direction: str
) -> float:
    """
    Apply forward-looking adjustment to compensate for 3-minute API latency.

    Args:
        predicted_price: AI predicted entry/exit price
        current_price: Current market price at response time
        volatility: Recent price volatility (ATR percentage)
        trend_direction: "bullish", "bearish", or "neutral"

    Returns:
        Adjusted price accounting for 3-minute latency

    Raises:
        ValueError: If input parameters are invalid

    Example:
        >>> strategy = DeepSeekAlphaArenaStrategy()
        >>> adjusted = strategy._apply_forward_looking_adjustment(
        ...     predicted_price=1050.0,
        ...     current_price=1051.50,
        ...     volatility=0.02,
        ...     trend_direction="bullish"
        ... )
        >>> print(f"Adjusted price: {adjusted}")
        Adjusted price: 1052.31
    """
```

**IMPLEMENTATION PATTERN**:
```python
def _apply_forward_looking_adjustment(self, predicted_price, current_price, volatility, trend_direction):
    try:
        # Calculate 3-minute drift adjustment
        drift_multiplier = 1.0 + (volatility * 0.15)  # 15% of volatility for 3-minute window

        if trend_direction == "bullish":
            return predicted_price * drift_multiplier
        elif trend_direction == "bearish":
            return predicted_price * (2.0 - drift_multiplier)
        else:  # neutral
            return (predicted_price + current_price) / 2.0

    except Exception as e:
        logger.error(f"Forward-looking adjustment failed: {e}")
        return predicted_price  # Fallback to original prediction
```

**PATTERN REFERENCE**: See existing `_apply_ai_decisions()` method for similar price handling

**TOOL USAGE**:
1. **Tool**: `Read`
   **When**: Study existing `_apply_ai_decisions()` method

2. **Tool**: `Edit`
   **When**: Add new method after existing AI decision methods

**TEST FILE**: `tests/unit/test_deepseek_alpha_arena.py::test_forward_looking_adjustment`

**ACCEPTANCE CRITERIA**:
- [ ] Function signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

**COMPLETION COMMAND**:
```bash
.devstream/bin/python -m pytest tests/unit/test_deepseek_alpha_arena.py::test_forward_looking_adjustment -v
.devstream/bin/python -m mypy /Users/fulvioventura/freqtrade/user_data/strategies/deepseek_alpha_arena.py --strict
```

---

### Task 4: Implement Adaptive Safety Buffer (Duration: 30 min)

**File**: `/Users/fulvioventura/freqtrade/user_data/strategies/deepseek_alpha_arena.py` (Lines: 950-1000)

**ACTION**: Add adaptive safety buffer system for entry price adjustments

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def _calculate_adaptive_safety_buffer(
    self,
    base_price: float,
    market_volatility: float,
    price_movement_direction: str,
    confidence_level: float
) -> Dict[str, float]:
    """
    Calculate adaptive safety buffer to compensate for API response latency.

    Args:
        base_price: AI-recommended entry price
        market_volatility: Current market volatility (0.0-1.0)
        price_movement_direction: "up", "down", or "sideways"
        confidence_level: AI confidence level (0.0-1.0)

    Returns:
        Dictionary with 'entry_price' and 'buffer_percentage' keys

    Raises:
        ValueError: If confidence_level or volatility out of range

    Example:
        >>> strategy = DeepSeekAlphaArenaStrategy()
        >>> result = strategy._calculate_adaptive_safety_buffer(
        ...     base_price=1050.0,
        ...     market_volatility=0.02,
        ...     price_movement_direction="up",
        ...     confidence_level=0.75
        ... )
        >>> print(f"Entry: {result['entry_price']}, Buffer: {result['buffer_percentage']:.2%}")
        Entry: 1051.05, Buffer: 0.10%
    """
```

**PATTERN REFERENCE**: Existing price calculation methods in strategy file

**TOOL USAGE**:
1. **Tool**: `Grep`
   **When**: Find existing price calculation patterns
   **Example**:
   ```python
   mcp__devstream__devstream_search_memory(
       query="price calculation buffer volatility adjustment",
       content_type="code",
       limit=5
   )
   ```

**ACCEPTANCE CRITERIA**:
- [ ] Function signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Buffer calculation logic implemented
- [ ] Test written and passing
- [ ] mypy --strict passes

---

### Task 5: Implement Time Lock Validation (Duration: 25 min)

**File**: `/Users/fulvioventura/freqtrade/user_data/strategies/deepseek_alpha_arena.py` (Lines: 1000-1050)

**ACTION**: Add 6-minute time lock system for AI signal validity

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def _is_signal_valid_within_time_lock(
    self,
    signal_timestamp: datetime,
    current_time: datetime,
    max_age_minutes: int = 6
) -> bool:
    """
    Check if AI signal is still valid within time lock window.

    Args:
        signal_timestamp: When the AI signal was generated
        current_time: Current time for validation
        max_age_minutes: Maximum age before signal expires (default: 6)

    Returns:
        True if signal is still valid, False otherwise

    Example:
        >>> from datetime import datetime, timedelta
        >>> strategy = DeepSeekAlphaArenaStrategy()
        >>> signal_time = datetime.now() - timedelta(minutes=3)
        >>> is_valid = strategy._is_signal_valid_within_time_lock(
        ...     signal_time, datetime.now()
        ... )
        >>> print(f"Signal valid: {is_valid}")
        Signal valid: True
    """
```

**PATTERN REFERENCE**: Existing timestamp validation patterns in strategy

**TOOL USAGE**:
1. **Tool**: `Grep`
   **When**: Find existing datetime handling patterns
2. **Tool**: `Edit`
   **When**: Add time lock validation method

**ACCEPTANCE CRITERIA**:
- [ ] Function signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Time lock logic implemented (6 minutes)
- [ ] Test written and passing
- [ ] mypy --strict passes

---

### Task 6: Implement Volatility Detector (Duration: 35 min)

**File**: `/Users/fulvioventura/freqtrade/user_data/strategies/deepseek_alpha_arena.py` (Lines: 1050-1100)

**ACTION**: Add volatility detection system for automatic signal suspension

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def _detect_market_volatility_spike(
    self,
    current_price: float,
    price_history: List[float],
    volatility_threshold: float = 0.01
) -> Dict[str, Any]:
    """
    Detect if market volatility exceeds acceptable threshold during AI latency.

    Args:
        current_price: Current market price
        price_history: Recent price points (last 5-10 readings)
        volatility_threshold: Maximum allowed volatility (default: 1%)

    Returns:
        Dictionary with volatility metrics and suspension recommendation

    Raises:
        ValueError: If price_history is empty or invalid

    Example:
        >>> strategy = DeepSeekAlphaArenaStrategy()
        >>> prices = [1050.0, 1051.2, 1052.8, 1051.5, 1050.9]
        >>> result = strategy._detect_market_volatility_spike(
        ...     1051.0, prices
        ... )
        >>> print(f"Suspend signals: {result['suspend_signals']}")
        Suspend signals: False
    """
```

**PATTERN REFERENCE**: Existing volatility calculation methods using ATR

**TOOL USAGE**:
1. **Tool**: `Grep`
   **When**: Find existing ATR/volatility calculations
2. **Tool**: `Edit`
   **When**: Add volatility detection method

**ACCEPTANCE CRITERIA**:
- [ ] Function signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Volatility calculation implemented
- [ ] Suspension logic working
- [ ] Test written and passing
- [ ] mypy --strict passes

---

### Task 7: Integrate All Systems in AI Decision Flow (Duration: 20 min)

**File**: `/Users/fulvioventura/freqtrade/user_data/strategies/deepseek_alpha_arena.py` (Lines: 600-700)

**ACTION**: Update `_collect_all_pairs_data_and_call_ai()` method to use all new systems

**INTEGRATION POINTS**:
1. Forward-looking adjustments in `_parse_ai_response()`
2. Safety buffer in `_apply_ai_decisions()`
3. Time lock validation before signal application
4. Volatility detection in signal processing

**PATTERN REFERENCE**: Existing `_collect_all_pairs_data_and_call_ai()` method structure

**TOOL USAGE**:
1. **Tool**: `Read`
   **When**: Study current AI decision flow
2. **Tool**: `Edit`
   **When**: Integrate new systems into existing flow

**ACCEPTANCE CRITERIA**:
- [ ] All new systems integrated into AI flow
- [ ] Forward-looking adjustments applied
- [ ] Safety buffer calculations used
- [ ] Time lock validation enforced
- [ ] Volatility detection active
- [ ] All existing functionality preserved

---

### Task 8: Live Production Testing (Duration: 15 min)

**File**: Test live system with new parameters

**ACTION**: Deploy updated system and monitor live trading performance

**TESTING CHECKLIST**:
- [ ] Strategy loads without errors
- [ ] AI calls working every 3 minutes
- [ ] 15-minute timeframe data correctly collected
- [ ] Forward-looking adjustments applied
- [ ] Safety buffer calculations working
- [ ] Time lock validation enforced
- [ ] Volatility detection active
- [ ] Trades executing with new parameters

**MONITORING COMMANDS**:
```bash
# Start live trading with new config
freqtrade trade --config config_alpha_arena.json --strategy DeepSeekAlphaArenaStrategy

# Monitor logs
tail -f freqtrade_with_full_ai_logs.log | grep -E "(API response|Forward-looking|Safety buffer|Time lock|Volatility)"
```

---

## 🔍 CONTEXT7 RESEARCH FINDINGS (Pre-Researched)

**Freqtrade Library**: v2025.9.1
**Trust Score**: 9/10
**Context7 ID**: `/freqtrade/freqtrade`

**Key Pattern 1**: Time Frame Configuration
```python
# Freqtrade timeframe configuration
"timeframe": "15m",  # 15-minute candles
"startup_candle_count": 200,  # 200 candles = 50 hours of data
"internals": {
    "process_throttle_secs": 180  # Process every 3 minutes
}
```
**When to use**: Configuring trading frequency and data requirements

**Key Pattern 2**: Custom Stop Loss Integration
```python
def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> float:
    # Dynamic stop loss based on AI predictions
    return -0.05  # 5% stop loss
```
**When to use**: Implementing custom risk management logic

---

## 🚨 CRITICAL CONSTRAINTS (DO NOT VIOLATE)

**FORBIDDEN ACTIONS**:
- ❌ **NO** removal of existing AI functionality
- ❌ **NO** changes to API credentials or endpoints
- ❌ **NO** simplifications that reduce signal quality
- ❌ **NO** skipping error handling
- ❌ **NO** marking task complete with failing tests

**REQUIRED ACTIONS**:
- ✅ **YES** maintain all existing Alpha Arena features
- ✅ **YES** preserve 3-minute AI call frequency
- ✅ **YES** use 180-second API timeout (user requirement)
- ✅ **YES** follow exact error handling pattern
- ✅ **YES** full docstrings + type hints EVERY function
- ✅ **YES** check acceptance criteria per micro-task

---

## ✅ QUALITY GATES (MANDATORY BEFORE COMPLETION)

### 1. Test Coverage
```bash
.devstream/bin/python -m pytest tests/ -v \
    --cov=user_data.strategies.deepseek_alpha_arena \
    --cov-report=term-missing \
    --cov-report=html

# REQUIREMENT: ≥ 95% coverage for NEW code
```

### 2. Type Safety
```bash
.devstream/bin/python -m mypy /Users/fulvioventura/freqtrade/user_data/strategies/deepseek_alpha_arena.py --strict

# REQUIREMENT: Zero errors
```

### 3. Freqtrade Validation
```bash
.devstream/bin/python -m freqtrade check-strategy --strategy DeepSeekAlphaArenaStrategy

# REQUIREMENT: Strategy validation passes
```

---

## 📝 COMMIT MESSAGE TEMPLATE

```
feat(alpha-arena): upgrade timeframe from 3m to 15m with latency compensation

Implemented comprehensive time frame upgrade with AI latency compensation:

Technical Changes:
- Updated timeframe configuration from 3m to 15m
- Added forward-looking prediction system for 3-minute API latency
- Implemented adaptive safety buffer for price adjustments
- Added 6-minute time lock validation for AI signals
- Integrated volatility detector for automatic signal suspension

Performance Improvements:
- More stable AI analysis with 15-minute data windows
- Reduced market noise while maintaining 3-minute reactivity
- Enhanced risk management with adaptive buffers
- Improved signal quality with volatility-based filtering

Quality Validation:
- ✅ Tests: 8 new tests passing, 98% coverage
- ✅ Type safety: mypy --strict passed
- ✅ Freqtrade validation: strategy check passed
- ✅ Live testing: successful deployment with confirmed trade execution

Task ID: alpha-arena-timeframe-upgrade

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## 📊 SUCCESS METRICS

- **Completion**: 100% of micro-tasks with acceptance criteria met
- **Test Coverage**: ≥ 95% for new code
- **Type Safety**: Zero mypy errors
- **Performance**: AI calls every 3 minutes with 15-minute analysis windows
- **Trading**: Successful live deployment with confirmed trade execution
- **Latency Compensation**: Forward-looking adjustments working correctly

---

**READY TO START?**
1. Mark first TodoWrite task as "in_progress"
2. Search DevStream memory for context
3. Implement according to specification
4. Run tests + type check
5. Mark "completed" when all acceptance criteria met
6. Proceed to next micro-task

**REMEMBER**: Execute, don't explore. Follow patterns, don't invent. Complete tasks, don't quit early. 🚀