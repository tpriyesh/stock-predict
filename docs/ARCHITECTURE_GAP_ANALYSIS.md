# Trading Signal Algorithm - Architecture Gap Analysis

## Executive Summary

This document provides a comprehensive analysis of the trading signal algorithm architecture, identifying gaps, edge cases, error conditions, and potential money-losing scenarios.

**Overall Assessment:** The system is well-designed with multiple safety layers, but has several critical gaps that could lead to financial losses.

---

## 1. Signal Generation Gaps

### 1.1 SELL Signals Disabled - Opportunity Cost
**File:** `src/models/enhanced_scoring.py`
**Risk Level:** MEDIUM

**Issue:** SELL signals are disabled due to 28.6% accuracy (inverted).

**Impact:**
- Cannot profit from down markets
- System is long-only, missing ~50% of potential trades
- In bear markets, system continuously tries failing BUY signals

**Gap:** No short-selling or inverse ETF strategy.

### 1.2 Model Accuracy Ceiling
**Risk Level:** HIGH

**Issue:** BUY accuracy is 57.6%-68%. Even at 68%, 32% of trades lose.

**Impact:**
- With 2% risk per trade, expected value is marginally positive
- Transaction costs (0.03% round-trip) can erase thin edges
- No room for execution slippage

**Gap:** No dynamic position sizing based on signal confidence.

### 1.3 Model Agreement False Confidence
**Risk Level:** MEDIUM

**Issue:** Models may be correlated (using similar underlying data).

**Impact:** "Agreement" may not indicate independent confirmation.

**Gap:** No correlation analysis between model predictions.

### 1.4 Stale Signal Execution
**File:** `agent/orchestrator.py`
**Risk Level:** HIGH

**Issue:** Signal TTL may be too long for fast-moving markets.

**Impact:** 5-minute old signal in a gap-down scenario is worthless.

### 1.5 News Sentiment Reliability
**Risk Level:** MEDIUM

**Issues:**
- News APIs can fail silently
- No news available overnight/weekends
- Sentiment analysis misinterprets sarcasm/context
- 4-hour old news is stale for intraday

---

## 2. Execution Engine Edge Cases

### 2.1 Double-Sell Race Condition (CRITICAL)
**File:** `agent/orchestrator.py` - `_exit_position()`
**Risk Level:** CRITICAL

**Issue:** Between checking SL status and placing exit order, SL could trigger.

**Impact:** Double sell creates unintended short position.

**Gap:** No locking mechanism or idempotency key for exits.

### 2.2 Partial Fill Handling
**Risk Level:** MEDIUM

**Issues:**
- Cancel can fail, leaving unexpected late fills
- Position tracking may be inconsistent with broker
- SL order quantity mismatch if partial fill occurs

### 2.3 SL Order Failure - Position Unprotected
**Risk Level:** CRITICAL

**Issue:** If SL placement fails AND exit fails, position has NO stop loss.

**Impact:** Unlimited downside risk.

### 2.4 Trailing Stop Activation Gap
**File:** `src/execution/trailing_stop.py`
**Risk Level:** MEDIUM

**Issue:** Default activation at 1R. Price can reverse before trailing activates.

**Impact:** Profitable trades become breakeven/losers.

### 2.5 Order Circuit Breaker Pause
**Risk Level:** MEDIUM

**Issue:** Circuit breaker prevents NEW entries but existing positions are still at risk if broker has issues.

---

## 3. Risk Management Gaps

### 3.1 Portfolio Value Calculation Failure
**File:** `risk/manager.py`
**Risk Level:** HIGH

**Issues:**
- Uses cached value which may be stale
- Cached unrealized P&L can be significantly wrong in volatile markets
- Kill switch on API failure may be too aggressive

### 3.2 Daily Loss Limit Bypass
**Risk Level:** MEDIUM

**Issue:** Unrealized P&L can hide losses until realized.

**Impact:** Trading may continue when it should stop.

### 3.3 Gap Risk Check Incomplete
**Risk Level:** HIGH

**Issues:**
- Only checks at entry, not for existing positions
- Gap threshold (3%) may be too wide
- No adjustment for gap-through-stop scenarios

### 3.4 Circuit Breaker Proximity
**Risk Level:** MEDIUM

**Issue:** Stock hitting circuit after entry means exit may be impossible.

**Gap:** No monitoring for circuit limit approach.

### 3.5 Position Sizing with Stale Funds
**Risk Level:** MEDIUM

**Issue:** If funds API fails, position sizing uses incorrect values.

**Impact:** Over-leveraging or missed trades.

---

## 4. Broker Integration Issues

### 4.1 Rate Limiting During Exit
**Risk Level:** HIGH

**Issue:** Rate limit hit during position exit = STUCK POSITION.

**Gap:** No prioritization for exit orders.

### 4.2 Token Expiry During Trading
**Risk Level:** HIGH

**Issues:**
- Token check interval (30 min) means undetected expiry
- Emergency exit may fail if token is expired
- No automatic token refresh

### 4.3 Broker Position Mismatch
**Risk Level:** MEDIUM

**Issue:** Orphaned positions are only LOGGED, not automatically managed.

**Impact:** No SL protection for orphaned positions.

### 4.4 SL Modify Race Condition
**Risk Level:** MEDIUM

**Issue:** Time window between cancel and new SL order = UNPROTECTED.

### 4.5 Unknown Order Statuses
**Risk Level:** MEDIUM

**Issue:** Unknown statuses default to PENDING, causing incorrect handling.

---

## 5. Data Provider Issues

### 5.1 yfinance Rate Limiting
**Risk Level:** HIGH (paper trading)

**Issues:**
- No official rate limit, can be blocked anytime
- Yahoo may return stale/incorrect data
- No data validation for reasonableness

### 5.2 Price Data Staleness
**Risk Level:** HIGH

**Issue:** Cached data can be up to 5 minutes old.

**Impact:** In fast markets, stale prices cause bad decisions.

### 5.3 Corporate Action Detection
**Risk Level:** HIGH

**Issue:** 40% overnight price change detection may miss stock splits.

**Impact:** Signals based on incorrect price data.

### 5.4 Volume Check Bypass
**Risk Level:** MEDIUM

**Issue:** Volume check failure doesn't prevent trading, just logs warning.

---

## 6. Error Handling Gaps

### 6.1 Circuit Breaker State Persistence
**File:** `utils/error_handler.py`
**Risk Level:** MEDIUM

**Issue:** Circuit breaker state is in-memory only.

**Impact:** Restart clears circuit breaker, allowing requests to failing service.

### 6.2 Exception Categorization False Positives
**Risk Level:** LOW

**Issue:** String matching for categorization may misclassify errors.

### 6.3 Retry During Market Close
**Risk Level:** MEDIUM

**Issue:** Retry logic doesn't consider market hours.

**Impact:** Retries wasting API quota after market close.

---

## 7. Timing & Race Conditions

### 7.1 Position Check Interval
**Risk Level:** HIGH

**Issue:** 60-second interval between position checks.

**Impact:** SL breach may not be detected for up to 60 seconds.

### 7.2 Signal Refresh vs Position Check
**Risk Level:** MEDIUM

**Issue:** Signal refresh (5 min) may be slow for changing conditions.

### 7.3 Shutdown Timeout
**Risk Level:** HIGH

**Issue:** 120-second shutdown timeout may not be enough for all exits.

**Impact:** Positions left open after shutdown.

### 7.4 SL Verification Grace Period
**Risk Level:** MEDIUM

**Issue:** 30-second grace period for stuck SL orders.

**Impact:** Additional losses during grace period.

---

## 8. Gap & Volatility Scenarios

### 8.1 Gap-Down Through Stop Loss
**Risk Level:** CRITICAL

**Scenario:** Stock opens 5% down, SL was at 2%.

**Impact:**
- SL triggers at open price (much worse than expected)
- 2% planned loss becomes 5% actual loss
- No gap handler in exit path

### 8.2 Volatility Spike
**Risk Level:** HIGH

**Issue:** ATR-based stops become too wide in high volatility.

**Impact:** Larger than expected losses.

### 8.3 Liquidity Crisis
**Risk Level:** HIGH

**Issue:** Wide bid-ask spreads during stress.

**Impact:**
- Market orders fill at terrible prices
- SL orders trigger but can't exit

### 8.4 Flash Crash
**Risk Level:** CRITICAL

**Issue:** Price drops and recovers within seconds.

**Impact:** SL triggers at bottom, position exits at loss, price recovers.

---

## 9. Paper vs Live Discrepancies

### 9.1 SL Trigger Simulation
**Risk Level:** HIGH

**Issue:** Paper broker checks SL on `get_positions()` call only.

**Impact:** Paper trading may not trigger SL as expected in live.

### 9.2 Slippage Simulation
**Risk Level:** MEDIUM

**Issue:** Paper uses 0.1% slippage. Live can be much worse.

### 9.3 Partial Fill Simulation
**Risk Level:** MEDIUM

**Issue:** Paper rarely simulates partial fills realistically.

### 9.4 Order Rejection
**Risk Level:** MEDIUM

**Issue:** Paper rarely rejects orders. Live broker rejects for many reasons.

---

## 10. Configuration & Operational Risks

### 10.1 HOLDING Phase Unreachable
**Risk Level:** LOW

**Issue:** `entry_window_end == square_off_start == 14:30`.

**Impact:** HOLDING phase logic never executes.

### 10.2 Hardcoded Holidays
**Risk Level:** MEDIUM

**Issue:** NSE holidays hardcoded for 2026.

**Impact:** Trading on holiday if list not updated.

### 10.3 Capital Not Validated
**Risk Level:** MEDIUM

**Issue:** No validation that broker has required capital.

### 10.4 Max Positions = 3
**Risk Level:** LOW

**Issue:** Too few positions for diversification.

---

## 11. Critical Bug Summary

| Severity | Issue | File | Potential Loss |
|----------|-------|------|----------------|
| CRITICAL | Double-sell race condition | orchestrator.py | Unlimited (short position) |
| CRITICAL | SL failure + exit failure | orchestrator.py | Unlimited downside |
| CRITICAL | Gap-down through SL | Multiple | 2x-3x expected loss |
| CRITICAL | Flash crash SL trigger | Multiple | Opportunity loss + actual loss |
| HIGH | Token expiry during trade | orchestrator.py | Cannot exit positions |
| HIGH | Rate limit during exit | broker/*.py | Stuck position |
| HIGH | Stale signal execution | orchestrator.py | Bad entry prices |
| HIGH | 60s position check interval | orchestrator.py | Late SL detection |
| HIGH | yfinance blocking | paper.py | No paper trading data |
| MEDIUM | Partial fill handling | orchestrator.py | SL quantity mismatch |
| MEDIUM | SL modify race | orchestrator.py | Brief unprotected period |
| MEDIUM | Model accuracy ceiling | enhanced_scoring.py | 32% trades lose |
| MEDIUM | SELL signals disabled | enhanced_scoring.py | Missed opportunities |

---

## 12. Recommendations

### Immediate (Critical)

1. **Add exit idempotency key** - Prevent double-sell with unique key per exit attempt
2. **Implement gap handler at exit** - Detect gap-through and adjust expectations
3. **Add emergency WebSocket price feed** - Backup for LTP failures
4. **Implement SL verification loop** - Continuous verification, not just on position check

### Short-term (High)

5. **Add exit order priority** - Bypass rate limits for exit orders
6. **Implement token health heartbeat** - Check every 5 min, not 30 min
7. **Add correlation analysis** - Detect model agreement independence
8. **Implement volatility-adjusted stops** - Tighter in high volatility

### Medium-term (Medium)

9. **Add short-selling strategy** - Profit from down markets
10. **Implement circuit limit monitoring** - Warn before hitting limits
11. **Add liquidity scoring** - Skip illiquid stocks
12. **Implement confidence-based sizing** - Size by signal quality

### Operational

13. **Daily pre-flight checks** - Verify broker connectivity, token, capital
14. **Real-time P&L dashboard** - Monitor exposure continuously
15. **Alert escalation** - Critical alerts via call/SMS, not just Telegram

---

## 13. Money-Losing Scenario Examples

### Scenario 1: Gap Down Disaster
```
1. Enter RELIANCE at 2400, SL at 2352 (2%)
2. Overnight news: Market crash
3. Opens at 2280 (5% gap down)
4. SL triggers at 2280, not 2352
5. Loss: 5% instead of planned 2%
6. With 2% risk per trade, actual loss = 5% of capital
```

### Scenario 2: Rate Limit Trap
```
1. Enter 3 positions at 10:00 AM
2. At 10:30, market reverses
3. Try to exit all positions
4. Hit rate limit after 1st exit
5. Circuit breaker opens
6. Other 2 positions continue losing
7. By time circuit recovers, losses doubled
```

### Scenario 3: Token Expiry Cascade
```
1. Token expires at 10:15 (was issued at 8:00)
2. Not detected until 10:45 (30-min check)
3. 3 positions open, market dropping
4. Emergency exit fails (token expired)
5. Must manually login and exit
6. By time fixed, significant losses
```

### Scenario 4: Double Sell
```
1. Position: 100 shares TCS
2. SL at 3200, current price 3190
3. Orchestrator checks SL status - NOT filled
4. Before exit order placed, SL triggers at broker
5. Exit order goes through
6. Result: Sold 200 shares (100 short)
7. Short position has no SL protection
8. If price rises, unlimited loss potential
```

### Scenario 5: Flash Crash
```
1. Position: INFY at 1400, SL at 1372
2. Flash crash: Price drops to 1350 in 1 second
3. SL triggers at 1350
4. Price recovers to 1405 in next second
5. Position exited at loss
6. Missed the recovery move
7. Would have been profitable if held
```

---

*Document Version: 1.0*
*Generated: February 14, 2026*
*Author: Architecture Analysis*