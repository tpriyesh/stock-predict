# Architecture Risk Analysis - Money-Losing Scenarios

**Document Purpose**: Comprehensive analysis of all gaps, edge cases, and error conditions that could result in financial losses.

**Analysis Date**: February 14, 2026
**System Version**: 3.0 (7-Model Ensemble)

---

## Executive Summary

### Critical Risk Areas (Immediate Attention Required)

| Priority | Risk Area | Potential Loss | Mitigation Status |
|----------|-----------|----------------|-------------------|
| 🔴 CRITICAL | Data staleness (yfinance delays) | High | Partial |
| 🔴 CRITICAL | News API failure silent trading | Medium | Missing |
| 🔴 CRITICAL | Slippage on large orders | High | Partial |
| 🟡 HIGH | SL stuck during gap-down | High | Implemented |
| 🟡 HIGH | Choppy market false signals | Medium | Partial |
| 🟡 HIGH | Model ensemble degradation | Medium | Missing |
| 🟢 MEDIUM | Single provider data failure | Low | Implemented |

---

## 1. SIGNAL GENERATION RISKS

### 1.1 Data Staleness (CRITICAL)

**Problem**: yfinance data can be delayed 15-30 minutes during market hours

**Code Location**:
```python
# src/data/price_fetcher.py - uses yfinance for live prices
price_fetcher.fetch_prices(symbol, period="1y")  # Historical
get_current_price(symbol)  # Current (may be delayed)
```

**Money-Losing Scenario**:
1. System generates BUY signal based on 20-minute-old data
2. Stock has already moved up 2% in that time
3. You enter at inflated price
4. SL is calculated based on old ATR (tighter than should be)
5. Normal noise triggers SL, small loss

**Risk**: Entering at worse prices, tighter SLs than intended
**Frequency**: Daily during high volatility
**Potential Loss**: 0.5-2% per trade

**Current Mitigation**:
- Data freshness validation in validators.py (lines 215-248)
- But only warns, doesn't block trading

**Gap**: No real-time data source (WebSocket)

---

### 1.2 Model Ensemble Failure Modes (HIGH)

**Problem**: 7-model ensemble may have correlated failures

**Code Location**:
```python
# src/models/enhanced_scoring.py lines 475-677
# Each model wrapped in try-except, falls back to 0.5 (neutral)
```

**Money-Losing Scenarios**:

#### Scenario A: Cascading Model Failures
1. Physics engine fails (exception) → defaults to 0.5
2. Math engine fails → defaults to 0.5
3. Only base model + regime remain
4. Signal still generated with reduced confidence
5. Confidence calibration doesn't account for model failures
6. Trade taken with effectively 2-model signal, not 7

**Gap**: No penalty for failed models in confidence calculation

#### Scenario B: Macro Data Staleness
```python
# Line 588: macro_signal = self.macro_engine.get_sector_macro_signal(sector)
# Macro data may be hours/days old
```

1. Macro signal shows bullish for IT sector (based on yesterday's data)
2. Overnight USD/INR crashed (bad for IT exports)
3. System still gets bullish macro vote
4. Buys IT stock into headwinds

---

### 1.3 Confidence Calibration Errors (HIGH)

**Problem**: Calibrated confidence may not reflect true win rate

**Code Location**:
```python
# config/thresholds.py lines 195-247
# Calibration based on historical 57.6% BUY accuracy
# But market regimes change
```

**Money-Losing Scenario**:
1. Backtest showed 65% accuracy for high-confidence signals
2. Market enters unusual regime (elections, war, etc.)
3. Calibration no longer valid
4. System continues trading at same confidence levels
5. Actual accuracy drops to 40%
6. Series of losses

**Gap**: No real-time calibration validation or regime-aware confidence adjustment

---

## 2. SIGNAL FILTERING RISKS

### 2.1 Supertrend Lag (MEDIUM)

**Problem**: Supertrend is a lagging indicator

**Code Location**:
```python
# agent/signal_adapter.py lines 279-284
st_info = get_supertrend_signal(df, period=7, multiplier=2.5)
```

**Money-Losing Scenario**:
1. Stock reverses sharply
2. Supertrend still shows uptrend (lag)
3. Signal passes filter
4. Buy right at top
5. Trend reverses, immediate loss

**Risk**: Late entries, buying at local tops
**Mitigation**: ADX filter helps but doesn't eliminate

---

### 2.2 Volume Filter Timing (MEDIUM)

**Problem**: Volume check uses historical average, not current

**Code Location**:
```python
# agent/signal_adapter.py lines 401-424
avg_volume = df['volume'].tail(5).mean()  # Past 5 days
```

**Money-Losing Scenario**:
1. Stock has low historical volume (illiquid)
2. Today it has news, volume spikes 10x
3. Volume filter passes (using old average)
4. You enter large position
5. When trying to exit, low liquidity = high slippage
6. Can't exit at target price

**Gap**: No real-time volume validation at entry

---

### 2.3 Corporate Action Detection Limitations (MEDIUM)

**Code**:
```python
# agent/signal_adapter.py lines 426-449
# Detects >40% overnight jumps
```

**Gap**: 
- Only checks 40% threshold (misses smaller splits)
- No actual adjustment for split-adjusted data
- Dividend impact not considered

**Risk**: Trading on distorted price data

---

## 3. ORDER EXECUTION RISKS

### 3.1 Market Order Slippage (CRITICAL)

**Problem**: Market orders can have significant slippage on illiquid stocks

**Code Location**:
```python
# agent/orchestrator.py lines 448-455
order = Order(
    order_type=OrderType.MARKET,  # Market order!
    ...
)
```

**Money-Losing Scenario**:
1. Signal generated for illiquid stock
2. Market order placed for 1000 shares
3. Bid-ask spread is 2%
4. Fill price 1% worse than expected
5. SL calculated based on expected price
6. SL hit immediately due to slippage
7. Loss on what should be scratch trade

**Risk**: 0.5-3% slippage on illiquid stocks
**Mitigation**: None (no limit order option implemented)

---

### 3.2 Partial Fill Handling (MEDIUM)

**Code**:
```python
# agent/orchestrator.py lines 501-515
if actual_quantity < quantity:
    # Cancels remainder
    if actual_quantity <= 0:
        return False
```

**Gap**: If 50% fills and 50% pending:
- Cancel the pending
- Trade with half position
- Risk/reward now different
- But still uses same SL/target levels

**Risk**: Risk management assumptions broken

---

### 3.3 Order Book Verification Missing (HIGH)

**Gap**: No check of order book depth before placing orders

**Risk**: Large orders in thin markets move price against you

---

## 4. POSITION MONITORING RISKS

### 4.1 LTP Failure Cascade (CRITICAL)

**Code**:
```python
# agent/orchestrator.py lines 672-688
current_price = self.broker.get_ltp(symbol)
if not current_price or current_price <= 0:
    # Skip this cycle
    continue
```

**Money-Losing Scenario**:
1. Broker API starts failing (500 errors)
2. LTP returns 0 or None
3. System skips position checks (line 685: continue)
4. Stock drops 5% through SL
5. System doesn't know (still skipping checks)
6. By the time API recovers, you're down 8%
7. SL triggers far below intended level

**Risk**: Unchecked positions during API outages
**Mitigation**: Partial (5-failure alert)

---

### 4.2 SL Stuck Detection Race Condition (MEDIUM)

**Code**:
```python
# agent/orchestrator.py lines 768-796
# 30-second grace period
if elapsed >= 30:
    # Force market sell
```

**Gap**: In fast-moving markets:
1. Price gaps through SL
2. SL order doesn't trigger (common in India)
3. 30-second grace period
4. Price drops another 2%
5. Market sell executes even lower
6. Total loss 3-4% more than expected

---

### 4.3 Trailing Stop Calculation Errors (MEDIUM)

**Code**:
```python
# src/execution/trailing_stop.py lines 80-84
risk = abs(entry_price - original_stop)
if risk == 0:
    return current_stop  # No trailing!
```

**Gap**: If risk calculation is 0 (data error), trailing stops disabled silently.

---

### 4.4 Position Reconciliation Delays (MEDIUM)

**Code**:
```python
# agent/orchestrator.py lines 1158-1165
# Reconciles every 2 minutes
if elapsed < 120:
    return
```

**Gap**: 
1. Position closed externally (broker app)
2. System doesn't know for up to 2 minutes
3. Tries to modify SL on non-existent position
4. Error, but position already closed
5. P&L tracking may be off

---

## 5. STOP LOSS PROTECTION RISKS

### 5.1 SL Order Type Risk (CRITICAL)

**Code**:
```python
# agent/orchestrator.py lines 613-621
order_type=OrderType.SL_M,  # SL-M (market when triggered)
```

**Money-Losing Scenario**:
1. SL-M order placed
2. Gap down open (common in India)
3. SL triggers at market price
4. Fill is 3-4% below SL price
5. Loss much larger than planned

**Risk**: Gap-down slippage
**Mitigation**: None (SL-M chosen over SL-L for reliability)

**Alternative Risk with SL-L**:
- SL-L (limit) may not fill in crash
- Position remains open
- Potentially unlimited loss

**Trade-off**: Guaranteed fill vs guaranteed price

---

### 5.2 Trailing Stop Update Failures (MEDIUM)

**Code**:
```python
# agent/orchestrator.py lines 854-891
try:
    result = self.broker.modify_order(...)
    if result:
        # Verify
        order_book = self.broker.get_order_status(sl_order_id)
        # ... fallback to cancel+replace
```

**Gap**: If modify fails and cancel+replace fails:
1. Old SL remains active
2. Price has moved up
3. Stock reverses
4. Hit old SL (lower than it should be)
5. Give back more profit than necessary

---

## 6. DATA QUALITY RISKS

### 6.1 Yahoo Finance Data Issues (CRITICAL)

**Known Issues**:
- Delays during market hours (15-30 min)
- Missing data for some stocks
- Incorrect splits handling
- After-hours data included

**Impact**: All technical indicators calculated on bad data

**Code**:
```python
# src/data/price_fetcher.py - no alternative to yfinance
# Primary: yfinance
# Fallback: yfinance (same source!)
```

**Gap**: No truly independent data source

---

### 6.2 Corporate Action Handling (MEDIUM)

**Current**: Detects >40% overnight jumps
**Gap**:
- No automatic adjustment
- No dividend impact modeling
- No rights issue handling
- Bonus issues detected late

**Risk**: Trading on distorted technicals

---

### 6.3 Validation Non-Blocking (MEDIUM)

**Code**:
```python
# src/data/validators.py lines 191-193
if not is_valid:
    for error in errors:
        audit_log("VALIDATION_WARNING", ...)  # Just warns!
```

**Gap**: Data validation warnings don't block trading
**Risk**: Trading on invalid data

---

## 7. NEWS SENTIMENT RISKS

### 7.1 News API Failures (CRITICAL)

**Code**:
```python
# src/data/news_fetcher.py lines 142-143
if not all_articles and providers_tried == 0:
    logger.warning("No news providers available")
# Continues trading without news!
```

**Money-Losing Scenario**:
1. All news APIs fail (rate limits, outages)
2. System generates signal without news context
3. Major negative news released
4. System buys into bad news
5. Stock tanks immediately

**Risk**: Blind trading during news events
**Gap**: No option to pause trading if news unavailable

---

### 7.2 Sentiment Analysis Errors (MEDIUM)

**Code**:
```python
# Sentiment via OpenAI
# src/features/news_features.py
```

**Risks**:
- OpenAI API errors → no sentiment
- Sarcasm not detected
- Context misunderstanding
- Language translation errors

**Gap**: No sentiment validation or confidence score

---

### 7.3 News Latency (MEDIUM)

**Problem**: News APIs have 5-15 minute delay

**Risk**: Trade on "old" news that's already priced in

---

## 8. MARKET REGIME RISKS

### 8.1 Choppy Market Detection (HIGH)

**Code**:
```python
# src/models/enhanced_scoring.py lines 565-567
if regime_result.current_regime == MarketRegime.CHOPPY:
    warnings.append("[REGIME] Choppy market - reduce position size")
    # But still trades!
```

**Gap**: Warns but doesn't block trading in choppy markets
**Risk**: False signals in sideways markets → repeated small losses

---

### 8.2 Regime Transition Lag (MEDIUM)

**Problem**: HMM regime detection has lookback period

**Risk**: Slow to detect regime changes
**Result**: Trend-following signals in new ranging market

---

### 8.3 No Market-Wide Circuit Breaker Check (MEDIUM)

**Gap**: System doesn't check if NSE is halted
**Risk**: Orders sent during trading halt, rejected or queued badly

---

## 9. BROKER INTEGRATION RISKS

### 9.1 Token Expiry Mid-Trade (CRITICAL)

**Code**:
```python
# agent/orchestrator.py lines 310-332
if not self.broker.check_token_valid():
    # Emergency exit and shutdown
```

**Gap**: 
1. Token valid at position entry
2. Token expires while position open
3. Can't place SL modification
4. Can't exit position
5. Position unprotected

**Risk**: Unprotected positions

---

### 9.2 Rate Limit Handling (MEDIUM)

**Code**:
```python
# broker/upstox.py lines 152-159
if response.status_code == 429:
    retry_after = int(response.headers.get("Retry-After", 60))
    raise RateLimitException(...)
```

**Gap**: Hard backoff, no prioritization
**Risk**: Critical orders (SL) delayed due to rate limits on non-critical requests

---

### 9.3 Partial Broker Outages (MEDIUM)

**Gap**: No health score tracking
**Risk**: Erratic behavior during partial API outages

---

## 10. RISK MANAGEMENT RISKS

### 10.1 Unrealized P&L Calculation (HIGH)

**Code**:
```python
# risk/manager.py lines 117-121
pnl = self.broker.get_pnl()
unrealized = pnl.unrealized
```

**Gap**: 
1. Broker API returns cached/delayed P&L
2. Actual losses larger than shown
3. Kill switch doesn't trigger
4. Continue trading while already below stop

---

### 10.2 Portfolio Value API Failure (MEDIUM)

**Code**:
```python
# risk/manager.py lines 136-169
if self._portfolio_api_failures >= 3:
    # Kill switch
```

**Gap**: After 3 failures, kills trading
**But**: What if position is open when this happens?
**Risk**: Can't exit, can't monitor

---

### 10.3 Concentration Risk (MEDIUM)

**Current**: Max 3 positions, 25% per position
**Gap**: No correlation check
**Risk**: All 3 positions in same sector (banking), sector crash = max loss

---

## 11. COMPOUND RISK SCENARIOS

### Scenario A: The Perfect Storm (Worst Case)

1. **09:30**: Market opens volatile
2. **09:31**: yfinance delays (data 20 min old)
3. **09:32**: Signal generated on stale data
4. **09:33**: News APIs down (missed negative news)
5. **09:34**: Market order placed, 1% slippage
6. **09:35**: Real data arrives, shows downtrend
7. **09:36**: SL placed
8. **09:37**: Broker API issues begin
9. **09:38**: Position monitoring skipping checks
10. **09:45**: Stock gaps down through SL
11. **09:46**: SL stuck, not triggering
12. **09:47**: Manual intervention required but unaware
13. **10:00**: Position down 8%

**Total Loss**: 8-10% on single trade

---

### Scenario B: Death by a Thousand Cuts

1. Choppy market (not detected)
2. 5 false signals in a row
3. Each loses 0.5-1% (slippage + whipsaw)
4. Total daily loss: 3-5%
5. No single big loss, but steady bleeding

---

## 12. MISSING SAFEGUARDS

### Critical Missing Features

| Feature | Risk Without It | Priority |
|---------|-----------------|----------|
| Real-time data source | Slippage, stale signals | 🔴 |
| Order book depth check | Market impact | 🔴 |
| News mandatory option | Blind trading | 🔴 |
| Live calibration tracking | Wrong confidence | 🟡 |
| Correlation matrix | Concentration risk | 🟡 |
| Pre-market risk report | Unknown exposure | 🟡 |
| Maximum daily trades | Overtrading | 🟡 |
| Volatility scaling | Wrong position sizes | 🟡 |
| Broker health score | Erratic execution | 🟡 |

---

## 13. RECOMMENDATIONS

### Immediate Actions (Before Trading)

1. **Add mandatory news check**
   ```python
   # config option: REQUIRE_NEWS=True
   if not news_available and REQUIRE_NEWS:
       skip_signal()
   ```

2. **Add real-time data validation**
   ```python
   if data_age > 5_minutes:
       block_trading()
   ```

3. **Add order book depth check**
   ```python
   if order_size > 10% of avg_volume:
       reject_or_reduce()
   ```

4. **Model failure penalty**
   ```python
   if failed_models > 2:
       confidence *= 0.8
   ```

### Short-term Improvements

1. WebSocket data feed integration
2. Live confidence calibration tracking
3. Sector correlation limits
4. Volatility-based position sizing
5. Pre-trade risk report

---

## 14. CONCLUSION

### System Strengths
- ✅ Good error handling and logging
- ✅ Crash recovery works
- ✅ SL stuck detection
- ✅ Position reconciliation
- ✅ Kill switch implemented

### Critical Gaps
- ❌ Data staleness not blocked
- ❌ News failure doesn't pause trading
- ❌ No real-time data
- ❌ Slippage not controlled
- ❌ Model failures don't reduce confidence

### Risk Rating: MEDIUM-HIGH

The system is functional but has several gaps that could result in:
- Slippage losses: 0.5-2% per trade
- False signals in choppy markets: 3-5% daily
- Gap-down losses: 5-10% (rare but possible)
- Data quality losses: 1-3% per trade

**Recommended**: Paper trade for 2 weeks, monitor all gaps, fix critical ones before live trading.

---

*End of Analysis*
