# Critical Fixes Required - Trading System

**Document**: Prioritized list of fixes to prevent money loss  
**Version**: 3.0  
**Date**: February 14, 2026

---

## 🔴 CRITICAL (Fix Before Live Trading)

### FIX-001: Data Staleness Blocking
**Risk**: Trading on delayed data causes bad entries and wrong SL levels  
**Potential Loss**: 0.5-2% per trade

**Implementation**:
```python
# src/data/price_fetcher.py
class PriceFetcher:
    MAX_DATA_AGE_SECONDS = 300  # 5 minutes
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        price, timestamp = self._get_cached_or_fetch(symbol)
        
        if price and timestamp:
            age = (datetime.now() - timestamp).total_seconds()
            if age > self.MAX_DATA_AGE_SECONDS:
                logger.error(f"Data too old: {age}s for {symbol}")
                alert_error("STALE_DATA", f"{symbol}: {age}s old")
                return None  # Block stale data
        
        return price
```

**Config**:
```bash
# .env
MAX_DATA_AGE_SECONDS=300
BLOCK_TRADING_ON_STALE_DATA=true
```

---

### FIX-002: Mandatory News Check
**Risk**: Trading without news context = blind trading  
**Potential Loss**: 2-5% on news events

**Implementation**:
```python
# agent/signal_adapter.py
REQUIRE_NEWS = os.getenv("REQUIRE_NEWS", "true").lower() == "true"
MIN_NEWS_ARTICLES = int(os.getenv("MIN_NEWS_ARTICLES", "2"))

def _passes_filters(self, signal: TradeSignal) -> bool:
    # ... existing filters ...
    
    if REQUIRE_NEWS:
        news = self._news_fetcher.fetch_for_symbol(signal.symbol, hours=4)
        if len(news) < MIN_NEWS_ARTICLES:
            logger.warning(f"{signal.symbol}: Insufficient news ({len(news)} < {MIN_NEWS_ARTICLES})")
            return False
```

---

### FIX-003: Model Failure Confidence Penalty
**Risk**: Trading with fewer models than expected  
**Potential Loss**: Reduced accuracy, more false signals

**Implementation**:
```python
# src/models/enhanced_scoring.py
def score_stock(...):
    failed_models = 0
    
    # Count failures
    if base_score == 0.5:  # Default on failure
        failed_models += 1
    if physics_score == 0.5:
        failed_models += 1
    # ... etc
    
    # Apply penalty
    if failed_models > 0:
        penalty = 0.05 * failed_models  # 5% per failed model
        confidence = max(0.35, confidence - penalty)
        warnings.append(f"[RISK] {failed_models} models failed - confidence reduced")
```

---

### FIX-004: Order Size Limits (Liquidity Check)
**Risk**: Market impact on illiquid stocks  
**Potential Loss**: 1-3% slippage

**Implementation**:
```python
# risk/manager.py
def validate_trade(...):
    # ... existing checks ...
    
    # Liquidity check
    avg_volume = self._get_avg_volume(symbol)
    if quantity > avg_volume * 0.1:  # Max 10% of daily volume
        return RiskCheck(
            allowed=False,
            reason=f"Order size {quantity} > 10% of avg volume {avg_volume}"
        )
```

---

### FIX-005: SL-L vs SL-M Configuration
**Risk**: Gap-down slippage vs no-fill risk  
**Potential Loss**: 2-5% on gaps

**Implementation**:
```python
# config/trading_config.py
USE_SL_LIMIT_ORDERS = os.getenv("USE_SL_LIMIT_ORDERS", "false").lower() == "true"
SL_LIMIT_BUFFER_PCT = float(os.getenv("SL_LIMIT_BUFFER_PCT", "0.5"))  # 0.5% buffer

# agent/orchestrator.py
def _place_stop_loss(self, trade: TradeRecord) -> Optional[str]:
    if USE_SL_LIMIT_ORDERS:
        # SL-L: Limit order at SL price - buffer
        limit_price = trade.stop_loss * (1 - SL_LIMIT_BUFFER_PCT/100)
        order_type = OrderType.SL  # SL with limit
        price = limit_price
    else:
        # SL-M: Market when triggered (default)
        order_type = OrderType.SL_M
        price = None
```

---

## 🟡 HIGH (Fix Within 1 Week)

### FIX-006: Real-Time Confidence Calibration
**Risk**: Confidence doesn't reflect current market conditions

**Implementation**:
```python
# Track recent performance
class ConfidenceTracker:
    def __init__(self):
        self.recent_signals = []  # Last 20 signals
        
    def update(self, signal, result):
        self.recent_signals.append({
            'confidence': signal.confidence,
            'pnl': result.pnl,
            'timestamp': datetime.now()
        })
        
    def get_calibration_factor(self):
        if len(self.recent_signals) < 10:
            return 1.0
        
        # Compare predicted vs actual
        high_conf_signals = [s for s in self.recent_signals if s['confidence'] > 0.6]
        if not high_conf_signals:
            return 1.0
            
        win_rate = sum(1 for s in high_conf_signals if s['pnl'] > 0) / len(high_conf_signals)
        
        # If win rate < 50%, reduce confidence
        if win_rate < 0.5:
            return 0.8
        return 1.0
```

---

### FIX-007: Choppy Market Trading Block
**Risk**: False signals in sideways markets

**Implementation**:
```python
# agent/signal_adapter.py
def _passes_filters(self, signal: TradeSignal) -> bool:
    # ... existing filters ...
    
    # Block choppy markets
    if signal.regime == 'choppy' and signal.regime_stability < 0.6:
        logger.info(f"{signal.symbol}: BLOCKED - Choppy market")
        return False
```

---

### FIX-008: Broker API Health Score
**Risk**: Trading during API degradation

**Implementation**:
```python
# broker/base.py
class BrokerHealthMonitor:
    def __init__(self):
        self.failure_count = 0
        self.last_failure = None
        
    def record_success(self):
        self.failure_count = max(0, self.failure_count - 1)
        
    def record_failure(self):
        self.failure_count += 1
        self.last_failure = datetime.now()
        
    def is_healthy(self):
        return self.failure_count < 3
        
# In orchestrator
def _check_positions(self):
    if not self.broker.health_monitor.is_healthy():
        logger.error("Broker unhealthy - pausing position checks")
        return
```

---

### FIX-009: Maximum Daily Trades Limit
**Risk**: Overtrading in choppy markets

**Implementation**:
```python
# config/trading_config.py
MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", "5"))

# risk/manager.py
def can_trade(self):
    if self.trades_today >= MAX_DAILY_TRADES:
        return False, f"Max daily trades reached: {MAX_DAILY_TRADES}"
```

---

### FIX-010: Position Correlation Check
**Risk**: All positions in same sector

**Implementation**:
```python
# risk/manager.py
def validate_trade(self, symbol, ...):
    positions = self.broker.get_positions()
    
    # Get sectors
    new_sector = get_sector(symbol)
    existing_sectors = [get_sector(p.symbol) for p in positions]
    
    # Check correlation
    if existing_sectors.count(new_sector) >= 2:
        return RiskCheck(
            allowed=False,
            reason=f"Max 2 positions per sector. Already have 2 in {new_sector}"
        )
```

---

## 🟢 MEDIUM (Fix Within 1 Month)

### FIX-011: Pre-Market Risk Report
**Risk**: Unknown overnight exposure

**Implementation**:
```python
# agent/orchestrator.py
def _handle_pre_market(self):
    # Generate risk report before trading
    report = {
        'market_regime': self._detect_market_regime(),
        'vix_level': self._get_vix(),
        'overnight_gaps': self._scan_overnight_gaps(),
        'news_sentiment': self._get_market_news_sentiment(),
    }
    
    if report['vix_level'] > 25:
        alert_warning("High volatility detected - reducing position size")
```

---

### FIX-012: Volatility-Based Position Sizing
**Risk**: Wrong position sizes for volatility

**Implementation**:
```python
# risk/manager.py
def calculate_position_size(self, entry_price, stop_loss, atr_pct):
    base_qty, _ = self._base_position_calc(entry_price, stop_loss)
    
    # Volatility scaling
    if atr_pct > 4:  # High volatility
        return int(base_qty * 0.5)  # Half size
    elif atr_pct > 3:
        return int(base_qty * 0.75)
    
    return base_qty
```

---

### FIX-013: Weekend/Overnight Position Limit
**Risk**: Gap risk over weekends

**Implementation**:
```python
# agent/orchestrator.py
def _handle_entry_window(self):
    # Check if Friday afternoon
    if self._is_friday_afternoon():
        if len(self.active_trades) >= 1:  # Already have position
            logger.info("Friday afternoon - no new positions")
            return  # Skip new entries
```

---

### FIX-014: Automated Data Quality Score
**Risk**: Gradual data degradation unnoticed

**Implementation**:
```python
# src/data/quality_monitor.py
class DataQualityMonitor:
    def __init__(self):
        self.quality_score = 1.0
        
    def check_quality(self, df):
        checks = {
            'completeness': self._check_completeness(df),
            'timeliness': self._check_timeliness(df),
            'consistency': self._check_consistency(df),
        }
        
        self.quality_score = sum(checks.values()) / len(checks)
        
        if self.quality_score < 0.7:
            alert_error("DATA_QUALITY", f"Score: {self.quality_score}")
```

---

## Implementation Priority Matrix

| Fix | Impact | Effort | Priority | Risk Reduction |
|-----|--------|--------|----------|----------------|
| FIX-001 | High | Low | 🔴 | 2% per trade |
| FIX-002 | High | Low | 🔴 | 3% on news events |
| FIX-003 | Medium | Low | 🔴 | 5% confidence accuracy |
| FIX-004 | High | Medium | 🔴 | 1-3% slippage |
| FIX-005 | High | Low | 🔴 | 2-5% gap risk |
| FIX-006 | Medium | High | 🟡 | Long-term accuracy |
| FIX-007 | Medium | Low | 🟡 | 0.5% daily bleeding |
| FIX-008 | Medium | Medium | 🟡 | Prevents erratic behavior |
| FIX-009 | Low | Low | 🟡 | Prevents overtrading |
| FIX-010 | Medium | Medium | 🟡 | Reduces concentration |

---

## Testing Strategy

### Unit Tests Required
```python
# tests/test_critical_fixes.py

def test_stale_data_blocked():
    fetcher = PriceFetcher()
    fetcher._cache.set("RELIANCE", (100.0, datetime.now() - timedelta(minutes=10)))
    result = fetcher.get_current_price("RELIANCE")
    assert result is None

def test_news_required():
    adapter = SignalAdapter()
    with patch.object(adapter._news_fetcher, 'fetch_for_symbol', return_value=[]):
        signal = adapter.get_single_signal("RELIANCE")
        assert signal is None  # Blocked due to no news

def test_model_failure_penalty():
    engine = EnhancedScoringEngine()
    # Simulate failures
    score = engine.score_stock(..., force_failures=['physics', 'math'])
    assert score.confidence < 0.6  # Penalty applied
```

### Integration Tests Required
```python
# tests/test_integration_safety.py

def test_end_to_end_stale_data():
    """Full system test with stale data"""
    with freeze_time("2026-02-14 10:00:00"):
        # Data from 9:30
        system.run()
        assert_no_trades_executed()

def test_sl_stuck_detection():
    """Verify SL stuck detection works"""
    broker.simulate_sl_stuck("RELIANCE")
    time.sleep(35)
    assert_position_exited_via_market_order()
```

---

## Deployment Checklist

Before each fix:
- [ ] Unit test written and passing
- [ ] Integration test written and passing  
- [ ] Config option added to .env.example
- [ ] Documentation updated
- [ ] Paper trade for 1 day to verify

Critical fixes must be paper-tested for 3 days minimum.

---

*End of Document*
