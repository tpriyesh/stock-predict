# Stock Prediction Engine: Improvement Plan

## Executive Summary

Based on comprehensive external review from multiple advanced AI systems, this document outlines a prioritized roadmap to transform the current stock prediction engine from a sophisticated prototype into a **production-robust trading system**.

### Current State Assessment

**Strengths (Keep):**
- Modular architecture with 4 prediction engines
- Walk-forward validation (addresses look-ahead bias)
- Ensemble aggregation with multiple signal sources
- Bias corrections (10 identified issues addressed)
- Human-readable explanations via InsightsEngine
- Kelly Criterion position sizing

**Critical Gaps (Fix Immediately):**
1. Data leakage in feature engineering (scaling on full dataset)
2. LLMs predicting prices instead of analyzing sentiment
3. Static ensemble weights (should be learned)
4. No transaction costs in backtesting
5. Overlapping 5-day labels without purged CV
6. No monitoring/drift detection for live trading

---

## Priority Matrix

| Priority | Issue | Impact | Effort | Risk if Ignored |
|----------|-------|--------|--------|-----------------|
| P0 | Feature Engineering Leakage | Critical | Medium | Silent prediction failures |
| P0 | LLM Role Misalignment | Critical | Medium | Hallucinated predictions |
| P0 | Static Ensemble Weights | High | Medium | Suboptimal aggregation |
| P1 | Missing Transaction Costs | High | Low | Phantom edge (looks good, loses money) |
| P1 | Purged K-Fold CV | High | Medium | Overfitting on overlapping data |
| P1 | Data Quality Checks | High | Low | Garbage in, garbage out |
| P2 | Meta-Learner Stacker | Medium | High | Fixed weights ignore regime |
| P2 | Monitoring & Kill Switch | Medium | Medium | No early warning system |
| P2 | Log Returns | Medium | Low | Non-stationary inputs |
| P3 | Live Performance Tracker | Medium | High | Can't learn from mistakes |
| P3 | Macro Factor Integration | Low | Medium | Missing context |

---

## Phase 1: Critical Fixes (Week 1-2)

### 1.1 Fix Feature Engineering Leakage

**Problem:** Features (scaling, normalization) computed on full dataset before train/test split allows future information to leak into training.

**Solution:** Move all scaling INSIDE the walk-forward loop.

```python
# BEFORE (WRONG):
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)  # Sees ALL data including future
X_train, X_test = split(df_scaled)

# AFTER (CORRECT):
X_train_raw, X_test_raw = split(df)  # Split FIRST
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)  # Fit ONLY on training
X_test = scaler.transform(X_test_raw)  # Apply same params to test
```

**Files to Modify:**
- `src/insights/robust_predictor.py` - ML training pipeline
- `src/insights/unified_predictor.py` - ML component
- `src/insights/advanced_predictor.py` - Walk-forward implementation

**Implementation:**
```python
# New: SafeFeatureScaler class
class SafeFeatureScaler:
    """Prevents data leakage by fitting only on training data."""

    def __init__(self, method='robust'):
        self.method = method
        self.scaler = None
        self.is_fitted = False

    def fit_transform_train(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler on training data and transform."""
        if self.method == 'robust':
            self.scaler = RobustScaler()
        elif self.method == 'standard':
            self.scaler = StandardScaler()
        self.is_fitted = True
        return pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

    def transform_test(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using ONLY training parameters."""
        if not self.is_fitted:
            raise ValueError("Must fit on training data first!")
        return pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
```

---

### 1.2 Redefine LLM Role: Sentiment Only

**Problem:** LLMs are terrible calculators. Asking GPT to interpret "RSI: 45" wastes tokens and invites hallucination. ML models do this infinitely better.

**Solution:** LLMs analyze TEXT (news, earnings, sentiment). ML handles NUMBERS (prices, indicators).

**Current (Wrong):**
```python
prompt = f"""
Analyze this stock data:
RSI: {rsi}, MACD: {macd}, P/E: {pe}
Predict: BUY/SELL/HOLD with probability
"""
```

**New (Correct):**
```python
prompt = f"""
You are a financial news analyst. Do NOT analyze numbers or charts.

Recent news for {symbol}:
{news_headlines}

Market context:
- Sector: {sector} ({sector_sentiment})
- Overall market: {market_regime}

Output JSON:
{{
  "sentiment_score": float (-1.0 to 1.0),
  "sentiment_label": "bullish" | "bearish" | "neutral",
  "key_catalysts": ["list of positive drivers"],
  "key_risks": ["list of negative factors"],
  "news_quality": "high" | "medium" | "low",
  "reasoning": "2-3 sentence summary"
}}
"""
```

**Files to Modify:**
- `src/insights/unified_predictor.py` - LLM prompt templates
- `src/insights/multi_model_judge.py` - Judge prompts
- `src/insights/insights_engine.py` - AI narrative generation

**New Architecture:**
```
TRACK A: NUMBERS (ML Models)
├── Input: OHLCV, Technical Indicators, Fundamentals
├── Models: XGBoost, RandomForest, GradientBoosting
└── Output: probability_up (float)

TRACK B: TEXT (LLM Analysts)
├── Input: News headlines, Earnings summaries, Market commentary
├── Models: GPT-4, Gemini, Claude
└── Output: sentiment_score (-1 to +1)

AGGREGATION:
├── ML probability: 60% weight
├── LLM sentiment: 20% weight (converted to probability)
├── Technical rules: 15% weight
└── Calendar: 5% weight
```

---

### 1.3 Add Data Quality Checks

**Problem:** No validation that input data is sufficient/clean before prediction.

**Solution:** Add `DataHealthChecker` at pipeline start.

```python
@dataclass
class DataHealth:
    status: Literal["OK", "WARNING", "INSUFFICIENT"]
    issues: List[str]
    metrics: Dict[str, float]

class DataHealthChecker:
    """Validates data quality before prediction."""

    MIN_ROWS = 252  # 1 year minimum
    MAX_MISSING_PCT = 0.05  # 5% max missing
    MIN_VOLUME_AVG = 100000  # Liquidity check

    def check(self, df: pd.DataFrame) -> DataHealth:
        issues = []

        # Check row count
        if len(df) < self.MIN_ROWS:
            issues.append(f"Insufficient history: {len(df)} rows (need {self.MIN_ROWS})")

        # Check missing data
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > self.MAX_MISSING_PCT:
            issues.append(f"High missing data: {missing_pct:.1%}")

        # Check volume (liquidity)
        avg_volume = df['Volume'].mean()
        if avg_volume < self.MIN_VOLUME_AVG:
            issues.append(f"Low liquidity: avg volume {avg_volume:,.0f}")

        # Check for corporate action anomalies
        returns = df['Close'].pct_change()
        extreme_moves = (returns.abs() > 0.20).sum()
        if extreme_moves > 5:
            issues.append(f"Potential corporate actions: {extreme_moves} extreme moves")

        # Determine status
        if len(issues) == 0:
            status = "OK"
        elif any("Insufficient" in i for i in issues):
            status = "INSUFFICIENT"
        else:
            status = "WARNING"

        return DataHealth(
            status=status,
            issues=issues,
            metrics={
                'rows': len(df),
                'missing_pct': missing_pct,
                'avg_volume': avg_volume,
                'extreme_moves': extreme_moves
            }
        )
```

**Files to Create:**
- `src/data/health_checker.py`

**Integration Points:**
- `src/signals/generator.py` - Check before scoring
- All predictor classes - Skip if INSUFFICIENT

---

### 1.4 Replace Static Weights with Initial Calibration

**Problem:** Weights like `ML: 0.15, LLM: 0.15, Technical: 0.12` are arbitrary.

**Immediate Fix:** Use historical walk-forward accuracy to set initial weights.

```python
class AdaptiveWeightCalculator:
    """Calculate ensemble weights based on historical accuracy."""

    def __init__(self, lookback_predictions: int = 100):
        self.lookback = lookback_predictions
        self.component_accuracy: Dict[str, List[bool]] = defaultdict(list)

    def update(self, component: str, predicted_up: bool, actual_up: bool):
        """Track prediction accuracy per component."""
        correct = (predicted_up == actual_up)
        self.component_accuracy[component].append(correct)
        # Keep only recent predictions
        if len(self.component_accuracy[component]) > self.lookback:
            self.component_accuracy[component].pop(0)

    def get_weights(self) -> Dict[str, float]:
        """Calculate weights proportional to accuracy."""
        accuracies = {}
        for component, results in self.component_accuracy.items():
            if len(results) >= 20:  # Minimum sample
                accuracies[component] = sum(results) / len(results)
            else:
                accuracies[component] = 0.5  # Default to neutral

        # Convert to weights (accuracy above 50% gets more weight)
        excess_acc = {k: max(0, v - 0.5) for k, v in accuracies.items()}
        total = sum(excess_acc.values()) or 1

        weights = {k: v / total for k, v in excess_acc.items()}

        # Ensure minimum weight
        for k in weights:
            weights[k] = max(0.05, weights[k])

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
```

---

## Phase 2: Robustness Improvements (Week 3-4)

### 2.1 Add Transaction Costs to Backtesting

**Problem:** Current backtesting ignores costs, creating phantom edges.

**Solution:** Add realistic cost model.

```python
@dataclass
class TransactionCosts:
    """Indian market transaction costs."""
    brokerage_pct: float = 0.0003  # 0.03% (discount broker)
    stt_pct: float = 0.001  # 0.1% STT on sell
    exchange_txn_pct: float = 0.0000345  # NSE charges
    gst_pct: float = 0.18  # 18% GST on brokerage
    stamp_duty_pct: float = 0.00015  # 0.015% stamp duty
    slippage_pct: float = 0.001  # 0.1% slippage estimate

    def total_round_trip(self, trade_value: float) -> float:
        """Calculate total cost for buy + sell."""
        buy_costs = (
            self.brokerage_pct +
            self.exchange_txn_pct +
            self.stamp_duty_pct +
            self.slippage_pct
        )
        sell_costs = (
            self.brokerage_pct +
            self.stt_pct +
            self.exchange_txn_pct +
            self.slippage_pct
        )
        gst = (self.brokerage_pct * 2) * self.gst_pct

        total_pct = buy_costs + sell_costs + gst
        return trade_value * total_pct

# Integration into backtester
class CostAwareBacktester:
    def simulate_trade(self, entry_price, exit_price, quantity):
        gross_pnl = (exit_price - entry_price) * quantity
        trade_value = entry_price * quantity
        costs = self.costs.total_round_trip(trade_value)
        net_pnl = gross_pnl - costs
        return net_pnl
```

**Files to Modify:**
- `src/models/backtester.py` - Add cost calculations
- `src/insights/robust_predictor.py` - Monte Carlo with costs

---

### 2.2 Implement Purged K-Fold Cross-Validation

**Problem:** 5-day forward labels overlap, causing data leakage between folds.

**Solution:** Purged K-Fold with embargo period.

```python
class PurgedKFold:
    """K-Fold CV with purging and embargo for overlapping labels."""

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame, y: pd.Series = None):
        """Generate train/test indices with purging."""
        n_samples = len(X)
        embargo = int(n_samples * self.embargo_pct)

        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples

            # Purge: Remove overlap region before test set
            train_end = max(0, test_start - 5)  # 5 = label horizon

            # Embargo: Additional gap after test set
            next_train_start = min(n_samples, test_end + embargo)

            train_indices = list(range(0, train_end)) + list(range(next_train_start, n_samples))
            test_indices = list(range(test_start, test_end))

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
```

**Files to Modify:**
- `src/insights/advanced_predictor.py` - Replace sklearn KFold
- `src/insights/robust_predictor.py` - ML training

---

### 2.3 Use Log Returns for Stationarity

**Problem:** Raw returns are asymmetric and non-additive.

**Solution:** Use log returns everywhere.

```python
# BEFORE:
df['returns'] = df['Close'].pct_change()

# AFTER:
df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

# Benefits:
# 1. Symmetric: +10% and -10% have equal magnitude
# 2. Additive: Multi-period return = sum of log returns
# 3. Better for statistical models (closer to normal)
```

**Files to Modify:**
- `src/features/technical.py` - Return calculations
- `src/insights/*.py` - All predictors

---

### 2.4 Add Monitoring & Kill Switch

**Problem:** No way to detect when model starts failing in production.

**Solution:** Drift detection and automatic disable.

```python
@dataclass
class MonitoringAlert:
    alert_type: Literal["DRIFT", "CALIBRATION", "PERFORMANCE", "DATA"]
    severity: Literal["WARNING", "CRITICAL"]
    message: str
    timestamp: datetime
    should_disable: bool

class PredictionMonitor:
    """Monitor live prediction performance."""

    def __init__(self):
        self.predictions: List[Dict] = []
        self.calibration_window = 50
        self.accuracy_threshold = 0.45  # Below this = disable
        self.calibration_threshold = 0.15  # Max deviation

    def record_prediction(self, symbol: str, predicted_prob: float,
                          predicted_signal: str, timestamp: datetime):
        """Record a prediction for later evaluation."""
        self.predictions.append({
            'symbol': symbol,
            'predicted_prob': predicted_prob,
            'predicted_signal': predicted_signal,
            'timestamp': timestamp,
            'actual_outcome': None  # Filled later
        })

    def record_outcome(self, symbol: str, timestamp: datetime, actual_up: bool):
        """Record actual outcome when known."""
        for pred in self.predictions:
            if pred['symbol'] == symbol and pred['timestamp'] == timestamp:
                pred['actual_outcome'] = actual_up
                break

    def check_health(self) -> List[MonitoringAlert]:
        """Check for issues."""
        alerts = []

        # Get recent predictions with outcomes
        evaluated = [p for p in self.predictions[-self.calibration_window:]
                     if p['actual_outcome'] is not None]

        if len(evaluated) < 20:
            return alerts  # Not enough data

        # Check accuracy
        correct = sum(1 for p in evaluated
                      if (p['predicted_prob'] > 0.5) == p['actual_outcome'])
        accuracy = correct / len(evaluated)

        if accuracy < self.accuracy_threshold:
            alerts.append(MonitoringAlert(
                alert_type="PERFORMANCE",
                severity="CRITICAL",
                message=f"Accuracy dropped to {accuracy:.1%} (threshold: {self.accuracy_threshold:.1%})",
                timestamp=datetime.now(),
                should_disable=True
            ))

        # Check calibration
        # Group by predicted probability buckets
        buckets = defaultdict(list)
        for p in evaluated:
            bucket = round(p['predicted_prob'], 1)
            buckets[bucket].append(p['actual_outcome'])

        for bucket, outcomes in buckets.items():
            if len(outcomes) >= 5:
                actual_rate = sum(outcomes) / len(outcomes)
                if abs(actual_rate - bucket) > self.calibration_threshold:
                    alerts.append(MonitoringAlert(
                        alert_type="CALIBRATION",
                        severity="WARNING",
                        message=f"Calibration drift: predicted {bucket:.0%}, actual {actual_rate:.0%}",
                        timestamp=datetime.now(),
                        should_disable=False
                    ))

        return alerts

    def should_trade(self) -> Tuple[bool, str]:
        """Check if trading should be enabled."""
        alerts = self.check_health()
        critical = [a for a in alerts if a.should_disable]
        if critical:
            return False, critical[0].message
        return True, "OK"
```

**Files to Create:**
- `src/monitoring/monitor.py`
- `src/monitoring/alerts.py`

---

## Phase 3: Advanced Enhancements (Week 5-8)

### 3.1 Meta-Learner Stacker

**Problem:** Static weights ignore market regime and component performance.

**Solution:** Train a meta-model that learns optimal weighting dynamically.

```python
class MetaLearnerStacker:
    """Learn optimal ensemble weights dynamically."""

    def __init__(self):
        self.meta_model = LogisticRegression(C=0.1)  # Regularized
        self.is_trained = False

    def prepare_meta_features(self, component_predictions: Dict) -> np.ndarray:
        """Create feature vector for meta-model."""
        features = [
            component_predictions.get('ml_rf_prob', 0.5),
            component_predictions.get('ml_gb_prob', 0.5),
            component_predictions.get('ml_xgb_prob', 0.5),
            component_predictions.get('llm_sentiment', 0.0),
            component_predictions.get('technical_score', 50) / 100,
            component_predictions.get('calendar_score', 50) / 100,
            component_predictions.get('volatility_regime', 0),
            component_predictions.get('trend_strength', 0),
        ]
        return np.array(features).reshape(1, -1)

    def train(self, historical_predictions: List[Dict], outcomes: List[bool]):
        """Train meta-model on historical component predictions."""
        X = np.vstack([self.prepare_meta_features(p) for p in historical_predictions])
        y = np.array(outcomes)

        # Time-ordered split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.meta_model.fit(X_train, y_train)

        # Validation
        val_acc = self.meta_model.score(X_val, y_val)
        print(f"Meta-model validation accuracy: {val_acc:.1%}")

        self.is_trained = True

    def predict_proba(self, component_predictions: Dict) -> float:
        """Get final probability from meta-model."""
        if not self.is_trained:
            # Fallback to simple average
            probs = [v for k, v in component_predictions.items()
                     if 'prob' in k or 'score' in k]
            return np.mean(probs)

        X = self.prepare_meta_features(component_predictions)
        return self.meta_model.predict_proba(X)[0, 1]
```

**Files to Create:**
- `src/models/meta_learner.py`

**Integration:**
- Replace fixed weights in `unified_predictor.py`
- Train meta-model during backtesting

---

### 3.2 Portfolio-Level Risk Management

**Problem:** Position sizing ignores correlation with existing positions.

**Solution:** Add correlation-aware portfolio manager.

```python
class PortfolioRiskManager:
    """Portfolio-level risk management."""

    def __init__(self, max_portfolio_risk: float = 0.02):  # 2% daily VaR
        self.max_portfolio_risk = max_portfolio_risk
        self.current_positions: Dict[str, float] = {}  # symbol -> allocation

    def calculate_correlation_penalty(self,
                                       new_symbol: str,
                                       sector: str,
                                       returns_matrix: pd.DataFrame) -> float:
        """Reduce allocation for correlated assets."""
        if not self.current_positions:
            return 1.0  # No penalty if no positions

        # Calculate correlation with existing positions
        correlations = []
        for existing_symbol in self.current_positions:
            if existing_symbol in returns_matrix.columns and new_symbol in returns_matrix.columns:
                corr = returns_matrix[existing_symbol].corr(returns_matrix[new_symbol])
                correlations.append(abs(corr))

        if not correlations:
            return 1.0

        avg_corr = np.mean(correlations)

        # Higher correlation = lower allocation
        # penalty = 1.0 at corr=0, 0.5 at corr=1
        penalty = 1.0 - (avg_corr * 0.5)
        return penalty

    def adjust_position_size(self,
                              symbol: str,
                              kelly_size: float,
                              sector: str,
                              returns_matrix: pd.DataFrame) -> float:
        """Adjust position size for portfolio risk."""
        # Correlation penalty
        corr_penalty = self.calculate_correlation_penalty(symbol, sector, returns_matrix)

        # Sector concentration limit
        sector_exposure = sum(alloc for sym, alloc in self.current_positions.items()
                              if self.get_sector(sym) == sector)
        if sector_exposure > 0.25:  # Max 25% per sector
            sector_penalty = max(0.5, 1.0 - (sector_exposure - 0.25))
        else:
            sector_penalty = 1.0

        # Total position limit
        total_exposure = sum(self.current_positions.values())
        if total_exposure > 0.8:  # Max 80% invested
            exposure_penalty = max(0.3, 1.0 - (total_exposure - 0.8))
        else:
            exposure_penalty = 1.0

        adjusted_size = kelly_size * corr_penalty * sector_penalty * exposure_penalty
        return min(adjusted_size, 0.15)  # Hard cap at 15%
```

**Files to Create:**
- `src/risk/portfolio_manager.py`

---

### 3.3 Live Performance Tracker

**Problem:** No systematic tracking of prediction outcomes.

**Solution:** Database-backed prediction tracking.

```python
# Database schema addition
class PredictionRecord(Base):
    __tablename__ = 'prediction_records'

    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, index=True)

    # Prediction details
    predicted_signal = Column(String)
    predicted_prob_up = Column(Float)
    confidence = Column(Float)

    # Component scores
    ml_score = Column(Float)
    llm_sentiment = Column(Float)
    technical_score = Column(Float)
    calendar_score = Column(Float)

    # Trade setup
    entry_price = Column(Float)
    stop_loss = Column(Float)
    target_price = Column(Float)
    position_size_pct = Column(Float)

    # Outcomes (filled later)
    actual_outcome = Column(Boolean, nullable=True)
    actual_return_pct = Column(Float, nullable=True)
    hit_target = Column(Boolean, nullable=True)
    hit_stop = Column(Boolean, nullable=True)

    # Attribution
    model_version = Column(String)
    feature_version = Column(String)

class PerformanceTracker:
    """Track and analyze prediction performance."""

    def get_accuracy_by_component(self, days: int = 30) -> Dict[str, float]:
        """Get accuracy breakdown by signal source."""
        # Query predictions with outcomes
        # Group by which component was strongest
        # Return accuracy per component
        pass

    def get_calibration_curve(self, days: int = 30) -> Dict[float, float]:
        """Get predicted vs actual probability mapping."""
        # Bucket predictions by probability
        # Calculate actual win rate per bucket
        pass

    def get_regime_performance(self, days: int = 90) -> Dict[str, Dict]:
        """Get performance by market regime."""
        # Group by regime (bull/bear/sideways)
        # Return accuracy, avg return, Sharpe per regime
        pass
```

**Files to Create:**
- `src/storage/prediction_records.py`
- `src/analytics/performance_tracker.py`

---

## Phase 4: Future Considerations (Optional)

### 4.1 Items to Consider but NOT Prioritize

Based on the reviews, these are interesting but lower priority:

1. **LSTM/Transformer models**: High complexity, marginal gains for daily predictions
2. **Alternative data** (satellite, social): High cost, uncertain signal quality
3. **Real-time streaming**: Overkill for daily/swing trading
4. **Custom LLM fine-tuning**: Expensive, current prompting is sufficient

### 4.2 Architecture Evolution Path

```
Current: Monolithic Predictor Classes
    ↓
Phase 1: Clean separation + data hygiene
    ↓
Phase 2: Cost-aware + purged validation
    ↓
Phase 3: Meta-learner + monitoring
    ↓
Future: Microservices (if scale requires)
```

---

## Implementation Checklist

### Phase 1 (Critical - Do First)
- [ ] Create `SafeFeatureScaler` class
- [ ] Refactor all predictors to use scaler correctly
- [ ] Rewrite LLM prompts for sentiment-only analysis
- [ ] Create `DataHealthChecker`
- [ ] Add data health check to pipeline entry
- [ ] Implement `AdaptiveWeightCalculator`
- [ ] Add unit tests for leakage prevention

### Phase 2 (Robustness)
- [ ] Create `TransactionCosts` class
- [ ] Integrate costs into backtester
- [ ] Implement `PurgedKFold`
- [ ] Replace sklearn KFold in all CV
- [ ] Convert all returns to log returns
- [ ] Create `PredictionMonitor`
- [ ] Add monitoring hooks to prediction pipeline
- [ ] Create kill switch logic

### Phase 3 (Advanced)
- [ ] Implement `MetaLearnerStacker`
- [ ] Train meta-model on historical data
- [ ] Create `PortfolioRiskManager`
- [ ] Implement correlation-aware sizing
- [ ] Create prediction records schema
- [ ] Build `PerformanceTracker`
- [ ] Add performance dashboards

---

## Success Metrics

After implementing these improvements, track:

| Metric | Current | Target |
|--------|---------|--------|
| Walk-forward accuracy | ~52% | >55% |
| Calibration error | Unknown | <10% |
| Backtest Sharpe (after costs) | Unknown | >1.0 |
| Max drawdown | Unknown | <15% |
| Prediction latency | ~5s | <3s |
| System uptime | N/A | >99% |

---

## Risk Warnings

Even with all improvements:

1. **Markets are adversarial**: Edge decays as others find it
2. **Black swans exist**: No model predicts wars, pandemics, fraud
3. **Past != Future**: Historical accuracy doesn't guarantee profits
4. **Costs compound**: Small costs become large over many trades
5. **Psychology matters**: Best signals fail if you can't follow them

**This is NOT financial advice. Use for educational purposes only.**
