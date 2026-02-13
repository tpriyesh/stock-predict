# Stock Prediction Engine - Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [7-Model Ensemble Architecture](#7-model-ensemble-architecture)
4. [Model Standardization (Adapters)](#model-standardization-adapters)
5. [Data Flow Pipeline](#data-flow-pipeline)
6. [Model Details](#model-details)
7. [Advanced Execution Rules](#advanced-execution-rules)
8. [RL Feedback Loop](#rl-feedback-loop)
9. [Regime-Specific Calibration](#regime-specific-calibration)
10. [Configuration System](#configuration-system)
11. [UI Architecture](#ui-architecture)
12. [Key Classes Reference](#key-classes-reference)
13. [Accuracy & Calibration](#accuracy--calibration)
14. [Execution Paths](#execution-paths)

---

## Overview

The Stock Prediction Engine is a **quantitative trading signal generation system** for Indian equity markets (NSE/BSE). It uses a **7-model ensemble** approach where independent prediction models vote on stock signals, achieving calibrated accuracy through backtest validation.

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STOCK PREDICTION ENGINE                               │
│                                                                              │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│   │  Model 1    │  │  Model 2    │  │  Model 3    │  │  Model 4    │       │
│   │   BASE      │  │  PHYSICS    │  │   MATH      │  │   REGIME    │       │
│   │   (25%)     │  │   (18%)     │  │   (14%)     │  │   (13%)     │       │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
│          │                │                │                │               │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                        │
│   │  Model 5    │  │  Model 6    │  │  Model 7    │                        │
│   │   MACRO     │  │  ALT DATA   │  │  ADVANCED   │                        │
│   │   (10%)     │  │   (10%)     │  │   (10%)     │                        │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                        │
│          │                │                │                                │
│          └────────────────┴────────────────┴───────────────┐               │
│                                                            │               │
│                                     ┌──────────────────────▼──────────┐    │
│                                     │    ENSEMBLE AGGREGATION         │    │
│                                     │  (Weighted Average + Voting)    │    │
│                                     └──────────────────────┬──────────┘    │
│                                                            │               │
│                                     ┌──────────────────────▼──────────┐    │
│                                     │   CONFIDENCE CALIBRATION        │    │
│                                     │  (Backtest-validated accuracy)  │    │
│                                     └──────────────────────┬──────────┘    │
│                                                            │               │
│                                     ┌──────────────────────▼──────────┐    │
│                                     │     SIGNAL + TRADE LEVELS       │    │
│                                     │   (BUY/HOLD, Entry/SL/Target)   │    │
│                                     └─────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Principles
- **One Engine**: All UI tabs use the same prediction engine for consistency
- **Model Agreement**: Signals require multiple models to agree (3/7 moderate, 5/7 strong)
- **Calibrated Confidence**: Probabilities reflect actual backtest accuracy
- **SELL Disabled**: SELL signals disabled due to 28.6% accuracy (inverted)
- **Risk Management**: Built-in filters for liquidity, volatility, and position sizing

### Performance Targets
| Metric | 4-Model (Baseline) | 7-Model (Current) |
|--------|-------------------|-------------------|
| BUY Accuracy | 57.6% | 65-68% (target) |
| SELL Accuracy | 28.6% | DISABLED |
| Calibration Error | 1.9% | <2% |

---

## Directory Structure

```
stock-predict/
├── main.py                          # CLI entry point
├── config/                          # Configuration layer
│   ├── settings.py                  # Environment & trading parameters
│   ├── thresholds.py                # Signal thresholds & calibration
│   ├── symbols.json                 # Stock universe (NIFTY 50)
│   └── sector_dependencies.json     # Macro sector relationships
│
├── src/                             # Core prediction engines
│   ├── data/                        # Data ingestion
│   │   ├── price_fetcher.py         # Yahoo Finance OHLCV
│   │   ├── news_fetcher.py          # News API + web scraping
│   │   └── fundamentals.py          # Company financial data
│   │
│   ├── features/                    # Feature engineering
│   │   ├── technical.py             # 20+ technical indicators
│   │   ├── news_features.py         # OpenAI sentiment analysis
│   │   └── market_features.py       # Sector & market context
│   │
│   ├── models/                      # Scoring engines
│   │   ├── scoring.py               # Base 4-model scoring
│   │   ├── enhanced_scoring.py      # 7-model ensemble (MAIN)
│   │   ├── backtester.py            # Walk-forward backtesting
│   │   └── calibration.py           # Confidence calibration
│   │
│   ├── signals/                     # Signal generation
│   │   ├── generator.py             # Basic signal generator
│   │   ├── enhanced_generator.py    # 7-model generator
│   │   └── explainer.py             # Reasoning & explanations
│   │
│   ├── physics/                     # MODEL 2: Physics models
│   │   ├── physics_engine.py        # Orchestrator
│   │   ├── momentum_conservation.py # Newton's momentum
│   │   ├── spring_reversion.py      # Mean reversion
│   │   ├── energy_clustering.py     # Phase states
│   │   └── network_propagation.py   # Wave interference
│   │
│   ├── math_models/                 # MODEL 3: Math models
│   │   ├── math_engine.py           # Orchestrator
│   │   ├── fourier_cycles.py        # Spectral analysis
│   │   ├── fractal_dimension.py     # Hurst exponent
│   │   ├── information_entropy.py   # Shannon entropy
│   │   └── statistical_mechanics.py # Phase transitions
│   │
│   ├── ml/                          # MODEL 4: ML models
│   │   └── hmm_regime_detector.py   # Hidden Markov Model
│   │
│   ├── macro/                       # MODEL 5: Macro indicators
│   │   ├── macro_engine.py          # Orchestrator
│   │   ├── commodities.py           # Oil, Gold, Copper
│   │   ├── currencies.py            # USD/INR, DXY
│   │   └── bonds.py                 # US 10Y, Yield curve
│   │
│   ├── alternative_data/            # MODEL 6: Alternative data
│   │   ├── alternative_engine.py    # Orchestrator
│   │   ├── earnings_events.py       # Earnings analysis
│   │   ├── options_flow.py          # Options positioning
│   │   ├── institutional_flow.py    # FII/DII flows
│   │   └── multi_timeframe.py       # Multi-timeframe confluence
│   │
│   ├── core/                        # Core infrastructure
│   │   ├── advanced_engine.py       # MODEL 7: 20 math models
│   │   ├── calibrated_scoring.py    # Production calibrated scoring
│   │   ├── regime_calibration.py    # Per-regime calibration curves
│   │   ├── transaction_costs.py     # Trading cost calculations
│   │   │
│   │   ├── interfaces/              # Standardized protocols
│   │   │   ├── __init__.py
│   │   │   └── model_output.py      # StandardizedModelOutput protocol
│   │   │
│   │   ├── adapters/                # Model output adapters
│   │   │   ├── __init__.py
│   │   │   ├── base_adapter.py      # Base adapter with fallback logic
│   │   │   ├── technical_adapter.py # Wraps ScoringEngine
│   │   │   ├── physics_adapter.py   # Wraps PhysicsEngine
│   │   │   ├── math_adapter.py      # Wraps MathEngine
│   │   │   ├── regime_adapter.py    # Wraps HMMRegimeDetector
│   │   │   ├── macro_adapter.py     # Wraps MacroEngine
│   │   │   ├── alternative_adapter.py # Wraps AlternativeDataEngine
│   │   │   └── advanced_adapter.py  # Wraps AdvancedPredictionEngine
│   │   │
│   │   └── advanced_math/           # Individual math modules
│   │       ├── kalman_filter.py
│   │       ├── wavelet_analysis.py
│   │       ├── pca_decomposition.py
│   │       ├── dqn_position_sizing.py
│   │       └── [16 more modules]
│   │
│   ├── execution/                   # Trade execution management
│   │   ├── __init__.py
│   │   ├── trade_executor.py        # Advanced trade executor
│   │   ├── trailing_stop.py         # Trailing stop calculator
│   │   ├── gap_handler.py           # Gap detection & handling
│   │   └── reentry_manager.py       # Re-entry rules after stops
│   │
│   ├── rl/                          # Reinforcement Learning
│   │   ├── __init__.py
│   │   ├── experience_collector.py  # Collects real trade experiences
│   │   ├── dqn_trainer.py           # DQN training scheduler
│   │   └── reward_shaping.py        # Configurable reward functions
│   │
│   ├── insights/                    # Analysis & tracking
│   │   └── trade_tracker.py         # Trade tracking with RL integration
│   │
│   └── storage/                     # Data persistence
│       ├── database.py              # DuckDB storage
│       └── models.py                # Pydantic schemas
│
├── ui/                              # Streamlit dashboard
│   ├── app.py                       # Main application
│   └── components/
│       └── engine_config.py         # Configuration UI
│
├── data/                            # Data storage (created at runtime)
│   ├── backtest_results/            # Backtest output
│   ├── predictions/                 # Saved predictions
│   ├── processed/                   # Processed data cache
│   ├── raw/                         # Raw data cache
│   ├── trades/                      # Trade tracking data
│   └── stock_predict.duckdb         # DuckDB database
│
└── docs/                            # Documentation
    └── ARCHITECTURE.md              # This file
```

---

## 7-Model Ensemble Architecture

### Weight Distribution

| Model | Weight | Source Module | Components |
|-------|--------|---------------|------------|
| **1. Base Technical** | 25% | `src/models/scoring.py` | RSI, MACD, BB, Volume, News |
| **2. Physics Engine** | 18% | `src/physics/physics_engine.py` | Momentum, Spring, Energy, Network |
| **3. Math Engine** | 14% | `src/math_models/math_engine.py` | Fourier, Fractal, Entropy, Mechanics |
| **4. HMM Regime** | 13% | `src/ml/hmm_regime_detector.py` | Bull/Bear/Choppy/Ranging |
| **5. Macro Engine** | 10% | `src/macro/macro_engine.py` | Commodities, Currencies, Bonds |
| **6. Alternative Data** | 10% | `src/alternative_data/alternative_engine.py` | Earnings, Options, Institutional |
| **7. Advanced Math** | 10% | `src/core/advanced_engine.py` | Kalman, Wavelet, PCA, + 14 more |

### Ensemble Calculation

```python
# From src/models/enhanced_scoring.py

ensemble_score = (
    0.25 * base_score +
    0.18 * physics_score +
    0.14 * math_score +
    0.13 * regime_score +
    0.10 * macro_score +
    0.10 * alternative_score +
    0.10 * advanced_score
)

# Model agreement (voting)
model_votes = {
    'base': 'BUY' if base_score > 0.55 else 'HOLD',
    'physics': 'BUY' if physics_score > 0.55 else 'HOLD',
    # ... same for all 7 models
}
agreement = sum(1 for v in model_votes.values() if v == 'BUY')

# Signal generation
if ensemble_score >= 0.70 and agreement >= 5:
    signal = 'STRONG_BUY', strength = 'strong'
elif ensemble_score >= 0.60 and agreement >= 3:
    signal = 'BUY', strength = 'moderate'
elif 0.45 < ensemble_score < 0.55:
    signal = 'HOLD', strength = 'none'
else:
    signal = 'HOLD'  # SELL disabled
```

### Signal Thresholds

| Agreement | Ensemble Score | Signal | Strength |
|-----------|---------------|--------|----------|
| 5+ / 7 | >= 0.70 | STRONG_BUY | strong (runtime uses strong_agreement=5) |
| 3+ / 7 | >= 0.60 | BUY | moderate |
| 2 / 7 | >= 0.55 | WEAK_BUY | weak |
| Any | 0.45-0.55 | HOLD | none |
| Any | < 0.45 | HOLD | none (SELL disabled) |

---

## Model Standardization (Adapters)

### Problem Solved
The 7 models have different output types:
- `PhysicsScore` - 4 sub-scores + strategy + reasons
- `MathScore` - 4 sub-scores + predictability
- `MacroSignal` - signal (-1 to +1) + drivers
- `RegimeResult` - regime + probability
- etc.

### Solution: Adapter Pattern

All models now produce a unified `StandardizedModelOutput` via adapters:

```python
# From src/core/interfaces/model_output.py

class UncertaintyType(Enum):
    CONFIDENCE_INTERVAL = "confidence_interval"
    STANDARD_DEVIATION = "standard_deviation"
    QUANTILES = "quantiles"
    NONE = "none"

@dataclass
class ModelUncertainty:
    type: UncertaintyType                          # Type of uncertainty estimate
    value: float                                   # Primary uncertainty value (e.g., std dev)
    lower_bound: Optional[float] = None            # For confidence intervals
    upper_bound: Optional[float] = None            # For confidence intervals
    quantiles: Optional[Dict[int, float]] = None   # For quantile estimates {5: 0.45, 95: 0.80}

@dataclass
class StandardizedModelOutput:
    p_buy: float              # 0-1 probability (primary signal)
    uncertainty: ModelUncertainty  # Uncertainty quantification
    coverage: float           # 0-1 data quality/availability
    reasoning: List[str]      # Human-readable explanations
    model_name: str = ""      # Identifier for logging
    timestamp: datetime       # When prediction was made
    raw_output: Optional[Dict[str, Any]] = None  # Model-specific extras
    warnings: List[str]       # Risk warnings
```

### Adapter Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ADAPTER LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  BaseModelAdapter (Abstract)                                    │
│  ├── predict_standardized() → StandardizedModelOutput           │
│  ├── _convert_to_standard()  # Template method                  │
│  └── _get_fallback_output()  # Error handling                   │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │TechnicalAdapter  │  │ PhysicsAdapter   │                     │
│  │                  │  │                  │                     │
│  │ ScoringEngine    │  │ PhysicsEngine    │                     │
│  │     ↓            │  │     ↓            │                     │
│  │ StandardizedOut  │  │ StandardizedOut  │                     │
│  └──────────────────┘  └──────────────────┘                     │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │  MathAdapter     │  │  RegimeAdapter   │                     │
│  │                  │  │                  │                     │
│  │ MathEngine       │  │ HMMRegimeDetector│                     │
│  │     ↓            │  │     ↓            │                     │
│  │ StandardizedOut  │  │ StandardizedOut  │                     │
│  └──────────────────┘  └──────────────────┘                     │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐   │
│  │  MacroAdapter    │  │ AlternativeAdapt │  │AdvancedAdapt│   │
│  │                  │  │                  │  │              │   │
│  │ MacroEngine      │  │ AltDataEngine    │  │ AdvancedEng  │   │
│  │     ↓            │  │     ↓            │  │     ↓        │   │
│  │ StandardizedOut  │  │ StandardizedOut  │  │StandardOut   │   │
│  └──────────────────┘  └──────────────────┘  └──────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Usage in EnhancedScoringEngine

```python
# From src/models/enhanced_scoring.py

class EnhancedScoringEngine:
    def _initialize_adapters(self):
        """Initialize standardized model adapters."""
        self.technical_adapter = TechnicalModelAdapter(self.base_engine)
        self.physics_adapter = PhysicsModelAdapter(self.physics_engine)
        self.math_adapter = MathModelAdapter(self.math_engine)
        self.regime_adapter = RegimeModelAdapter(self.hmm_detector)
        self.macro_adapter = MacroModelAdapter(self.macro_engine)
        self.alternative_adapter = AlternativeDataAdapter(self.alternative_engine)
        self.advanced_adapter = AdvancedMathAdapter(self.advanced_engine)

    def get_standardized_predictions(self, symbol, df, **kwargs) -> EnsembleInput:
        """Get standardized outputs from all models."""
        outputs = []

        # Always available
        outputs.append(self.technical_adapter.predict_standardized(symbol, df))
        outputs.append(self.physics_adapter.predict_standardized(symbol, df))
        outputs.append(self.math_adapter.predict_standardized(symbol, df))
        outputs.append(self.regime_adapter.predict_standardized(symbol, df))

        # Optional (may not be initialized)
        if self.macro_adapter:
            outputs.append(self.macro_adapter.predict_standardized(symbol, df))
        if self.alternative_adapter:
            outputs.append(self.alternative_adapter.predict_standardized(symbol, df))
        if self.advanced_adapter:
            outputs.append(self.advanced_adapter.predict_standardized(symbol, df))

        return EnsembleInput(outputs=outputs, symbol=symbol, timestamp=datetime.now())
```

### Benefits
- **Consistent interface**: All models return the same structure
- **Coverage tracking**: Know when models have insufficient data
- **Uncertainty quantification**: Confidence bounds on predictions
- **Graceful degradation**: Fallback outputs when models fail
- **Reasoning transparency**: Human-readable explanations

---

## Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Yahoo Finance API              News API + Firecrawl        │
│       ↓                              ↓                      │
│  OHLCV (1 year)                 Headlines (7 days)          │
│  - Open, High, Low, Close       - Sentiment extraction      │
│  - Volume, Returns              - Source validation         │
│  - ATR calculation              - Relevance scoring         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING LAYER                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  TechnicalIndicators.calculate_all()                        │
│  ↓                                                           │
│  - RSI (14)            - Volume ratios                      │
│  - MACD (12,26,9)      - Support/Resistance                 │
│  - Bollinger Bands     - SMA (20, 50, 200)                  │
│  - ATR (14)            - Momentum metrics                   │
│                                                              │
│  NewsFeatureExtractor (OpenAI)                              │
│  ↓                                                           │
│  - Sentiment score (-1 to +1)                               │
│  - Key claims extraction                                    │
│  - Time decay (recency weighting)                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 7-MODEL ENSEMBLE SCORING                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  EnhancedScoringEngine.score_stock(symbol, df, trade_type) │
│                                                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Model 1 │  │ Model 2 │  │ Model 3 │  │ Model 4 │        │
│  │  BASE   │  │ PHYSICS │  │  MATH   │  │ REGIME  │        │
│  │  25%    │  │  18%    │  │  14%    │  │  13%    │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │            │              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │
│  │ Model 5 │  │ Model 6 │  │ Model 7 │                     │
│  │  MACRO  │  │ALT DATA │  │ADVANCED │                     │
│  │  10%    │  │  10%    │  │  10%    │                     │
│  └────┬────┘  └────┬────┘  └────┬────┘                     │
│       │            │            │                           │
│       └────────────┴────────────┴──────────→ Aggregate     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    SIGNAL GENERATION                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Calculate ensemble_score (weighted average)             │
│  2. Count model agreement (votes for BUY)                   │
│  3. Generate signal based on thresholds                     │
│  4. Calibrate confidence from lookup tables                 │
│  5. Calculate trade levels (entry, stop, target)            │
│  6. Collect reasons and warnings from all models            │
│                                                              │
│  Output: EnhancedStockScore dataclass                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      FILTERING & OUTPUT                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Risk Filters:                                              │
│  - Confidence >= min_confidence (55%)                       │
│  - Liquidity >= min_liquidity_cr (5 Cr)                     │
│  - ATR% <= max_atr_pct (6%)                                 │
│  - Model Agreement >= min_model_agreement (3)               │
│                                                              │
│  Ranking: by (confidence DESC, signal_strength DESC)        │
│                                                              │
│  Final Output:                                              │
│  - Signal, Confidence, Model Agreement                      │
│  - Entry, Stop Loss, Target prices                          │
│  - Model votes breakdown                                    │
│  - Reasons and warnings                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Model Details

### Model 1: Base Technical (25%)

**Location**: `src/models/scoring.py`

**Components**:
```python
WEIGHTS = {
    'technical': 0.35,   # RSI, MACD, BB, MA crossovers
    'momentum': 0.25,    # Price acceleration, ROC
    'news': 0.15,        # OpenAI sentiment
    'sector': 0.15,      # Relative sector strength
    'volume': 0.10       # Volume confirmation
}
```

**Key Indicators**:
- RSI (14): Overbought/oversold detection
- MACD (12,26,9): Trend direction and momentum
- Bollinger Bands (20,2): Volatility and mean reversion
- Moving Averages: SMA 20/50/200 crossovers
- Volume Ratio: Current vs 20-day average

---

### Model 2: Physics Engine (18%)

**Location**: `src/physics/physics_engine.py`

**Sub-Models**:

| Model | Concept | Application |
|-------|---------|-------------|
| MomentumConservation | Newton's 1st Law | Price momentum persistence |
| SpringReversion | Hooke's Law | Mean reversion elasticity |
| EnergyClustering | Thermodynamics | Solid/Liquid/Gas phases |
| NetworkPropagation | Wave Interference | Peer correlation impact |

**Output**: `PhysicsScore` with recommended strategy (momentum/reversion/breakout/avoid)

---

### Model 3: Math Engine (14%)

**Location**: `src/math_models/math_engine.py`

**Sub-Models**:

| Model | Technique | Purpose |
|-------|-----------|---------|
| FourierCycles | FFT | Identify dominant price cycles |
| FractalDimension | Hurst Exponent | Trend persistence (H>0.5 = trending) |
| InformationEntropy | Shannon Entropy | Market randomness/predictability |
| StatisticalMechanics | Phase Transitions | Market state changes |

**Output**: `MathScore` with market_character (trending/random/mean_reverting)

---

### Model 4: HMM Regime (13%)

**Location**: `src/ml/hmm_regime_detector.py`

**Detected Regimes**:
- `TRENDING_BULL`: Strong uptrend
- `TRENDING_BEAR`: Strong downtrend
- `CHOPPY`: High volatility, no direction
- `RANGING`: Mean-reverting, sideways

**Impact**: Adjusts weights for physics/math models based on current regime

---

### Model 5: Macro Engine (10%)

**Location**: `src/macro/macro_engine.py`

**Data Sources**:
- **Commodities**: Oil, Gold, Copper, Steel, Natural Gas
- **Currencies**: USD/INR, DXY (Dollar Index)
- **Bonds**: US 10Y Yield, Yield Curve
- **Semiconductors**: SOX Index

**Sector Mappings** (from `config/sector_dependencies.json`):
```
IT         ← Semiconductors (high correlation)
Banking    ← Interest rates, Gold (carry trade)
Oil_Gas    ← Oil prices (direct)
Pharma     ← Gold (risk-off), USD/INR
Auto       ← Oil, Steel, Semiconductors
FMCG       ← USD/INR (import costs)
```

---

### Model 6: Alternative Data (10%)

**Location**: `src/alternative_data/alternative_engine.py`

**Sub-Components**:

| Source | Weight | Accuracy |
|--------|--------|----------|
| Multi-Timeframe Confluence | 35% | 72% (all aligned) |
| Earnings Events | 25% | 65-70% (event days) |
| Options Flow | 20% | 62-68% (extreme signals) |
| Institutional Flow | 20% | 60-65% |

**Signal Strength**:
- STRONG: 3+ sources agree
- MODERATE: 2 sources agree
- WEAK: 1 source
- NEUTRAL: Conflicting or <0.55

---

### Model 7: Advanced Math (10%)

**Location**: `src/core/advanced_engine.py`

**20 Mathematical Models**:

| Category | Models |
|----------|--------|
| Filtering | Kalman Filter, Wavelet Analysis |
| Dimensionality | PCA Decomposition, Factor Model |
| Regime | Markov Regime (MLE) |
| Optimization | DQN Position Sizing, Bellman Optimal |
| Geometry | Information Geometry, Optimal Transport |
| Advanced Stats | Heavy-Tailed, Copula, Taylor Series, Stochastic DE |
| Validation | Statistical Validation, Measurement Metrics, Multidim Confidence |
| Graph/Tensor | Spectral Graph, Tensor Decomposition |
| Quantum | Quantum-Inspired Optimizer |
| Unified | Unified Advanced Engine |

**Pipeline**:
```
Raw Data → PCA → Wavelet/Kalman → Markov Regime
    → Factor Analysis → Ensemble → DQN Position → Final Score
```

---

## Advanced Execution Rules

### Location: `src/execution/`

The execution module provides sophisticated trade management beyond simple entry/exit:

### Components

| File | Class | Purpose |
|------|-------|---------|
| `trade_executor.py` | `AdvancedTradeExecutor` | Orchestrates trailing stops, partial exits |
| `trailing_stop.py` | `TrailingStopCalculator` | Multiple trailing stop methods |
| `gap_handler.py` | `GapHandler` | Gap detection and stop adjustment |
| `reentry_manager.py` | `ReentryManager` | Re-entry rules after stops |

### Configuration

```python
# From src/execution/trade_executor.py

@dataclass
class ExecutionConfig:
    # Trailing Stops
    use_trailing_stop: bool = True
    trailing_stop_atr_multiplier: float = 1.5  # Trail at 1.5 × ATR
    trailing_activation_r: float = 1.0         # Activate after 1R profit

    # Partial Profit Booking
    use_partial_exits: bool = True
    partial_exits: List[Tuple[float, float]] = [
        (1.0, 0.25),  # Book 25% at 1R
        (2.0, 0.50),  # Book 50% at 2R (target)
    ]
    # Remaining 25% trails with stop

    # Gap Handling
    skip_if_gap_exceeds_atr_multiple: float = 2.0  # Skip entry if gap > 2×ATR
    adjust_sl_on_gap_through: bool = True
    gap_sl_buffer_pct: float = 0.5  # 0.5% buffer below gap

    # Re-entry Rules
    allow_reentry_after_sl: bool = True
    reentry_cooldown_bars: int = 3
    reentry_requires_new_signal: bool = True
    reentry_position_scale: float = 0.5  # 50% size on re-entry

    # Time-based Rules
    max_holding_days_intraday: int = 1
    max_holding_days_swing: int = 5
    max_holding_days_positional: int = 20
```

### Trailing Stop Methods

```python
# From src/execution/trailing_stop.py

class TrailingMethod(Enum):
    ATR_MULTIPLE = "atr_multiple"   # Trail at N × ATR from high
    PERCENTAGE = "percentage"        # Trail at N% from high
    CHANDELIER = "chandelier"        # Classic 3×ATR from high
    STEP = "step"                    # Move stop in fixed R increments
```

### Example: Advanced Trade Simulation

```python
# From src/models/backtester.py

backtester = Backtester(
    use_advanced_execution=True,
    execution_config=ExecutionConfig(
        use_trailing_stop=True,
        trailing_stop_atr_multiplier=1.5,
        partial_exits=[(1.0, 0.25), (2.0, 0.50)]
    )
)

result = backtester.simulate_trade(
    prices=df,
    entry_date=signal_date,
    signal=SignalType.BUY,
    stop_loss=stop,
    target=target,
    regime="trending_bull"
)

# Result includes:
# - partial_exits: [(date1, price1, 0.25), (date2, price2, 0.50)]
# - trailing_stop_activated: True
# - final_trailing_stop: 105.50
```

### Gap Handling Logic

```python
# From src/execution/gap_handler.py

class GapType(Enum):
    NONE = "none"
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    GAP_UP_CONTINUATION = "gap_up_continuation"
    GAP_DOWN_CONTINUATION = "gap_down_continuation"
    GAP_UP_EXHAUSTION = "gap_up_exhaustion"
    GAP_DOWN_EXHAUSTION = "gap_down_exhaustion"

# Skip entry if gap > 2×ATR
should_skip, reason = gap_handler.should_skip_entry(
    prev_close=100.0,
    current_open=105.0,
    atr=2.5
)
if should_skip:
    return None  # Skip this trade

# Adjust stop if price gaps through
adjusted_stop = gap_handler.adjust_stop_for_gap(
    gap_type=GapType.GAP_DOWN,
    original_sl=100.0,
    open_price=95.0,
    buffer_pct=0.5,
    is_long=True
)
```

---

## RL Feedback Loop

### Location: `src/rl/`

The RL module connects real trade outcomes to DQN position sizing training:

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL FEEDBACK LOOP                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Signal Generated                                            │
│       ↓                                                          │
│  2. TradeTracker.record_trade()                                 │
│       ↓                                                          │
│  3. ExperienceCollector.record_trade_entry()                    │
│       → Extracts TradingState from price DataFrame              │
│       → Stores (state, action, position_size)                   │
│       ↓                                                          │
│  4. Trade Executes in Market                                    │
│       ↓                                                          │
│  5. TradeTracker.update_trade_outcome()                         │
│       ↓                                                          │
│  6. ExperienceCollector.record_trade_exit()                     │
│       → Captures exit state                                      │
│       → Creates RealTradeExperience                              │
│       ↓                                                          │
│  7. DQNTrainingScheduler.should_train()                         │
│       → Checks: min_experiences >= 30                           │
│       → Checks: training frequency (daily/weekly)               │
│       ↓                                                          │
│  8. DQNTrainingScheduler.train_batch()                          │
│       → RewardCalculator shapes rewards                         │
│       → DQNPositionAgent.train_step()                           │
│       → Saves weights to checkpoint                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Structures

```python
# From src/rl/experience_collector.py

@dataclass
class TradingState:
    """State representation for DQN (10 features)."""
    price_change_1d: float    # Normalized
    price_change_5d: float
    price_change_20d: float
    volatility: float         # 20-day annualized
    rsi: float                # 0-1 normalized
    macd_signal: float        # -1 to +1
    volume_ratio: float       # vs 20-day average
    regime: int               # 0-4 (bull/bear/ranging/choppy/transition)
    current_position: float   # -1 to +1
    unrealized_pnl: float     # Normalized

@dataclass
class RealTradeExperience:
    """Complete trade experience for RL training."""
    trade_id: str
    symbol: str
    entry_state: TradingState
    action: int               # 0-6: HOLD, BUY_25, BUY_50, BUY_100, SELL_25, SELL_50, SELL_100
    position_size: float      # Actual position (0-1)
    pnl_pct: float            # Realized P&L
    holding_days: int
    hit_target: bool
    hit_stop: bool
    exit_state: TradingState
    regime: str
```

### Reward Shaping

```python
# From src/rl/reward_shaping.py

@dataclass
class RewardConfig:
    pnl_scale: float = 10.0           # Scale PnL percentage
    target_bonus: float = 0.5         # Bonus for hitting target
    stop_penalty: float = -0.3        # Penalty for hitting stop
    time_penalty_per_day: float = -0.02  # Encourage efficiency
    max_time_penalty: float = -0.20
    loss_multiplier: float = 1.5      # Penalize losses more than wins

# Reward calculation:
reward = pnl_pct * 10 / 100           # Base: scaled P&L (2% → 0.2, not 20)
if reward < 0:
    reward *= loss_multiplier          # Asymmetric loss aversion
reward += 0.5 if hit_target else 0     # Target bonus
reward -= 0.3 if hit_stop else 0       # Stop penalty
reward -= 0.02 * holding_days          # Time efficiency
# Additional components:
reward += small_position_bonus or large_position_penalty  # Position size adjustment
reward *= risk_factor                  # Risk-aware scaling
reward += regime_adjustment            # Bull bonus / Bear penalty / Choppy penalty
reward = clip(reward, -5, +5)          # Bound to prevent extreme values
```

### Training Scheduler

```python
# From src/rl/dqn_trainer.py

trainer = DQNTrainingScheduler(
    agent=dqn_agent,
    experience_collector=collector,
    training_frequency="daily",     # or "weekly", "on_trade_close"
    min_experiences=30,             # Minimum before training
    batch_size=32,
    epochs_per_batch=5
)

# After trade closes:
if trainer.should_train(date.today()):
    result = trainer.train_batch()
    print(f"Loss: {result.avg_loss:.4f}, Reward: {result.avg_reward:.3f}")
```

### Integration with TradeTracker

```python
# From src/insights/trade_tracker.py

tracker = TradeTracker(
    enable_rl=True,
    enable_regime_calibration=True,
    dqn_agent=agent  # Optional DQNPositionAgent instance
)

# When trade is recorded with price DataFrame:
trade_id = tracker.record_trade(
    symbol="RELIANCE",
    predicted_probability=0.65,
    predicted_return=2.5,
    signal_strength="moderate",
    timeframe="intraday",
    entry_price=2500.0,
    stop_loss=2475.0,
    target_price=2537.5,
    rsi=55.0,
    volume_ratio=1.3,
    trend_score=0.7,
    factors_bullish=3,
    factors_bearish=1,
    regime="trending_bull",
    position_size=0.5,
    df=price_df,            # For RL state extraction
)

# When outcome is known:
tracker.update_trade_outcome(
    trade_id=trade_id,
    exit_price=105.0,
    exit_reason="HIT_TARGET",
    df=exit_df,
    holding_days=3
)
# Automatically:
# 1. Records to ExperienceCollector
# 2. Triggers DQN training if scheduled
# 3. Updates regime calibration
```

---

## Regime-Specific Calibration

### Location: `src/core/regime_calibration.py`

### Problem Solved
A single global calibration curve doesn't account for different market conditions:
- 65% confidence in BULL market may be 70% actual win rate
- 65% confidence in CHOPPY market may be 55% actual win rate

### Solution: Per-Regime Calibration

```python
# From src/core/regime_calibration.py

class CalibrationRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    RANGING = "ranging"
    CHOPPY = "choppy"
    HIGH_VOL = "high_volatility"
    UNKNOWN = "unknown"

@dataclass
class RegimeCalibrationData:
    regime: CalibrationRegime
    history: List[Tuple[float, int]]  # (raw_score, outcome)
    n_samples: int
    accuracy: float                   # Calibration accuracy
    last_fit: Optional[datetime]
    _calibrator: IsotonicRegression   # Fitted model
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              REGIME-SPECIFIC CALIBRATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw Score (0.65) + Regime (BULL)                               │
│       ↓                                                          │
│  RegimeSpecificCalibrator.calibrate()                           │
│       ↓                                                          │
│  ┌─────────────────────────────────────────┐                     │
│  │  Check regime-specific calibrator        │                     │
│  │  - Has enough samples (≥30)?            │                     │
│  │  - IsotonicRegression fitted?           │                     │
│  └─────────────────────────────────────────┘                     │
│       ↓                                                          │
│  ┌─────────────┐        ┌─────────────┐                         │
│  │ YES: Use    │        │ NO: Fallback │                         │
│  │ BULL curve  │        │ to GLOBAL    │                         │
│  │ (fitted on  │        │ calibrator   │                         │
│  │ BULL trades)│        │              │                         │
│  └──────┬──────┘        └──────┬───────┘                         │
│         │                      │                                  │
│         └──────────┬───────────┘                                 │
│                    ↓                                              │
│  Calibrated Confidence (0.68)                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Usage

```python
# From src/core/regime_calibration.py

calibrator = RegimeSpecificCalibrator(
    min_regime_samples=30,  # Min samples before using regime-specific
    refit_interval=50       # Refit every 50 new samples
)

# Record outcome
calibrator.record_outcome(
    raw_score=0.65,
    actual_outcome=1,  # 1=win, 0=loss
    regime="bull"
)

# Calibrate with regime awareness
calibrated = calibrator.calibrate(
    raw_score=0.70,
    regime="bull"
)

# Get stats per regime
stats = calibrator.get_regime_stats()
# Returns: {
#   'bull': {'n_samples': 150, 'accuracy': 0.72, ...},
#   'bear': {'n_samples': 45, 'accuracy': 0.58, ...},
#   ...
# }

# Compare same score across regimes
comparison = calibrator.compare_regimes(raw_score=0.65)
# Returns: {
#   'bull': 0.68,
#   'bear': 0.58,
#   'ranging': 0.62,
#   'choppy': 0.55,
# }
```

### Integration Points

**1. CalibratedScoringEngine** (`src/core/calibrated_scoring.py`):
```python
class CalibratedScoringEngine:
    def __init__(self, use_regime_calibration=True):
        if use_regime_calibration:
            self.regime_calibrator = get_regime_calibrator()

    def score_stock(self, symbol, df, ...):
        # After calculating raw_score...
        if self.use_regime_calibration and self.regime_calibrator:
            calibrated = self.regime_calibrator.calibrate(raw_score, regime)
        else:
            calibrated = self.calibrator.calibrate(raw_score)  # Global
```

**2. TradeTracker** (`src/insights/trade_tracker.py`):
```python
def update_trade_outcome(self, trade_id, exit_price, ...):
    # After determining outcome...
    if self.enable_regime_calibration and self.regime_calibrator:
        self.regime_calibrator.record_outcome(
            raw_score=trade['predicted_probability'],
            actual_outcome=1 if was_correct else 0,
            regime=trade['regime']
        )
```

### Storage

Calibration data persists to `data/calibration/regime_calibration.json`:
```json
{
  "regimes": [
    {
      "regime": "bull",
      "history": [[0.65, 1], [0.70, 1], [0.55, 0], ...],
      "n_samples": 150,
      "accuracy": 0.72,
      "last_fit": "2026-01-19T20:55:00"
    },
    ...
  ],
  "global_history": [[0.60, 1], ...],
  "last_updated": "2026-01-19T20:58:00"
}
```

---

## Configuration System

### UI Configuration (`ui/components/engine_config.py`)

**EngineConfig Dataclass**:
```python
@dataclass
class EngineConfig:
    use_7_model: bool = True           # Use 7-model vs 4-model
    min_model_agreement: int = 3       # Min models agreeing (1-7)
    capital: float = 100000.0          # Trading capital (INR)
    risk_per_trade: float = 0.02       # Risk per trade (2%)
    min_confidence: float = 0.55       # Min confidence to show
    min_liquidity_cr: float = 5.0      # Min liquidity (Crores)
    max_atr_pct: float = 6.0           # Max volatility (ATR%)
    strong_buy_agreement: int = 5      # For strong buy
    moderate_buy_agreement: int = 3    # For moderate buy
    top_n_picks: int = 10              # Display count
    show_warnings: bool = True
```

**Preset Configurations**:

| Preset | Agreement | Confidence | Risk | Use Case |
|--------|-----------|------------|------|----------|
| **Conservative** | 5/7 | 60% | 1% | Safe, high-probability |
| **Balanced** (Recommended) | 3/7 | 55% | 2% | Default |
| **Aggressive** | 2/7 | 50% | 3% | More signals |
| **Basic 4-Model** | 2/4 | 55% | 2% | Fallback mode |

---

### Thresholds Configuration (`config/thresholds.py`)

**Signal Thresholds**:
```python
@dataclass(frozen=True)
class SignalThresholds:
    strong_buy: float = 0.70      # 5+ agreement required
    buy: float = 0.60             # 3+ agreement required
    weak_buy: float = 0.55        # Smaller position
    hold_upper: float = 0.55
    hold_lower: float = 0.45
    sell_signals_enabled: bool = False  # DISABLED
```

**Ensemble Weights**:
```python
@dataclass(frozen=True)
class EnsembleWeights:
    # 7-model weights
    base_technical_7m: float = 0.25
    physics_7m: float = 0.18
    math_7m: float = 0.14
    regime_7m: float = 0.13
    macro_7m: float = 0.10
    alternative_7m: float = 0.10
    advanced_7m: float = 0.10
```

**Confidence Calibration**:
```python
@dataclass(frozen=True)
class ConfidenceCalibration:
    # Score range → Expected accuracy
    very_high_score_accuracy: float = 0.65   # 75-100%
    high_score_accuracy: float = 0.60        # 65-75%
    moderate_score_accuracy: float = 0.55    # 55-65%
    neutral_score_accuracy: float = 0.50     # 45-55%
    below_neutral_accuracy: float = 0.45     # 0-45%

    # Model agreement bonuses (base 4-model values)
    four_model_agreement_bonus: float = 0.08   # All 4 agree
    three_model_agreement_bonus: float = 0.05  # 3 agree
    two_model_agreement_bonus: float = 0.02    # 2 agree
    one_model_agreement_bonus: float = 0.00    # 1 only
    no_agreement_penalty: float = -0.05        # No agreement

    # For 7-model ensemble, bonuses are derived at runtime:
    # 7/7 = four_model_bonus + 0.03 = 0.11
    # 6/7 = four_model_bonus = 0.08
    # 5/7 = three_model_bonus + 0.01 = 0.06
    # 4/7 = three_model_bonus = 0.05
    # 3/7 = two_model_bonus = 0.02

    max_confidence: float = 0.80  # Cap at 80%
    min_confidence: float = 0.35  # Floor at 35%
```

---

## UI Architecture

### Main Application (`ui/app.py`)

**Tabs**:
1. **Top Picks** - Auto-scan for best opportunities
2. **Deep Analysis** - Single stock comprehensive analysis
3. **Sectors** - Sector-wise stock analysis
4. **Screener** - Basic filtering (price, cap, momentum)
5. **About** - Documentation

**Sidebar Settings**:
- Configuration preset selector
- 7-Model toggle
- Min model agreement slider
- Filter settings (confidence, liquidity, volatility)
- Trading settings (capital, risk per trade)
- Display settings

### Unified Prediction Flow

All tabs use the same prediction path:
```python
from ui.components.engine_config import get_unified_prediction

# In any tab:
result = get_unified_prediction(
    symbol="RELIANCE",
    df=price_data,
    trade_type="INTRADAY"
)

# Result contains:
# - signal, confidence, model_agreement
# - model_votes: {model_name: 'BUY'/'HOLD'}
# - entry_price, stop_loss, target_price
# - reasons, warnings
# - all 7 individual model scores
```

---

## Key Classes Reference

### Scoring Classes

| Class | File | Purpose |
|-------|------|---------|
| `EnhancedScoringEngine` | `src/models/enhanced_scoring.py` | Main 7-model ensemble |
| `ScoringEngine` | `src/models/scoring.py` | Base technical scoring |
| `CalibratedScoringEngine` | `src/core/calibrated_scoring.py` | Production scoring with regime calibration |
| `PhysicsEngine` | `src/physics/physics_engine.py` | Physics models orchestrator |
| `MathEngine` | `src/math_models/math_engine.py` | Math models orchestrator |
| `HMMRegimeDetector` | `src/ml/hmm_regime_detector.py` | Regime detection |
| `MacroEngine` | `src/macro/macro_engine.py` | Macro analysis |
| `AlternativeDataEngine` | `src/alternative_data/alternative_engine.py` | Alt data analysis |
| `AdvancedPredictionEngine` | `src/core/advanced_engine.py` | Advanced math |

### Adapter Classes

| Class | File | Purpose |
|-------|------|---------|
| `BaseModelAdapter` | `src/core/adapters/base_adapter.py` | Abstract base with fallback logic |
| `TechnicalModelAdapter` | `src/core/adapters/technical_adapter.py` | Wraps ScoringEngine |
| `PhysicsModelAdapter` | `src/core/adapters/physics_adapter.py` | Wraps PhysicsEngine |
| `MathModelAdapter` | `src/core/adapters/math_adapter.py` | Wraps MathEngine |
| `RegimeModelAdapter` | `src/core/adapters/regime_adapter.py` | Wraps HMMRegimeDetector |
| `MacroModelAdapter` | `src/core/adapters/macro_adapter.py` | Wraps MacroEngine |
| `AlternativeDataAdapter` | `src/core/adapters/alternative_adapter.py` | Wraps AlternativeDataEngine |
| `AdvancedMathAdapter` | `src/core/adapters/advanced_adapter.py` | Wraps AdvancedPredictionEngine |

### Execution Classes

| Class | File | Purpose |
|-------|------|---------|
| `AdvancedTradeExecutor` | `src/execution/trade_executor.py` | Trade lifecycle management |
| `TrailingStopCalculator` | `src/execution/trailing_stop.py` | Multiple trailing stop methods |
| `GapHandler` | `src/execution/gap_handler.py` | Gap detection and handling |
| `ReentryManager` | `src/execution/reentry_manager.py` | Re-entry rules after stops |
| `Backtester` | `src/models/backtester.py` | Walk-forward backtesting with advanced execution |

### RL Classes

| Class | File | Purpose |
|-------|------|---------|
| `ExperienceCollector` | `src/rl/experience_collector.py` | Collects real trade experiences |
| `DQNTrainingScheduler` | `src/rl/dqn_trainer.py` | Manages periodic DQN training |
| `RewardCalculator` | `src/rl/reward_shaping.py` | Configurable reward functions |
| `TradingState` | `src/rl/experience_collector.py` | 10-feature state for DQN |
| `RealTradeExperience` | `src/rl/experience_collector.py` | Complete trade experience |

### Calibration Classes

| Class | File | Purpose |
|-------|------|---------|
| `RegimeSpecificCalibrator` | `src/core/regime_calibration.py` | Per-regime IsotonicRegression |
| `EmpiricalCalibrator` | `src/core/calibrated_scoring.py` | Global calibration curve |
| `AdaptiveWeightManager` | `src/core/calibrated_scoring.py` | Learns optimal model weights |

### Data Classes

| Class | File | Fields |
|-------|------|--------|
| `EnhancedStockScore` | `enhanced_scoring.py` | All 7 scores, signal, confidence, trade levels |
| `StandardizedModelOutput` | `src/core/interfaces/model_output.py` | p_buy, uncertainty, coverage, reasoning |
| `ModelUncertainty` | `src/core/interfaces/model_output.py` | type, value, lower_bound, upper_bound, quantiles |
| `PhysicsScore` | `physics_engine.py` | 4 sub-scores, strategy, reasons |
| `MathScore` | `math_engine.py` | 4 sub-scores, predictability |
| `RegimeResult` | `hmm_regime_detector.py` | Regime, probability, stability |
| `MacroSignal` | `macro_engine.py` | Signal (-1 to +1), drivers |
| `AlternativeDataScore` | `alternative_engine.py` | Probability, signal strength |
| `AdvancedPredictionResult` | `advanced_engine.py` | Probability, regime, features |
| `ExecutionConfig` | `src/execution/trade_executor.py` | Trailing stops, partials, gaps |
| `TradeResult` | `src/models/backtester.py` | Trade outcome with partial_exits, trailing info |
| `RewardConfig` | `src/rl/reward_shaping.py` | Reward function parameters |

### Configuration Classes

| Class | File | Purpose |
|-------|------|---------|
| `EngineConfig` | `ui/components/engine_config.py` | UI settings |
| `SignalThresholds` | `config/thresholds.py` | Signal generation |
| `EnsembleWeights` | `config/thresholds.py` | Model weights |
| `ConfidenceCalibration` | `config/thresholds.py` | Score to confidence |

---

## Accuracy & Calibration

### Backtest Results

| Metric | Value | Notes |
|--------|-------|-------|
| BUY Accuracy (4-model) | 57.6% | 330 trades |
| BUY Accuracy (7-model) | 65-68% | Target with 5+ agreement |
| SELL Accuracy | 28.6% | DISABLED (inverted) |
| Calibration Error | 1.9% | Confidence vs actual |

### Confidence Calibration Table

| Ensemble Score | Model Agreement | Expected Accuracy |
|---------------|-----------------|-------------------|
| 0.75+ | 7/7 | 76% |
| 0.75+ | 6/7 | 73% |
| 0.75+ | 5/7 | 71% |
| 0.65-0.75 | 5+/7 | 68% |
| 0.65-0.75 | 3-4/7 | 65% |
| 0.55-0.65 | 3+/7 | 60% |
| 0.55-0.65 | <3/7 | 55% |
| 0.45-0.55 | Any | 50% (neutral) |
| <0.45 | Any | 45% |

---

## Execution Paths

### CLI Analysis
```bash
python main.py analyze RELIANCE
```
Flow: `main.py` → `EnhancedSignalGenerator` → `EnhancedScoringEngine` → Display

### CLI Batch Run
```bash
python main.py run --symbols TCS,INFY,RELIANCE
```
Flow: `main.py` → Iterate symbols → Score each → Filter → Rank → Output top picks

### UI Dashboard
```bash
python main.py ui
```
Flow: `streamlit run ui/app.py` → User selects stock → `get_unified_prediction()` → Display

### Configuration Change
1. User adjusts sidebar settings
2. `EngineConfig` updated in session state
3. `get_scoring_engine()` cache invalidated
4. New engine created with updated config
5. All subsequent predictions use new settings

---

## Quick Reference

### Starting the System
```bash
# CLI - Single stock
python main.py analyze RELIANCE

# CLI - Batch analysis
python main.py run --symbols TCS,INFY

# UI Dashboard
python main.py ui
```

### Key Files to Modify

| Change | File |
|--------|------|
| Model weights | `config/thresholds.py` |
| Signal thresholds | `config/thresholds.py` |
| UI presets | `ui/components/engine_config.py` |
| Add new model | `src/models/enhanced_scoring.py` |
| Technical indicators | `src/features/technical.py` |
| Stock universe | `config/symbols.json` |
| Macro sector mappings | `config/sector_dependencies.json` |

### Recommended Configuration
- **Preset**: Balanced
- **Model**: 7-Model Ensemble
- **Agreement**: 3/7 minimum
- **Confidence**: 55% minimum
- **Risk**: 2% per trade

---

## Mathematical Formulas

### Kelly Criterion (Position Sizing)
```
f* = (b × p - q) / b

where:
  f* = optimal fraction of capital
  b  = odds (reward / risk)
  p  = probability of winning
  q  = 1 - p
```

### RSI (Relative Strength Index)
```
RS = Average Gain / Average Loss
RSI = 100 - (100 / (1 + RS))
```

### Ensemble Score Calculation
```
ensemble_score = Σ(weight_i × model_score_i)

where weights sum to 1.0 and model_scores ∈ [0, 1]
```

### Confidence Calibration
```
confidence = base_accuracy(score_range) + agreement_bonus(model_agreement)
confidence = max(0.35, min(0.80, confidence))
```

---

## Warnings and Limitations

1. **Past performance does not guarantee future results**
2. **SELL signals are disabled due to poor accuracy (28.6%)**
3. **Maximum confidence capped at 80%**
4. **Requires 60+ days of historical data**
5. **Indian market hours (9:15 AM - 3:30 PM IST) only**
6. **News sentiment requires OpenAI API key**
7. **Macro data requires internet connectivity**

---

---

## Trading Agent Integration

The signal engine is now fully integrated with an automated trading agent (see root `ARCHITECTURE.md`):

```
Signal Engine (this document)          Trading Agent (ARCHITECTURE.md)
+---------------------------+          +---------------------------+
| src/signals/generator.py  |          | agent/orchestrator.py     |
| src/models/scoring.py     |   --->   | agent/signal_adapter.py   |
| src/features/technical.py |          | broker/upstox.py          |
| src/execution/trailing_stop|   --->   | risk/manager.py           |
| src/data/news_fetcher.py  |   --->   | config/trading_config.py  |
+---------------------------+          +---------------------------+
```

### Connected Components (as of Feb 2026)

| Component | Status | Connection |
|-----------|--------|------------|
| `src/signals/generator.py` | CONNECTED | Called by `agent/signal_adapter.py` |
| `src/features/technical.py` | CONNECTED | Used for Supertrend/ADX filtering |
| `src/execution/trailing_stop.py` | CONNECTED | Used by orchestrator for position monitoring |
| `src/data/news_fetcher.py` | CONNECTED | Used by orchestrator for live news monitoring |
| `src/features/news_features.py` | CONNECTED | Imported by orchestrator.py for held position sentiment |
| `src/models/enhanced_scoring.py` | NOT CONNECTED | 7-model ensemble (future deployment) |
| `src/execution/trade_executor.py` | NOT CONNECTED | Advanced executor (future) |
| `src/execution/gap_handler.py` | NOT CONNECTED | Gap handling (future) |
| `src/execution/reentry_manager.py` | NOT CONNECTED | Re-entry rules (future) |
| `src/rl/` | NOT CONNECTED | RL feedback loop (future) |

### Signal Filter Pipeline (Trading Agent)

After the signal engine produces predictions, the trading agent applies additional filters:

```
SignalGenerator.run() -> predictions
    -> SignalAdapter._convert_prediction() -> TradeSignal
    -> SignalAdapter._passes_filters():
        1. Decision must be BUY or SELL
        2. Confidence >= CONFIG.signals.min_confidence (default 0.60)
        3. Risk:Reward >= CONFIG.signals.min_risk_reward (default 1.5)
        4. Within entry window (CONFIG.hours, default 9:30-14:30)
        5. Valid prices (entry, SL, target > 0)
        6. SL < entry and target > entry (for BUY)
        7. Supertrend confirmation (UP for BUY, DOWN for SELL)
        8. ADX >= CONFIG.strategy.adx_strong_trend (default 25)
    -> TradingOrchestrator._enter_position()
```

### Key Fixes Applied to Signal Engine

| Fix | File | Description |
|-----|------|-------------|
| VWAP daily reset | `src/features/technical.py` | Changed from cumulative to daily-reset via `groupby(date)` |
| DX division by zero | `src/features/technical.py` | `replace(0, np.nan)` guards on ATR and DI sum |
| Trailing stop zero guard | `src/execution/trailing_stop.py` | Division by zero protection in `calculate()` method |
| Orchestrator risk_pct guard | `agent/orchestrator.py` | `CONFIG.capital.max_per_trade_risk_pct or 0.02` in init |

---

*Last Updated: February 5, 2026*
*Version: 4.1 - 7-Model Ensemble with Advanced Execution, RL Feedback, Regime Calibration, and Trading Agent Integration*
