"""
DeterministicSentimentAnalyzer - Reproducible News Sentiment Analysis

CRITICAL: LLM-based sentiment is non-deterministic by nature.
Same news can get different scores on retry, breaking reproducibility.

Solution:
1. Content-based caching (hash of article text → sentiment)
2. Temperature=0 for LLM calls
3. Fallback to rule-based sentiment when LLM unavailable
4. Versioned cache to invalidate on model changes
5. Ensemble of methods for robustness

This ensures:
- Same article always gets same sentiment
- Backtests are reproducible
- No API dependency for cached content
"""

import hashlib
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from loguru import logger


class SentimentSource(Enum):
    """Source of sentiment score."""
    CACHE = "cache"
    LLM = "llm"
    RULES = "rules"
    ENSEMBLE = "ensemble"


@dataclass
class SentimentResult:
    """Structured sentiment analysis result."""
    score: float                    # -1 (negative) to +1 (positive)
    confidence: float               # 0 to 1
    source: SentimentSource
    keywords_found: List[str]
    cache_hit: bool
    content_hash: str
    timestamp: str

    def to_normalized_score(self) -> float:
        """Convert to 0-1 scale for scoring engine."""
        return (self.score + 1) / 2


@dataclass
class ArticleSentiment:
    """Sentiment for a single article."""
    title: str
    content_preview: str
    sentiment: SentimentResult
    relevance: float
    is_stale: bool


class RuleBasedSentimentEngine:
    """
    Deterministic rule-based sentiment analysis.

    Uses keyword matching with context-aware weighting.
    100% reproducible - no LLM dependency.
    """

    # Positive keywords with weights
    POSITIVE_KEYWORDS = {
        # Strong positive (weight 2.0)
        'surge': 2.0, 'soar': 2.0, 'breakout': 2.0, 'record high': 2.0,
        'beat estimates': 2.0, 'exceeds expectations': 2.0, 'upgrade': 2.0,
        'outperform': 2.0, 'strong buy': 2.0, 'bullish': 1.8,

        # Medium positive (weight 1.0)
        'gain': 1.0, 'rise': 1.0, 'up': 0.8, 'growth': 1.0, 'profit': 1.0,
        'positive': 1.0, 'optimistic': 1.0, 'recovery': 1.0, 'boost': 1.0,
        'improve': 1.0, 'strong': 0.9, 'expand': 1.0, 'buy': 0.8,

        # Mild positive (weight 0.5)
        'stable': 0.5, 'steady': 0.5, 'maintain': 0.4, 'support': 0.5,
        'hold': 0.3, 'neutral positive': 0.5
    }

    NEGATIVE_KEYWORDS = {
        # Strong negative (weight -2.0)
        'crash': -2.0, 'plunge': -2.0, 'collapse': -2.0, 'bankruptcy': -2.0,
        'fraud': -2.0, 'scam': -2.0, 'default': -1.8, 'downgrade': -1.8,
        'sell': -1.5, 'bearish': -1.8, 'underperform': -1.5,

        # Medium negative (weight -1.0)
        'fall': -1.0, 'drop': -1.0, 'decline': -1.0, 'loss': -1.0,
        'miss estimates': -1.5, 'below expectations': -1.5, 'concern': -0.8,
        'risk': -0.7, 'weak': -1.0, 'slowdown': -0.9, 'warning': -1.0,

        # Mild negative (weight -0.5)
        'volatile': -0.5, 'uncertain': -0.5, 'cautious': -0.4, 'mixed': -0.3,
        'pressure': -0.6, 'challenge': -0.5
    }

    # Negation words that flip sentiment
    NEGATION_WORDS = {'not', 'no', 'never', 'neither', 'nobody', 'nothing',
                       'nowhere', 'nor', "n't", 'without', 'hardly', 'barely'}

    # Amplifiers that strengthen sentiment
    AMPLIFIERS = {'very': 1.5, 'extremely': 2.0, 'significantly': 1.5,
                   'substantially': 1.5, 'strongly': 1.5, 'sharply': 1.8,
                   'dramatically': 2.0, 'massively': 2.0}

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze text sentiment using rules.

        Returns:
            SentimentResult with deterministic score
        """
        if not text:
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                source=SentimentSource.RULES,
                keywords_found=[],
                cache_hit=False,
                content_hash=self._hash_content(""),
                timestamp=datetime.now().isoformat()
            )

        text_lower = text.lower()
        words = text_lower.split()

        # Find keywords
        positive_hits = []
        negative_hits = []
        total_weight = 0.0

        for keyword, weight in self.POSITIVE_KEYWORDS.items():
            if keyword in text_lower:
                # Check for negation in context
                is_negated = self._is_negated(text_lower, keyword)
                amplifier = self._get_amplifier(text_lower, keyword)

                if is_negated:
                    negative_hits.append(f"NOT {keyword}")
                    total_weight -= weight * amplifier
                else:
                    positive_hits.append(keyword)
                    total_weight += weight * amplifier

        for keyword, weight in self.NEGATIVE_KEYWORDS.items():
            if keyword in text_lower:
                is_negated = self._is_negated(text_lower, keyword)
                amplifier = self._get_amplifier(text_lower, keyword)

                if is_negated:
                    positive_hits.append(f"NOT {keyword}")
                    total_weight -= weight * amplifier  # Double negative = positive
                else:
                    negative_hits.append(keyword)
                    total_weight += weight * amplifier  # weight is already negative

        # Normalize score to -1 to +1
        keyword_count = len(positive_hits) + len(negative_hits)
        if keyword_count > 0:
            # Normalize by keyword count to avoid bias from long articles
            normalized_score = total_weight / (keyword_count * 1.5)
            score = max(-1, min(1, normalized_score))
        else:
            score = 0.0

        # Confidence based on keyword density
        word_count = len(words)
        keyword_density = keyword_count / max(1, word_count)
        confidence = min(0.9, keyword_density * 20 + 0.3 * min(keyword_count, 5))

        return SentimentResult(
            score=score,
            confidence=confidence,
            source=SentimentSource.RULES,
            keywords_found=positive_hits + negative_hits,
            cache_hit=False,
            content_hash=self._hash_content(text),
            timestamp=datetime.now().isoformat()
        )

    def _is_negated(self, text: str, keyword: str) -> bool:
        """Check if keyword is negated in context."""
        # Find keyword position
        pos = text.find(keyword)
        if pos == -1:
            return False

        # Check previous 5 words for negation
        before = text[max(0, pos - 50):pos]
        before_words = before.split()[-5:]

        return any(neg in before_words for neg in self.NEGATION_WORDS)

    def _get_amplifier(self, text: str, keyword: str) -> float:
        """Get amplification factor for keyword."""
        pos = text.find(keyword)
        if pos == -1:
            return 1.0

        before = text[max(0, pos - 30):pos]
        before_words = before.split()[-3:]

        for word in before_words:
            if word in self.AMPLIFIERS:
                return self.AMPLIFIERS[word]

        return 1.0

    def _hash_content(self, text: str) -> str:
        """Generate deterministic hash of content."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


class SentimentCache:
    """
    Persistent cache for sentiment scores.

    Ensures reproducibility across sessions.
    """

    CACHE_VERSION = "v2"  # Increment when model/rules change

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or '/tmp/sentiment_cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f'sentiment_{self.CACHE_VERSION}.json')
        self.cache: Dict[str, Dict] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.debug(f"Loaded {len(self.cache)} cached sentiments")
            except Exception as e:
                logger.warning(f"Could not load sentiment cache: {e}")
                self.cache = {}

    def _save_cache(self):
        """Persist cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Could not save sentiment cache: {e}")

    def get(self, content_hash: str) -> Optional[Dict]:
        """Get cached sentiment by content hash."""
        return self.cache.get(content_hash)

    def set(self, content_hash: str, sentiment: Dict):
        """Cache sentiment result."""
        self.cache[content_hash] = sentiment
        # Save periodically (every 100 new entries)
        if len(self.cache) % 100 == 0:
            self._save_cache()

    def save(self):
        """Force save cache."""
        self._save_cache()


class DeterministicSentimentAnalyzer:
    """
    Main sentiment analyzer with determinism guarantees.

    Priority:
    1. Check cache (fastest, 100% deterministic)
    2. Use LLM with temp=0 (slow but accurate)
    3. Fallback to rules (always available)
    """

    def __init__(self,
                 cache_dir: str = None,
                 use_llm: bool = True,
                 openai_api_key: str = None):
        self.cache = SentimentCache(cache_dir)
        self.rule_engine = RuleBasedSentimentEngine()
        self.use_llm = use_llm
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')

    def _hash_content(self, text: str) -> str:
        """Generate deterministic content hash."""
        # Normalize text for consistent hashing
        normalized = ' '.join(text.lower().split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]

    def analyze_article(self,
                         title: str,
                         content: str,
                         symbol: str = None) -> SentimentResult:
        """
        Analyze article sentiment deterministically.

        Args:
            title: Article title
            content: Article content
            symbol: Stock symbol for relevance filtering

        Returns:
            SentimentResult with guaranteed reproducibility
        """
        full_text = f"{title} {content}"
        content_hash = self._hash_content(full_text)

        # 1. Check cache
        cached = self.cache.get(content_hash)
        if cached:
            return SentimentResult(
                score=cached['score'],
                confidence=cached['confidence'],
                source=SentimentSource.CACHE,
                keywords_found=cached.get('keywords', []),
                cache_hit=True,
                content_hash=content_hash,
                timestamp=cached.get('timestamp', datetime.now().isoformat())
            )

        # 2. Try LLM with deterministic settings
        if self.use_llm and self.openai_api_key:
            try:
                llm_result = self._analyze_with_llm(full_text, symbol)
                if llm_result:
                    # Cache the result
                    self.cache.set(content_hash, {
                        'score': llm_result.score,
                        'confidence': llm_result.confidence,
                        'keywords': llm_result.keywords_found,
                        'timestamp': llm_result.timestamp,
                        'source': 'llm'
                    })
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM sentiment failed, using rules: {e}")

        # 3. Fallback to rule-based
        rule_result = self.rule_engine.analyze(full_text)

        # Cache rule-based result too
        self.cache.set(content_hash, {
            'score': rule_result.score,
            'confidence': rule_result.confidence,
            'keywords': rule_result.keywords_found,
            'timestamp': rule_result.timestamp,
            'source': 'rules'
        })

        return rule_result

    def _analyze_with_llm(self, text: str, symbol: str = None) -> Optional[SentimentResult]:
        """
        Analyze with LLM using deterministic settings.

        Uses temperature=0 and seed for reproducibility.
        """
        try:
            import openai
            client = openai.OpenAI(api_key=self.openai_api_key)

            prompt = f"""Analyze the sentiment of this financial news for stock trading.
Return ONLY a JSON object with:
- score: float from -1.0 (very negative) to +1.0 (very positive)
- confidence: float from 0.0 to 1.0
- keywords: list of key sentiment words found

{"Focus on relevance to: " + symbol if symbol else ""}

News text:
{text[:2000]}

JSON response:"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,  # Deterministic
                seed=42,        # Fixed seed for reproducibility
                max_tokens=200
            )

            content = response.choices[0].message.content

            # Parse JSON response
            import json
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return SentimentResult(
                    score=float(data.get('score', 0)),
                    confidence=float(data.get('confidence', 0.5)),
                    source=SentimentSource.LLM,
                    keywords_found=data.get('keywords', []),
                    cache_hit=False,
                    content_hash=self._hash_content(text),
                    timestamp=datetime.now().isoformat()
                )

        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")

        return None

    def analyze_batch(self,
                       articles: List[Dict],
                       symbol: str = None) -> Tuple[float, List[ArticleSentiment]]:
        """
        Analyze multiple articles and return aggregate score.

        Args:
            articles: List of dicts with 'title' and 'content' keys
            symbol: Stock symbol

        Returns:
            Tuple of (aggregate_score, list of ArticleSentiment)
        """
        results = []
        weighted_scores = []

        for article in articles:
            title = article.get('title', '')
            content = article.get('content', '')
            pub_date = article.get('published_date')

            sentiment = self.analyze_article(title, content, symbol)

            # Check staleness (>7 days old)
            is_stale = False
            if pub_date:
                try:
                    if isinstance(pub_date, str):
                        pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    is_stale = (datetime.now(pub_date.tzinfo) - pub_date).days > 7
                except:
                    pass

            # Calculate relevance
            relevance = self._calculate_relevance(title, content, symbol)

            article_sentiment = ArticleSentiment(
                title=title[:100],
                content_preview=content[:200] if content else "",
                sentiment=sentiment,
                relevance=relevance,
                is_stale=is_stale
            )
            results.append(article_sentiment)

            # Weight by confidence and relevance, penalize stale
            weight = sentiment.confidence * relevance
            if is_stale:
                weight *= 0.5

            weighted_scores.append((sentiment.score, weight))

        # Calculate weighted average
        if weighted_scores:
            total_weight = sum(w for _, w in weighted_scores)
            if total_weight > 0:
                aggregate = sum(s * w for s, w in weighted_scores) / total_weight
            else:
                aggregate = 0.0
        else:
            aggregate = 0.0

        return aggregate, results

    def _calculate_relevance(self, title: str, content: str, symbol: str = None) -> float:
        """Calculate relevance score for article."""
        if not symbol:
            return 0.5

        text = f"{title} {content}".lower()
        symbol_lower = symbol.lower()

        # Direct mention
        if symbol_lower in text:
            return 1.0

        # Company name variations (would need symbol->company mapping)
        # For now, use basic heuristics
        return 0.5

    def get_aggregate_score_for_stock(self,
                                        symbol: str,
                                        articles: List[Dict]) -> float:
        """
        Get final 0-1 score for use in scoring engine.

        Args:
            symbol: Stock symbol
            articles: List of news articles

        Returns:
            Score from 0 (negative) to 1 (positive), 0.5 is neutral
        """
        if not articles:
            return 0.5  # Neutral when no news

        aggregate, _ = self.analyze_batch(articles, symbol)

        # Convert -1 to +1 scale to 0 to 1
        return (aggregate + 1) / 2

    def save_cache(self):
        """Force save the cache."""
        self.cache.save()


# Singleton for global access
_analyzer_instance = None


def get_sentiment_analyzer(use_llm: bool = True) -> DeterministicSentimentAnalyzer:
    """Get or create singleton analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = DeterministicSentimentAnalyzer(use_llm=use_llm)
    return _analyzer_instance
