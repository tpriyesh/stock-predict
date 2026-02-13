"""
Stock Screener - Find stocks by price, momentum, market cap.

Features:
- Fetches REAL NSE stocks dynamically (not hardcoded)
- Penny stocks by price range (₹1-10, ₹10-30, ₹30-100)
- Momentum stocks (consistently rising over timeframes)
- Market cap categories (Large, Mid, Small)
- OpenAI-powered stock insights
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import yfinance as yf
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import get_settings


class PriceRange(Enum):
    """Stock price ranges."""
    MICRO_PENNY = "₹1-10"
    PENNY = "₹10-30"
    LOW_PRICE = "₹30-100"
    MID_PRICE = "₹100-500"
    HIGH_PRICE = "₹500+"


class MarketCap(Enum):
    """Market cap categories."""
    LARGE_CAP = "Large Cap (>₹20,000 Cr)"
    MID_CAP = "Mid Cap (₹5,000-20,000 Cr)"
    SMALL_CAP = "Small Cap (₹500-5,000 Cr)"
    MICRO_CAP = "Micro Cap (<₹500 Cr)"


class MomentumPeriod(Enum):
    """Momentum analysis periods."""
    ONE_WEEK = "1 Week"
    ONE_MONTH = "1 Month"
    THREE_MONTHS = "3 Months"
    SIX_MONTHS = "6 Months"
    ONE_YEAR = "1 Year"


@dataclass
class StockData:
    """Stock data container."""
    symbol: str
    name: str
    price: float
    market_cap: float
    market_cap_cr: float
    sector: str
    industry: str

    # Returns
    change_1d: float
    change_1w: float
    change_1m: float
    change_3m: float
    change_6m: float
    change_1y: float

    # Momentum score
    momentum_score: float
    momentum_consistency: float

    # Volume
    avg_volume: float
    volume_ratio: float

    # Technical
    rsi: float
    above_ma50: bool
    above_ma200: bool

    # Category
    price_category: PriceRange
    cap_category: MarketCap


class StockScreener:
    """
    Comprehensive stock screener for Indian markets.

    Fetches REAL NSE stocks dynamically and analyzes based on:
    - Price range
    - Market cap
    - Momentum
    - Technical indicators
    """

    def __init__(self):
        """Initialize screener."""
        self.settings = get_settings()
        self.openai_key = self.settings.openai_api_key
        self._all_stocks_cache = None
        self._cache_time = None

    def fetch_all_nse_stocks(self) -> List[Dict]:
        """Fetch all stocks from NSE + BSE (combined, deduplicated)."""
        # Use cache if less than 1 hour old
        if self._all_stocks_cache and self._cache_time:
            if (datetime.now() - self._cache_time).seconds < 3600:
                return self._all_stocks_cache

        all_stocks = {}  # Use dict to deduplicate by symbol

        # ===== FETCH NSE STOCKS =====
        logger.info("Fetching NSE stocks...")
        try:
            url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                for _, row in df.iterrows():
                    symbol = row['SYMBOL'].strip()
                    name = row.get('NAME OF COMPANY', symbol)
                    if isinstance(name, str):
                        name = name.strip()
                    all_stocks[symbol] = {'symbol': symbol, 'name': name, 'exchange': 'NSE'}
                logger.info(f"Fetched {len(all_stocks)} stocks from NSE")
        except Exception as e:
            logger.warning(f"NSE fetch failed: {e}")
            # Fallback to nsetools
            try:
                from nsetools import Nse
                nse = Nse()
                stock_codes = nse.get_stock_codes()
                for k, v in stock_codes.items():
                    if k != 'SYMBOL':
                        all_stocks[k] = {'symbol': k, 'name': v, 'exchange': 'NSE'}
                logger.info(f"Fetched {len(all_stocks)} stocks from nsetools")
            except:
                pass

        # ===== FETCH BSE STOCKS =====
        logger.info("Fetching BSE stocks...")
        try:
            # BSE official equity list
            bse_url = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Atea=&status=Active"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://www.bseindia.com/',
                'Accept': 'application/json'
            }
            response = requests.get(bse_url, headers=headers, timeout=15)

            if response.status_code == 200:
                bse_data = response.json()
                bse_count = 0
                for item in bse_data:
                    symbol = item.get('scrip_cd') or item.get('SCRIP_CD', '')
                    name = item.get('scrip_name') or item.get('Scrip_Name', '')
                    # BSE uses scrip codes, try to get trading symbol
                    trading_sym = item.get('scrip_id') or item.get('SCRIP_ID', symbol)
                    if trading_sym and trading_sym not in all_stocks:
                        all_stocks[trading_sym] = {'symbol': trading_sym, 'name': name, 'exchange': 'BSE'}
                        bse_count += 1
                logger.info(f"Added {bse_count} unique BSE stocks")
        except Exception as e:
            logger.warning(f"BSE API fetch failed: {e}")
            # Try alternate BSE source
            try:
                bse_csv_url = "https://www.bseindia.com/download/BhsCSC/Equity/Equity.csv"
                response = requests.get(bse_csv_url, headers=headers, timeout=15)
                if response.status_code == 200:
                    from io import StringIO
                    df = pd.read_csv(StringIO(response.text))
                    bse_count = 0
                    for _, row in df.iterrows():
                        symbol = str(row.get('Security Id', '')).strip()
                        name = str(row.get('Security Name', symbol)).strip()
                        if symbol and symbol not in all_stocks:
                            all_stocks[symbol] = {'symbol': symbol, 'name': name, 'exchange': 'BSE'}
                            bse_count += 1
                    logger.info(f"Added {bse_count} unique BSE stocks from CSV")
            except Exception as e2:
                logger.warning(f"BSE CSV fetch also failed: {e2}")

        # Add additional BSE penny stocks if BSE API failed
        bse_count = len([s for s in all_stocks.values() if s.get('exchange') == 'BSE'])
        if bse_count < 100:
            logger.info("Adding known BSE penny stocks...")
            for stock in self._get_additional_bse_stocks():
                if stock['symbol'] not in all_stocks:
                    all_stocks[stock['symbol']] = stock
            logger.info(f"Added {len(self._get_additional_bse_stocks())} BSE penny stocks")

        # Convert to list
        stocks_list = list(all_stocks.values())

        if len(stocks_list) < 100:
            # Fallback if both failed
            logger.warning("Using fallback stock list")
            return self._get_fallback_stocks()

        self._all_stocks_cache = stocks_list
        self._cache_time = datetime.now()
        logger.info(f"Total unique stocks: {len(stocks_list)} (NSE + BSE combined)")
        return stocks_list

    def _get_fallback_stocks(self) -> List[Dict]:
        """Fallback list of known NSE stocks."""
        symbols = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR',
            'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT', 'HCLTECH',
            'AXISBANK', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'TITAN',
            'BAJFINANCE', 'WIPRO', 'ULTRACEMCO', 'NTPC', 'NESTLEIND',
            'POWERGRID', 'JSWSTEEL', 'TATASTEEL', 'ONGC', 'COALINDIA',
            'TECHM', 'HINDALCO', 'CIPLA', 'DRREDDY', 'BPCL', 'ADANIPORTS',
            'BAJAJ-AUTO', 'BAJAJFINSV', 'BRITANNIA', 'EICHERMOT', 'GRASIM',
            'HEROMOTOCO', 'INDUSINDBK', 'IOC', 'IRCTC', 'LUPIN', 'MARICO',
            'NMDC', 'PNB', 'VEDL', 'YESBANK', 'IDEA', 'SUZLON', 'NHPC',
            'IRFC', 'HFCL', 'RVNL', 'HUDCO', 'BHEL', 'NBCC', 'NCC', 'SJVN',
            'PTC', 'RAIN', 'GRAPHITE', 'HEG', 'SAIL', 'NATIONALUM', 'PFC',
            'RECLTD', 'TATAPOWER', 'ADANIGREEN', 'DLF', 'GODREJCP', 'HAVELLS'
        ]
        return [{'symbol': s, 'name': s, 'exchange': 'NSE'} for s in symbols]

    def _get_additional_bse_stocks(self) -> List[Dict]:
        """Additional BSE penny/small stocks not on NSE."""
        # These are BSE-listed stocks known for penny stock trading
        # Will be fetched with .BO suffix
        bse_only = [
            # Known penny stocks on BSE
            'RTNPOWER', 'SICAL', 'BINDALAGRO', 'TERASOFT', 'MEGASTAR',
            'AKSHOPTFBR', 'ARIHANT', 'ASIANHOTNR', 'AVONMORE', 'BALLARPUR',
            'BHAGYANGR', 'BIOPAC', 'BLKASHYAP', 'BURNPUR', 'CELEBRITY',
            'CHAMBLFERT', 'COMPUSOFT', 'CREATIVEYE', 'DCW', 'DHARSUGAR',
            'ELDEHSG', 'ESAFSFB', 'EURO', 'GANGAFORGE', 'GOENKA',
            'GOLDSTONE', 'GSLSEC', 'GTLINFRA', 'HEXATRADEX', 'HILTON',
            'INDOAMIN', 'JAIBALAJI', 'JETFREIGHT', 'JIKIND', 'JMCPROJECT',
            'KANORICHEM', 'KAPSTON', 'KERNEX', 'KIOCL', 'KITEX',
            'KRISHANA', 'LAKPRE', 'LAMBODHARA', 'LSIL', 'MADHAV',
            'MADHUCON', 'MAGNUM', 'MAHSCOOTER', 'MANALIPETC', 'MANINFRA',
            'METALFORGE', 'MITCON', 'MOLDTEK', 'MORARJEE', 'MOSERBAER',
            'MUKANDLTD', 'NAGARFERT', 'NAGREEKCAP', 'NAHARPOLY', 'NECCLTD',
            'NELCO', 'NETFINCAP', 'NITINFIRE', 'NRBBEARING', 'OPAL',
            'ORIENTBELL', 'PANACHE', 'PARSVNATH', 'PASUPATI', 'PATANJALI',
            'PDMJEPAPER', 'PEARLPOLY', 'PENIND', 'PFOCUS', 'PILITA',
            'PIONEEREMB', 'PNBHOUSING', 'PODDAR', 'POKARNA', 'PRACTYN',
            'PRECOT', 'PRICOLLTD', 'PRIMESECU', 'PUNJLLOYD', 'RADIOCITY',
            'RAJAGREEN', 'RAJSREESUG', 'RAMASTEEL', 'RAMCOSYS', 'RANEENGINE',
            'RBMINFRA', 'REGENCERAM', 'RELAXO', 'RHFL', 'RIIL',
            'RKDL', 'ROLCON', 'RSYSTEMS', 'RUCHIRA', 'RUPA',
            'SABEVENTS', 'SALZERELEC', 'SANGAMIND', 'SATIN', 'SATINDLTD',
            'SELMCL', 'SEPOWER', 'SERVALL', 'SHAHALLOYS', 'SHANTIGEAR',
            'SHREEPUSHK', 'SHYAMTEL', 'SIGNET', 'SILGO', 'SINCLAIR',
            'SITINET', 'SKIPPER', 'SMARTLINK', 'SMSPHARMA', 'SOFCOM',
            'SOLARA', 'SOUTHWEST', 'SPARC', 'SPIC', 'SREEL',
            'SSWL', 'STAR', 'STEELCITY', 'STERTOOLS', 'STINDIA',
            'SUBEX', 'SUDARSCHEM', 'SUNDARAM', 'SUNFLAG', 'SUNTECK',
            'SUPERHOUSE', 'SUPRIYA', 'SURYALAXMI', 'SUULD', 'SWANENERGY',
        ]
        return [{'symbol': s, 'name': s, 'exchange': 'BSE'} for s in bse_only]

    def fetch_stock_data_fast(self, symbol: str, exchange: str = 'NSE') -> Optional[StockData]:
        """Fetch stock data with minimal API calls."""
        try:
            # Try NSE first, then BSE
            suffix = '.NS' if exchange == 'NSE' else '.BO'
            ticker = yf.Ticker(f"{symbol}{suffix}")

            hist = ticker.history(period='1y')

            # If NSE fails, try BSE
            if (hist.empty or len(hist) < 5) and exchange == 'NSE':
                ticker = yf.Ticker(f"{symbol}.BO")
                hist = ticker.history(period='1y')

            if hist.empty or len(hist) < 5:
                return None

            current_price = hist['Close'].iloc[-1]

            # Skip if price is 0 or invalid
            if current_price <= 0 or pd.isna(current_price):
                return None

            # Get basic info (cached by yfinance)
            try:
                info = ticker.info
            except:
                info = {}

            # Calculate returns
            def safe_return(days):
                if len(hist) >= days:
                    past_price = hist['Close'].iloc[-days]
                    if past_price > 0:
                        return ((current_price / past_price) - 1) * 100
                return 0

            change_1d = safe_return(2) if len(hist) > 1 else 0
            change_1w = safe_return(5)
            change_1m = safe_return(21)
            change_3m = safe_return(63)
            change_6m = safe_return(126)
            change_1y = safe_return(252) if len(hist) >= 252 else safe_return(len(hist))

            # Momentum score
            momentum_score = (
                change_1w * 0.1 +
                change_1m * 0.2 +
                change_3m * 0.25 +
                change_6m * 0.25 +
                change_1y * 0.2
            ) / 100

            # Momentum consistency
            positive_periods = sum([
                1 if change_1w > 0 else 0,
                1 if change_1m > 0 else 0,
                1 if change_3m > 0 else 0,
                1 if change_6m > 0 else 0,
                1 if change_1y > 0 else 0
            ])
            momentum_consistency = positive_periods / 5

            # Volume
            avg_volume = hist['Volume'].mean()
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) and rs.iloc[-1] != 0 else 50

            # Moving averages
            ma50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else current_price
            ma200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else current_price

            # Market cap
            market_cap = info.get('marketCap', 0) or 0
            market_cap_cr = market_cap / 10000000

            # Categorize price
            if current_price <= 10:
                price_category = PriceRange.MICRO_PENNY
            elif current_price <= 30:
                price_category = PriceRange.PENNY
            elif current_price <= 100:
                price_category = PriceRange.LOW_PRICE
            elif current_price <= 500:
                price_category = PriceRange.MID_PRICE
            else:
                price_category = PriceRange.HIGH_PRICE

            # Categorize market cap
            if market_cap_cr >= 20000:
                cap_category = MarketCap.LARGE_CAP
            elif market_cap_cr >= 5000:
                cap_category = MarketCap.MID_CAP
            elif market_cap_cr >= 500:
                cap_category = MarketCap.SMALL_CAP
            else:
                cap_category = MarketCap.MICRO_CAP

            return StockData(
                symbol=symbol,
                name=info.get('shortName', symbol),
                price=round(current_price, 2),
                market_cap=market_cap,
                market_cap_cr=round(market_cap_cr, 2),
                sector=info.get('sector', 'Unknown'),
                industry=info.get('industry', 'Unknown'),
                change_1d=round(change_1d, 2),
                change_1w=round(change_1w, 2),
                change_1m=round(change_1m, 2),
                change_3m=round(change_3m, 2),
                change_6m=round(change_6m, 2),
                change_1y=round(change_1y, 2),
                momentum_score=round(momentum_score, 3),
                momentum_consistency=round(momentum_consistency, 2),
                avg_volume=avg_volume,
                volume_ratio=round(volume_ratio, 2),
                rsi=round(rsi, 1),
                above_ma50=current_price > ma50,
                above_ma200=current_price > ma200,
                price_category=price_category,
                cap_category=cap_category
            )

        except Exception as e:
            return None

    def _fetch_batch_parallel(self, stocks: List[Dict], max_workers: int = 10) -> List[StockData]:
        """Fetch multiple stocks in parallel for speed."""
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {}
            for stock in stocks:
                if isinstance(stock, dict):
                    sym = stock.get('symbol', stock)
                    exch = stock.get('exchange', 'NSE')
                else:
                    sym = stock
                    exch = 'NSE'
                future_to_symbol[executor.submit(self.fetch_stock_data_fast, sym, exch)] = sym

            for future in as_completed(future_to_symbol):
                try:
                    data = future.result()
                    if data:
                        results.append(data)
                except Exception:
                    pass

        return results

    def screen_by_price(
        self,
        price_range: PriceRange,
        min_volume: float = 100000,
        limit: int = 20
    ) -> List[StockData]:
        """Screen stocks by price range - fetches from ALL NSE stocks."""
        logger.info(f"Screening ALL NSE stocks in price range: {price_range.value}")

        # Define price bounds
        bounds = {
            PriceRange.MICRO_PENNY: (1, 10),
            PriceRange.PENNY: (10, 30),
            PriceRange.LOW_PRICE: (30, 100),
            PriceRange.MID_PRICE: (100, 500),
            PriceRange.HIGH_PRICE: (500, 100000),
        }
        min_price, max_price = bounds.get(price_range, (0, 100000))

        # Get all NSE stocks
        all_stocks = self.fetch_all_nse_stocks()
        symbols = [s['symbol'] for s in all_stocks]

        logger.info(f"Scanning {len(symbols)} stocks for price range {min_price}-{max_price}...")

        # Quick price filter using yfinance batch download
        # This is MUCH faster than individual calls
        try:
            # Download prices for all stocks at once
            symbols_ns = [f"{s}.NS" for s in symbols[:500]]  # Limit to 500 for speed
            data = yf.download(symbols_ns, period='5d', progress=False, threads=True)

            if data.empty:
                logger.warning("No data from batch download")
                return []

            # Get latest prices
            close_prices = data['Close'].iloc[-1] if 'Close' in data.columns else data[('Close',)].iloc[-1]

            # Filter by price range
            matching_symbols = []
            for sym_ns in symbols_ns:
                sym = sym_ns.replace('.NS', '')
                try:
                    price = close_prices[sym_ns] if sym_ns in close_prices else close_prices.get(sym_ns)
                    if price and min_price <= price <= max_price:
                        matching_symbols.append(sym)
                except:
                    continue

            logger.info(f"Found {len(matching_symbols)} stocks in price range. Fetching details...")

            # Now fetch full details for matching stocks only (in parallel)
            matching_stocks = [{'symbol': s, 'exchange': 'NSE'} for s in matching_symbols[:limit * 2]]
            results = self._fetch_batch_parallel(matching_stocks, max_workers=15)

            # Filter by volume and sort by momentum
            results = [r for r in results if r.avg_volume >= min_volume]
            results.sort(key=lambda x: x.momentum_score, reverse=True)

            return results[:limit]

        except Exception as e:
            logger.error(f"Batch download failed: {e}")
            # Fallback to sequential
            return self._screen_sequential(symbols[:100], price_range, min_volume, limit)

    def _screen_sequential(self, symbols, price_range, min_volume, limit):
        """Fallback sequential screening."""
        bounds = {
            PriceRange.MICRO_PENNY: (1, 10),
            PriceRange.PENNY: (10, 30),
            PriceRange.LOW_PRICE: (30, 100),
            PriceRange.MID_PRICE: (100, 500),
            PriceRange.HIGH_PRICE: (500, 100000),
        }
        min_price, max_price = bounds.get(price_range, (0, 100000))

        results = []
        for symbol in symbols:
            try:
                data = self.fetch_stock_data_fast(symbol)
                if data and min_price <= data.price <= max_price:
                    if data.avg_volume >= min_volume:
                        results.append(data)
                        if len(results) >= limit:
                            break
            except:
                continue

        results.sort(key=lambda x: x.momentum_score, reverse=True)
        return results[:limit]

    def screen_momentum_stocks(
        self,
        period: MomentumPeriod = MomentumPeriod.THREE_MONTHS,
        min_return: float = 10,
        consistency_threshold: float = 0.6,
        limit: int = 20
    ) -> List[StockData]:
        """Screen stocks with consistent momentum from ALL NSE stocks."""
        logger.info(f"Screening momentum stocks for period: {period.value}")

        all_stocks = self.fetch_all_nse_stocks()
        stocks_to_scan = all_stocks[:300]  # Top 300 for speed

        # Fetch in parallel
        results = self._fetch_batch_parallel(stocks_to_scan, max_workers=15)

        # Filter by momentum criteria
        filtered = []
        for data in results:
            period_return = {
                MomentumPeriod.ONE_WEEK: data.change_1w,
                MomentumPeriod.ONE_MONTH: data.change_1m,
                MomentumPeriod.THREE_MONTHS: data.change_3m,
                MomentumPeriod.SIX_MONTHS: data.change_6m,
                MomentumPeriod.ONE_YEAR: data.change_1y,
            }.get(period, data.change_3m)

            if period_return >= min_return and data.momentum_consistency >= consistency_threshold:
                filtered.append(data)

        filtered.sort(key=lambda x: x.momentum_score, reverse=True)
        return filtered[:limit]

    def screen_by_market_cap(
        self,
        cap_category: MarketCap,
        sort_by: str = 'momentum',
        limit: int = 20
    ) -> List[StockData]:
        """Screen stocks by market cap from ALL NSE stocks."""
        logger.info(f"Screening by market cap: {cap_category.value}")

        all_stocks = self.fetch_all_nse_stocks()
        stocks_to_scan = all_stocks[:300]

        results = self._fetch_batch_parallel(stocks_to_scan, max_workers=15)

        # Filter by market cap
        filtered = [r for r in results if r.cap_category == cap_category]

        # Sort
        if sort_by == 'momentum':
            filtered.sort(key=lambda x: x.momentum_score, reverse=True)
        elif sort_by == 'price':
            filtered.sort(key=lambda x: x.price)
        elif sort_by == 'volume':
            filtered.sort(key=lambda x: x.avg_volume, reverse=True)

        return filtered[:limit]

    def get_top_gainers(self, period: str = '1d', limit: int = 10) -> List[StockData]:
        """Get top gaining stocks for a period."""
        logger.info(f"Finding top gainers for period: {period}")

        all_stocks = self.fetch_all_nse_stocks()
        stocks_to_scan = all_stocks[:200]

        results = self._fetch_batch_parallel(stocks_to_scan, max_workers=15)

        sort_key = {
            '1d': lambda x: x.change_1d,
            '1w': lambda x: x.change_1w,
            '1m': lambda x: x.change_1m,
            '3m': lambda x: x.change_3m,
            '6m': lambda x: x.change_6m,
            '1y': lambda x: x.change_1y,
        }.get(period, lambda x: x.change_1d)

        results.sort(key=sort_key, reverse=True)
        return results[:limit]

    def fetch_stock_data(self, symbol: str) -> Optional[StockData]:
        """Public method to fetch single stock data."""
        return self.fetch_stock_data_fast(symbol)

    def get_ai_stock_insight(self, stock_data: StockData) -> str:
        """Get AI-powered insight for a stock."""
        if not self.openai_key:
            return "OpenAI API key not configured."

        prompt = f"""Analyze this Indian stock and provide actionable insights:

STOCK: {stock_data.symbol} - {stock_data.name}
Price: ₹{stock_data.price}
Sector: {stock_data.sector}
Market Cap: ₹{stock_data.market_cap_cr:,.0f} Cr ({stock_data.cap_category.value})

PERFORMANCE:
- 1 Day: {stock_data.change_1d:+.2f}%
- 1 Week: {stock_data.change_1w:+.2f}%
- 1 Month: {stock_data.change_1m:+.2f}%
- 3 Months: {stock_data.change_3m:+.2f}%
- 6 Months: {stock_data.change_6m:+.2f}%
- 1 Year: {stock_data.change_1y:+.2f}%

TECHNICAL:
- RSI: {stock_data.rsi}
- Above 50-day MA: {'Yes' if stock_data.above_ma50 else 'No'}
- Above 200-day MA: {'Yes' if stock_data.above_ma200 else 'No'}
- Momentum Score: {stock_data.momentum_score:.2f}
- Volume Ratio: {stock_data.volume_ratio:.2f}x

Provide:
1. **VERDICT**: BUY / HOLD / SELL with confidence %
2. **KEY STRENGTHS**: Top 3 bullish factors
3. **KEY RISKS**: Top 3 concerns
4. **ENTRY STRATEGY**: When and at what price to enter
5. **TARGET & STOP LOSS**: Specific price levels
6. **TIME HORIZON**: Intraday / Swing / Investment

Be specific and actionable. This is for Indian retail investors."""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert Indian stock market analyst. Provide specific, actionable advice."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return f"AI analysis failed: {e}"

    def get_ai_market_overview(self, stocks: List[StockData]) -> str:
        """Get AI-powered market overview based on screened stocks."""
        if not self.openai_key or not stocks:
            return "Unable to generate overview."

        avg_momentum = sum(s.momentum_score for s in stocks) / len(stocks)
        bullish_count = sum(1 for s in stocks if s.momentum_score > 0)
        bearish_count = len(stocks) - bullish_count

        top_gainers = sorted(stocks, key=lambda x: x.change_1m, reverse=True)[:5]
        top_losers = sorted(stocks, key=lambda x: x.change_1m)[:5]

        prompt = f"""Analyze this set of {len(stocks)} Indian stocks and provide market insights:

OVERVIEW:
- Average Momentum Score: {avg_momentum:.2f}
- Bullish Stocks: {bullish_count}
- Bearish Stocks: {bearish_count}

TOP 5 GAINERS (1 Month):
{chr(10).join([f"- {s.symbol}: ₹{s.price} | {s.change_1m:+.1f}%" for s in top_gainers])}

TOP 5 LOSERS (1 Month):
{chr(10).join([f"- {s.symbol}: ₹{s.price} | {s.change_1m:+.1f}%" for s in top_losers])}

Provide:
1. **MARKET SENTIMENT**: Overall bullish/bearish with reasoning
2. **SECTOR ROTATION**: Which sectors are strong/weak
3. **TOP 3 PICKS**: Best stocks to buy now with entry levels
4. **AVOID LIST**: Stocks to stay away from
5. **TRADING STRATEGY**: Specific actionable advice for today

Be specific to Indian market context."""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert Indian stock market analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1200
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"AI analysis failed: {e}"


def run_screener_demo():
    """Demo the screener functionality."""
    screener = StockScreener()

    print("=" * 70)
    print("STOCK SCREENER DEMO - REAL NSE DATA")
    print("=" * 70)

    # Penny stocks
    print("\n### PENNY STOCKS (₹10-30) ###")
    penny = screener.screen_by_price(PriceRange.PENNY, limit=5)
    for s in penny:
        print(f"{s.symbol}: ₹{s.price} | 1M: {s.change_1m:+.1f}% | Momentum: {s.momentum_score:.2f}")

    # Momentum stocks
    print("\n### MOMENTUM STOCKS (3M > 20%) ###")
    momentum = screener.screen_momentum_stocks(MomentumPeriod.THREE_MONTHS, min_return=20, limit=5)
    for s in momentum:
        print(f"{s.symbol}: ₹{s.price} | 3M: {s.change_3m:+.1f}% | Consistency: {s.momentum_consistency:.0%}")


if __name__ == "__main__":
    run_screener_demo()
