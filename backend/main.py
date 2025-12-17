"""
UK Share Analyzer - Backend API
FastAPI application for analyzing UK stocks
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import re
import pandas as pd
import json
import os
import resend
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

def strip_html_tags(text):
    """Remove HTML tags from text"""
    if not text:
        return text
    
    # Remove HTML tags using regex
    clean_text = re.sub(r'<[^>]+>', '', text)
    
    # Replace common HTML entities
    clean_text = clean_text.replace('&nbsp;', ' ')
    clean_text = clean_text.replace('&amp;', '&')
    clean_text = clean_text.replace('&lt;', '<')
    clean_text = clean_text.replace('&gt;', '>')
    clean_text = clean_text.replace('&quot;', '"')
    clean_text = clean_text.replace('&#39;', "'")
    
    # Remove extra whitespace
    clean_text = ' '.join(clean_text.split())
    
    return clean_text.strip()

# Email capture configuration
EMAIL_FILE = "captured_emails.json"
# Email configuration - load from environment variable
RESEND_API_KEY = os.environ.get('RESEND_API_KEY')
if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY
    print("✓ Resend API key loaded from environment")
else:
    print("⚠️ WARNING: RESEND_API_KEY not set - email sending disabled")

app = FastAPI(title="UK Share Analyzer API", version="1.0.0")

# CORS configuration for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000", "https://magnificent-figolla-37a50b.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory cache
cache = {}
CACHE_DURATION = timedelta(hours=1)


class AnalysisRequest(BaseModel):
    """Request model for stock analysis"""
    ticker: str


class KPICalculator:
    """Calculate all 12 KPIs and their scores"""
    
    @staticmethod
    def get_stock_data(ticker: str):
        """Fetch stock data with caching"""
        cache_key = f"{ticker}_data"
        
        # Check cache
        if cache_key in cache:
            cached_data, timestamp = cache[cache_key]
            if datetime.now() - timestamp < CACHE_DURATION:
                print(f"Using cached data for {ticker}")
                return cached_data
        
        # Fetch fresh data
        try:
            print(f"Fetching fresh data for {ticker}")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if we got valid data
            if not info or len(info) == 0:
                raise ValueError(f"No data returned for ticker {ticker}")
            
            hist = stock.history(period="1y")
            
            # Check if history is valid
            if hist is None:
                hist = stock.history(period="1y")  # Try again
            
            if hist is None or hist.empty:
                print(f"Warning: No historical data for {ticker}, creating empty DataFrame")
                import pandas as pd
                hist = pd.DataFrame()
            
            # Fetch news - Updated for current yfinance structure
            news = []
            try:
                print(f"Attempting to fetch news for {ticker}...")
                
                # Try to get news from yfinance
                if hasattr(stock, 'news'):
                    news_data = stock.news
                    print(f"Raw news data type: {type(news_data)}")
                    print(f"News data length: {len(news_data) if news_data else 0}")
                    
                    if news_data and len(news_data) > 0:
                        # Print first article structure to debug
                        print(f"First article keys: {news_data[0].keys() if news_data else 'None'}")
                        print(f"First article sample: {news_data[0] if news_data else 'None'}")
                        
                        for idx, article in enumerate(news_data[:5]):  # Limit to 5
                            # yfinance news has nested 'content' structure
                            # Check if article has 'content' key and use that, otherwise use article directly
                            article_data = article.get('content', article)
                            
                            # DEBUG: Print the article_data structure
                            print(f"\n=== Article {idx} Debug ===")
                            print(f"article_data keys: {article_data.keys() if isinstance(article_data, dict) else 'Not a dict'}")
                            print(f"Has 'title': {article_data.get('title', 'NO TITLE KEY')}")
                            print(f"Has 'provider': {article_data.get('provider', 'NO PROVIDER KEY')}")
                            print(f"Has 'canonicalUrl': {article_data.get('canonicalUrl', 'NO URL KEY')}")
                            
                            # Try different possible field names from the nested content
                            title = (article_data.get('title') or 
                                   article_data.get('headline') or 
                                   article_data.get('summary') or 
                                   'No title available')
                            
                            # Strip HTML tags from title
                            if title and title != 'No title available':
                                title = strip_html_tags(title)
                            
                            # Publisher can be nested in provider object
                            provider = article_data.get('provider', {})
                            if isinstance(provider, dict):
                                publisher = provider.get('displayName', 'Unknown source')
                            else:
                                publisher = (article_data.get('publisher') or 
                                           article_data.get('source') or 
                                           'Unknown source')
                            
                            # Link can be in canonicalUrl or clickThroughUrl
                            canonical = article_data.get('canonicalUrl', {})
                            click_through = article_data.get('clickThroughUrl', {})
                            
                            if isinstance(canonical, dict):
                                link = canonical.get('url', '')
                            elif isinstance(click_through, dict):
                                link = click_through.get('url', '')
                            else:
                                link = (article_data.get('link') or 
                                      article_data.get('url') or 
                                      '')
                            
                            # Try to get timestamp - can be in multiple formats
                            pub_time = (article_data.get('providerPublishTime') or 
                                      article_data.get('publish_time') or 
                                      article_data.get('publishedAt') or 
                                      article_data.get('pubDate') or 
                                      article_data.get('displayTime') or
                                      0)
                            
                            # Convert to int if not already
                            if isinstance(pub_time, str):
                                try:
                                    from dateutil import parser
                                    pub_time = int(parser.parse(pub_time).timestamp())
                                except:
                                    pub_time = int(datetime.now().timestamp())
                            else:
                                pub_time = int(pub_time) if pub_time else int(datetime.now().timestamp())
                            
                            # If timestamp is in milliseconds, convert to seconds
                            if pub_time > 10000000000:
                                pub_time = pub_time // 1000
                            
                            # Extract thumbnail image
                            thumbnail_url = None
                            thumbnail_data = article_data.get('thumbnail', {})
                            if isinstance(thumbnail_data, dict):
                                # Try to get the best resolution thumbnail
                                thumbnail_url = thumbnail_data.get('originalUrl')
                                if not thumbnail_url and 'resolutions' in thumbnail_data:
                                    resolutions = thumbnail_data.get('resolutions', [])
                                    if resolutions and len(resolutions) > 0:
                                        # Get the largest resolution available
                                        thumbnail_url = resolutions[-1].get('url') if isinstance(resolutions[-1], dict) else None
                            
                            # Extract article summary/description
                            summary = (article_data.get('summary') or 
                                     article_data.get('description') or 
                                     article_data.get('snippet') or 
                                     None)
                            
                            # Strip HTML tags from summary
                            if summary:
                                summary = strip_html_tags(summary)
                            
                            # Analyze sentiment based on title and summary
                            sentiment_text = f"{title} {summary if summary else ''}"
                            sentiment_result = KPICalculator.analyze_sentiment(sentiment_text)
                            
                            # Calculate read time based on summary length
                            read_time = KPICalculator.calculate_read_time(summary if summary else title)
                            
                            # Classify article category
                            category = KPICalculator.classify_category(sentiment_text)
                            
                            # DEBUG: Show what we extracted
                            print(f"EXTRACTED -> Title: {title}")
                            print(f"EXTRACTED -> Publisher: {publisher}")
                            print(f"EXTRACTED -> Link: {link}")
                            print(f"EXTRACTED -> Thumbnail: {thumbnail_url[:50] if thumbnail_url else 'None'}")
                            print(f"EXTRACTED -> Summary: {summary[:80] if summary else 'None'}...")
                            print(f"EXTRACTED -> Sentiment: {sentiment_result['sentiment']} (score: {sentiment_result['score']})")
                            print(f"EXTRACTED -> Read Time: {read_time} min")
                            print(f"EXTRACTED -> Category: {category}")
                            print(f"=== End Article {idx} ===\n")
                            
                            print(f"Article {idx}: title={title[:50] if len(title) > 50 else title}, publisher={publisher}, link={link[:50] if link and len(link) > 50 else link}")
                            
                            news.append({
                                'title': title,
                                'publisher': publisher,
                                'link': link,
                                'publish_time': pub_time,
                                'type': article_data.get('type', 'article'),
                                'thumbnail': thumbnail_url,
                                'summary': summary,
                                'sentiment': sentiment_result['sentiment'],
                                'sentiment_score': sentiment_result['score'],
                                'read_time': read_time,
                                'category': category
                            })
                        
                        print(f"Successfully parsed {len(news)} news articles")
                    else:
                        print("News data is empty or None")
                else:
                    print("Stock object has no 'news' attribute")
                
                # If still no news, provide a fallback
                if len(news) == 0:
                    print("No news found, creating fallback link")
                    news = [{
                        'title': f'View latest news for {ticker} on Yahoo Finance',
                        'publisher': 'Yahoo Finance',
                        'link': f'https://finance.yahoo.com/quote/{ticker}/news',
                        'publish_time': int(datetime.now().timestamp()),
                        'type': 'link'
                    }]
                    
            except Exception as e:
                print(f"Error fetching news: {e}")
                import traceback
                print(traceback.format_exc())
                news = [{
                    'title': f'Unable to fetch news. View on Yahoo Finance',
                    'publisher': 'System',
                    'link': f'https://finance.yahoo.com/quote/{ticker}/news',
                    'publish_time': int(datetime.now().timestamp()),
                    'type': 'error'
                }]
            
            data = {'info': info, 'history': hist, 'news': news}
            cache[cache_key] = (data, datetime.now())
            
            print(f"Successfully fetched data for {ticker}")
            print(f"Info keys: {list(info.keys())[:10]}...")  # Show first 10 keys
            print(f"History shape: {hist.shape if not hist.empty else 'empty'}")
            
            return data
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            raise HTTPException(
                status_code=404, 
                detail=f"Unable to fetch data for {ticker}. Error: {str(e)}"
            )
    
    @staticmethod
    def safe_get(data: dict, key: str, default=None):
        """Safely get value from dictionary"""
        return data.get(key, default)
    
    @staticmethod
    def determine_exchange_and_index(info: dict, ticker: str):
        """Determine exchange and stock index (UK or US) based on market cap and ticker"""
        exchange = info.get('exchange', '')
        market_cap = info.get('marketCap', 0)
        
        # Get full exchange name
        exchange_map = {
            'LSE': 'London Stock Exchange',
            'LON': 'London Stock Exchange',
            'NYSE': 'New York Stock Exchange',
            'NMS': 'NASDAQ',
            'NGM': 'NASDAQ',
            'NYQ': 'New York Stock Exchange',
        }
        
        exchange_full = exchange_map.get(exchange, exchange or 'Unknown')
        
        # Determine index based on market cap and ticker
        index_name = None
        
        # UK indices (for .L tickers)
        if ticker.endswith('.L'):
            if market_cap > 7_000_000_000:  # > £7B
                index_name = 'FTSE 100'
            elif market_cap > 500_000_000:  # £500M - £7B
                index_name = 'FTSE 250'
            elif market_cap > 0:  # < £500M
                index_name = 'FTSE Small Cap / AIM'
        
        # US indices (for non-.L tickers with US exchanges)
        elif exchange in ['NYSE', 'NMS', 'NGM', 'NYQ', 'NASDAQ']:
            # Convert market cap to billions for easier comparison
            market_cap_b = market_cap / 1_000_000_000
            
            if market_cap_b > 15:  # > $15B
                index_name = 'S&P 500'
            elif market_cap_b > 2:  # $2B - $15B
                index_name = 'S&P 400 (Mid Cap)'
            elif market_cap_b > 0.3:  # $300M - $2B
                index_name = 'S&P 600 (Small Cap)'
            elif market_cap > 0:  # < $300M
                index_name = 'Micro Cap'
        
        return {
            'exchange': exchange,
            'exchange_full': exchange_full,
            'index': index_name
        }
    
    @staticmethod
    def analyze_sentiment(text: str) -> dict:
        """Analyze sentiment of news text using keyword matching"""
        if not text:
            return {'sentiment': 'neutral', 'score': 0}
        
        text_lower = text.lower()
        
        # Positive keywords (financial/business context)
        positive_keywords = [
            'profit', 'growth', 'gain', 'rise', 'surge', 'beat', 'exceed', 'strong',
            'up', 'upgrade', 'boost', 'record', 'high', 'improved', 'expansion',
            'success', 'positive', 'outperform', 'rally', 'soar', 'jump', 'climb',
            'recovery', 'breakthrough', 'win', 'award', 'innovation', 'bullish'
        ]
        
        # Negative keywords (financial/business context)
        negative_keywords = [
            'loss', 'drop', 'fall', 'decline', 'plunge', 'miss', 'below', 'weak',
            'down', 'downgrade', 'cut', 'low', 'concern', 'warning', 'risk',
            'fail', 'negative', 'underperform', 'crash', 'tumble', 'slide', 'slump',
            'crisis', 'trouble', 'investigation', 'fine', 'penalty', 'bearish'
        ]
        
        # Count keyword matches
        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        
        # Calculate sentiment score (-1 to +1)
        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            return {'sentiment': 'neutral', 'score': 0}
        
        score = (positive_count - negative_count) / total_keywords
        
        # Determine sentiment category
        if score > 0.2:
            sentiment = 'positive'
        elif score < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {'sentiment': sentiment, 'score': round(score, 2)}
    
    @staticmethod
    def calculate_read_time(text: str) -> int:
        """Calculate estimated reading time in minutes based on text length"""
        if not text:
            return 3  # Default 3 minutes for articles without summary
        
        # Average reading speed: 200 words per minute
        # Financial articles might be slower, but we'll use standard rate
        words_per_minute = 200
        
        # Count words in text
        word_count = len(text.split())
        
        # Calculate read time in minutes (minimum 1 minute)
        read_time = max(1, round(word_count / words_per_minute))
        
        # Cap at reasonable maximum (15 minutes)
        # Most news articles are shorter than this
        read_time = min(15, read_time)
        
        return read_time
    
    @staticmethod
    def classify_category(text: str) -> str:
        """Classify news article into category based on keywords"""
        if not text:
            return 'general'
        
        text_lower = text.lower()
        
        # Category keywords (ordered by priority)
        category_keywords = {
            'earnings': [
                'earnings', 'profit', 'revenue', 'quarterly', 'results', 'q1', 'q2', 'q3', 'q4',
                'forecast', 'guidance', 'eps', 'beat', 'miss', 'expectation', 'fiscal',
                'income', 'sales', 'performance', 'outlook'
            ],
            'ma': [  # M&A
                'merger', 'acquisition', 'takeover', 'deal', 'acquire', 'merge', 'buyout',
                'bid', 'offer', 'purchase', 'buy', 'sell', 'divestiture', 'spin-off',
                'consolidation', 'combination'
            ],
            'regulation': [
                'regulation', 'regulatory', 'compliance', 'investigation', 'probe', 'fine',
                'penalty', 'lawsuit', 'legal', 'court', 'sec', 'fca', 'regulator',
                'violation', 'enforcement', 'inquiry', 'charges', 'settlement'
            ],
            'markets': [
                'shares', 'stock', 'trading', 'market', 'price', 'rally', 'surge', 'plunge',
                'drop', 'rise', 'fall', 'gain', 'loss', 'volume', 'index', 'ftse',
                'dow', 'nasdaq', 'valuation', 'analyst', 'rating', 'upgrade', 'downgrade'
            ],
            'leadership': [
                'ceo', 'cfo', 'chairman', 'chief', 'executive', 'appoint', 'resign',
                'departure', 'successor', 'management', 'board', 'director', 'hire',
                'promote', 'retire', 'succession'
            ]
        }
        
        # Count keyword matches for each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, or 'general' if no matches
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return 'general'
    
    @staticmethod
    def calculate_score(value: float, thresholds: list, reverse: bool = False) -> int:
        """
        Calculate score (1-10) based on thresholds
        thresholds: [(max_val, score), ...]
        reverse: True if lower values are better
        """
        if value is None:
            return 5
        
        if reverse:
            thresholds = [(t[0], 11 - t[1]) for t in thresholds]
        
        for threshold, score in thresholds:
            if value <= threshold:
                return score
        return thresholds[-1][1]
    
    def calculate_kpis(self, ticker: str) -> Dict[str, Any]:
        """Calculate all 12 KPIs"""
        data = self.get_stock_data(ticker)
        info = data['info']
        
        # Extract basic company info
        company_name = self.safe_get(info, 'longName', ticker)
        current_price = self.safe_get(info, 'currentPrice') or self.safe_get(info, 'regularMarketPrice')
        currency = self.safe_get(info, 'currency', 'GBP')
        
        # 1. P/E Ratio (Price-to-Earnings)
        pe_ratio = self.safe_get(info, 'trailingPE')
        pe_score = self.calculate_score(
            pe_ratio if pe_ratio else 999,
            [(10, 10), (15, 8), (20, 6), (30, 4), (999, 2)]
        ) if pe_ratio else 5
        
        # 2. P/B Ratio (Price-to-Book)
        pb_ratio = self.safe_get(info, 'priceToBook')
        pb_score = self.calculate_score(
            pb_ratio if pb_ratio else 999,
            [(1.0, 10), (2.0, 8), (3.0, 6), (5.0, 4), (999, 2)]
        ) if pb_ratio else 5
        
        # 3. ROE (Return on Equity)
        roe = self.safe_get(info, 'returnOnEquity')
        roe_pct = roe * 100 if roe else None
        roe_score = self.calculate_score(
            roe_pct if roe_pct else -999,
            [(0, 2), (10, 5), (15, 7), (20, 9), (999, 10)]
        ) if roe_pct else 5
        
        # 4. Debt-to-Equity
        debt_to_equity = self.safe_get(info, 'debtToEquity')
        if debt_to_equity:
            debt_to_equity = debt_to_equity / 100
        de_score = self.calculate_score(
            debt_to_equity if debt_to_equity else 999,
            [(0.3, 10), (0.6, 8), (1.0, 6), (2.0, 4), (999, 2)],
            reverse=True
        ) if debt_to_equity else 5
        
        # 5. Current Ratio
        current_ratio = self.safe_get(info, 'currentRatio')
        cr_score = self.calculate_score(
            current_ratio if current_ratio else 0,
            [(0.5, 2), (1.0, 5), (1.5, 8), (2.0, 10), (999, 9)]
        ) if current_ratio else 5
        
        # 6. Revenue Growth
        revenue_growth = self.safe_get(info, 'revenueGrowth')
        revenue_growth_pct = revenue_growth * 100 if revenue_growth else None
        rg_score = self.calculate_score(
            revenue_growth_pct if revenue_growth_pct else -999,
            [(-10, 2), (0, 4), (5, 6), (10, 8), (999, 10)]
        ) if revenue_growth_pct else 5
        
        # 7. Profit Margin
        profit_margin = self.safe_get(info, 'profitMargins')
        profit_margin_pct = profit_margin * 100 if profit_margin else None
        pm_score = self.calculate_score(
            profit_margin_pct if profit_margin_pct else -999,
            [(0, 2), (5, 4), (10, 6), (15, 8), (999, 10)]
        ) if profit_margin_pct else 5
        
        # 8. Dividend Yield
        # 8. Dividend Yield
        dividend_yield = self.safe_get(info, 'dividendYield')

        # Smart conversion - handle both decimal and percentage formats
        if dividend_yield:
            # If value is already > 1, it's likely already a percentage
            if dividend_yield > 1:
                dividend_yield_pct = dividend_yield
            else:
                # If value is < 1, it's a decimal, convert to percentage
                dividend_yield_pct = dividend_yield * 100
        else:
            dividend_yield_pct = None

        dy_score = self.calculate_score(
            dividend_yield_pct if dividend_yield_pct else 0,
            [(0, 3), (2, 6), (4, 9), (6, 10), (999, 8)]
        ) if dividend_yield_pct is not None else 5
        
        
        # 9. EPS Growth (Earnings Per Share)
        earnings_growth = self.safe_get(info, 'earningsGrowth')
        earnings_growth_pct = earnings_growth * 100 if earnings_growth else None
        eg_score = self.calculate_score(
            earnings_growth_pct if earnings_growth_pct else -999,
            [(-10, 2), (0, 4), (10, 6), (20, 8), (999, 10)]
        ) if earnings_growth_pct else 5
        
        # 10. Beta (Volatility measure)
        beta = self.safe_get(info, 'beta')
        beta_score = self.calculate_score(
            abs(beta - 1.0) if beta else 999,
            [(0.2, 10), (0.5, 8), (0.8, 6), (1.5, 4), (999, 2)]
        ) if beta else 5
        
        # 11. 52-Week Price Position
        fifty_two_week_high = self.safe_get(info, 'fiftyTwoWeekHigh')
        fifty_two_week_low = self.safe_get(info, 'fiftyTwoWeekLow')
        
        if current_price and fifty_two_week_high and fifty_two_week_low:
            price_position = ((current_price - fifty_two_week_low) / 
                            (fifty_two_week_high - fifty_two_week_low)) * 100
            pp_score = self.calculate_score(
                price_position,
                [(20, 10), (40, 8), (60, 6), (80, 4), (100, 2)]
            )
        else:
            price_position = None
            pp_score = 5
        
        # 12. Operating Margin
        operating_margin = self.safe_get(info, 'operatingMargins')
        operating_margin_pct = operating_margin * 100 if operating_margin else None
        om_score = self.calculate_score(
            operating_margin_pct if operating_margin_pct else -999,
            [(0, 2), (5, 4), (10, 6), (15, 8), (999, 10)]
        ) if operating_margin_pct else 5
        
        # Compile all KPIs into structured format
        kpis = {
            'valuation': {
                'pe_ratio': {
                    'value': round(pe_ratio, 2) if pe_ratio else None, 
                    'score': pe_score
                },
                'pb_ratio': {
                    'value': round(pb_ratio, 2) if pb_ratio else None, 
                    'score': pb_score
                },
            },
            'profitability': {
                'roe': {
                    'value': round(roe_pct, 2) if roe_pct else None, 
                    'score': roe_score
                },
                'profit_margin': {
                    'value': round(profit_margin_pct, 2) if profit_margin_pct else None, 
                    'score': pm_score
                },
                'operating_margin': {
                    'value': round(operating_margin_pct, 2) if operating_margin_pct else None, 
                    'score': om_score
                },
            },
            'health': {
                'debt_to_equity': {
                    'value': round(debt_to_equity, 2) if debt_to_equity else None, 
                    'score': de_score
                },
                'current_ratio': {
                    'value': round(current_ratio, 2) if current_ratio else None, 
                    'score': cr_score
                },
            },
            'growth': {
                'revenue_growth': {
                    'value': round(revenue_growth_pct, 2) if revenue_growth_pct else None, 
                    'score': rg_score
                },
                'eps_growth': {
                    'value': round(earnings_growth_pct, 2) if earnings_growth_pct else None, 
                    'score': eg_score
                },
            },
            'technical': {
                'beta': {
                    'value': round(beta, 2) if beta else None, 
                    'score': beta_score
                },
                'price_position': {
                    'value': round(price_position, 2) if price_position else None, 
                    'score': pp_score
                },
                'dividend_yield': {
                    'value': round(dividend_yield_pct, 2) if dividend_yield_pct else None, 
                    'score': dy_score
                },
            }
        }
        
        # Get 12-month historical price data WITH MOVING AVERAGES
        historical_prices = []
        try:
            hist_data = data.get('history')
            if hist_data is not None and not hist_data.empty and 'Close' in hist_data.columns:
                hist_12m = hist_data.tail(252)
                
                # Calculate moving averages
                # 50-day MA
                ma_50 = hist_12m['Close'].rolling(window=50, min_periods=1).mean()
                # 200-day MA  
                ma_200 = hist_12m['Close'].rolling(window=200, min_periods=1).mean()
                
                for i, (date, row) in enumerate(hist_12m.iterrows()):
                    historical_prices.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'price': round(float(row['Close']), 2),
                        'ma_50': round(float(ma_50.iloc[i]), 2) if not pd.isna(ma_50.iloc[i]) else None,
                        'ma_200': round(float(ma_200.iloc[i]), 2) if not pd.isna(ma_200.iloc[i]) else None
                    })
                print(f"Fetched {len(historical_prices)} historical price points with moving averages")
            else:
                print("No historical data available")
        except Exception as e:
            print(f"Warning: Could not fetch historical prices: {e}")
            historical_prices = []
        
        # Get news from data
        news = data.get('news', [])
        
        # Determine exchange and index
        exchange_info = self.determine_exchange_and_index(info, ticker)
        
        # Extract company overview information
        company_overview = {
            'description': self.safe_get(info, 'longBusinessSummary'),
            'sector': self.safe_get(info, 'sector'),
            'industry': self.safe_get(info, 'industry'),
            'employees': self.safe_get(info, 'fullTimeEmployees'),
            'headquarters': f"{self.safe_get(info, 'city', '')}, {self.safe_get(info, 'country', '')}".strip(', '),
            'website': self.safe_get(info, 'website'),
            'exchange': exchange_info['exchange'],
            'exchange_full': exchange_info['exchange_full'],
            'index': exchange_info['index'],
            'market_cap': self.safe_get(info, 'marketCap'),
            'enterprise_value': self.safe_get(info, 'enterpriseValue'),
            'previous_close': self.safe_get(info, 'previousClose'),
            'day_range': {
                'low': self.safe_get(info, 'dayLow'),
                'high': self.safe_get(info, 'dayHigh')
            },
            'week_52_range': {
                'low': self.safe_get(info, 'fiftyTwoWeekLow'),
                'high': self.safe_get(info, 'fiftyTwoWeekHigh')
            },
            'volume': self.safe_get(info, 'volume'),
            'avg_volume': self.safe_get(info, 'averageVolume')
        }
        
        return {
            'company_name': company_name,
            'ticker': ticker,
            'current_price': round(current_price, 2) if current_price else None,
            'currency': currency,
            'kpis': kpis,
            'historical_prices': historical_prices,
            'news': news,
            'company_overview': company_overview
        }

class RecommendationEngine:
    """Generate Buy/Hold/Sell recommendation based on weighted KPI scores"""
    
    # Category weights (must sum to 1.0)
    WEIGHTS = {
        'valuation': 0.25,
        'profitability': 0.25,
        'health': 0.20,
        'growth': 0.20,
        'technical': 0.10
    }
    
    @staticmethod
    def calculate_category_score(category_data: dict) -> float:
        """Calculate average score for a category"""
        scores = [item['score'] for item in category_data.values()]
        return sum(scores) / len(scores) if scores else 5.0
    
    def generate_recommendation(self, kpis: dict) -> dict:
        """Generate overall Buy/Hold/Sell recommendation"""
        category_scores = {}
        
        # Calculate score for each category
        for category in self.WEIGHTS.keys():
            category_scores[category] = self.calculate_category_score(kpis[category])
        
        # Calculate weighted total score (0-10 scale)
        total_score = sum(
            score * self.WEIGHTS[category] 
            for category, score in category_scores.items()
        )
        
        # Convert to 0-100 scale for display
        total_score_100 = (total_score / 10) * 100
        
        # Determine recommendation based on score
        if total_score_100 >= 70:
            recommendation = "BUY"
            color = "green"
            message = "Strong fundamentals indicate good investment value"
        elif total_score_100 >= 40:
            recommendation = "HOLD"
            color = "yellow"
            message = "Moderate fundamentals, monitor for changes"
        else:
            recommendation = "SELL"
            color = "red"
            message = "Weak fundamentals suggest high risk"
        
        return {
            'recommendation': recommendation,
            'score': round(total_score_100, 1),
            'color': color,
            'message': message,
            'category_scores': {
                k: round(v, 1) for k, v in category_scores.items()
            }
        }


# API Routes

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "UK Share Analyzer API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/api/analyze",
            "health": "/api/health",
            "docs": "/docs"
        }
    }


@app.post("/api/analyze")
def analyze_stock(request: AnalysisRequest):
    """
    Analyze a UK stock and return comprehensive KPI analysis with recommendation
    
    Args:
        request: AnalysisRequest containing ticker symbol
        
    Returns:
        Complete analysis including company info, KPIs, and recommendation
    """
    try:
        print(f"\n{'='*50}")
        print(f"Analyzing ticker: {request.ticker}")
        print(f"{'='*50}")
        
        calculator = KPICalculator()
        engine = RecommendationEngine()
        
        # Calculate all KPIs
        print("Step 1: Calculating KPIs...")
        analysis = calculator.calculate_kpis(request.ticker)
        print("Step 1: Complete ✓")
        
        # Check what we got
        print(f"Analysis keys: {list(analysis.keys())}")
        print(f"KPIs keys: {list(analysis['kpis'].keys())}")
        print(f"Historical prices count: {len(analysis.get('historical_prices', []))}")
        
        # Generate recommendation
        print("Step 2: Generating recommendation...")
        recommendation = engine.generate_recommendation(analysis['kpis'])
        print("Step 2: Complete ✓")
        
        print(f"✓ Analysis complete")
        print(f"  Recommendation: {recommendation['recommendation']}")
        print(f"  Score: {recommendation['score']}/100")
        
        return {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'company': {
                'name': analysis['company_name'],
                'ticker': analysis['ticker'],
                'price': analysis['current_price'],
                'currency': analysis['currency']
            },
            'kpis': analysis['kpis'],
            'recommendation': recommendation,
            'historical_prices': analysis.get('historical_prices', []),
            'news': analysis.get('news', []),
            'company_overview': analysis.get('company_overview', {})
        }
    
    except HTTPException as he:
        print(f"✗ HTTP Error: {he.detail}")
        raise he
    except Exception as e:
        import traceback
        print(f"✗ Error: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_size": len(cache)
    }

@app.get("/api/check-env")
def check_environment():
    """Check if environment variables are set (for debugging)"""
    return {
        'resend_api_key_set': bool(os.environ.get('RESEND_API_KEY')),
        'resend_api_key_length': len(os.environ.get('RESEND_API_KEY', '')) if os.environ.get('RESEND_API_KEY') else 0,
        # Never return the actual key!
        'environment': os.environ.get('RENDER', 'local')
    }

@app.post("/api/clear-cache")
def clear_cache():
    """Clear the cache - useful for debugging"""
    cache.clear()
    return {
        "success": True,
        "message": "Cache cleared",
        "timestamp": datetime.now().isoformat()
    }
def send_welcome_email(first_name: str, email: str):
    """
    Send welcome email to new user
    """
    try:
        # HTML email template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                    border-radius: 10px 10px 0 0;
                }}
                .content {{
                    background: #f9fafb;
                    padding: 30px;
                    border-radius: 0 0 10px 10px;
                }}
                .button {
                    display: inline-block;
                    background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
                    color: #ffffff !important;
                    padding: 12px 30px;
                    text-decoration: none;
                    border-radius: 5px;
                    margin: 20px 0;
                    font-weight: bold;
                    font-size: 16px;
                    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
                }
                .footer {{
                    text-align: center;
                    color: #666;
                    font-size: 12px;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📈 Welcome to Stock Analyzer!</h1>
            </div>
            <div class="content">
                <h2>Hi {first_name},</h2>
                <p>Thank you for signing up! You now have <strong>unlimited access</strong> to:</p>
                <ul>
                    <li>✅ Analyze 200+ UK & US stocks (FTSE 100/250, S&P 500)</li>
                    <li>✅ 12 comprehensive KPIs with intelligent scoring</li>
                    <li>✅ Price charts with 50-day & 200-day moving averages</li>
                    <li>✅ Latest news with AI sentiment analysis</li>
                    <li>✅ Buy/Hold/Sell recommendations</li>
                    <li>✅ PDF report generation</li>
                </ul>
                <p style="text-align: center;">
                    <a href="https://magnificent-figolla-37a50b.netlify.app" class="button">Start Analyzing Stocks</a>
                </p>
                <p><strong>Pro Tips:</strong></p>
                <ul>
                    <li>📊 Check the Overall Score (0-100) for quick insights</li>
                    <li>📰 Use news filters to focus on specific sentiment or categories</li>
                    <li>💾 Download PDF reports for offline analysis</li>
                    <li>📈 Watch for moving average crossovers for trading signals</li>
                </ul>
                <p>Have questions or feedback? Just reply to this email - we'd love to hear from you!</p>
                <p>Happy analyzing!</p>
                <p><strong>The Stock Analyzer Team</strong></p>
            </div>
            <div class="footer">
                <p>You're receiving this email because you signed up for Stock Analyzer.</p>
                <p>Stock Analyzer | <a href="https://magnificent-figolla-37a50b.netlify.app">magnificent-figolla-37a50b.netlify.app</a></p>
            </div>
        </body>
        </html>
        """
        
        # Send email via Resend
        params = {
            "from": "Stock Analyzer <onboarding@resend.dev>",  # Use resend.dev for testing
            "to": [email],
            "subject": f"Welcome to Stock Analyzer, {first_name}! 🎉",
            "html": html_content
        }
        
        response = resend.Emails.send(params)
        print(f"✓ Welcome email sent to {email} (ID: {response['id']})")
        return True
        
    except Exception as e:
        print(f"✗ Failed to send email to {email}: {str(e)}")
        # Don't raise exception - we don't want email failure to break signup
        return False
    
@app.post("/api/capture-email")
def capture_email(email_data: dict):
    """
    Capture user information for lead generation
    
    Args:
        email_data: dict with 'firstName', 'lastName', 'email', 'source', 'timestamp'
        
    Returns:
        Success response
    """
    try:
        first_name = email_data.get('firstName', '').strip()
        last_name = email_data.get('lastName', '').strip()
        email = email_data.get('email', '').strip().lower()
        source = email_data.get('source', 'unknown')
        timestamp = email_data.get('timestamp', datetime.now().isoformat())
        
        # Validate required fields
        if not first_name:
            raise HTTPException(status_code=400, detail="First name is required")
        
        if len(first_name) < 2:
            raise HTTPException(status_code=400, detail="First name must be at least 2 characters")
        
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        # Validate email format
        email_regex = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
        if not re.match(email_regex, email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        # Load existing emails
        if os.path.exists(EMAIL_FILE):
            with open(EMAIL_FILE, 'r') as f:
                leads = json.load(f)
        else:
            leads = []
        
        # Check if email already exists
        existing_lead = next((lead for lead in leads if lead['email'] == email), None)
        
        if existing_lead:
            print(f"Email already exists: {email}")
            return {
                'success': True,
                'message': 'Email already registered',
                'new_signup': False
            }
        
        # Add new lead
        lead_entry = {
            'firstName': first_name,
            'lastName': last_name,
            'email': email,
            'source': source,
            'timestamp': timestamp,
            'captured_at': datetime.now().isoformat()
        }
        
        leads.append(lead_entry)
        
        # Save to file
        with open(EMAIL_FILE, 'w') as f:
            json.dump(leads, f, indent=2)
        
        print(f"✓ Captured new lead: {first_name} {last_name} <{email}> (source: {source})")
        print(f"  Total leads: {len(leads)}")
        
        # Send welcome email
        email_sent = send_welcome_email(first_name, email)

        return {
            'success': True,
            'message': 'Information captured successfully',
            'new_signup': True,
            'total_leads': len(leads),
            'email_sent': email_sent
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"✗ Error capturing lead: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to capture information: {str(e)}")


@app.get("/api/leads")
def get_leads():
    """
    Get all captured leads (admin only - add authentication later)
    Returns leads with firstName, lastName, email
    """
    try:
        if os.path.exists(EMAIL_FILE):
            with open(EMAIL_FILE, 'r') as f:
                leads = json.load(f)
            return {
                'success': True,
                'total': len(leads),
                'leads': leads
            }
        return {
            'success': True,
            'total': 0,
            'leads': []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)