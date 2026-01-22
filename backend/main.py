"""
UK Share Analyzer - Backend API
FastAPI application for analyzing UK stocks
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from numpy import info
from pydantic import BaseModel
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import re
import pandas as pd
import json
import os
import requests
from mailerlite import MailerLiteApi
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
MAILERLITE_API_KEY = os.environ.get('MAILERLITE_API_KEY')
MAILERLITE_GROUP_ID = os.environ.get('MAILERLITE_GROUP_ID')

if MAILERLITE_API_KEY and MAILERLITE_GROUP_ID:
    mailerlite = MailerLiteApi(MAILERLITE_API_KEY)
    print("✓ MailerLite API key loaded from environment")
    print(f"✓ MailerLite Group ID: {MAILERLITE_GROUP_ID}")
else:
    mailerlite = None
    print("⚠️ WARNING: MailerLite not configured - email sending disabled")

app = FastAPI(title="UK Share Analyzer API", version="1.0.0")

# CORS configuration for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000", "https://magnificent-figolla-37a50b.netlify.app", "https://app.smartstockinsights.com"],
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

# Industry/Sector Benchmark Averages
# These are approximate averages as of late 2024/early 2025
# Updated periodically based on market conditions
SECTOR_BENCHMARKS = {
    'Technology': {
        'pe_ratio': 28.5,
        'pb_ratio': 6.5,
        'roe': 18.5,
        'profit_margin': 15.2,
        'operating_margin': 18.5,
        'debt_to_equity': 0.45,
        'current_ratio': 1.4,
        'revenue_growth': 12.5,
        'eps_growth': 15.0,
        'beta': 1.15,
        'price_position': 50.0,
        'dividend_yield': 1.2
    },
    'Financial Services': {
        'pe_ratio': 12.5,
        'pb_ratio': 1.2,
        'roe': 12.0,
        'profit_margin': 20.5,
        'operating_margin': 25.0,
        'debt_to_equity': 1.8,
        'current_ratio': 1.1,
        'revenue_growth': 6.5,
        'eps_growth': 8.0,
        'beta': 1.05,
        'price_position': 50.0,
        'dividend_yield': 2.8
    },
    'Healthcare': {
        'pe_ratio': 22.0,
        'pb_ratio': 4.2,
        'roe': 15.5,
        'profit_margin': 12.5,
        'operating_margin': 16.0,
        'debt_to_equity': 0.55,
        'current_ratio': 1.6,
        'revenue_growth': 8.5,
        'eps_growth': 10.5,
        'beta': 0.95,
        'price_position': 50.0,
        'dividend_yield': 1.8
    },
    'Consumer Cyclical': {
        'pe_ratio': 18.5,
        'pb_ratio': 3.8,
        'roe': 16.0,
        'profit_margin': 8.5,
        'operating_margin': 11.0,
        'debt_to_equity': 0.75,
        'current_ratio': 1.3,
        'revenue_growth': 9.0,
        'eps_growth': 11.0,
        'beta': 1.10,
        'price_position': 50.0,
        'dividend_yield': 1.5
    },
    'Consumer Defensive': {
        'pe_ratio': 20.5,
        'pb_ratio': 4.5,
        'roe': 14.0,
        'profit_margin': 6.5,
        'operating_margin': 9.5,
        'debt_to_equity': 0.65,
        'current_ratio': 1.2,
        'revenue_growth': 4.5,
        'eps_growth': 6.0,
        'beta': 0.70,
        'price_position': 50.0,
        'dividend_yield': 2.5
    },
    'Energy': {
        'pe_ratio': 15.5,
        'pb_ratio': 1.8,
        'roe': 12.5,
        'profit_margin': 8.0,
        'operating_margin': 12.5,
        'debt_to_equity': 0.85,
        'current_ratio': 1.4,
        'revenue_growth': 5.5,
        'eps_growth': 7.5,
        'beta': 1.20,
        'price_position': 50.0,
        'dividend_yield': 3.5
    },
    'Industrials': {
        'pe_ratio': 19.0,
        'pb_ratio': 3.2,
        'roe': 13.5,
        'profit_margin': 7.5,
        'operating_margin': 10.5,
        'debt_to_equity': 0.70,
        'current_ratio': 1.5,
        'revenue_growth': 7.0,
        'eps_growth': 9.0,
        'beta': 1.05,
        'price_position': 50.0,
        'dividend_yield': 2.0
    },
    'Basic Materials': {
        'pe_ratio': 14.5,
        'pb_ratio': 2.1,
        'roe': 11.0,
        'profit_margin': 9.0,
        'operating_margin': 13.5,
        'debt_to_equity': 0.60,
        'current_ratio': 1.8,
        'revenue_growth': 6.0,
        'eps_growth': 8.5,
        'beta': 1.15,
        'price_position': 50.0,
        'dividend_yield': 2.8
    },
    'Communication Services': {
        'pe_ratio': 16.5,
        'pb_ratio': 2.8,
        'roe': 14.5,
        'profit_margin': 11.5,
        'operating_margin': 15.5,
        'debt_to_equity': 0.95,
        'current_ratio': 1.2,
        'revenue_growth': 8.0,
        'eps_growth': 10.0,
        'beta': 0.90,
        'price_position': 50.0,
        'dividend_yield': 1.6
    },
    'Utilities': {
        'pe_ratio': 17.5,
        'pb_ratio': 1.5,
        'roe': 9.5,
        'profit_margin': 10.5,
        'operating_margin': 15.0,
        'debt_to_equity': 1.2,
        'current_ratio': 0.9,
        'revenue_growth': 3.5,
        'eps_growth': 4.5,
        'beta': 0.65,
        'price_position': 50.0,
        'dividend_yield': 3.8
    },
    'Real Estate': {
        'pe_ratio': 25.0,
        'pb_ratio': 2.0,
        'roe': 8.5,
        'profit_margin': 18.5,
        'operating_margin': 25.0,
        'debt_to_equity': 1.5,
        'current_ratio': 1.0,
        'revenue_growth': 5.0,
        'eps_growth': 6.5,
        'beta': 0.85,
        'price_position': 50.0,
        'dividend_yield': 3.2
    }
}

# Market-wide benchmark averages (S&P 500 / FTSE 100 approximation)
MARKET_BENCHMARKS = {
    'US': {  # S&P 500 averages
        'pe_ratio': 22.1,
        'pb_ratio': 4.2,
        'roe': 15.2,
        'profit_margin': 10.5,
        'operating_margin': 14.0,
        'debt_to_equity': 0.80,
        'current_ratio': 1.3,
        'revenue_growth': 7.5,
        'eps_growth': 9.5,
        'beta': 1.0,
        'price_position': 50.0,
        'dividend_yield': 1.8
    },
    'UK': {  # FTSE 100 averages
        'pe_ratio': 14.5,
        'pb_ratio': 1.8,
        'roe': 12.0,
        'profit_margin': 8.5,
        'operating_margin': 12.0,
        'debt_to_equity': 0.70,
        'current_ratio': 1.2,
        'revenue_growth': 4.5,
        'eps_growth': 6.0,
        'beta': 0.95,
        'price_position': 50.0,
        'dividend_yield': 3.5
    }
}

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
            ticker_upper = ticker.upper()
        
            # Known FTSE 100 stocks (for edge cases with market cap < £7B)
            known_ftse_100 = {
                'CTEC.L',   # Convatec - £4.8B market cap
                'RR.L',     # Rolls-Royce - can be below £7B
                'BA.L',     # BAE Systems
                'SMDS.L',   # Smith & Nephew
                'PSON.L',   # Pearson
                'WEIR.L',   # Weir Group
                'CRDA.L',   # Croda International
                'IMI.L',    # IMI plc
                'SMIN.L',   # Smiths Group
                'DCC.L',    # DCC plc
                'HWDN.L',   # Howden Joinery
                # Add more as you encounter misclassifications
            }
            
            # Check if it's a known FTSE 100 stock first
            if ticker_upper in known_ftse_100:
                index_name = 'FTSE 100'
            # Then use market cap thresholds
            elif market_cap > 7_000_000_000:  # > £7B
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
        anomalies = []  # Store all detected anomalies
    
        # Extract basic company info
        company_name = self.safe_get(info, 'longName', ticker)
        current_price = self.safe_get(info, 'currentPrice') or self.safe_get(info, 'regularMarketPrice')
        currency = self.safe_get(info, 'currency', 'GBP')
    
        # 1. P/E Ratio (Price-to-Earnings)
        pe_ratio = self.safe_get(info, 'trailingPE')

        # === ANOMALY DETECTION - START ===
        if pe_ratio and pe_ratio > 50:
            anomalies.append({
                'metric': 'P/E Ratio',
                'anomaly': {
                    'type': 'extreme_high',
                    'severity': 'medium',
                    'message': 'Very high P/E ratio may indicate growth expectations or overvaluation',
                    'context': f'P/E of {round(pe_ratio, 2)} is more than 2x the typical range (15-25). This could mean the market expects strong future growth, or the stock is overvalued.',
                    'investor_note': 'High P/E stocks can deliver great returns if growth materializes, but carry higher risk if expectations aren\'t met.'
                }
            })
        elif pe_ratio and pe_ratio < 5:
            anomalies.append({
                'metric': 'P/E Ratio',
                'anomaly': {
                    'type': 'extreme_low',
                    'severity': 'medium',
                    'message': 'Very low P/E ratio may indicate value opportunity or fundamental problems',
                    'context': f'P/E of {round(pe_ratio, 2)} is unusually low. This could be a value opportunity, or indicate declining earnings/negative sentiment.',
                    'investor_note': 'Verify why P/E is so low - is it a bargain or a value trap?'
                }
            })
# === ANOMALY DETECTION - END ===

        pe_score = self.calculate_score(
            pe_ratio if pe_ratio else 999,
            [(10, 10), (15, 8), (20, 6), (30, 4), (999, 2)]
        ) if pe_ratio else 5
    
        # 2. P/B Ratio (Price-to-Book)
        pb_ratio = self.safe_get(info, 'priceToBook')
    
        # Fix for UK stocks (.L suffix) - yfinance reports prices in pence (GBp)
        # but book value in pounds, causing P/B to be 100x too high
        if pb_ratio and ticker.endswith('.L'):
            pb_ratio = pb_ratio / 100

        # === ANOMALY DETECTION - START ===
        if pb_ratio and pb_ratio < 0.5:
            anomalies.append({
                'metric': 'P/B Ratio',
                'anomaly': {
                    'type': 'extreme_low',
                    'severity': 'high',
                    'message': 'Trading below half of book value - potential distressed situation',
                    'context': f'P/B of {round(pb_ratio, 2)} is extremely low. This often indicates bankruptcy concerns, asset writedowns, or severe market pessimism.',
                    'investor_note': 'Verify the company\'s financial health and why the market is pricing it so low. Could be a turnaround opportunity or legitimate distress.'
                }
            })
        elif pb_ratio and pb_ratio > 10:
            anomalies.append({
                'metric': 'P/B Ratio',
                'anomaly': {
                    'type': 'extreme_high',
                    'severity': 'medium',
                    'message': 'Trading at 10x+ book value - asset-light business or high growth expectations',
                    'context': f'P/B of {round(pb_ratio, 2)} is very high. Normal for tech/software companies with few physical assets, but may indicate overvaluation for traditional businesses.',
                    'investor_note': 'High P/B is acceptable for asset-light businesses (software, services) but scrutinize carefully for capital-intensive industries.'
                }
            })
        # === ANOMALY DETECTION - END ===

        pb_score = self.calculate_score(
            pb_ratio if pb_ratio else 999,
            [(1.0, 10), (2.0, 8), (3.0, 6), (5.0, 4), (999, 2)]
        ) if pb_ratio else 5
    
        # 3. ROE (Return on Equity)
        roe = self.safe_get(info, 'returnOnEquity')
        
        # === FORMAT HANDLING - START ===
        # Yahoo Finance sometimes returns ROE as decimal (0.15) or percentage (15 or even higher like 58.43)
        if roe is not None:
            if abs(roe) < 1:
                # It's a decimal (0.15 = 15%)
                roe_pct = roe * 100
            else:
                # It's already a percentage (15 = 15% or 58.43 = 58.43%)
                roe_pct = roe
        else:
            roe_pct = None
        # === FORMAT HANDLING - END ===

        # ====== ADD THIS DEBUG LINE ======
        # print(f"[DEBUG] ROE - Raw: {roe}, Converted: {roe_pct}, Will check anomaly: {roe_pct and roe_pct > 100}")
        # ==================================

        # === ANOMALY DETECTION - START ===
        if roe_pct and roe_pct > 100:
            # This is the Rolls-Royce scenario!
            severity = 'high' if roe_pct > 1000 else 'medium'
            
            anomalies.append({
                'metric': 'ROE',
                'anomaly': {
                    'type': 'extreme_high',
                    'severity': severity,
                    'message': 'Extremely high ROE indicates very low shareholder equity',
                    'context': f'ROE of {round(roe_pct, 2)}% suggests the company has very low equity, often due to:\n\n• Recent losses or restructuring (e.g., post-COVID recovery)\n• Aggressive share buybacks reducing equity\n• Previous writedowns or asset sales\n\nWhile the company may be profitable now, the balance sheet is weak. This is a mathematical result, not operational excellence.',
                    'investor_note': 'Suitable for turnaround/recovery plays, NOT conservative portfolios. Verify balance sheet strength and understand the company\'s history before investing. High risk, high potential reward.'
                }
            })
        elif roe_pct and roe_pct < -20:
            anomalies.append({
                'metric': 'ROE',
                'anomaly': {
                    'type': 'extreme_low',
                    'severity': 'high',
                    'message': 'Negative ROE indicates company is losing money',
                    'context': f'ROE of {round(roe_pct, 2)}% means shareholders are seeing negative returns. Company is destroying equity value.',
                    'investor_note': 'Avoid unless you understand the turnaround story and have high risk tolerance.'
                }
            })
        # === ANOMALY DETECTION - END ===

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
            [(0.3, 10), (0.6, 8), (1.0, 6), (2.0, 4), (999, 2)]  
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
        # === ANOMALY DETECTION - START ===
        if profit_margin_pct and profit_margin_pct > 40:
            anomalies.append({
                'metric': 'Profit Margin',
                'anomaly': {
                    'type': 'extreme_high',
                    'severity': 'low',
                    'message': 'Exceptionally high profit margins',
                    'context': f'Profit margin of {round(profit_margin_pct, 2)}% is exceptional. Common in software, pharmaceuticals, or monopolistic businesses with strong pricing power and low variable costs.',
                    'investor_note': 'High margins are great but verify sustainability - can competitors erode this advantage?'
                }
            })
        elif profit_margin_pct and profit_margin_pct < 0:
            anomalies.append({
                'metric': 'Profit Margin',
                'anomaly': {
                    'type': 'negative',
                    'severity': 'high',
                    'message': 'Company is unprofitable',
                    'context': f'Profit margin of {round(profit_margin_pct, 2)}% means expenses exceed revenue. Company is losing money on operations.',
                    'investor_note': 'Unprofitable companies can be good investments if they\'re in growth phase with path to profitability. Verify the business model and cash runway.'
                }
            })
        # === ANOMALY DETECTION - END ===

        pm_score = self.calculate_score(
            profit_margin_pct if profit_margin_pct else -999,
            [(0, 2), (5, 4), (10, 6), (15, 8), (999, 10)]
        ) if profit_margin_pct else 5
    
        # 8. Dividend Yield
        dividend_yield = self.safe_get(info, 'dividendYield')

        # Yahoo Finance is inconsistent with dividend yield format
        # Strategy: Most dividend yields are 0-15%, so:
        # - If value < 0.1 (like 0.077): multiply by 100 (it's decimal)
        # - If value < 15 (like 0.77, 7.7): use as-is (already percentage)
        # - If value >= 15 (like 77): divide by 10 (something went wrong)
        if dividend_yield is not None:
            if dividend_yield < 0.1:
                # Value like 0.077 means 7.7% in decimal format
                dividend_yield_pct = dividend_yield * 100
            elif dividend_yield < 15:
                # Value like 0.77 or 7.7 is already percentage
                dividend_yield_pct = dividend_yield
            else:
                # Value like 77 or 770 is wrong, divide by 10
                dividend_yield_pct = dividend_yield / 10
        else:
            dividend_yield_pct = None
        # === ANOMALY DETECTION - START ===
        if dividend_yield_pct and dividend_yield_pct > 10:
            anomalies.append({
                'metric': 'Dividend Yield',
                'anomaly': {
                    'type': 'extreme_high',
                    'severity': 'high',
                    'message': 'Extremely high dividend yield - potential dividend cut risk',
                    'context': f'Dividend yield of {round(dividend_yield_pct, 2)}% is unusually high. This often indicates:\n\n• Stock price has fallen dramatically (yield rises as price falls)\n• Market expects dividend cut\n• Unsustainable payout ratio\n\nVerify the payout ratio and company\'s cash flow.',
                    'investor_note': 'High dividend yields above 10% are often "yield traps" - the dividend gets cut and stock falls further. Proceed with extreme caution.'
                }
            })
        # === ANOMALY DETECTION - END ===

        dy_score = self.calculate_score(
            dividend_yield_pct if dividend_yield_pct else 0,
            [(0, 3), (2, 6), (4, 9), (6, 10), (999, 8)]
        ) if dividend_yield_pct is not None else 5

        # 9. EPS Growth (Earnings Per Share)
        earnings_growth = self.safe_get(info, 'earningsGrowth')
        earnings_growth_pct = earnings_growth * 100 if earnings_growth else None

        # ====== ADD THIS DEBUG LINE ======
        # print(f"[DEBUG] Earnings Growth - Raw: {earnings_growth}, Converted: {earnings_growth_pct}, Will check anomaly: {earnings_growth_pct and earnings_growth_pct > 200}")
        # ==================================

        # === ANOMALY DETECTION - START ===
        if earnings_growth_pct and earnings_growth_pct > 200:
            anomalies.append({
                'metric': 'EPS Growth',
                'anomaly': {
                    'type': 'extreme_high',
                    'severity': 'medium',
                    'message': 'Exceptional EPS growth - verify sustainability',
                    'context': f'EPS growth of {round(earnings_growth_pct, 2)}% is exceptional. This often indicates:\n\n• Recovery from very low base (like Rolls-Royce post-COVID)\n• One-time events or accounting changes\n• Small company in hypergrowth phase\n\nSuch high growth is rarely sustainable long-term.',
                    'investor_note': 'While impressive, growth above 200% is rarely sustainable. Verify the driver is operational improvement, not accounting or one-time gains. Future growth will likely normalize.'
                }
            })
        elif earnings_growth_pct and earnings_growth_pct < -50:
            anomalies.append({
                'metric': 'EPS Growth',
                'anomaly': {
                    'type': 'extreme_low',
                    'severity': 'high',
                    'message': 'Severe earnings decline',
                    'context': f'EPS growth of {round(earnings_growth_pct, 2)}% represents a severe earnings collapse. Investigate the cause - is this temporary (cyclical downturn) or structural (broken business model)?',
                    'investor_note': 'Major earnings declines can be buying opportunities IF the business is intact and the cause is temporary. But if structural, avoid.'
                }
            })
        # === ANOMALY DETECTION - END ===

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
            
            # Score based on distance from 50% (middle is best, extremes are risky)
            # This avoids bias toward value investing (low prices) or momentum (high prices)
            distance_from_middle = abs(price_position - 50)
            
            # Thresholds: ≤10% from middle = 10, ≤20% = 8, ≤30% = 6, ≤40% = 4, >40% = 2
            pp_score = self.calculate_score(
                distance_from_middle,
                [(10, 10), (20, 8), (30, 6), (40, 4), (50, 2)]
            )
        else:
            price_position = None
            pp_score = 5
    
        # 12. Operating Margin
        operating_margin = self.safe_get(info, 'operatingMargins')
        operating_margin_pct = operating_margin * 100 if operating_margin else None

        # === ANOMALY DETECTION - START ===
        if operating_margin_pct and operating_margin_pct > 40:
            anomalies.append({
                'metric': 'Operating Margin',
                'anomaly': {
                    'type': 'extreme_high',
                    'severity': 'low',
                    'message': 'Exceptionally high operating margins',
                    'context': f'Operating margin of {round(operating_margin_pct, 2)}% is exceptional, indicating strong operational efficiency and pricing power.',
                    'investor_note': 'High margins attract competition. Check for moats (patents, network effects, brand) that protect this advantage.'
                }
            })
        elif operating_margin_pct and operating_margin_pct < -10:
            anomalies.append({
                'metric': 'Operating Margin',
                'anomaly': {
                    'type': 'extreme_low',
                    'severity': 'high',
                    'message': 'Severely negative operating margin',
                    'context': f'Operating margin of {round(operating_margin_pct, 2)}% indicates core operations are deeply unprofitable.',
                    'investor_note': 'Heavy losses at operating level are concerning unless this is a deliberate growth investment (like Amazon in early years).'
                }
            })
# === ANOMALY DETECTION - END ===

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
    
        # Add benchmark comparisons
        sector = company_overview.get('sector')
        kpis_with_benchmarks = self.add_benchmarks(kpis, sector, ticker)
        
        return {
            'company_name': company_name,
            'ticker': ticker,
            'current_price': round(current_price, 2) if current_price else None,
            'currency': currency,
            'kpis': kpis_with_benchmarks,
            'anomalies': anomalies,  # ← ADD THIS LINE
            'historical_prices': historical_prices,
            'news': news,
            'company_overview': company_overview
        }
    def add_benchmarks(self, kpis: dict, sector: str, ticker: str) -> dict:
        """Add benchmark comparisons to KPIs"""
        
        # Determine market (US or UK)
        market = 'UK' if ticker.endswith('.L') else 'US'
        market_name = 'FTSE 100' if market == 'UK' else 'S&P 500'
        
        # Get market benchmarks
        market_bench = MARKET_BENCHMARKS.get(market, MARKET_BENCHMARKS['US'])
        
        # Get sector benchmarks (fallback to market if sector not found)
        sector_bench = SECTOR_BENCHMARKS.get(sector, None)
        
        # Add benchmarks to each KPI
        for category, metrics in kpis.items():
            for kpi_key, kpi_data in metrics.items():
                if kpi_data['value'] is not None:
                    # Get market benchmark
                    market_value = market_bench.get(kpi_key)
                    
                    # Get sector benchmark
                    sector_value = sector_bench.get(kpi_key) if sector_bench else None
                    
                    # Calculate comparisons
                    benchmarks = {}
                    
                    if market_value:
                        # Determine if higher is better or lower is better
                        lower_is_better = kpi_key in ['pe_ratio', 'pb_ratio', 'debt_to_equity', 'beta', 'price_position']
                        
                        # Calculate percentage difference
                        diff_pct = ((kpi_data['value'] - market_value) / market_value) * 100
                        
                        # Determine if this is good or bad
                        if lower_is_better:
                            is_better = kpi_data['value'] < market_value
                            # Fix the status wording based on is_better
                            if kpi_key in ['pe_ratio', 'pb_ratio']:
                                status = 'cheaper' if is_better else 'more expensive'
                            else:
                                status = 'lower' if is_better else 'higher'
                        else:
                            is_better = kpi_data['value'] > market_value
                            status = 'higher' if is_better else 'lower'
                        
                        benchmarks['market'] = {
                            'name': market_name,
                            'value': round(market_value, 2),
                            'diff_pct': round(abs(diff_pct), 1),  # Always positive for display
                            'is_better': is_better,
                            'status': status
                        }
                    
                    if sector_value and sector:
                        # Calculate percentage difference
                        diff_pct = ((kpi_data['value'] - sector_value) / sector_value) * 100
                        
                        # Determine if higher is better or lower is better
                        lower_is_better = kpi_key in ['pe_ratio', 'pb_ratio', 'debt_to_equity', 'beta', 'price_position']
                        
                        # Determine if this is good or bad
                        if lower_is_better:
                            is_better = kpi_data['value'] < sector_value
                            # Fix the status wording based on is_better
                            if kpi_key in ['pe_ratio', 'pb_ratio']:
                                status = 'cheaper' if is_better else 'more expensive'
                            else:
                                status = 'lower' if is_better else 'higher'
                        else:
                            is_better = kpi_data['value'] > sector_value
                            status = 'higher' if is_better else 'lower'
                        
                        benchmarks['sector'] = {
                            'name': sector,
                            'value': round(sector_value, 2),
                            'diff_pct': round(abs(diff_pct), 1),  # Always positive for display
                            'is_better': is_better,
                            'status': status
                        }
                    
                    # Add benchmarks to KPI data
                    kpi_data['benchmarks'] = benchmarks
        
        return kpis
        

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
        """Generate overall Buy/Hold/Sell analysis rating"""
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
        elif total_score_100 >= 50:
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
            'anomalies': analysis.get('anomalies', []),  # ← ADD THIS LINE
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
def send_welcome_email(first_name: str, last_name: str, email: str):
    """
    Add subscriber to MailerLite group using REST API (triggers welcome email automation)
    """
    try:
        if not MAILERLITE_API_KEY or not MAILERLITE_GROUP_ID:
            print("⚠️ MailerLite not configured")
            return False
        
        # MailerLite API endpoint
        url = "https://connect.mailerlite.com/api/subscribers"
        
        # Headers for authentication
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MAILERLITE_API_KEY}"
        }
        
        # Subscriber data with group assignment
        data = {
            "email": email,
            "fields": {
                "name": first_name,
                "last_name": last_name
            },
            "groups": [MAILERLITE_GROUP_ID]  # Assign to group directly
        }
        
        # Make API request
        response = requests.post(url, headers=headers, json=data)
        
        # Check response
        if response.status_code in [200, 201]:
            print(f"✓ Subscriber added to MailerLite: {email}")
            print(f"✓ Added to group: {MAILERLITE_GROUP_ID}")
            print(f"  Welcome email will be sent automatically by automation")
            return True
        else:
            print(f"✗ Failed to add subscriber to MailerLite: {email}")
            print(f"  Status code: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
        
    except Exception as e:
        print(f"✗ Failed to add subscriber to MailerLite: {email}")
        print(f"  Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
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
        
        # Add subscriber to MailerLite (triggers welcome email automation)
        email_sent = send_welcome_email(first_name, last_name, email)

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