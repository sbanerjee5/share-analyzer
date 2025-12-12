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

app = FastAPI(title="UK Share Analyzer API", version="1.0.0")

# CORS configuration - UPDATED FOR NETLIFY DEPLOYMENT
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174", 
        "http://localhost:3000",
        "https://magnificent-figolla-37a50b.netlify.app"  # Your Netlify domain
    ],
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
            hist = stock.history(period="1y")
            
            data = {
                'info': info,
                'history': hist,
                'financials': stock.financials,
                'balance_sheet': stock.balance_sheet,
                'cashflow': stock.cashflow,
                'recommendations': stock.recommendations,
                'news': stock.news if hasattr(stock, 'news') else []
            }
            
            # Cache the data
            cache[cache_key] = (data, datetime.now())
            print(f"Successfully fetched and cached data for {ticker}")
            return data
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch stock data: {str(e)}")
    
    @staticmethod
    def calculate_valuation_kpis(data: Dict) -> Dict:
        """Calculate P/E and P/B ratios"""
        info = data.get('info', {})
        
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        pb_ratio = info.get('priceToBook')
        
        # Score P/E (lower is better, but not too low)
        pe_score = 5
        if pe_ratio:
            if 10 <= pe_ratio <= 20:
                pe_score = 9
            elif 20 < pe_ratio <= 30:
                pe_score = 7
            elif pe_ratio < 10:
                pe_score = 6
            else:
                pe_score = 4
        
        # Score P/B (lower is better)
        pb_score = 5
        if pb_ratio:
            if pb_ratio < 1:
                pb_score = 10
            elif pb_ratio < 3:
                pb_score = 8
            elif pb_ratio < 5:
                pb_score = 6
            else:
                pb_score = 4
        
        return {
            'pe_ratio': {
                'value': round(pe_ratio, 2) if pe_ratio else None,
                'score': pe_score
            },
            'pb_ratio': {
                'value': round(pb_ratio, 2) if pb_ratio else None,
                'score': pb_score
            }
        }
    
    @staticmethod
    def calculate_profitability_kpis(data: Dict) -> Dict:
        """Calculate ROE, Profit Margin, and Operating Margin"""
        info = data.get('info', {})
        
        roe = info.get('returnOnEquity')
        profit_margin = info.get('profitMargins')
        operating_margin = info.get('operatingMargins')
        
        # Convert to percentages
        roe_pct = round(roe * 100, 2) if roe else None
        profit_pct = round(profit_margin * 100, 2) if profit_margin else None
        operating_pct = round(operating_margin * 100, 2) if operating_margin else None
        
        # Score ROE
        roe_score = 5
        if roe_pct:
            if roe_pct >= 20:
                roe_score = 10
            elif roe_pct >= 15:
                roe_score = 8
            elif roe_pct >= 10:
                roe_score = 6
            else:
                roe_score = 4
        
        # Score Profit Margin
        profit_score = 5
        if profit_pct:
            if profit_pct >= 20:
                profit_score = 10
            elif profit_pct >= 10:
                profit_score = 8
            elif profit_pct >= 5:
                profit_score = 6
            else:
                profit_score = 4
        
        # Score Operating Margin
        operating_score = 5
        if operating_pct:
            if operating_pct >= 20:
                operating_score = 10
            elif operating_pct >= 15:
                operating_score = 8
            elif operating_pct >= 10:
                operating_score = 6
            else:
                operating_score = 4
        
        return {
            'roe': {
                'value': roe_pct,
                'score': roe_score
            },
            'profit_margin': {
                'value': profit_pct,
                'score': profit_score
            },
            'operating_margin': {
                'value': operating_pct,
                'score': operating_score
            }
        }
    
    @staticmethod
    def calculate_health_kpis(data: Dict) -> Dict:
        """Calculate Debt-to-Equity and Current Ratio"""
        info = data.get('info', {})
        
        debt_to_equity = info.get('debtToEquity')
        current_ratio = info.get('currentRatio')
        
        # Convert debt to equity from percentage to ratio
        de_ratio = round(debt_to_equity / 100, 2) if debt_to_equity else None
        
        # Score Debt-to-Equity (lower is better)
        de_score = 5
        if de_ratio is not None:
            if de_ratio < 0.5:
                de_score = 10
            elif de_ratio < 1:
                de_score = 8
            elif de_ratio < 2:
                de_score = 6
            else:
                de_score = 3
        
        # Score Current Ratio (around 1.5-2 is ideal)
        cr_score = 5
        if current_ratio:
            if 1.5 <= current_ratio <= 2.5:
                cr_score = 10
            elif 1 <= current_ratio < 1.5:
                cr_score = 7
            elif current_ratio >= 2.5:
                cr_score = 8
            else:
                cr_score = 4
        
        return {
            'debt_to_equity': {
                'value': de_ratio,
                'score': de_score
            },
            'current_ratio': {
                'value': round(current_ratio, 2) if current_ratio else None,
                'score': cr_score
            }
        }
    
    @staticmethod
    def calculate_growth_kpis(data: Dict) -> Dict:
        """Calculate Revenue Growth and EPS Growth"""
        info = data.get('info', {})
        
        revenue_growth = info.get('revenueGrowth')
        earnings_growth = info.get('earningsGrowth')
        
        # Convert to percentages
        revenue_pct = round(revenue_growth * 100, 2) if revenue_growth else None
        earnings_pct = round(earnings_growth * 100, 2) if earnings_growth else None
        
        # Score Revenue Growth
        rev_score = 5
        if revenue_pct is not None:
            if revenue_pct >= 20:
                rev_score = 10
            elif revenue_pct >= 10:
                rev_score = 8
            elif revenue_pct >= 5:
                rev_score = 6
            elif revenue_pct >= 0:
                rev_score = 5
            else:
                rev_score = 3
        
        # Score EPS Growth
        eps_score = 5
        if earnings_pct is not None:
            if earnings_pct >= 20:
                eps_score = 10
            elif earnings_pct >= 10:
                eps_score = 8
            elif earnings_pct >= 5:
                eps_score = 6
            elif earnings_pct >= 0:
                eps_score = 5
            else:
                eps_score = 3
        
        return {
            'revenue_growth': {
                'value': revenue_pct,
                'score': rev_score
            },
            'eps_growth': {
                'value': earnings_pct,
                'score': eps_score
            }
        }
    
    @staticmethod
    def calculate_technical_kpis(data: Dict) -> Dict:
        """Calculate Beta, 52-Week Position, and Dividend Yield"""
        info = data.get('info', {})
        hist = data.get('history')
        
        beta = info.get('beta')
        dividend_yield = info.get('dividendYield')
        
        # Calculate 52-week price position
        price_position = None
        if hist is not None and not hist.empty and 'Close' in hist.columns:
            current_price = hist['Close'].iloc[-1]
            low_52 = hist['Close'].min()
            high_52 = hist['Close'].max()
            
            if high_52 > low_52:
                price_position = round(((current_price - low_52) / (high_52 - low_52)) * 100, 2)
        
        # Convert dividend yield to percentage
        div_yield_pct = round(dividend_yield * 100, 2) if dividend_yield else None
        
        # Score Beta (close to 1 is good, very high or low is concerning)
        beta_score = 5
        if beta:
            if 0.8 <= beta <= 1.2:
                beta_score = 8
            elif 0.5 <= beta < 0.8 or 1.2 < beta <= 1.5:
                beta_score = 6
            else:
                beta_score = 4
        
        # Score Price Position (mid-range is safer)
        pos_score = 5
        if price_position is not None:
            if 30 <= price_position <= 70:
                pos_score = 8
            elif 20 <= price_position < 30 or 70 < price_position <= 80:
                pos_score = 6
            else:
                pos_score = 4
        
        # Score Dividend Yield
        div_score = 5
        if div_yield_pct is not None:
            if div_yield_pct >= 4:
                div_score = 9
            elif div_yield_pct >= 2:
                div_score = 7
            elif div_yield_pct >= 1:
                div_score = 5
            elif div_yield_pct > 0:
                div_score = 3
            else:
                div_score = 2
        
        return {
            'beta': {
                'value': round(beta, 2) if beta else None,
                'score': beta_score
            },
            'price_position': {
                'value': price_position,
                'score': pos_score
            },
            'dividend_yield': {
                'value': div_yield_pct,
                'score': div_score
            }
        }
    
    @staticmethod
    def calculate_overall_recommendation(kpis: Dict) -> Dict:
        """Calculate overall recommendation based on all KPIs"""
        # Calculate category averages
        valuation_avg = (kpis['valuation']['pe_ratio']['score'] + 
                        kpis['valuation']['pb_ratio']['score']) / 2
        
        profitability_avg = (kpis['profitability']['roe']['score'] + 
                           kpis['profitability']['profit_margin']['score'] + 
                           kpis['profitability']['operating_margin']['score']) / 3
        
        health_avg = (kpis['health']['debt_to_equity']['score'] + 
                     kpis['health']['current_ratio']['score']) / 2
        
        growth_avg = (kpis['growth']['revenue_growth']['score'] + 
                     kpis['growth']['eps_growth']['score']) / 2
        
        technical_avg = (kpis['technical']['beta']['score'] + 
                        kpis['technical']['price_position']['score'] + 
                        kpis['technical']['dividend_yield']['score']) / 3
        
        # Overall score (weighted average)
        overall_score = (
            valuation_avg * 0.25 +
            profitability_avg * 0.25 +
            health_avg * 0.20 +
            growth_avg * 0.20 +
            technical_avg * 0.10
        )
        
        # Convert to 100-point scale
        overall_score = round(overall_score * 10)
        
        # Determine recommendation
        if overall_score >= 70:
            recommendation = "BUY"
            message = "Strong fundamentals indicate a good investment opportunity."
        elif overall_score >= 50:
            recommendation = "HOLD"
            message = "Mixed signals. Suitable for existing positions but wait for better entry point for new positions."
        else:
            recommendation = "SELL"
            message = "Weak fundamentals suggest caution. Consider reducing exposure."
        
        return {
            'recommendation': recommendation,
            'score': overall_score,
            'message': message,
            'category_scores': {
                'valuation': round(valuation_avg, 1),
                'profitability': round(profitability_avg, 1),
                'health': round(health_avg, 1),
                'growth': round(growth_avg, 1),
                'technical': round(technical_avg, 1)
            }
        }
    
    @staticmethod
    def analyze_sentiment(text: str) -> Dict:
        """Simple sentiment analysis based on keywords"""
        positive_words = ['strong', 'growth', 'profit', 'gain', 'success', 'increase', 
                         'positive', 'beat', 'exceed', 'outperform', 'upgraded', 'bullish']
        negative_words = ['weak', 'loss', 'decline', 'fall', 'drop', 'concern', 
                         'negative', 'miss', 'underperform', 'downgraded', 'bearish', 'risk']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return {'sentiment': 'positive', 'score': min(pos_count, 5)}
        elif neg_count > pos_count:
            return {'sentiment': 'negative', 'score': min(neg_count, 5)}
        else:
            return {'sentiment': 'neutral', 'score': 0}
    
    @staticmethod
    def calculate_read_time(text: str) -> int:
        """Calculate estimated read time in minutes (average 200 words per minute)"""
        if not text:
            return 1
        
        word_count = len(text.split())
        read_time = max(1, round(word_count / 200))
        return min(read_time, 10)  # Cap at 10 minutes
    
    @staticmethod
    def classify_category(text: str) -> str:
        """Classify article into categories based on content"""
        text_lower = text.lower()
        
        # Check for earnings-related keywords
        if any(word in text_lower for word in ['earnings', 'profit', 'revenue', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'fiscal']):
            return 'earnings'
        
        # Check for M&A keywords
        if any(word in text_lower for word in ['acquisition', 'merger', 'acquired', 'deal', 'takeover', 'buyout']):
            return 'ma'
        
        # Check for regulation keywords
        if any(word in text_lower for word in ['regulation', 'regulatory', 'compliance', 'lawsuit', 'legal', 'sec', 'fca']):
            return 'regulation'
        
        # Check for market-related keywords
        if any(word in text_lower for word in ['market', 'index', 'stocks', 'trading', 'investor', 'wall street']):
            return 'markets'
        
        # Check for leadership keywords
        if any(word in text_lower for word in ['ceo', 'executive', 'management', 'board', 'director', 'appointed', 'resignation']):
            return 'leadership'
        
        # Default to general
        return 'general'

    @staticmethod
    def get_company_overview(data: Dict) -> Dict:
        """Extract company overview information"""
        info = data.get('info', {})
        
        # Get exchange information
        exchange = info.get('exchange', '')
        exchange_full = info.get('fullExchangeName', exchange)
        
        # Determine index based on exchange and market cap
        index = None
        market_cap = info.get('marketCap', 0)
        
        if 'LON' in exchange or 'LSE' in exchange_full:
            # UK stocks - determine FTSE 100 or FTSE 250
            if market_cap > 5_000_000_000:  # Over £5B typically FTSE 100
                index = 'FTSE 100'
            else:
                index = 'FTSE 250'
        elif exchange in ['NMS', 'NYQ', 'NGM']:
            # US stocks
            if market_cap > 10_000_000_000:  # Over $10B
                index = 'S&P 500'
            else:
                index = 'S&P MidCap'
        
        # Get day range
        day_low = info.get('dayLow')
        day_high = info.get('dayHigh')
        day_range = None
        if day_low and day_high:
            day_range = {'low': day_low, 'high': day_high}
        
        # Get 52-week range
        fifty_two_low = info.get('fiftyTwoWeekLow')
        fifty_two_high = info.get('fiftyTwoWeekHigh')
        week_52_range = None
        if fifty_two_low and fifty_two_high:
            week_52_range = {'low': fifty_two_low, 'high': fifty_two_high}
        
        return {
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'description': info.get('longBusinessSummary'),
            'website': info.get('website'),
            'employees': info.get('fullTimeEmployees'),
            'headquarters': f"{info.get('city', '')}, {info.get('country', '')}".strip(', '),
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
            'exchange': exchange,
            'exchange_full': exchange_full,
            'index': index,
            'day_range': day_range,
            'week_52_range': week_52_range,
            'volume': info.get('volume'),
            'avg_volume': info.get('averageVolume')
        }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "UK Share Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/api/analyze",
            "health": "/api/health",
            "clear_cache": "/api/clear-cache"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_size": len(cache)
    }


@app.post("/api/clear-cache")
async def clear_cache():
    """Clear the cache"""
    cache.clear()
    return {
        "success": True,
        "message": "Cache cleared successfully",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/analyze")
async def analyze_stock(request: AnalysisRequest):
    """
    Analyze a stock and return comprehensive metrics
    """
    try:
        ticker = request.ticker.strip().upper()
        print(f"\n{'='*50}")
        print(f"Analyzing: {ticker}")
        print(f"{'='*50}")
        
        # Fetch stock data
        data = KPICalculator.get_stock_data(ticker)
        info = data.get('info', {})
        
        # Basic company info
        company_name = info.get('longName') or info.get('shortName') or ticker
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        currency = info.get('currency', 'USD')
        
        # Handle UK pence notation
        if currency == 'GBp':
            currency = 'GBp'
        
        print(f"Company: {company_name}")
        print(f"Price: {currency} {current_price}")
        
        # Calculate all KPIs
        print("\nCalculating KPIs...")
        kpis = {
            'valuation': KPICalculator.calculate_valuation_kpis(data),
            'profitability': KPICalculator.calculate_profitability_kpis(data),
            'health': KPICalculator.calculate_health_kpis(data),
            'growth': KPICalculator.calculate_growth_kpis(data),
            'technical': KPICalculator.calculate_technical_kpis(data)
        }
        
        # Calculate recommendation
        recommendation = KPICalculator.calculate_overall_recommendation(kpis)
        print(f"\nRecommendation: {recommendation['recommendation']} (Score: {recommendation['score']}/100)")
        
        # Get 12-month historical price data WITH MOVING AVERAGES
        historical_prices = []
        try:
            hist_data = data.get('history')
            if hist_data is not None and not hist_data.empty and 'Close' in hist_data.columns:
                hist_12m = hist_data.tail(252)
                
                # Calculate moving averages
                ma_50 = hist_12m['Close'].rolling(window=50, min_periods=1).mean()
                ma_200 = hist_12m['Close'].rolling(window=200, min_periods=1).mean()
                
                for i, (date, row) in enumerate(hist_12m.iterrows()):
                    historical_prices.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'price': round(float(row['Close']), 2),
                        'ma_50': round(float(ma_50.iloc[i]), 2) if not pd.isna(ma_50.iloc[i]) else None,
                        'ma_200': round(float(ma_200.iloc[i]), 2) if not pd.isna(ma_200.iloc[i]) else None
                    })
                
                print(f"Historical prices: {len(historical_prices)} data points with moving averages")
        except Exception as e:
            print(f"Warning: Could not fetch historical prices: {str(e)}")
        
        # Get company overview
        company_overview = KPICalculator.get_company_overview(data)
        
        # Process news
        print("\nProcessing news articles...")
        news = []
        try:
            news_data = data.get('news', [])
            if news_data and len(news_data) > 0:
                print(f"Found {len(news_data)} news articles")
                
                for idx, article_data in enumerate(news_data[:10]):
                    try:
                        print(f"\n=== Processing Article {idx + 1} ===")
                        
                        if not isinstance(article_data, dict):
                            print(f"Skipping article {idx}: not a dictionary")
                            continue
                        
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
                                  '#')
                        
                        pub_time = article_data.get('providerPublishTime', 0)
                        
                        thumbnail_url = None
                        thumbnail_data = article_data.get('thumbnail', {})
                        if isinstance(thumbnail_data, dict):
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
                        
                    except Exception as e:
                        print(f"Error processing article {idx}: {str(e)}")
                        continue
                
                print(f"\nSuccessfully processed {len(news)} articles")
            else:
                print("No news data available")
        except Exception as e:
            print(f"Warning: Could not process news: {str(e)}")
        
        response_data = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'company': {
                'name': company_name,
                'ticker': ticker,
                'price': round(current_price, 2) if current_price else None,
                'currency': currency
            },
            'kpis': kpis,
            'recommendation': recommendation,
            'historical_prices': historical_prices,
            'company_overview': company_overview,
            'news': news
        }
        
        print(f"\n{'='*50}")
        print(f"Analysis complete for {ticker}")
        print(f"{'='*50}\n")
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in analyze_stock: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
