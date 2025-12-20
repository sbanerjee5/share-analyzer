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

app = FastAPI(title="UK Share Analyzer API", version="1.0.0")

# CORS configuration for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
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
                            
                            # Try different possible field names from the nested content
                            title = (article_data.get('title') or 
                                   article_data.get('headline') or 
                                   article_data.get('summary') or 
                                   'No title available')
                            
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
                            
                            print(f"Article {idx}: title={title[:50] if len(title) > 50 else title}, publisher={publisher}, link={link[:50] if link and len(link) > 50 else link}")
                            
                            news.append({
                                'title': title,
                                'publisher': publisher,
                                'link': link,
                                'publish_time': pub_time,
                                'type': article_data.get('type', 'article')
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
        dividend_yield = self.safe_get(info, 'dividendYield')
        dividend_yield_pct = dividend_yield * 100 if dividend_yield else None
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
        
        # Get 12-month historical price data
        historical_prices = []
        try:
            hist_data = data.get('history')
            if hist_data is not None and not hist_data.empty and 'Close' in hist_data.columns:
                hist_12m = hist_data.tail(252)
                for date, row in hist_12m.iterrows():
                    historical_prices.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'price': round(float(row['Close']), 2)
                    })
                print(f"Fetched {len(historical_prices)} historical price points")
            else:
                print("No historical data available")
        except Exception as e:
            print(f"Warning: Could not fetch historical prices: {e}")
            historical_prices = []
        
        # Get news from data
        news = data.get('news', [])
        
        return {
            'company_name': company_name,
            'ticker': ticker,
            'current_price': round(current_price, 2) if current_price else None,
            'currency': currency,
            'kpis': kpis,
            'historical_prices': historical_prices,
            'news': news
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
            'news': analysis.get('news', [])
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)