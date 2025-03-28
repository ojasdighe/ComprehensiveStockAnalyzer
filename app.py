#################################################################################################
# File Name - app.py
# Author - Ojas Ulhas Dighe (Updated)
# Date - 18th Mar 2025
# Description - This file contains the code for the Flask application that serves the frontend
#               and now includes functionality to fetch and analyze top gainers
#################################################################################################

from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import click
import requests
import json
import time
import random
from bs4 import BeautifulSoup

# Initialize Flask app and set up logging  

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#################################################################################################
# Class Name - IndianStockAnalyzer
# Author - Ojas Ulhas Dighe (Updated)
# Date - 18th Mar 2025
# Description - This class contains the logic for analyzing Indian stocks using technical indicators
#################################################################################################

class IndianStockAnalyzer:
    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 risk_free_rate: float = 0.05):
        """
        Initialize the Indian Stock Analyzer.
        """
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.risk_free_rate = risk_free_rate
        self.stock_data = None
        self.stock_info = None

        # Set up a session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Referer': 'https://www.nseindia.com/',
            'DNT': '1',
            'Upgrade-Insecure-Requests': '1'
        })
        self.session.get('https://www.nseindia.com/')  # Initial request to set cookies
        time.sleep(2)  # Sleep to avoid being blocked
        self.session.get('https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY')  # Another request to set cookies
        time.sleep(2)  # Sleep to avoid being blocked
        self.session.get('https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY')  # Another request to set cookies
        time.sleep(2)  # Sleep to avoid being blocked
        self.session.get('https://www.nseindia.com/api/option-chain-indices?symbol=FINNIFTY')  # Another request to set cookies
        time.sleep(2)  # Sleep to avoid being blocked

        # Set up a session for yfinance
        self.yf_session = requests.Session()
        self.yf_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Referer': 'https://www.nseindia.com/',
            'DNT': '1',
            'Upgrade-Insecure-Requests': '1'
        })
        self.yf_session.get('https://www.nseindia.com/')  # Initial request to set cookies
        time.sleep(2)  # Sleep to avoid being blocked
        self.yf_session.get('https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY')  # Another request to set cookies
        time.sleep(2)  # Sleep to avoid being blocked
        self.yf_session.get('https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY')  # Another request to set cookies
        time.sleep(2)  # Sleep to avoid being blocked
        self.yf_session.get('https://www.nseindia.com/api/option-chain-indices?symbol=FINNIFTY')  # Another request to set cookies
        time.sleep(2)  # Sleep to avoid being blocked

        # Set up a session for BeautifulSoup
        self.bs_session = requests.Session()
        self.bs_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Referer': 'https://www.nseindia.com/',
            'DNT': '1',
            'Upgrade-Insecure-Requests': '1'
        })
        self.bs_session.get('https://www.nseindia.com/')  # Initial request to set cookies
        time.sleep(2)  # Sleep to avoid being blocked
        self.bs_session.get('https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY')  # Another request to set cookies
        time.sleep(2)  # Sleep to avoid being blocked
        self.bs_session.get('https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY')  # Another request to set cookies
        time.sleep(2)  # Sleep to avoid being blocked
        self.bs_session.get('https://www.nseindia.com/api/option-chain-indices?symbol=FINNIFTY')  # Another request to set cookies
        time.sleep(2)  # Sleep to avoid being blocked

################################################################################################
# Function Name - fetch_fundamental_data
# Author - Ojas Ulhas Dighe (Updated)
# Date - 28th Mar 2025
# Description - This function fetches fundamental data for a given stock symbol using web scraping and APIs
#              and performs comprehensive fundamental analysis.
#              It uses yfinance for basic fundamental data and additional metrics.
#              The function returns a dictionary containing the fundamental data and analysis results.
#              The scoring criteria for fundamental metrics are defined within the function.
#################################################################################################


    def fetch_fundamental_data(self, symbol: str) -> dict:
        """
    Fetch fundamental data for a given stock symbol using web scraping and APIs.
    
    Parameters:
    symbol (str): Stock symbol to analyze
    
    Returns:
    dict: Comprehensive fundamental analysis data
    """
        try:
            # Construct full symbol for NSE
            full_symbol = f"{symbol}.NS"
            
            # Use yfinance for basic fundamental data
            stock = yf.Ticker(full_symbol)
            info = stock.info
            
            # Fetch additional fundamental details
            fundamental_data = {
                'basic_info': {
                    'company_name': info.get('longName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                },
                'valuation_metrics': {
                    'market_cap': info.get('marketCap', 0),
                    'enterprise_value': info.get('enterpriseValue', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'forward_pe': info.get('forwardPE', 0),
                    'price_to_book': info.get('priceToBook', 0),
                    'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                    'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
                },
                'financial_health': {
                    'total_revenue': info.get('totalRevenue', 0),
                    'gross_profit': info.get('grossProfit', 0),
                    'net_income': info.get('netIncomeToCommon', 0),
                    'total_debt': info.get('totalDebt', 0),
                    'debt_to_equity': info.get('debtToEquity', 0),
                    'return_on_equity': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
                },
                'growth_metrics': {
                    'revenue_growth': info.get('revenueGrowth', 0) * 100,
                    'earnings_growth': info.get('earningsGrowth', 0) * 100,
                    'profit_margins': info.get('profitMargins', 0) * 100
                }
            }
        
            return fundamental_data
        
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
            return {}

################################################################################################
# Function Name - perform_fundamental_analysis
# Author - Ojas Ulhas Dighe (Updated)
# Date - 28th Mar 2025
# Description - This function performs comprehensive fundamental analysis on a given stock symbol.
#             It uses the fetch_fundamental_data function to get the data and then applies scoring criteria
#            to generate a recommendation.
#            The scoring criteria for fundamental metrics are defined within the function. 
# #################################################################################################

    def perform_fundamental_analysis(self, symbol: str) -> dict:
        """
        Perform comprehensive fundamental analysis.
        
        Parameters:
        symbol (str): Stock symbol to analyze
        
        Returns:
        dict: Fundamental analysis insights and recommendations
        """
        try:
            fundamental_data = self.fetch_fundamental_data(symbol)
            
            # Define fundamental analysis scoring criteria
            def score_fundamental_metrics(data):
                score = 0
                
                # PE Ratio Scoring
                pe_ratio = data['valuation_metrics']['pe_ratio']
                if 0 < pe_ratio < 15:
                    score += 2  # Undervalued
                elif 15 <= pe_ratio <= 25:
                    score += 1  # Fairly valued
                else:
                    score -= 1  # Potentially overvalued
                
                # Debt to Equity Scoring
                debt_to_equity = data['financial_health']['debt_to_equity']
                if debt_to_equity < 0.5:
                    score += 2  # Low debt
                elif debt_to_equity < 1:
                    score += 1  # Moderate debt
                else:
                    score -= 1  # High debt
                
                # Growth Metrics Scoring
                revenue_growth = data['growth_metrics']['revenue_growth']
                earnings_growth = data['growth_metrics']['earnings_growth']
                if revenue_growth > 10 and earnings_growth > 10:
                    score += 2  # Strong growth
                elif revenue_growth > 5 and earnings_growth > 5:
                    score += 1  # Moderate growth
                else:
                    score -= 1  # Low growth
                
                # Profitability Scoring
                profit_margins = data['growth_metrics']['profit_margins']
                if profit_margins > 15:
                    score += 2  # High profitability
                elif profit_margins > 10:
                    score += 1  # Good profitability
                else:
                    score -= 1  # Low profitability
                
                # Dividend Yield Scoring
                dividend_yield = data['valuation_metrics']['dividend_yield']
                if dividend_yield > 3:
                    score += 1  # Good dividend
                
                return score
            
            # Generate fundamental recommendation
            fundamental_score = score_fundamental_metrics(fundamental_data)
            
            if fundamental_score >= 4:
                recommendation = "Strong Buy"
            elif fundamental_score >= 2:
                recommendation = "Buy"
            elif fundamental_score >= 0:
                recommendation = "Hold"
            elif fundamental_score >= -2:
                recommendation = "Sell"
            else:
                recommendation = "Strong Sell"
            
            fundamental_data['recommendation'] = recommendation
            fundamental_data['score'] = fundamental_score
            
            return fundamental_data
        
        except Exception as e:
            logger.error(f"Fundamental analysis error for {symbol}: {str(e)}")
            return {}

#################################################################################################
# Function Name - fetch_top_gainers
# Author - Updated
# Date - 18th Mar 2025
# Description - This function fetches top gainers from NSE India or fallback to yfinance for ^NSEI
#################################################################################################

    def fetch_top_gainers(self, limit: int = 5) -> list:
        """
        Fetch top gainers data from NSE India API or fallback to yfinance for ^NSEI.
        
        Parameters:
        limit (int): Number of top gainers to return
        
        Returns:
        list: List of top gainers with symbol and other information
        """
        try:
            # Try direct NSE API approach first
            try:
                # Headers to mimic a browser request
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Connection': 'keep-alive',
                    'Referer': 'https://www.nseindia.com/',
                    'DNT': '1',
                    'Upgrade-Insecure-Requests': '1'
                }
                
                # First make a request to the NSE homepage to get cookies
                session = requests.Session()
                homepage_response = session.get('https://www.nseindia.com/', headers=headers, timeout=10)
                
                # Sleep to avoid being blocked
                time.sleep(2)
                
                # Now fetch the top gainers data
                api_url = "https://www.nseindia.com/api/live-analysis-variations?index=gainers"
                response = session.get(api_url, headers=headers, timeout=10)
                
                if response.status_code != 200:
                    logger.error(f"NSE API returned status code: {response.status_code}")
                    logger.info("Response content preview: " + response.text[:100])
                    raise Exception(f"Failed to fetch data: Status code {response.status_code}")
                
                try:
                    data = response.json()
                    # Extract the top gainers
                    gainers = data.get('NIFTY', {}).get('data', [])[:limit]
                    
                    # Format the data
                    formatted_gainers = []
                    for gainer in gainers:
                        formatted_gainers.append({
                            'symbol': gainer.get('symbol', ''),
                            'series': gainer.get('series', ''),
                            'ltp': gainer.get('ltp', 0),
                            'percentChange': gainer.get('perChange', 0),
                            'tradedQuantity': gainer.get('tradedQuantity', 0),
                            'value': gainer.get('value', 0),
                            'open': gainer.get('open', 0),
                            'high': gainer.get('high', 0),
                            'low': gainer.get('low', 0),
                            'previousClose': gainer.get('previousClose', 0)
                        })
                    
                    return formatted_gainers
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON: {str(e)}")
                    logger.info("Response content preview: " + response.text[:100])
                    raise
                
            except Exception as e:
                logger.error(f"NSE API approach failed: {str(e)}")
                logger.info("Falling back to yfinance for top Nifty stocks")
                raise
                
        except Exception as e:
            logger.warning(f"Error fetching from NSE API: {str(e)}")
            logger.info("Fallback: Using yfinance to get top Nifty stocks")
            
            # Fallback: Use yfinance to get top Nifty stocks
            # This won't give us real-time gainers but will provide some stocks to analyze
            try:
                # Get Nifty 50 components
                nifty_stocks = [
                    "RELIANCE", "HDFCBANK", "INFY", "ICICIBANK", "TCS", 
                    "KOTAKBANK", "HINDUNILVR", "LT", "AXISBANK", "BAJFINANCE",
                    "SBIN", "MARUTI", "ASIANPAINT", "HDFC", "BHARTIARTL",
                    "TITAN", "BAJAJFINSV", "TATASTEEL", "INDUSINDBK", "ULTRACEMCO"
                ]
                
                # Shuffle and take the top 'limit' stocks
                random.shuffle(nifty_stocks)
                selected_stocks = nifty_stocks[:limit]
                
                formatted_gainers = []
                for symbol in selected_stocks:
                    try:
                        stock = yf.Ticker(f"{symbol}.NS")
                        info = stock.info
                        history = stock.history(period="2d")
                        
                        if len(history) >= 2:
                            prev_close = history['Close'].iloc[-2]
                            current = history['Close'].iloc[-1]
                            percent_change = ((current - prev_close) / prev_close) * 100
                        else:
                            percent_change = 0
                            
                        formatted_gainers.append({
                            'symbol': symbol,
                            'series': 'EQ',
                            'ltp': current if 'current' in locals() else info.get('previousClose', 0),
                            'percentChange': round(percent_change, 2),
                            'tradedQuantity': info.get('volume', 0),
                            'value': info.get('marketCap', 0),
                            'open': info.get('open', 0),
                            'high': info.get('dayHigh', 0),
                            'low': info.get('dayLow', 0),
                            'previousClose': info.get('previousClose', 0)
                        })
                    except Exception as e:
                        logger.error(f"Error fetching data for {symbol}: {str(e)}")
                
                # Sort by percent change (descending)
                formatted_gainers.sort(key=lambda x: x['percentChange'], reverse=True)
                return formatted_gainers[:limit]
                
            except Exception as e:
                logger.error(f"Fallback method failed: {str(e)}")
                raise Exception(f"Failed to fetch top gainers: {str(e)}")

#################################################################################################
# Function Name - fetch_stock_data
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function fetches stock data from Yahoo Finance
#################################################################################################

    def fetch_stock_data(self, symbol: str, exchange: str = "NS", 
                        start_date: str = None,
                        interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        """
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                
            stock_symbol = f"{symbol}.{exchange}"
            stock = yf.Ticker(stock_symbol)
            
            data = stock.history(start=start_date, interval=interval)
            if data.empty:
                raise ValueError(f"No data found for symbol {stock_symbol}")
                
            self.stock_data = data
            self.stock_info = stock.info
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise

#################################################################################################
# Function Name - calculate_rsi
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function calculates the Relative Strength Index (RSI) technical indicator
#################################################################################################

    def calculate_rsi(self, data: pd.Series) -> pd.Series:
        """Calculate RSI technical indicator."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

#################################################################################################
# Function Name - calculate_macd
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function calculates the Moving Average Convergence Divergence (MACD) technical indicator
#################################################################################################

    def calculate_macd(self, data: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal line, and Histogram."""
        exp1 = data.ewm(span=12, adjust=False).mean()
        exp2 = data.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

#################################################################################################
# Function Name - calculate_bollinger_bands
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function calculates the Bollinger Bands technical indicator
#################################################################################################

    def calculate_bollinger_bands(self, data: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle_band = data.rolling(window=window).mean()
        std_dev = data.rolling(window=window).std()
        upper_band = middle_band + (std_dev * 2)
        lower_band = middle_band - (std_dev * 2)
        return upper_band, middle_band, lower_band

#################################################################################################
# Function Name - calculate_technical_indicators
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function calculates various technical indicators
#################################################################################################

    def calculate_technical_indicators(self) -> pd.DataFrame:
        """Calculate various technical indicators."""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data first.")
            
        df = self.stock_data.copy()
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.calculate_macd(df['Close'])
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Volume Indicators
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        
        return df

#################################################################################################
# Function Name - calculate_risk_metrics
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function calculates various risk metrics
#################################################################################################

    def calculate_risk_metrics(self) -> dict:
        """Calculate various risk metrics."""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data first.")
            
        daily_returns = self.stock_data['Close'].pct_change().dropna()
        
        # Sharpe Ratio
        excess_returns = daily_returns - (self.risk_free_rate / 252)
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        
        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Maximum Drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        # Beta (using Nifty 50 as benchmark)
        try:
            nifty = yf.download('^NSEI', start=self.stock_data.index[0])
            nifty_returns = nifty['Close'].pct_change().dropna()
            beta = np.cov(daily_returns, nifty_returns)[0][1] / np.var(nifty_returns)
        except:
            beta = None
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'beta': beta
        }

#################################################################################################
# Function Name - generate_trading_signals
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function generates trading signals based on technical indicators
#################################################################################################

    def generate_trading_signals(self) -> tuple[str, list]:
        """Generate trading signals based on technical indicators."""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data first.")
            
        df = self.calculate_technical_indicators()
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        score = 0
        
        # RSI Signals
        if current['RSI'] < self.rsi_oversold:
            signals.append(f"Oversold (RSI: {current['RSI']:.2f})")
            score += 1
        elif current['RSI'] > self.rsi_overbought:
            signals.append(f"Overbought (RSI: {current['RSI']:.2f})")
            score -= 1
            
        # MACD Signals
        if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            signals.append("MACD Bullish Crossover")
            score += 1
        elif current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            signals.append("MACD Bearish Crossover")
            score -= 1
            
        # Moving Average Signals
        if current['Close'] > current['SMA_200']:
            signals.append("Price above 200 SMA (Bullish)")
            score += 0.5
        else:
            signals.append("Price below 200 SMA (Bearish)")
            score -= 0.5
            
        # Bollinger Bands Signals
        if current['Close'] < current['BB_Lower']:
            signals.append("Price below lower Bollinger Band (Potential Buy)")
            score += 1
        elif current['Close'] > current['BB_Upper']:
            signals.append("Price above upper Bollinger Band (Potential Sell)")
            score -= 1
            
        # Generate recommendation based on score
        if score >= 2:
            recommendation = "Strong Buy"
        elif score > 0:
            recommendation = "Buy"
        elif score == 0:
            recommendation = "Hold"
        elif score > -2:
            recommendation = "Sell"
        else:
            recommendation = "Strong Sell"
            
        return recommendation, signals

#################################################################################################
# Function Name - analyze_top_gainers
# Author - Updated
# Date - 18th Mar 2025
# Description - This function analyzes the top gainers stocks
#################################################################################################

    def analyze_top_gainers(self, limit: int = 5) -> list:
        """
        Fetch and analyze top gainers stocks.
        
        Parameters:
        limit (int): Number of top gainers to analyze
        
        Returns:
        list: List of analysis results for each top gainer
        """
        try:
            # Fetch top gainers
            top_gainers = self.fetch_top_gainers(limit)
            
            # Analyze each stock
            results = []
            
            for gainer in top_gainers:
                symbol = gainer['symbol']
                logger.info(f"Analyzing top gainer: {symbol}")
                
                # Skip non-equity series if the data comes from NSE API
                if 'series' in gainer and gainer['series'] != 'EQ':
                    continue
                    
                try:
                    # Fetch stock data
                    self.fetch_stock_data(symbol)
                    
                    # Generate trading signals
                    recommendation, signals = self.generate_trading_signals()
                    
                    # Calculate risk metrics
                    risk_metrics = self.calculate_risk_metrics()
                    
                    # Add analysis results
                    results.append({
                        'symbol': symbol,
                        'currentPrice': round(float(self.stock_data['Close'][-1]), 2),
                        'percentChange': gainer['percentChange'],
                        'recommendation': recommendation,
                        'signals': signals,
                        'riskMetrics': {
                            'sharpeRatio': round(float(risk_metrics['sharpe_ratio']), 2),
                            'volatility': round(float(risk_metrics['volatility']), 2),
                            'maxDrawdown': round(float(risk_metrics['max_drawdown']), 2),
                            'beta': round(float(risk_metrics['beta']), 2) if risk_metrics['beta'] else None
                        }
                    })
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing top gainers: {str(e)}")
            raise

#################################################################################################
# Function Name - index
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function renders the index.html template
#################################################################################################

@app.route('/')
def index():
    return render_template('index.html')

#################################################################################################
# Function Name - analyze
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function analyzes the stock data and returns the analysis results
#################################################################################################

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        exchange = data.get('exchange', 'NS')
        start_date = data.get('startDate')

        analyzer = IndianStockAnalyzer()
        stock_data = analyzer.fetch_stock_data(symbol, exchange, start_date)
        
        # Existing technical analysis
        tech_data = analyzer.calculate_technical_indicators()
        recommendation, signals = analyzer.generate_trading_signals()
        risk_metrics = analyzer.calculate_risk_metrics()
        
        # New fundamental analysis
        fundamental_analysis = analyzer.perform_fundamental_analysis(symbol)

        # Prepare chart data (existing code remains the same)
        chart_data = []
        for date, row in tech_data.iterrows():
            chart_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': round(float(row['Close']), 2),
                'sma20': round(float(row['SMA_20']), 2) if pd.notnull(row['SMA_20']) else None,
                'sma50': round(float(row['SMA_50']), 2) if pd.notnull(row['SMA_50']) else None,
                'rsi': round(float(row['RSI']), 2) if pd.notnull(row['RSI']) else None,
                'macd': round(float(row['MACD']), 2) if pd.notnull(row['MACD']) else None,
                'signal': round(float(row['MACD_Signal']), 2) if pd.notnull(row['MACD_Signal']) else None
            })

        response = {
            'success': True,
            'data': {
                'recommendation': recommendation,
                'fundamentalAnalysis': fundamental_analysis,
                'currentPrice': round(float(stock_data['Close'][-1]), 2),
                'signals': signals,
                'riskMetrics': {
                    'sharpeRatio': round(float(risk_metrics['sharpe_ratio']), 2),
                    'volatility': round(float(risk_metrics['volatility']), 2),
                    'maxDrawdown': round(float(risk_metrics['max_drawdown']), 2),
                    'beta': round(float(risk_metrics['beta']), 2) if risk_metrics['beta'] else None
                },
                'chartData': chart_data
            }
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

#################################################################################################
# Function Name - top_gainers
# Author - Updated
# Date - 18th Mar 2025
# Description - This function returns the top gainers analysis
#################################################################################################

@app.route('/top-gainers', methods=['GET'])
def top_gainers():
    try:
        limit = request.args.get('limit', default=5, type=int)
        
        analyzer = IndianStockAnalyzer()
        results = analyzer.analyze_top_gainers(limit)
        
        # Fetch fundamental data for each top gainer
        for result in results:
            fundamental_data = analyzer.perform_fundamental_analysis(result['symbol'])
            result['fundamentalAnalysis'] = fundamental_data
        
        return jsonify({
            'success': True,
            'data': results
        })
    except Exception as e:
        logger.error(f"Top gainers analysis error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

#################################################################################################
# Function Name - analyze_top_gainers_cli
# Author - Updated
# Date - 18th Mar 2025
# Description - This function runs top gainers analysis and prints results to console
#################################################################################################

@click.command()
@click.option('--limit', default=5, help='Number of top gainers to analyze')
def analyze_top_gainers_cli(limit):
    """Command line function to analyze top gainers and print results."""
    try:
        print(f"Fetching and analyzing top {limit} gainers from NSE...")
        
        analyzer = IndianStockAnalyzer()
        results = analyzer.analyze_top_gainers(limit)
        
        print("\n=================== TOP GAINERS ANALYSIS ===================\n")
        
        for idx, result in enumerate(results, 1):
            print(f"#{idx}: {result['symbol']} - ₹{result['currentPrice']} ({result['percentChange']}%)")
            print(f"Recommendation: {result['recommendation']}")
            print("Signals:")
            for signal in result['signals']:
                print(f"  - {signal}")
            print("Risk Metrics:")
            print(f"  - Sharpe Ratio: {result['riskMetrics']['sharpeRatio']}")
            print(f"  - Volatility: {result['riskMetrics']['volatility']}%")
            print(f"  - Max Drawdown: {result['riskMetrics']['maxDrawdown']}%")
            if result['riskMetrics']['beta']:
                print(f"  - Beta: {result['riskMetrics']['beta']}")
            print("\n" + "-" * 60 + "\n")
        
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    # Run the Flask app if no arguments are provided
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--analyze-top-gainers':
        # Remove the custom argument before passing to Click
        sys.argv.pop(1)
        analyze_top_gainers_cli()
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)