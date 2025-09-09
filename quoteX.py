from flask import Flask, request, jsonify
import requests
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import logging
from functools import wraps
import threading
import time
import json
from typing import Dict, List, Tuple
import yfinance as yf
import ccxt
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class TradingSignalsGenerator:
    def __init__(self):
        # API Configuration - Replace with your actual API keys
        self.config = {
            'alpha_vantage_key': 'YOUR_ALPHA_VANTAGE_KEY',
            'news_api_key': 'YOUR_NEWS_API_KEY',
            'fmp_api_key': 'YOUR_FMP_KEY',
            'polygon_api_key': 'YOUR_POLYGON_KEY'
        }
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize crypto exchange (Binance)
        self.exchange = ccxt.binance({
            'apiKey': 'YOUR_BINANCE_API_KEY',
            'secret': 'YOUR_BINANCE_SECRET',
            'sandbox': True,  # Set to False for live trading
        })
        
        # Initialize news client
        self.news_client = NewsApiClient(api_key=self.config['news_api_key'])
        
        # Cache for storing recent signals
        self.signals_cache = {}
        
        # Asset mapping for supported assets
        self.asset_mapping = {
            # USD Pairs
            'USD/BDT': {'type': 'forex', 'yf_symbol': 'USDBDT=X', 'av_symbol': 'USD/BDT'},
            'USD/PKR': {'type': 'forex', 'yf_symbol': 'USDPKR=X', 'av_symbol': 'USD/PKR'},
            'USD/INR': {'type': 'forex', 'yf_symbol': 'USDINR=X', 'av_symbol': 'USD/INR'},
            'USD/EGP': {'type': 'forex', 'yf_symbol': 'USDEGP=X', 'av_symbol': 'USD/EGP'},
            'USD/BLR': {'type': 'forex', 'yf_symbol': 'USDBLR=X', 'av_symbol': 'USD/BLR'},
            'USD/MXN': {'type': 'forex', 'yf_symbol': 'USDMXN=X', 'av_symbol': 'USD/MXN'},
            'USD/ARS': {'type': 'forex', 'yf_symbol': 'USDARS=X', 'av_symbol': 'USD/ARS'},
            'USD/DZD': {'type': 'forex', 'yf_symbol': 'USDDZD=X', 'av_symbol': 'USD/DZD'},
            'USD/JPY': {'type': 'forex', 'yf_symbol': 'USDJPY=X', 'av_symbol': 'USD/JPY'},
            'USD/ZAR': {'type': 'forex', 'yf_symbol': 'USDZAR=X', 'av_symbol': 'USD/ZAR'},
            'USD/NGN': {'type': 'forex', 'yf_symbol': 'USDNGN=X', 'av_symbol': 'USD/NGN'},
            'USD/AUD': {'type': 'forex', 'yf_symbol': 'USDAUD=X', 'av_symbol': 'USD/AUD'},
            'USD/CAD': {'type': 'forex', 'yf_symbol': 'USDCAD=X', 'av_symbol': 'USD/CAD'},
            
            # EUR Pairs
            'EUR/GBP': {'type': 'forex', 'yf_symbol': 'EURGBP=X', 'av_symbol': 'EUR/GBP'},
            'EUR/JPY': {'type': 'forex', 'yf_symbol': 'EURJPY=X', 'av_symbol': 'EUR/JPY'},
            'EUR/CAD': {'type': 'forex', 'yf_symbol': 'EURCAD=X', 'av_symbol': 'EUR/CAD'},
            'EUR/AUD': {'type': 'forex', 'yf_symbol': 'EURAUD=X', 'av_symbol': 'EUR/AUD'},
            'EUR/NZD': {'type': 'forex', 'yf_symbol': 'EURNZD=X', 'av_symbol': 'EUR/NZD'},
            'EUR/USD': {'type': 'forex', 'yf_symbol': 'EURUSD=X', 'av_symbol': 'EUR/USD'},
            
            # CAD Pairs
            'CAD/AUD': {'type': 'forex', 'yf_symbol': 'CADAUD=X', 'av_symbol': 'CAD/AUD'},
            'CAD/JPY': {'type': 'forex', 'yf_symbol': 'CADJPY=X', 'av_symbol': 'CAD/JPY'},
            'CAD/NZD': {'type': 'forex', 'yf_symbol': 'CADNZD=X', 'av_symbol': 'CAD/NZD'},
            'CAD/EUR': {'type': 'forex', 'yf_symbol': 'CADEUR=X', 'av_symbol': 'CAD/EUR'},
            
            # Commodities
            'UK Brent': {'type': 'commodity', 'yf_symbol': 'BZ=F', 'av_symbol': 'BRENT'},
            'US Crude': {'type': 'commodity', 'yf_symbol': 'CL=F', 'av_symbol': 'WTI'},
            'Gold': {'type': 'commodity', 'yf_symbol': 'GC=F', 'av_symbol': 'GOLD'},
            'Silver': {'type': 'commodity', 'yf_symbol': 'SI=F', 'av_symbol': 'SILVER'},
            
            # Stocks
            'Microsoft': {'type': 'stock', 'yf_symbol': 'MSFT', 'av_symbol': 'MSFT'},
            'Intel': {'type': 'stock', 'yf_symbol': 'INTC', 'av_symbol': 'INTC'},
            'Facebook': {'type': 'stock', 'yf_symbol': 'META', 'av_symbol': 'META'},
            'McDonald\'s': {'type': 'stock', 'yf_symbol': 'MCD', 'av_symbol': 'MCD'},
            'Boeing': {'type': 'stock', 'yf_symbol': 'BA', 'av_symbol': 'BA'}
        }
        
    def fetch_ohlcv_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from multiple sources for supported assets"""
        if symbol not in self.asset_mapping:
            logger.error(f"Unsupported asset: {symbol}")
            return pd.DataFrame()
            
        asset_info = self.asset_mapping[symbol]
        asset_type = asset_info['type']
        yf_symbol = asset_info['yf_symbol']
        
        # Convert timeframe for Yahoo Finance
        yf_interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '1h', '1d': '1d'
        }
        yf_interval = yf_interval_map.get(timeframe, '1h')
        
        try:
            # Fetch from Yahoo Finance
            if asset_type in ['forex', 'commodity', 'stock']:
                # Determine period based on timeframe and limit
                if timeframe in ['1m', '5m']:
                    period = '5d'
                elif timeframe in ['15m', '30m']:
                    period = '30d'
                else:
                    period = '90d'
                
                data = yf.download(yf_symbol, period=period, interval=yf_interval, progress=False)
                if not data.empty:
                    data = data.reset_index()
                    # Handle different column structures
                    if 'Adj Close' in data.columns:
                        data = data[['Datetime', 'Open', 'High', 'Low', 'Adj Close', 'Volume']]
                        data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    else:
                        data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    
                    # Handle missing volume for forex
                    if asset_type == 'forex' and data['volume'].isna().all():
                        data['volume'] = 1000000  # Default volume for forex
                        
                    return data.tail(limit).reset_index(drop=True)
                
        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance for {symbol}: {e}")
            
        # Fallback to Alpha Vantage for forex
        if asset_type == 'forex':
            try:
                return self.fetch_forex_alphavantage(asset_info['av_symbol'], timeframe, limit)
            except Exception as e:
                logger.error(f"Error fetching forex data from Alpha Vantage: {e}")
                
        return pd.DataFrame()
    
    def fetch_forex_alphavantage(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch forex data from Alpha Vantage"""
        # Map timeframes to Alpha Vantage intervals
        interval_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '60min', '4h': '60min', '1d': 'daily'
        }
        
        interval = interval_map.get(timeframe, '60min')
        base_url = 'https://www.alphavantage.co/query'
        
        if interval == 'daily':
            function = 'FX_DAILY'
            params = {
                'function': function,
                'from_symbol': symbol.split('/')[0],
                'to_symbol': symbol.split('/')[1],
                'apikey': self.config['alpha_vantage_key']
            }
        else:
            function = 'FX_INTRADAY'
            params = {
                'function': function,
                'from_symbol': symbol.split('/')[0],
                'to_symbol': symbol.split('/')[1],
                'interval': interval,
                'apikey': self.config['alpha_vantage_key']
            }
        
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if 'Time Series' in str(data):
            # Find the time series key
            time_series_key = [key for key in data.keys() if 'Time Series' in key][0]
            time_series = data[time_series_key]
            
            df_data = []
            for timestamp, values in time_series.items():
                df_data.append({
                    'timestamp': pd.to_datetime(timestamp),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': 1000000  # Default volume for forex
                })
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('timestamp').tail(limit).reset_index(drop=True)
            return df
        
        return pd.DataFrame()

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if df.empty or len(df) < 20:
            return df
            
        # Price data
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # Trend Indicators
        df['sma_20'] = talib.SMA(close, timeperiod=20)
        df['sma_50'] = talib.SMA(close, timeperiod=50)
        df['ema_12'] = talib.EMA(close, timeperiod=12)
        df['ema_26'] = talib.EMA(close, timeperiod=26)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)
        
        # RSI
        df['rsi'] = talib.RSI(close, timeperiod=14)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close, timeperiod=20)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close)
        
        # Williams %R
        df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # ADX (Average Directional Index)
        df['adx'] = talib.ADX(high, low, close, timeperiod=14)
        
        # CCI (Commodity Channel Index)
        df['cci'] = talib.CCI(high, low, close, timeperiod=14)
        
        # Volume indicators
        df['obv'] = talib.OBV(close, volume)
        df['ad'] = talib.AD(high, low, close, volume)
        
        # Volatility indicators
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        
        # Support and Resistance
        df['pivot'] = (high + low + close) / 3
        df['resistance_1'] = 2 * df['pivot'] - low
        df['support_1'] = 2 * df['pivot'] - high
        
        return df

    def fetch_market_sentiment(self, symbol: str) -> Dict:
        """Fetch market sentiment from news and social media"""
        sentiment_data = {
            'news_sentiment': 0,
            'social_sentiment': 0,
            'overall_sentiment': 0,
            'sentiment_strength': 'neutral'
        }
        
        try:
            # Fetch news articles
            query = symbol.replace('/', ' ') if '/' in symbol else symbol
            news = self.news_client.get_everything(
                q=query,
                language='en',
                sort_by='publishedAt',
                from_param=(datetime.now() - timedelta(days=1)).isoformat(),
                page_size=20
            )
            
            if news['articles']:
                sentiments = []
                for article in news['articles']:
                    text = f"{article['title']} {article.get('description', '')}"
                    sentiment = self.sentiment_analyzer.polarity_scores(text)
                    sentiments.append(sentiment['compound'])
                
                sentiment_data['news_sentiment'] = np.mean(sentiments)
                
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            
        # Calculate overall sentiment
        sentiment_data['overall_sentiment'] = sentiment_data['news_sentiment']
        
        if sentiment_data['overall_sentiment'] > 0.1:
            sentiment_data['sentiment_strength'] = 'bullish'
        elif sentiment_data['overall_sentiment'] < -0.1:
            sentiment_data['sentiment_strength'] = 'bearish'
        else:
            sentiment_data['sentiment_strength'] = 'neutral'
            
        return sentiment_data

    def generate_signal_score(self, df: pd.DataFrame, asset_type: str) -> Dict:
        """Generate comprehensive signal score based on multiple factors and asset type"""
        if df.empty or len(df) < 20:
            return {'signal': 'HOLD', 'confidence': 0, 'score': 0}
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Initialize scores
        trend_score = 0
        momentum_score = 0
        volume_score = 0
        volatility_score = 0
        
        # Asset-specific weight adjustments
        weights = self.get_asset_weights(asset_type)
        
        # Trend Analysis
        if latest['close'] > latest['sma_20'] and latest['sma_20'] > latest['sma_50']:
            trend_score += 2
        elif latest['close'] < latest['sma_20'] and latest['sma_20'] < latest['sma_50']:
            trend_score -= 2
            
        if latest['ema_12'] > latest['ema_26']:
            trend_score += 1
        else:
            trend_score -= 1
            
        # MACD trend
        if latest['macd'] > latest['macd_signal'] and latest['macd_hist'] > prev['macd_hist']:
            trend_score += 1
        elif latest['macd'] < latest['macd_signal'] and latest['macd_hist'] < prev['macd_hist']:
            trend_score -= 1
            
        # Momentum Analysis - Enhanced for different asset types
        rsi = latest['rsi']
        
        # Adjust RSI thresholds based on asset type
        if asset_type == 'forex':
            oversold, overbought = 25, 75
        elif asset_type == 'commodity':
            oversold, overbought = 30, 70
        else:  # stocks
            oversold, overbought = 30, 70
            
        if oversold < rsi < overbought:  # Normal range
            if rsi > 50:
                momentum_score += 1
            else:
                momentum_score -= 1
        elif rsi <= oversold:  # Oversold - potential buy
            momentum_score += 2
        elif rsi >= overbought:  # Overbought - potential sell
            momentum_score -= 2
            
        # Stochastic
        if latest['stoch_k'] < 20 and latest['stoch_k'] > prev['stoch_k']:
            momentum_score += 1
        elif latest['stoch_k'] > 80 and latest['stoch_k'] < prev['stoch_k']:
            momentum_score -= 1
            
        # Williams %R
        williams = latest['williams_r']
        if williams < -80:  # Oversold
            momentum_score += 1
        elif williams > -20:  # Overbought
            momentum_score -= 1
            
        # Volume Analysis - More important for stocks
        if asset_type != 'forex':  # Volume is less meaningful for forex
            volume_ma = df['volume'].rolling(window=20).mean().iloc[-1]
            current_volume = latest['volume']
            
            if current_volume > volume_ma * 1.5:  # High volume
                volume_score += 2
            elif current_volume > volume_ma * 1.2:
                volume_score += 1
            elif current_volume < volume_ma * 0.5:  # Low volume
                volume_score -= 1
                
            # OBV trend
            if latest['obv'] > prev['obv']:
                volume_score += 1
            else:
                volume_score -= 1
        else:
            # For forex, use price action volume proxy
            price_range = latest['high'] - latest['low']
            avg_range = df['high'].subtract(df['low']).rolling(window=20).mean().iloc[-1]
            if price_range > avg_range * 1.2:
                volume_score += 1
            
        # Volatility Analysis - Asset specific
        atr_ma = df['atr'].rolling(window=14).mean().iloc[-1]
        current_atr = latest['atr']
        
        if asset_type == 'commodity':
            # Commodities can be more volatile
            if current_atr > atr_ma * 1.3:
                volatility_score += 1
            elif current_atr < atr_ma * 0.7:
                volatility_score -= 1
        else:
            if current_atr > atr_ma * 1.2:
                volatility_score += 1
            elif current_atr < atr_ma * 0.8:
                volatility_score -= 1
            
        # Bollinger Bands position
        bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
        if bb_position > 0.8:  # Near upper band
            volatility_score -= 1
        elif bb_position < 0.2:  # Near lower band
            volatility_score += 1
            
        # Support/Resistance levels
        support_resistance_score = 0
        if latest['close'] > latest['resistance_1']:
            support_resistance_score += 1
        elif latest['close'] < latest['support_1']:
            support_resistance_score -= 1
            
        # Calculate weighted final score using asset-specific weights
        final_score = (
            trend_score * weights['trend'] +
            momentum_score * weights['momentum'] +
            volume_score * weights['volume'] +
            volatility_score * weights['volatility'] +
            support_resistance_score * weights['support_resistance']
        )
        
        # Generate signal with asset-specific thresholds
        thresholds = self.get_signal_thresholds(asset_type)
        
        if final_score >= thresholds['call']:
            signal = 'CALL'
            confidence = min(abs(final_score) * 20, 100)
        elif final_score <= thresholds['put']:
            signal = 'PUT'
            confidence = min(abs(final_score) * 20, 100)
        else:
            signal = 'HOLD'
            confidence = 30
            
        return {
            'signal': signal,
            'confidence': round(confidence, 2),
            'score': round(final_score, 2),
            'components': {
                'trend': round(trend_score, 2),
                'momentum': round(momentum_score, 2),
                'volume': round(volume_score, 2),
                'volatility': round(volatility_score, 2),
                'support_resistance': round(support_resistance_score, 2)
            }
        }
    
    def get_asset_weights(self, asset_type: str) -> Dict:
        """Get asset-specific weights for signal components"""
        if asset_type == 'forex':
            return {
                'trend': 0.35,
                'momentum': 0.30,
                'volume': 0.10,  # Less important for forex
                'volatility': 0.15,
                'support_resistance': 0.10
            }
        elif asset_type == 'commodity':
            return {
                'trend': 0.30,
                'momentum': 0.25,
                'volume': 0.20,
                'volatility': 0.15,
                'support_resistance': 0.10
            }
        else:  # stocks
            return {
                'trend': 0.30,
                'momentum': 0.25,
                'volume': 0.25,  # Very important for stocks
                'volatility': 0.10,
                'support_resistance': 0.10
            }
    
    def get_signal_thresholds(self, asset_type: str) -> Dict:
        """Get asset-specific signal thresholds"""
        if asset_type == 'forex':
            return {'call': 1.2, 'put': -1.2}  # More conservative for forex
        elif asset_type == 'commodity':
            return {'call': 1.0, 'put': -1.0}  # Standard thresholds
        else:  # stocks
            return {'call': 1.5, 'put': -1.5}  # More aggressive for stocks

    def get_risk_management(self, df: pd.DataFrame, signal: Dict) -> Dict:
        """Calculate risk management parameters"""
        if df.empty:
            return {}
            
        latest = df.iloc[-1]
        atr = latest['atr']
        current_price = latest['close']
        
        # Stop loss and take profit based on ATR
        if signal['signal'] == 'CALL':
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 3)
        elif signal['signal'] == 'PUT':
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 3)
        else:
            stop_loss = current_price
            take_profit = current_price
            
        risk_reward_ratio = abs(take_profit - current_price) / abs(current_price - stop_loss) if abs(current_price - stop_loss) > 0 else 0
        
        return {
            'stop_loss': round(stop_loss, 4),
            'take_profit': round(take_profit, 4),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'position_size': round(100 / signal['confidence'] if signal['confidence'] > 0 else 1, 2)
        }

    def generate_comprehensive_signal(self, symbol: str, timeframe: str = '1h') -> Dict:
        """Generate comprehensive trading signal"""
        try:
            # Fetch market data
            df = self.fetch_ohlcv_data(symbol, timeframe)
            if df.empty:
                return {'error': 'Unable to fetch market data'}
                
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Generate signal score
            signal_data = self.generate_signal_score(df)
            
            # Fetch market sentiment
            sentiment_data = self.fetch_market_sentiment(symbol)
            
            # Adjust signal based on sentiment
            sentiment_adjustment = sentiment_data['overall_sentiment'] * 0.5
            adjusted_score = signal_data['score'] + sentiment_adjustment
            
            # Recalculate signal with sentiment
            if adjusted_score >= 1.0:
                final_signal = 'CALL'
                confidence = min((abs(adjusted_score) * 15) + signal_data['confidence'], 100)
            elif adjusted_score <= -1.0:
                final_signal = 'PUT'
                confidence = min((abs(adjusted_score) * 15) + signal_data['confidence'], 100)
            else:
                final_signal = 'HOLD'
                confidence = max(signal_data['confidence'] - 10, 20)
                
            final_signal_data = {
                'signal': final_signal,
                'confidence': round(confidence, 2),
                'score': round(adjusted_score, 2)
            }
            
            # Get risk management
            risk_mgmt = self.get_risk_management(df, final_signal_data)
            
            # Get latest price data
            latest = df.iloc[-1]
            
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'signal': final_signal_data['signal'],
                'confidence': final_signal_data['confidence'],
                'score': final_signal_data['score'],
                'current_price': round(latest['close'], 4),
                'technical_analysis': {
                    'rsi': round(latest['rsi'], 2),
                    'macd': round(latest['macd'], 4),
                    'macd_signal': round(latest['macd_signal'], 4),
                    'bb_position': round((latest['close'] - latest['bb_lower']) / 
                                      (latest['bb_upper'] - latest['bb_lower']), 2),
                    'adx': round(latest['adx'], 2),
                    'atr': round(latest['atr'], 4)
                },
                'sentiment_analysis': sentiment_data,
                'risk_management': risk_mgmt,
                'signal_components': signal_data['components']
            }
            
            # Cache the signal
            self.signals_cache[f"{symbol}_{timeframe}"] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {'error': str(e)}

# Initialize the signals generator
signals_generator = TradingSignalsGenerator()

def require_auth(f):
    """Simple authentication decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != 'your-secret-api-key':  # Change this
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return jsonify({
        'name': 'Binary Trading Signals API',
        'version': '1.0.0',
        'endpoints': {
            '/signal/<symbol>': 'GET - Generate trading signal for symbol',
            '/signals/batch': 'POST - Generate signals for multiple symbols',
            '/signals/stream/<symbol>': 'GET - Stream live signals',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/signal/<symbol>')
def get_signal(symbol):
    """Get trading signal for a specific symbol"""
    timeframe = request.args.get('timeframe', '1h')
    
    # Validate timeframe
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    if timeframe not in valid_timeframes:
        return jsonify({'error': 'Invalid timeframe'}), 400
        
    signal = signals_generator.generate_comprehensive_signal(symbol, timeframe)
    return jsonify(signal)

@app.route('/signals/batch', methods=['POST'])
def get_batch_signals():
    """Get trading signals for multiple symbols"""
    data = request.get_json()
    if not data or 'symbols' not in data:
        return jsonify({'error': 'Symbols list required'}), 400
        
    symbols = data['symbols']
    timeframe = data.get('timeframe', '1h')
    
    if len(symbols) > 10:  # Limit batch size
        return jsonify({'error': 'Maximum 10 symbols per batch'}), 400
        
    results = {}
    for symbol in symbols:
        results[symbol] = signals_generator.generate_comprehensive_signal(symbol, timeframe)
        
    return jsonify({'results': results, 'total': len(results)})

@app.route('/signals/stream/<symbol>')
def stream_signals(symbol):
    """Stream live signals for a symbol"""
    def generate_stream():
        while True:
            timeframe = request.args.get('timeframe', '1h')
            signal = signals_generator.generate_comprehensive_signal(symbol, timeframe)
            yield f"data: {json.dumps(signal)}\n\n"
            time.sleep(30)  # Update every 30 seconds
            
    return app.response_class(
        generate_stream(),
        mimetype='text/plain'
    )

@app.route('/signals/cache')
def get_cached_signals():
    """Get cached signals"""
    return jsonify(signals_generator.signals_cache)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Binary Trading Signals API...")
    print("Make sure to install required packages:")
    print("pip install flask pandas numpy talib yfinance ccxt newsapi-python vaderSentiment requests")
    print("\nDon't forget to add your API keys in the TradingSignalsGenerator.__init__() method")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)