import asyncio
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Union
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import joblib
import argparse
import sys
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
from collections import Counter
import talib
import warnings
from tabulate import tabulate
import colorama
from colorama import Fore, Style

# Initialize colorama for colored console output
colorama.init()

# Suppress warnings
warnings.filterwarnings("ignore")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_system.log")
    ]
)
logger = logging.getLogger()

def setup_logging(level):
    logger.setLevel(level)

# Configuration
# Configuration
class Config:
    INITIAL_CAPITAL = 100000  # Rs.100,000 starting capital
    BACKTEST_DAYS = 365
    RISK_FACTOR = 0.004  # 0.5% risk per trade (reduced from 1%)
    MAX_POSITION_FRACTION = 0.12  # 10% max per trade (increased from 5%)
    ATR_MULTIPLIER = 1.7  # Adjusted from 2.0 for tighter stops
    CONFIDENCE_THRESHOLD = 0.68  # Increased confidence threshold for better trade quality
    FEATURE_WINDOW = 250
    MODEL_DIR = "models"
    RESULTS_DIR = "results"
    DATA_DIR = "historical_data"
    FORCE_REFRESH_DATA = True  # Always download fresh data
    UPSTOX_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI3NEEyUjIiLCJqdGkiOiI2N2Q0ZWYzNmRjYTc3ZTUzNGNjOWYwNDQiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaWF0IjoxNzQyMDA4MTE4LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NDIwNzYwMDB9.14-uaIt_9tR2zc2WUU25p0TJoTcZ4hRuH8uiCfudPLY"  # Replace with your token
    
    # Optimized model parameters
    RF_PARAMS = {
        'n_estimators': 400,
        'max_depth': 9,  # Increased from 6 to prevent underfitting
        'min_samples_split': 10,  # Reduced for more splits
        'min_samples_leaf': 6,  # Adjusted to improve generalization
        'class_weight': 'balanced_subsample',
        'random_state': 42,
        'n_jobs': -1,
        'max_features': 'sqrt'
    }
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'max_depth': 7,  # Increased from 6
        'learning_rate': 0.005,  # Slightly increased learning rate for better adaptability
        'subsample': 0.75,  # Increased from 0.8
        'colsample_bytree': 0.7,
        'min_child_weight': 6,  # Reduced to allow more splits
        'gamma': 0.15,  # Lowered for more flexibility
        'seed': 42,
        'eval_metric': 'auc',
        'scale_pos_weight': 1.0,
        'reg_alpha': 0.1,  # L1 regularization to handle sparse features
        'reg_lambda': 1.0   # L2 regularization
    }
    LGBM_PARAMS = {
        'objective': 'binary',
        'max_depth': 8,  # Increased from 6
        'learning_rate': 0.008,
        'subsample': 0.75,
        'colsample_bytree': 0.7,
        'min_data_in_leaf': 30,  # Lowered to capture more patterns
        'seed': 42,
        'metric': 'auc',
        'verbose': -1,
        'num_leaves': 64,  # Optimized for Indian market patterns
        'bagging_freq': 5,  # Frequent bagging helps with market regime changes
        'feature_fraction': 0.8
    }
    GB_PARAMS = {
        'n_estimators': 250,  # Increased from 200 for better prediction accuracy
        'max_depth': 6,  # Increased slightly
        'learning_rate': 0.015,  # Improved learning rate
        'subsample': 0.85,
        'min_samples_split': 15,  # Adjusted for better data split
        'min_samples_leaf': 8,  # Reduced to allow finer leaf nodes
        'random_state': 42
    }
    
    # Trading parameters
    MIN_TRADES_REQUIRED = 20  # Minimum number of trades for valid backtest
    MAX_DRAWDOWN_THRESHOLD = 0.25  # Maximum allowed drawdown (15%)
    PROFIT_TAKE_RATIO_1 = 1.8 # Risk:reward for first target
    PROFIT_TAKE_RATIO_2 = 4.5  # Risk:reward for second target
    TRAILING_STOP_ACTIVATION = 1.2  # Adjusted ATR multiplier to activate trailing stop earlier
    MAX_TRADES_PER_DAY = 8  # Increased from 3 to maximize opportunities

    # Feature importance
    MIN_FEATURE_IMPORTANCE = 0.015  # Minimum feature importance to keep

# Instrument Registry (example, expand as needed)
class InstrumentRegistry:
    INSTRUMENTS = {
        "NSE_EQ|INE002A01018": {"symbol": "RELIANCE", "lot_size": 250, "tick_size": 0.05},
        "NSE_EQ|INE040A01034": {"symbol": "HDFCBANK", "lot_size": 100, "tick_size": 0.05},
        "NSE_EQ|INE009A01021": {"symbol": "INFY", "lot_size": 300, "tick_size": 0.05},
        "NSE_EQ|INE062A01020": {"symbol": "TCS", "lot_size": 150, "tick_size": 0.05},
        # Add more instruments as needed
    }
    
    @staticmethod
    def get_instrument_key(symbol):
        for key, details in InstrumentRegistry.INSTRUMENTS.items():
            if details["symbol"] == symbol:
                return key
        return None

# Data classes
@dataclass
class OHLCV:
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"DataFrame missing required columns: {required_cols}")
            return None
        return cls(
            timestamp=df['timestamp'].values,
            open=df['open'].values,
            high=df['high'].values,
            low=df['low'].values,
            close=df['close'].values,
            volume=df['volume'].values
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        })

@dataclass
class Position:
    symbol: str
    entry_time: Union[int, float, pd.Timestamp]
    entry_price: float
    size: int
    signal: int  # 0 for long, 1 for short
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    exit_time: Optional[Union[int, float, pd.Timestamp]] = None
    exit_price: Optional[float] = None
    profit: Optional[float] = None
    exit_reason: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary with serializable values."""
        result = asdict(self)
        # Convert timestamps to string format
        if isinstance(result['entry_time'], pd.Timestamp):
            result['entry_time'] = result['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
        if result['exit_time'] is not None and isinstance(result['exit_time'], pd.Timestamp):
            result['exit_time'] = result['exit_time'].strftime('%Y-%m-%d %H:%M:%S')
        return result

# Data Manager
class DataManager:
    @staticmethod
    async def get_historical_data(symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data for the specified symbol and date range."""
        logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
        instrument_key = InstrumentRegistry.get_instrument_key(symbol)
        if not instrument_key:
            logger.error(f"No instrument key for {symbol}")
            return None

        os.makedirs(Config.DATA_DIR, exist_ok=True)
        file_path = os.path.join(Config.DATA_DIR, f"{symbol}_historical.csv")
        
        # Check if we can use cached data
        if os.path.exists(file_path) and not Config.FORCE_REFRESH_DATA:
            try:
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                min_date = df['timestamp'].min()
                max_date = df['timestamp'].max()
                
                if min_date <= pd.Timestamp(start_date) and max_date >= pd.Timestamp(end_date):
                    logger.info(f"Using cached data for {symbol} ({len(df)} rows)")
                    mask = (df['timestamp'] >= pd.Timestamp(start_date)) & (df['timestamp'] <= pd.Timestamp(end_date))
                    return df[mask].reset_index(drop=True)
                logger.info(f"Cached data for {symbol} doesn't cover required date range")
            except Exception as e:
                logger.warning(f"Failed to load cached data for {symbol}: {e}")
        
        # Fetch from API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/1minute/{to_date}/{from_date}"
        headers = {'Authorization': f'Bearer {Config.UPSTOX_TOKEN}'}
        
        try:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            
            data = resp.json()
            if 'data' not in data or 'candles' not in data['data']:
                logger.error(f"Invalid API response for {symbol}: {data}")
                return None
            
            df = pd.DataFrame(data['data']['candles'], columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df['timestamp'] = pd.to_datetime(df['ts'])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].sort_values('timestamp', ascending=True).reset_index(drop=True)
            
            # Convert price columns to float
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].astype(np.float64)
            df['volume'] = df['volume'].astype(int)
            
            df.to_csv(file_path, index=False)
            logger.info(f"Fetched and saved historical data for {symbol} with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            
            # Fallback to sample data for testing if API fails
            if not os.path.exists(file_path):
                logger.warning(f"Generating sample data for {symbol} (for testing only)")
                # Generate sample data for testing
                days = (end_date - start_date).days + 1
                periods = days * 390  # ~390 minutes in a trading day
                base_price = 1000.0
                
                timestamps = [start_date + timedelta(minutes=i) for i in range(periods)]
                prices = [base_price * (1 + 0.1 * np.sin(i/1000) + 0.01 * np.random.randn()) for i in range(periods)]
                
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'open': prices,
                    'high': [p * (1 + 0.005 * np.random.rand()) for p in prices],
                    'low': [p * (1 - 0.005 * np.random.rand()) for p in prices],
                    'close': [p * (1 + 0.002 * np.random.randn()) for p in prices],
                    'volume': [int(1000 * (1 + np.random.rand())) for _ in range(periods)]
                })
                df.to_csv(file_path, index=False)
                return df
            else:
                # Try to use existing file as fallback
                try:
                    df = pd.read_csv(file_path)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    logger.info(f"Using existing file as fallback for {symbol}")
                    return df
                except Exception as e2:
                    logger.error(f"Fallback also failed for {symbol}: {e2}")
                    return None

    @staticmethod
    async def get_training_data(symbol: str) -> Optional[pd.DataFrame]:
        """Get data for training ML models."""
        start_date = datetime.now() - timedelta(days=Config.BACKTEST_DAYS)
        end_date = datetime.now()
        return await DataManager.get_historical_data(symbol, start_date, end_date)

# Technical Analysis Engine
class TechnicalAnalysisEngine:
    @staticmethod
    def calculate_features(ohlcv: OHLCV) -> pd.DataFrame:
        """Calculate technical indicators and features for model training and prediction."""
        logger.debug(f"Calculating features for OHLCV with {len(ohlcv.timestamp)} bars")
        df = ohlcv.to_dataframe()
        if len(df) < Config.FEATURE_WINDOW + 1:
            logger.warning(f"Insufficient data: {len(df)} < {Config.FEATURE_WINDOW + 1}")
            return pd.DataFrame()
        
        # Ensure data types
        close = df['close'].astype(np.float64).values
        open_prices = df['open'].astype(np.float64).values
        high = df['high'].astype(np.float64).values
        low = df['low'].astype(np.float64).values
        volume = df['volume'].astype(np.float64).values
        
        # === TREND INDICATORS ===
        # Moving averages
        df['sma10'] = talib.SMA(close, timeperiod=10)
        df['sma20'] = talib.SMA(close, timeperiod=20)
        df['sma50'] = talib.SMA(close, timeperiod=50)
        df['sma100'] = talib.SMA(close, timeperiod=100)
        df['sma200'] = talib.SMA(close, timeperiod=200)
        df['ema9'] = talib.EMA(close, timeperiod=9)
        df['ema21'] = talib.EMA(close, timeperiod=21)
        df['ema55'] = talib.EMA(close, timeperiod=55)
        
        # Trend direction and strength
        df['adx'] = talib.ADX(high, low, close, timeperiod=14)
        df['di_plus'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['di_minus'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        df['trend_strength'] = df['adx'] / 100.0  # Normalized
        
        # Simple trend direction indicators
        df['close_over_sma50'] = (df['close'] > df['sma50']).astype(int)
        df['close_over_sma200'] = (df['close'] > df['sma200']).astype(int)
        df['sma50_over_sma200'] = (df['sma50'] > df['sma200']).astype(int)
        df['ema_trend'] = (df['ema9'] > df['ema21']).astype(int)
        
        # Trend classification
        df['market_regime'] = np.where(
            (df['sma20'] > df['sma50']) & (df['sma50'] > df['sma200']), 
            2,  # Strong uptrend
            np.where(
                (df['sma20'] < df['sma50']) & (df['sma50'] < df['sma200']),
                -2,  # Strong downtrend
                np.where(
                    (df['sma20'] > df['sma50']) & (df['sma50'] < df['sma200']),
                    1,  # Weak uptrend/recovery
                    -1  # Weak downtrend/correction
                )
            )
        )
        
        # === MOMENTUM INDICATORS ===
        # RSI and stochastic
        df['rsi'] = talib.RSI(close, timeperiod=14)
        df['rsi_sma'] = talib.SMA(df['rsi'].values.astype(np.float64), timeperiod=3)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close, 
                                                   fastk_period=14, 
                                                   slowk_period=3, 
                                                   slowd_period=3)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd_normalized'] = df['macd'] / df['close']  # Normalized MACD
        
        # Price momentum
        df['roc_5'] = talib.ROC(close, timeperiod=5)
        df['roc_10'] = talib.ROC(close, timeperiod=10)
        df['roc_20'] = talib.ROC(close, timeperiod=20)
        df['willr'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # Momentum classification
        df['rsi_regime'] = np.where(df['rsi'] > 70, -1, np.where(df['rsi'] < 30, 1, 0))
        df['macd_regime'] = np.sign(df['macd_hist'])
        
        # === VOLATILITY INDICATORS ===
        # ATR and normalized variants
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        df['atr_normalized'] = df['atr'] / df['close']
        df['natr'] = talib.NATR(high, low, close, timeperiod=14)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Historical volatility
        df['close_pct_change'] = df['close'].pct_change()
        df['volatility_10'] = df['close_pct_change'].rolling(window=10).std() * np.sqrt(252)
        df['volatility_20'] = df['close_pct_change'].rolling(window=20).std() * np.sqrt(252)
        df['volatility_ratio'] = df['volatility_10'] / df['volatility_20'].replace(0, np.nan)
        
        # === VOLUME INDICATORS ===
        # Volume moving averages
        df['volume_sma10'] = talib.SMA(volume, timeperiod=10)
        df['volume_sma20'] = talib.SMA(volume, timeperiod=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma20']
        
        # OBV and Chaikin
        df['obv'] = talib.OBV(close, volume)
        df['obv_sma'] = talib.SMA(df['obv'].values.astype(np.float64), timeperiod=20)
        df['obv_trend'] = (df['obv'] > df['obv_sma']).astype(int)
        df['ad'] = talib.AD(high, low, close, volume)
        df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        
        # Volume classification
        df['volume_trend'] = np.where(df['volume'] > df['volume_sma20'], 1, -1)
        
        # === PRICE PATTERNS ===
        # Candle patterns (selected important ones)
        df['doji'] = talib.CDLDOJI(open_prices, high, low, close)
        df['engulfing'] = talib.CDLENGULFING(open_prices, high, low, close)
        df['hammer'] = talib.CDLHAMMER(open_prices, high, low, close)
        df['shooting_star'] = talib.CDLSHOOTINGSTAR(open_prices, high, low, close)
        df['morning_star'] = talib.CDLMORNINGSTAR(open_prices, high, low, close)
        df['evening_star'] = talib.CDLEVENINGSTAR(open_prices, high, low, close)
        
        # === FEATURE ENGINEERING ===
        # Trend-momentum agreement
        df['trend_momentum_agree'] = np.where(
            (df['market_regime'] > 0) & (df['macd'] > 0) | 
            (df['market_regime'] < 0) & (df['macd'] < 0), 
            1, 0
        )
        
        # RSI divergence approximation - Fixed to use pandas Series methods
        df['price_higher_high'] = (df['close'] > df['close'].shift(5)).astype(int)
        df['rsi_higher_high'] = (df['rsi'] > df['rsi'].shift(5)).astype(int)
        df['rsi_divergence'] = np.where(
            (df['price_higher_high'] == 1) & (df['rsi_higher_high'] == 0),
            -1,  # Bearish divergence
            np.where(
                (df['price_higher_high'] == 0) & (df['rsi_higher_high'] == 1),
                1,  # Bullish divergence
                0
            )
        )
        
        # Support/resistance proximity
        df['pivot_high'] = df['high'].rolling(window=20, center=True).max()
        df['pivot_low'] = df['low'].rolling(window=20, center=True).min()
        df['dist_to_resistance'] = (df['pivot_high'] - df['close']) / df['close']
        df['dist_to_support'] = (df['close'] - df['pivot_low']) / df['close']
        
        # Volatility regime
        df['volatility_regime'] = np.where(
            df['bb_width'] > df['bb_width'].rolling(window=100).mean(),
            1,  # High volatility
            0   # Low volatility
        )
        
        # Combined signals
        df['buy_signal_strength'] = (
            (df['rsi'] < 40).astype(int) +
            (df['macd'] > df['macd_signal']).astype(int) +
            (df['close'] > df['sma50']).astype(int) +
            (df['obv_trend'] == 1).astype(int) +
            (df['volume_ratio'] > 1.5).astype(int)
        ) / 5.0  # Normalize to 0-1 range
        
        df['sell_signal_strength'] = (
            (df['rsi'] > 60).astype(int) +
            (df['macd'] < df['macd_signal']).astype(int) +
            (df['close'] < df['sma50']).astype(int) +
            (df['obv_trend'] == 0).astype(int) +
            (df['volume_ratio'] > 1.5).astype(int)
        ) / 5.0  # Normalize to 0-1 range
        
        # Recent returns
        for period in [1, 3, 5, 10, 20]:
            df[f'return_{period}d'] = df['close'].pct_change(periods=period)
        
        # === SHIFT FEATURES (to avoid lookahead bias) ===
        feature_cols = [col for col in df.columns if col not in [
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'bb_upper', 'bb_middle', 'bb_lower'
        ]]
        
        for col in feature_cols:
            df[col] = df[col].shift(1)
        
        # Drop NaN values
        df = df.dropna()
        logger.debug(f"Features calculated: {len(feature_cols)} features")
        return df


    def add_target_variables(self, df: pd.DataFrame, forward_periods: int = 20) -> pd.DataFrame:
        """Add target variables for ML training based on forward returns."""
        df = df.copy()
        
        # Calculate forward returns for different periods
        for period in [1, 5, 10, 20]:
            df[f'future_return_{period}'] = df['close'].pct_change(periods=period).shift(-period)
        
        # Calculate ATR-based targets (more adaptive to volatility)
        atr = df['atr'].rolling(window=14).mean()
        close = df['close']
        
        # Primary target: significant move in next 20 periods
        df['target_down'] = (df['future_return_20'] < -0.5 * (atr / close)).astype(int)
        
        # Alternative targets for model evaluation
        df['target_up'] = (df['future_return_20'] > 0.5 * (atr / close)).astype(int)
        df['target_flat'] = ((df['future_return_20'].abs() <= 0.5 * (atr / close))).astype(int)
        
        # Multi-class target
        df['target_multi'] = np.where(
            df['future_return_20'] > 1.0 * (atr / close), 
            2,  # Strong up
            np.where(
                df['future_return_20'] > 0.5 * (atr / close),
                1,  # Moderate up
                np.where(
                    df['future_return_20'] < -1.0 * (atr / close),
                    -2,  # Strong down
                    np.where(
                        df['future_return_20'] < -0.5 * (atr / close),
                        -1,  # Moderate down
                        0    # Flat
                    )
                )
            )
        )
        
        # Log target distribution
        target_dist = df['target_down'].value_counts(normalize=True)
        logger.info(f"Target distribution: Down={target_dist.get(1, 0):.2%}, Up/Flat={target_dist.get(0, 0):.2%}")
        
        return df


# Feature Selection
class FeatureSelector:
    @staticmethod
    def select_features(X_train, y_train, feature_names, model_type='rf'):
        """Select important features using model-based feature importance."""
        if model_type == 'rf':
            model = RandomForestClassifier(**Config.RF_PARAMS)
        elif model_type == 'xgb':
            model = xgb.XGBClassifier(objective='binary:logistic', **Config.XGB_PARAMS)
        else:
            model = GradientBoostingClassifier(**Config.GB_PARAMS)
            
        model.fit(X_train, y_train)
        
        if model_type == 'xgb':
            importance = model.feature_importances_
        else:
            importance = model.feature_importances_
            
        # Create DataFrame with feature importances
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Filter features based on importance threshold
        selected_features = feature_importance[
            feature_importance['importance'] > Config.MIN_FEATURE_IMPORTANCE
        ]['feature'].tolist()
        
        logger.info(f"Selected {len(selected_features)}/{len(feature_names)} features")
        return selected_features, feature_importance

# Machine Learning Engine
class MachineLearningEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        self.model_metrics = {}
        self.feature_importance = {}
        self.selected_features = {}

    async def train_model_for_symbol(self, symbol: str) -> bool:
        """Train ML models for a symbol using ensemble approach."""
        logger.info(f"Training model for {symbol}")
        
        # Get data and check if sufficient
        raw_data = await DataManager.get_training_data(symbol)
        if raw_data is None or len(raw_data) < Config.FEATURE_WINDOW + 100:
            logger.warning(f"Insufficient data for {symbol}: {len(raw_data) if raw_data is not None else 0} rows")
            return False
        
        # Split data for training, validation, and testing
        total_rows = len(raw_data)
        train_size = int(total_rows * 0.7)
        val_size = int(total_rows * 0.15)
        
        train_raw = raw_data.iloc[:train_size].copy()
        val_raw = raw_data.iloc[train_size:train_size + val_size].copy()
        test_raw = raw_data.iloc[train_size + val_size:].copy()
        logger.info(f"Data split: Train={len(train_raw)}, Val={len(val_raw)}, Test={len(test_raw)}")
        
        # Calculate features and add target variables
        ta_engine = TechnicalAnalysisEngine()
        train_features = ta_engine.calculate_features(OHLCV.from_dataframe(train_raw))
        train_with_targets = ta_engine.add_target_variables(train_features)
        
        val_features = ta_engine.calculate_features(OHLCV.from_dataframe(val_raw))
        val_with_targets = ta_engine.add_target_variables(val_features)
        
        test_features = ta_engine.calculate_features(OHLCV.from_dataframe(test_raw))
        test_with_targets = ta_engine.add_target_variables(test_features)
        
        # Handle empty dataframes
        for df, name in [(train_with_targets, "train"), (val_with_targets, "validation"), (test_with_targets, "test")]:
            if df.empty:
                logger.error(f"Empty {name} dataset after feature calculation")
                return False

        # Select features (exclude non-feature columns)
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'bb_upper', 'bb_middle', 'bb_lower',
                        'future_return_1', 'future_return_5', 'future_return_10', 'future_return_20',
                        'target_down', 'target_up', 'target_flat', 'target_multi']

        feature_cols = [col for col in train_with_targets.columns if col not in exclude_cols]

        # Feature selection using Random Forest importance
        X_train = train_with_targets[feature_cols].copy()
        y_train = train_with_targets['target_down'].copy()

        # Select important features to reduce dimensionality
        feature_selector = FeatureSelector()
        selected_features, feature_importance = feature_selector.select_features(
            X_train.values, y_train.values, feature_cols, model_type='rf')

        self.feature_importance[symbol] = feature_importance
        self.selected_features[symbol] = selected_features

        logger.info(f"Selected {len(selected_features)} important features for {symbol}")
        logger.info(f"Top 10 features: {', '.join(selected_features[:10])}")

        # Use only selected features
        X_train = train_with_targets[selected_features].copy()
        y_train = train_with_targets['target_down'].copy()
        X_val = val_with_targets[selected_features].copy()
        y_val = val_with_targets['target_down'].copy()
        X_test = test_with_targets[selected_features].copy()
        y_test = test_with_targets['target_down'].copy()

        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Calculate class weights to handle imbalance
        class_counts = np.bincount(y_train)
        total = len(y_train)
        class_weights = {
            0: total / (2 * class_counts[0]),
            1: total / (2 * class_counts[1])
        }
        logger.info(f"Using class weights: {class_weights}")

        # Train Random Forest model
        rf_params = Config.RF_PARAMS.copy()
        rf_params['class_weight'] = class_weights
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train_scaled, y_train)

        # Train XGBoost model
        dtrain_xgb = xgb.DMatrix(X_train_scaled, label=y_train)
        dval_xgb = xgb.DMatrix(X_val_scaled, label=y_val)
        dtest_xgb = xgb.DMatrix(X_test_scaled)

        xgb_params = Config.XGB_PARAMS.copy()
        xgb_params['scale_pos_weight'] = class_weights[1] / class_weights[0]

        xgb_model = xgb.train(
            xgb_params,
            dtrain_xgb,
            num_boost_round=1000,
            evals=[(dtrain_xgb, 'train'), (dval_xgb, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=100
        )

        # Train LightGBM model
        dtrain_lgb = lgb.Dataset(X_train_scaled, label=y_train)
        dval_lgb = lgb.Dataset(X_val_scaled, label=y_val, reference=dtrain_lgb)

        lgbm_params = Config.LGBM_PARAMS.copy()
        lgbm_params['scale_pos_weight'] = class_weights[1] / class_weights[0]

        lgbm_model = lgb.train(
            lgbm_params,
            dtrain_lgb,
            num_boost_round=1000,
            valid_sets=[dtrain_lgb, dval_lgb],
            valid_names=['train', 'validation'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ],
        )

        # Train Gradient Boosting model as fourth ensemble member
        gb_params = Config.GB_PARAMS.copy()
        gb_model = GradientBoostingClassifier(**gb_params)
        gb_model.fit(X_train_scaled, y_train)

        # Get validation predictions
        rf_val_probs = rf_model.predict_proba(X_val_scaled)[:, 1]
        xgb_val_probs = xgb_model.predict(dval_xgb)
        lgbm_val_probs = lgbm_model.predict(X_val_scaled)
        gb_val_probs = gb_model.predict_proba(X_val_scaled)[:, 1]

        # Create weighted ensemble (give more weight to better models)
        # Start with equal weights and adjust based on performance
        val_metrics = {}
        models = {
            'rf': (rf_model, rf_val_probs),
            'xgb': (xgb_model, xgb_val_probs),
            'lgbm': (lgbm_model, lgbm_val_probs),
            'gb': (gb_model, gb_val_probs)
        }

        # Calculate AUC for each model to determine weights
        weights = {}
        for name, (_, probs) in models.items():
            auc = roc_auc_score(y_val, probs)
            val_metrics[f"{name}_auc"] = auc
            # Use AUC as weight, higher AUC = higher weight
            weights[name] = max(0.1, auc - 0.5)  # Minimum weight of 0.1

        # Normalize weights to sum to 1
        weight_sum = sum(weights.values())
        weights = {k: v / weight_sum for k, v in weights.items()}
        logger.info(f"Ensemble weights: {weights}")

        # Calculate weighted ensemble probabilities
        ensemble_val_proba = (
            weights['rf'] * rf_val_probs + 
            weights['xgb'] * xgb_val_probs + 
            weights['lgbm'] * lgbm_val_probs + 
            weights['gb'] * gb_val_probs
        )

        # Find optimal threshold on validation set
        best_threshold = self._optimize_threshold(ensemble_val_proba, y_val)
        self.thresholds[symbol] = best_threshold

        # Evaluate on test set
        rf_test_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
        xgb_test_probs = xgb_model.predict(dtest_xgb)
        lgbm_test_probs = lgbm_model.predict(X_test_scaled)
        gb_test_probs = gb_model.predict_proba(X_test_scaled)[:, 1]

        ensemble_test_proba = (
            weights['rf'] * rf_test_probs + 
            weights['xgb'] * xgb_test_probs + 
            weights['lgbm'] * lgbm_test_probs + 
            weights['gb'] * gb_test_probs
        )
        
        # Make predictions using optimal threshold
        ensemble_test_preds = (ensemble_test_proba > best_threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, ensemble_test_preds),
            'f1': f1_score(y_test, ensemble_test_preds),
            'precision': precision_score(y_test, ensemble_test_preds),
            'recall': recall_score(y_test, ensemble_test_preds),
            'roc_auc': roc_auc_score(y_test, ensemble_test_proba)
        }
        
        # Add confusion matrix
        cm = confusion_matrix(y_test, ensemble_test_preds)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Log results
        logger.info(f"Test metrics for {symbol} (ensemble, threshold={best_threshold:.4f}):")
        for metric, value in metrics.items():
            if metric != 'confusion_matrix':
                logger.info(f"  {metric.capitalize()}: {value:.4f}")
        
        # Save models and metadata
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        joblib.dump(rf_model, os.path.join(Config.MODEL_DIR, f"{symbol}_rf_model.pkl"))
        xgb_model.save_model(os.path.join(Config.MODEL_DIR, f"{symbol}_xgb_model.model"))
        lgbm_model.save_model(os.path.join(Config.MODEL_DIR, f"{symbol}_lgbm_model.txt"))
        joblib.dump(gb_model, os.path.join(Config.MODEL_DIR, f"{symbol}_gb_model.pkl"))
        joblib.dump(scaler, os.path.join(Config.MODEL_DIR, f"{symbol}_scaler.pkl"))
        
        # Save weights and threshold
        with open(os.path.join(Config.MODEL_DIR, f"{symbol}_ensemble_config.json"), 'w') as f:
            json.dump({
                'weights': weights,
                'threshold': best_threshold,
                'selected_features': selected_features
            }, f, indent=2)
        
        # Store models and metadata in memory
        self.models[symbol] = {
            'rf': rf_model, 
            'xgb': xgb_model, 
            'lgbm': lgbm_model, 
            'gb': gb_model,
            'weights': weights
        }
        self.scalers[symbol] = scaler
        self.model_metrics[symbol] = {
            'test_metrics': metrics,
            'val_metrics': val_metrics
        }
        
        # Plot and save feature importance
        plt.figure(figsize=(12, 10))
        top_features = feature_importance.head(20)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top 20 Feature Importance for {symbol}')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.MODEL_DIR, f"{symbol}_feature_importance.png"))
        plt.close()
        
        return True

    def _optimize_threshold(self, y_proba: np.ndarray, y_true: np.ndarray) -> float:
        """Find optimal threshold to maximize F1 score."""
        # Try different thresholds and find the one with the best F1 score
        thresholds = np.arange(0.3, 0.7, 0.02)
        best_f1 = 0
        best_threshold = 0.5
        
        # Calculate metrics for each threshold
        results = []
        for t in thresholds:
            preds = (y_proba > t).astype(int)
            f1 = f1_score(y_true, preds)
            precision = precision_score(y_true, preds)
            recall = recall_score(y_true, preds)
            
            results.append({
                'threshold': t,
                'f1': f1,
                'precision': precision,
                'recall': recall
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        
        # Log threshold optimization results
        logger.info(f"Optimized threshold: {best_threshold:.4f}, F1 score: {best_f1:.4f}")
        
        # Create and save threshold optimization plot
        results_df = pd.DataFrame(results)
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['threshold'], results_df['f1'], label='F1 Score', marker='o')
        plt.plot(results_df['threshold'], results_df['precision'], label='Precision', marker='s')
        plt.plot(results_df['threshold'], results_df['recall'], label='Recall', marker='^')
        plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold: {best_threshold:.2f}')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold Optimization')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.MODEL_DIR, f"threshold_optimization.png"))
        plt.close()
        
        return best_threshold

    def load_models(self, symbol: str) -> bool:
        """Load pre-trained models from disk."""
        rf_path = os.path.join(Config.MODEL_DIR, f"{symbol}_rf_model.pkl")
        xgb_path = os.path.join(Config.MODEL_DIR, f"{symbol}_xgb_model.model")
        lgbm_path = os.path.join(Config.MODEL_DIR, f"{symbol}_lgbm_model.txt")
        gb_path = os.path.join(Config.MODEL_DIR, f"{symbol}_gb_model.pkl")
        scaler_path = os.path.join(Config.MODEL_DIR, f"{symbol}_scaler.pkl")
        config_path = os.path.join(Config.MODEL_DIR, f"{symbol}_ensemble_config.json")

        if not all(os.path.exists(p) for p in [rf_path, xgb_path, lgbm_path, gb_path, scaler_path, config_path]):
            logger.error(f"Missing model files for {symbol}. Please train first using --mode train or --mode both.")
            return False

        try:
            # Load models
            rf_model = joblib.load(rf_path)
            xgb_model = xgb.Booster(model_file=xgb_path)
            lgbm_model = lgb.Booster(model_file=lgbm_path)
            gb_model = joblib.load(gb_path)
            scaler = joblib.load(scaler_path)
            
            # Load configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
                weights = config.get('weights', {'rf': 0.25, 'xgb': 0.25, 'lgbm': 0.25, 'gb': 0.25})
                threshold = config.get('threshold', 0.5)
                selected_features = config.get('selected_features', [])
            
            # Store in memory
            self.models[symbol] = {
                'rf': rf_model,
                'xgb': xgb_model,
                'lgbm': lgbm_model,
                'gb': gb_model,
                'weights': weights
            }
            self.scalers[symbol] = scaler
            self.thresholds[symbol] = threshold
            self.selected_features[symbol] = selected_features
            
            logger.info(f"Loaded pre-trained models for {symbol} from disk.")
            logger.info(f"Using threshold: {threshold:.4f}, Selected features: {len(selected_features)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models for {symbol}: {e}")
            return False

    def predict(self, symbol: str, features: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """Generate predictions using the ensemble model."""
        if symbol not in self.models:
            logger.error(f"No trained model for {symbol}")
            return {'signal': -1, 'confidence': 0.0}
        
        # Extract only the selected features
        selected_features = self.selected_features.get(symbol, [])
        if not selected_features:
            logger.warning(f"No selected features for {symbol}, using all available features")
            X = features.drop(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'], errors='ignore')
        else:
            missing_features = [f for f in selected_features if f not in features.columns]
            if missing_features:
                logger.warning(f"Missing features for {symbol}: {missing_features}")
                # Use available features
                available_features = [f for f in selected_features if f in features.columns]
                X = features[available_features]
            else:
                X = features[selected_features]
        
        # Scale features
        X_scaled = self.scalers[symbol].transform(X)
        
        # Get predictions from each model
        weights = self.models[symbol]['weights']
        rf_proba = self.models[symbol]['rf'].predict_proba(X_scaled)[:, 1]
        xgb_proba = self.models[symbol]['xgb'].predict(xgb.DMatrix(X_scaled))
        lgbm_proba = self.models[symbol]['lgbm'].predict(X_scaled)
        gb_proba = self.models[symbol]['gb'].predict_proba(X_scaled)[:, 1]
        
        # Weighted ensemble
        ensemble_proba = (
            weights['rf'] * rf_proba + 
            weights['xgb'] * xgb_proba + 
            weights['lgbm'] * lgbm_proba + 
            weights['gb'] * gb_proba
        )
        
        # Apply threshold
        threshold = self.thresholds.get(symbol, 0.5)
        signal = 1 if ensemble_proba[0] > threshold else 0  # 1=down, 0=up (for trading)
        confidence = ensemble_proba[0] if signal == 1 else 1 - ensemble_proba[0]
        
        # Calculate individual model signals for comparison
        rf_signal = 1 if rf_proba[0] > threshold else 0
        xgb_signal = 1 if xgb_proba[0] > threshold else 0
        lgbm_signal = 1 if lgbm_proba[0] > threshold else 0
        gb_signal = 1 if gb_proba[0] > threshold else 0
        
        # Check for model agreement
        agreement_count = sum([
            rf_signal == signal,
            xgb_signal == signal,
            lgbm_signal == signal,
            gb_signal == signal
        ])
        
        agreement_ratio = agreement_count / 4.0
        
        # Higher confidence if models agree
        adjusted_confidence = confidence * (0.75 + 0.25 * agreement_ratio)
        
        return {
            'signal': signal,
            'confidence': adjusted_confidence,
            'raw_probability': float(ensemble_proba[0]),
            'threshold': threshold,
            'model_agreement': agreement_ratio
        }

# Strategy Engine
class StrategyEngine:
    def __init__(self, ml_engine: MachineLearningEngine):
        self.ml_engine = ml_engine
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD
        self.daily_trades = {}  # Track daily trades per symbol
        
    def _reset_daily_trades(self):
        """Reset daily trade counter at the start of a new day."""
        self.daily_trades = {}
    
    def _can_trade_today(self, symbol: str, timestamp: pd.Timestamp) -> bool:
        """Check if we can make more trades for this symbol today."""
        today = timestamp.date()
        if today not in self.daily_trades:
            self.daily_trades[today] = {}
        
        if symbol not in self.daily_trades[today]:
            self.daily_trades[today][symbol] = 0
            
        return self.daily_trades[today][symbol] < Config.MAX_TRADES_PER_DAY

    def generate_signals(self, symbol: str, ohlcv_data: pd.DataFrame, portfolio: Dict) -> List[Dict]:
        """Generate trading signals based on ML predictions and technical conditions."""
        if ohlcv_data.empty:
            logger.warning(f"Empty OHLCV data for {symbol}")
            return []
        
        # Get latest timestamp
        timestamp = ohlcv_data['timestamp'].iloc[-1]
        
        # Check if we can trade today
        if not self._can_trade_today(symbol, timestamp):
            logger.debug(f"Max daily trades reached for {symbol} on {timestamp.date()}")
            return []
        
        # Calculate technical features
        ta_engine = TechnicalAnalysisEngine()
        features = ta_engine.calculate_features(OHLCV.from_dataframe(ohlcv_data))
        if features.empty:
            logger.warning(f"Failed to calculate features for {symbol}")
            return []
        
        # Get latest features and current market conditions
        latest_features = features.iloc[-1:].copy()
        current_price = ohlcv_data['close'].iloc[-1]
        
        # Get ML prediction
        prediction = self.ml_engine.predict(symbol, latest_features)
        if prediction['signal'] == -1:
            logger.warning(f"Invalid prediction for {symbol}")
            return []
        
        # Extract prediction details
        signal_type = prediction['signal']  # 1 for down, 0 for up
        confidence = prediction['confidence']
        model_agreement = prediction.get('model_agreement', 1.0)
        
        # Convert to trading signal (0 for long, 1 for short)
        trading_signal = 1 if signal_type == 1 else 0
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            logger.debug(f"Confidence {confidence:.4f} below threshold {self.confidence_threshold:.4f} for {symbol}")
            return []
        
        # Check model agreement
        if model_agreement < 0.75:  # At least 3/4 models should agree
            logger.debug(f"Insufficient model agreement ({model_agreement:.2f}) for {symbol}")
            return []
        
        # Get market conditions from features for additional filtering
        market_regime = latest_features['market_regime'].iloc[0] if 'market_regime' in latest_features else 0
        trend_strength = latest_features['trend_strength'].iloc[0] if 'trend_strength' in latest_features else 0.5
        
        # Only trade with the trend in strong trend environments
        if abs(market_regime) >= 2 and np.sign(market_regime) != np.sign(1 - 2*trading_signal):
            logger.debug(f"Signal against strong trend for {symbol}, skipping")
            return []
        
        # Enhanced position sizing and risk management
        atr = features['atr'].iloc[-1] if 'atr' in features else current_price * 0.01
        volatility = features['volatility_20'].iloc[-1] if 'volatility_20' in features else 0.02
        
        # Dynamic ATR multiplier based on volatility
        atr_multiplier = min(3.0, max(1.5, Config.ATR_MULTIPLIER * (1 + volatility)))
        
        # Improved stop loss placement
        if trading_signal == 0:  # Long
            # Use recent swing low for long stop loss
            stop_window = min(20, len(ohlcv_data) - 1)
            recent_low = ohlcv_data['low'].iloc[-stop_window:].min()
            stop_loss = max(current_price - (atr_multiplier * atr), recent_low * 0.995)
        else:  # Short
            # Use recent swing high for short stop loss
            stop_window = min(20, len(ohlcv_data) - 1)
            recent_high = ohlcv_data['high'].iloc[-stop_window:].max()
            stop_loss = min(current_price + (atr_multiplier * atr), recent_high * 1.005)
        
        # Calculate risk per unit with improved stop loss
        risk_per_unit = abs(current_price - stop_loss)
        
        # Adaptive position sizing based on model confidence and market regime
        regime_factor = 1.0
        if market_regime == 2:  # Strong uptrend
            regime_factor = 1.2 if trading_signal == 0 else 0.8
        elif market_regime == -2:  # Strong downtrend
            regime_factor = 0.8 if trading_signal == 0 else 1.2
        
        # Calculate position size based on risk
        risk_amount = Config.RISK_FACTOR * portfolio['capital']
        confidence_factor = min(1.5, max(0.5, confidence))
        
        # Adjust risk based on confidence and regime
        adjusted_risk = risk_amount * confidence_factor * regime_factor
        
        # Calculate base position size
        base_size = adjusted_risk / risk_per_unit
        
        # Adjust based on historical model precision
        precision_adjustment = self.ml_engine.model_metrics.get(symbol, {}).get('test_metrics', {}).get('precision', 0.5)
        precision_factor = min(1.5, max(0.5, precision_adjustment / 0.5))  # Normalize around 0.5
        
        # Final position size calculation
        position_size = int(base_size * precision_factor)
        
        # Apply position limits
        max_position_value = portfolio['capital'] * Config.MAX_POSITION_FRACTION
        position_value = position_size * current_price
        
        if position_value > max_position_value:
            position_size = int(max_position_value / current_price)
        
        # Ensure minimum position size
        position_size = max(1, position_size)
        
        # Calculate take profit levels
        take_profit_1 = current_price - (Config.PROFIT_TAKE_RATIO_1 * atr_multiplier * atr) if trading_signal == 1 else \
                        current_price + (Config.PROFIT_TAKE_RATIO_1 * atr_multiplier * atr)
        
        take_profit_2 = current_price - (Config.PROFIT_TAKE_RATIO_2 * atr_multiplier * atr) if trading_signal == 1 else \
                        current_price + (Config.PROFIT_TAKE_RATIO_2 * atr_multiplier * atr)
        
        # Create signal dictionary
        signal = {
            'symbol': symbol,
            'timestamp': timestamp,
            'price': current_price,
            'signal': trading_signal,
            'confidence': confidence,
            'stop_loss': stop_loss,
            'take_profit_1': take_profit_1,
            'take_profit_2': take_profit_2,
            'position_size': position_size,
            'atr': atr,
            'market_regime': market_regime,
            'trend_strength': trend_strength,
            'model_agreement': model_agreement,
            'source': 'ML_Ensemble'
        }
        
        # Update daily trade counter
        today = timestamp.date()
        self.daily_trades[today][symbol] = self.daily_trades[today].get(symbol, 0) + 1
        
        # Log signal details
        logger.info(f"Signal generated for {symbol}: {'SHORT' if trading_signal == 1 else 'LONG'}")
        logger.info(f"  Price: {current_price:.2f}, Stop: {stop_loss:.2f}, Size: {position_size}")
        logger.info(f"  Confidence: {confidence:.4f}, ATR: {atr:.2f}, Regime: {market_regime}")
        
        return [signal]

# Backtest Engine
class BacktestEngine:
    def __init__(self, strategy_engine: StrategyEngine):
        self.strategy_engine = strategy_engine
        self.trailing_stops = {}  # Track trailing stops for open positions

    async def run_backtest(self, symbol: str, ohlcv_data: pd.DataFrame, portfolio: Dict) -> Dict:
        """Run backtest simulation for the given symbol and data."""
        trades = []
        cash = portfolio['capital']
        position = None
        equity = [cash]  # Start equity curve with initial capital
        daily_returns = []
        trade_dates = set()  # Track unique trading days
        
        # Save backtest data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        data_file = os.path.join(Config.RESULTS_DIR, f"backtest_data_{symbol}_{timestamp}.csv")
        ohlcv_data.to_csv(data_file, index=False)
        logger.info(f"Saved backtest data to {data_file}")

        # Reset trailing stop tracker
        self.trailing_stops = {}
        
        # For progress tracking
        progress_bar = tqdm(range(Config.FEATURE_WINDOW, len(ohlcv_data)), desc=f"Backtesting {symbol}")
        
        for i in progress_bar:
            window = ohlcv_data.iloc[:i+1].copy()
            current_bar = window.iloc[-1]
            
            current_price = current_bar['close']
            current_time = current_bar['timestamp']
            current_date = pd.Timestamp(current_time).date()
            trade_dates.add(current_date)
            
            # Update trailing stop if position is open
            if position and symbol in self.trailing_stops:
                if position.signal == 0:  # Long position
                    # Update trailing stop if price moves in our favor
                    if current_price > self.trailing_stops[symbol]:
                        # Move stop loss up as price increases
                        atr = window['atr'].iloc[-1] if 'atr' in window.columns else current_price * 0.01
                        new_stop = current_price - (Config.ATR_MULTIPLIER * atr)
                        if new_stop > position.stop_loss:
                            position.stop_loss = new_stop
                            self.trailing_stops[symbol] = new_stop
                            logger.debug(f"Updated trailing stop for {symbol} LONG to {new_stop:.2f}")
                
                elif position.signal == 1:  # Short position
                    # Update trailing stop if price moves in our favor
                    if current_price < self.trailing_stops[symbol]:
                        # Move stop loss down as price decreases
                        atr = window['atr'].iloc[-1] if 'atr' in window.columns else current_price * 0.01
                        new_stop = current_price + (Config.ATR_MULTIPLIER * atr)
                        if new_stop < position.stop_loss:
                            position.stop_loss = new_stop
                            self.trailing_stops[symbol] = new_stop
                            logger.debug(f"Updated trailing stop for {symbol} SHORT to {new_stop:.2f}")
            
            # Check for position exit conditions
            if position:
                exit_reason = None
                
                # Check stop loss
                if (position.signal == 0 and current_price <= position.stop_loss) or \
                   (position.signal == 1 and current_price >= position.stop_loss):
                    exit_reason = "Stop Loss"
                
                # Check take profit levels
                elif (position.signal == 0 and current_price >= position.take_profit_1) or \
                     (position.signal == 1 and current_price <= position.take_profit_1):
                    exit_reason = "Take Profit"
                
                # Execute exit if conditions are met
                if exit_reason:
                    position.exit_price = current_price
                    position.exit_time = current_time
                    
                    # Calculate profit/loss
                    if position.signal == 0:  # Long
                        profit = (position.exit_price - position.entry_price) * position.size
                    else:  # Short
                        profit = (position.entry_price - position.exit_price) * position.size
                    
                    position.profit = profit
                    position.exit_reason = exit_reason
                    
                    # Update cash
                    cash += profit + (position.size * position.entry_price)  # Return initial capital + profit
                    
                    # Save trade
                    trades.append(position)
                    logger.info(f"Closed trade for {symbol}: {exit_reason}, Profit={profit:.2f}, Cash={cash:.2f}")
                    
                    # Remove trailing stop
                    if symbol in self.trailing_stops:
                        del self.trailing_stops[symbol]
                    
                    # Reset position
                    position = None
            
            # Generate new signals if no position is open
            if not position:
                signals = self.strategy_engine.generate_signals(symbol, window, {'capital': cash})
                
                if signals:
                    signal = signals[0]
                    position_value = signal['position_size'] * current_price
                    
                    # Open position if cash allows
                    if cash >= position_value:
                        position = Position(
                            symbol=symbol,
                            entry_time=current_time,
                            entry_price=current_price,
                            size=signal['position_size'],
                            signal=signal['signal'],
                            stop_loss=signal['stop_loss'],
                            take_profit_1=signal['take_profit_1'],
                            take_profit_2=signal['take_profit_2']
                        )
                        
                        # Set initial trailing stop
                        self.trailing_stops[symbol] = position.stop_loss
                        
                        # Deduct position value from cash
                        cash -= position_value
                        
                        logger.info(f"Opened {signal['signal']} trade for {symbol}: Size={position.size}, " +
                                    f"Entry={current_price:.2f}, Stop={position.stop_loss:.2f}, Cash={cash:.2f}")
                    else:
                        logger.warning(f"Insufficient cash for {symbol}. Required={position_value:.2f}, Available={cash:.2f}")
            
            # Update equity curve
            current_equity = cash
            if position:
                # Mark-to-market position value
                if position.signal == 0:  # Long
                    position_value = position.size * current_price
                else:  # Short
                    position_value = position.size * (2 * position.entry_price - current_price)
                current_equity += position_value
            
            equity.append(current_equity)
            
            # Calculate daily return if this is a new day
            if len(equity) > 1:
                daily_return = (current_equity / equity[-2]) - 1
                daily_returns.append(daily_return)

        # Close any remaining position at the end of the backtest
        if position:
            position.exit_price = current_price
            position.exit_time = current_time
            position.exit_reason = "End of Backtest"
            
            if position.signal == 0:  # Long
                profit = (position.exit_price - position.entry_price) * position.size
            else:  # Short
                profit = (position.entry_price - position.exit_price) * position.size
            
            position.profit = profit
            cash += profit + (position.size * position.entry_price)
            trades.append(position)
            logger.info(f"Closed final trade for {symbol}: Profit={profit:.2f}, Cash={cash:.2f}")

        # Calculate performance metrics
        trades_df = pd.DataFrame([t.to_dict() for t in trades]) if trades else pd.DataFrame()

        # Calculate basic metrics
        net_profit = cash - portfolio['capital']
        net_profit_pct = net_profit / portfolio['capital'] * 100

        # Handle empty trades_df gracefully
        if trades_df.empty:
            logger.warning(f"No trades executed for {symbol}")
            result = {
                'symbol': symbol,
                'net_profit': net_profit,
                'net_profit_pct': net_profit_pct,
                'win_rate': 0,
                'winning_trades': 0,
                'total_trades': 0,
                'profit_factor': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'cagr': 0,
                'equity_curve': equity,
                'trades': [],
                'backtest_days': len(trade_dates),
                'avg_trade_duration': 0,
                'avg_profit_per_trade': 0,
                'max_consecutive_losses': 0
            }
        else:
            # Convert trades dataframe to include proper types
            trades_df['profit'] = trades_df['profit'].astype(float)
            trades_df['entry_price'] = trades_df['entry_price'].astype(float)
            trades_df['exit_price'] = trades_df['exit_price'].astype(float)
            
            # Calculate trade metrics
            winning_trades = len(trades_df[trades_df['profit'] > 0])
            total_trades = len(trades_df)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Profit factor
            gross_profit = trades_df[trades_df['profit'] > 0]['profit'].sum() if winning_trades > 0 else 0
            gross_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].sum()) if len(trades_df[trades_df['profit'] < 0]) > 0 else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate returns and risk metrics
            equity_series = pd.Series(equity)
            returns = equity_series.pct_change().fillna(0)
            
            # Sharpe and Sortino ratios
            risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate, daily
            excess_returns = returns - risk_free_rate
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0
            
            # Sortino ratio (downside risk only)
            downside_returns = excess_returns[excess_returns < 0]
            sortino_ratio = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() != 0 else 0
            
            # Maximum drawdown
            rolling_max = equity_series.cummax()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())
            
            # CAGR (Compound Annual Growth Rate)
            years = len(trade_dates) / 252  # Trading days per year
            cagr = ((cash / portfolio['capital']) ** (1 / years) - 1) * 100 if years > 0 else 0
            
            # Additional metrics
            # Trade duration
            if 'entry_time' in trades_df and 'exit_time' in trades_df:
                trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
                trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
                trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600  # hours
                avg_trade_duration = trades_df['duration'].mean()
            else:
                avg_trade_duration = 0
            
            # Average profit per trade
            avg_profit_per_trade = trades_df['profit'].mean() if total_trades > 0 else 0
            
            # Maximum consecutive losses
            if total_trades > 0:
                # Create a series of 1 (win) and 0 (loss)
                win_loss = (trades_df['profit'] > 0).astype(int)
                # Find runs of losses (0s)
                loss_runs = []
                current_run = 0
                for result in win_loss:
                    if result == 0:  # Loss
                        current_run += 1
                    else:  # Win
                        if current_run > 0:
                            loss_runs.append(current_run)
                        current_run = 0
                # Add the last run if it's a loss
                if current_run > 0:
                    loss_runs.append(current_run)
                
                max_consecutive_losses = max(loss_runs) if loss_runs else 0
            else:
                max_consecutive_losses = 0
            
            # Compile all metrics into result dictionary
            result = {
                'symbol': symbol,
                'net_profit': net_profit,
                'net_profit_pct': net_profit_pct,
                'win_rate': win_rate,
                'winning_trades': winning_trades,
                'total_trades': total_trades,
                'profit_factor': profit_factor,
                'max_drawdown_pct': max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'cagr': cagr,
                'equity_curve': equity,
                'trades': trades_df.to_dict('records'),
                'backtest_days': len(trade_dates),
                'avg_trade_duration': avg_trade_duration,
                'avg_profit_per_trade': avg_profit_per_trade,
                'max_consecutive_losses': max_consecutive_losses
            }
        
        logger.info(f"Backtest completed for {symbol}: Net Profit=${net_profit:.2f} ({net_profit_pct:.2f}%), " +
                   f"Win Rate={win_rate*100:.1f}%, Trades={total_trades}")
        
        # Save results to file
        try:
            # Create serializable result (handle numpy types, timestamps, etc.)
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(i) for i in obj]
                return obj
            
            serializable_result = convert_to_serializable(result)
            
            # Save JSON results
            result_file = os.path.join(Config.RESULTS_DIR, f"backtest_{symbol}_{timestamp}.json")
            with open(result_file, 'w') as f:
                json.dump(serializable_result, f, indent=2)
            
            # Save trades CSV
            if not trades_df.empty:
                trades_file = os.path.join(Config.RESULTS_DIR, f"trades_{symbol}_{timestamp}.csv")
                trades_df.to_csv(trades_file, index=False)
            
            logger.info(f"Saved backtest results to {result_file}")
            
        except Exception as e:
            logger.error(f"Failed to save backtest results for {symbol}: {e}", exc_info=True)
        
        # Generate equity curve chart
        self._plot_equity_curve(equity, symbol, timestamp)
        
        # Generate drawdown chart
        self._plot_drawdown(equity, symbol, timestamp)
        
        # Generate trade distribution chart if trades exist
        if not trades_df.empty:
            self._plot_trade_distribution(trades_df, symbol, timestamp)
        
        return result
    
    def _plot_equity_curve(self, equity, symbol, timestamp):
        """Plot and save equity curve chart."""
        plt.figure(figsize=(12, 6))
        plt.plot(equity, label='Equity Curve', color='blue')
        plt.title(f'Equity Curve for {symbol}')
        plt.xlabel('Trade Bars')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add initial capital line
        plt.axhline(y=equity[0], color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        
        # Save chart
        equity_chart = os.path.join(Config.RESULTS_DIR, f"equity_curve_{symbol}_{timestamp}.png")
        plt.savefig(equity_chart, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_drawdown(self, equity, symbol, timestamp):
        """Plot and save drawdown chart."""
        equity_series = pd.Series(equity)
        rolling_max = equity_series.cummax()
        drawdowns = (equity_series - rolling_max) / rolling_max * 100
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(range(len(drawdowns)), drawdowns, 0, color='red', alpha=0.3)
        plt.plot(drawdowns, color='red', label='Drawdown %')
        plt.title(f'Drawdown Chart for {symbol}')
        plt.xlabel('Trade Bars')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save chart
        drawdown_chart = os.path.join(Config.RESULTS_DIR, f"drawdown_{symbol}_{timestamp}.png")
        plt.savefig(drawdown_chart, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_trade_distribution(self, trades_df, symbol, timestamp):
        """Plot and save trade profit distribution chart."""
        plt.figure(figsize=(12, 6))
        
        # Histogram of trade profits
        sns.histplot(trades_df['profit'], bins=20, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title(f'Trade Profit Distribution for {symbol}')
        plt.xlabel('Profit/Loss ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Save chart
        trades_chart = os.path.join(Config.RESULTS_DIR, f"trade_distribution_{symbol}_{timestamp}.png")
        plt.savefig(trades_chart, dpi=300, bbox_inches='tight')
        plt.close()

# Trading System
class TradingSystem:
    def __init__(self):
        self.ml_engine = MachineLearningEngine()
        self.strategy_engine = StrategyEngine(self.ml_engine)
        self.backtest_engine = BacktestEngine(self.strategy_engine)
        self.portfolio = {
            'capital': Config.INITIAL_CAPITAL,
            'positions': {},
            'max_position_fraction': Config.MAX_POSITION_FRACTION,
            'max_drawdown': Config.MAX_DRAWDOWN_THRESHOLD
        }

    async def train_models(self, symbols: List[str]):
        """Train ML models for all symbols."""
        logger.info(f"Training models for {len(symbols)} symbols: {', '.join(symbols)}")
        
        results = []
        for symbol in symbols:
            try:
                success = await self.ml_engine.train_model_for_symbol(symbol)
                results.append((symbol, success))
            except Exception as e:
                logger.error(f"Error training model for {symbol}: {e}", exc_info=True)
                results.append((symbol, False))
        
        # Summarize training results
        successful = [symbol for symbol, success in results if success]
        failed = [symbol for symbol, success in results if not success]
        
        logger.info(f"Training complete: {len(successful)}/{len(symbols)} successful")
        if failed:
            logger.warning(f"Failed to train models for: {', '.join(failed)}")
        
        return successful

    async def run_backtest(self, symbols: List[str]) -> Dict[str, Dict]:
        """Run backtest for all symbols."""
        logger.info(f"Running backtest for symbols: {', '.join(symbols)}")
        results = {}
        
        # Set date range for backtest
        start_date = datetime.now() - timedelta(days=Config.BACKTEST_DAYS)
        end_date = datetime.now()
        
        # Load models if not already loaded
        for symbol in symbols:
            if not self.ml_engine.models.get(symbol) and not self.ml_engine.load_models(symbol):
                logger.error(f"Cannot run backtest for {symbol} without trained models. Run with --mode train or --mode both first.")
                return {}
        
        # Initialize portfolio tracking
        portfolio_value = Config.INITIAL_CAPITAL
        total_equity_curve = [portfolio_value]
        
        # Run backtest for each symbol
        for symbol in symbols:
            # Get historical data
            ohlcv_data = await DataManager.get_historical_data(symbol, start_date, end_date)
            if ohlcv_data is None or ohlcv_data.empty:
                logger.warning(f"No data for {symbol}, skipping backtest")
                continue
            
            # Run backtest
            backtest_result = await self.backtest_engine.run_backtest(symbol, ohlcv_data, self.portfolio)
            results[symbol] = backtest_result
            
            # Update portfolio value
            portfolio_value += backtest_result['net_profit']
            
            # Check drawdown threshold
            if 'equity_curve' in backtest_result:
                equity_curve = backtest_result['equity_curve']
                peak = max(equity_curve)
                current = equity_curve[-1]
                drawdown = (peak - current) / peak
                
                if drawdown > self.portfolio['max_drawdown']:
                    logger.warning(f"Max drawdown {drawdown:.2%} exceeded threshold {self.portfolio['max_drawdown']:.2%}")
        
        # Update portfolio capital for future runs
        self.portfolio['capital'] = portfolio_value
        
        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_results = {
            'symbols': symbols,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'initial_capital': Config.INITIAL_CAPITAL,
            'final_capital': portfolio_value,
            'total_return_pct': (portfolio_value / Config.INITIAL_CAPITAL - 1) * 100,
            'individual_results': {symbol: {
                'net_profit': result.get('net_profit', 0),
                'net_profit_pct': result.get('net_profit_pct', 0),
                'win_rate': result.get('win_rate', 0),
                'total_trades': result.get('total_trades', 0)
            } for symbol, result in results.items()}
        }
        
        # Save combined results to file
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        with open(os.path.join(Config.RESULTS_DIR, f"combined_results_{timestamp}.json"), 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        return results

    def print_backtest_summary(self, results: Dict[str, Dict], detailed: bool = False):
        """Print formatted backtest summary to console."""
        if not results:
            print(f"\n{Fore.RED}No backtest results available.{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}BACKTEST RESULTS SUMMARY{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        
        # Calculate overall metrics
        total_profit = sum(result.get('net_profit', 0) for result in results.values())
        total_trades = sum(result.get('total_trades', 0) for result in results.values())
        winning_trades = sum(result.get('winning_trades', 0) for result in results.values())
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Overall summary
        print(f"\n{Fore.GREEN}OVERALL PERFORMANCE:{Style.RESET_ALL}")
        print(f"Initial Capital: ${Config.INITIAL_CAPITAL:,.2f}")
        print(f"Final Capital: ${(Config.INITIAL_CAPITAL + total_profit):,.2f}")
        print(f"Net Profit: {Fore.GREEN if total_profit >= 0 else Fore.RED}${total_profit:,.2f} ({total_profit/Config.INITIAL_CAPITAL*100:.2f}%){Style.RESET_ALL}")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate*100:.1f}% ({winning_trades}/{total_trades})")
        
        # Per-symbol summary
        print(f"\n{Fore.YELLOW}SYMBOL PERFORMANCE:{Style.RESET_ALL}")
        
        # Create table data
        headers = ["Symbol", "Net Profit", "Win Rate", "Trades", "Profit Factor", "Max DD", "Sharpe"]
        table_data = []
        
        for symbol, result in results.items():
            net_profit = result.get('net_profit', 0)
            net_profit_pct = result.get('net_profit_pct', 0)
            win_rate = result.get('win_rate', 0)
            total_trades = result.get('total_trades', 0)
            profit_factor = result.get('profit_factor', 0)
            max_drawdown = result.get('max_drawdown_pct', 0)
            sharpe = result.get('sharpe_ratio', 0)
            
            profit_color = Fore.GREEN if net_profit >= 0 else Fore.RED
            
            # Format row
            row = [
                symbol,
                f"{profit_color}${net_profit:,.2f} ({net_profit_pct:.1f}%){Style.RESET_ALL}",
                f"{win_rate*100:.1f}%",
                f"{total_trades}",
                f"{profit_factor:.2f}",
                f"{max_drawdown:.1f}%",
                f"{sharpe:.2f}"
            ]
            table_data.append(row)
        
        # Print table
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Detailed trade information if requested
        if detailed and any(result.get('trades') for result in results.values()):
            print(f"\n{Fore.MAGENTA}RECENT TRADES:{Style.RESET_ALL}")
            
            all_trades = []
            for symbol, result in results.items():
                trades = result.get('trades', [])
                for trade in trades:
                    trade['symbol'] = symbol
                    all_trades.append(trade)
            
            # Sort trades by exit time, most recent first
            all_trades = sorted(all_trades, key=lambda x: x.get('exit_time', ''), reverse=True)
            
            # Take most recent 10 trades
            recent_trades = all_trades[:10]
            
            # Create trade table
            trade_headers = ["Symbol", "Type", "Entry", "Exit", "P/L", "Duration", "Reason"]
            trade_data = []
            
            for trade in recent_trades:
                signal = trade.get('signal', 0)
                trade_type = "LONG" if signal == 0 else "SHORT"
                profit = trade.get('profit', 0)
                profit_color = Fore.GREEN if profit >= 0 else Fore.RED
                
                # Format trade row
                trade_row = [
                    trade.get('symbol', ''),
                    f"{Fore.BLUE if signal == 0 else Fore.RED}{trade_type}{Style.RESET_ALL}",
                    f"${trade.get('entry_price', 0):,.2f}",
                    f"${trade.get('exit_price', 0):,.2f}",
                    f"{profit_color}${profit:,.2f}{Style.RESET_ALL}",
                    f"{trade.get('duration', 0):.1f}h" if 'duration' in trade else 'N/A',
                    trade.get('exit_reason', 'Unknown')
                ]
                trade_data.append(trade_row)
            
            # Print trade table
            print(tabulate(trade_data, headers=trade_headers, tablefmt="grid"))
        
        print(f"\n{Fore.CYAN}Detailed reports and charts saved to: {Config.RESULTS_DIR}{Style.RESET_ALL}")

# Main
async def main():
    """Main entry point for the trading system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Advanced ML-based Trading System")
    parser.add_argument("--mode", choices=["train", "backtest", "both"], required=True,
                      help="Operation mode: train models, run backtest, or both")
    parser.add_argument("--symbols", type=str, default="RELIANCE",
                      help="Comma-separated list of symbols to trade")
    parser.add_argument("--days", type=int, default=Config.BACKTEST_DAYS,
                      help="Number of days for backtesting")
    parser.add_argument("--capital", type=float, default=Config.INITIAL_CAPITAL,
                      help="Initial capital for backtesting")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    parser.add_argument("--detailed", action="store_true",
                      help="Show detailed trade information in summary")
    
    args = parser.parse_args()
    
    # Configure logging
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    
    # Update configuration
    Config.BACKTEST_DAYS = args.days
    Config.INITIAL_CAPITAL = args.capital
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Initialize trading system
    system = TradingSystem()
    
    try:
        # Display startup banner
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}ADVANCED ML-BASED TRADING SYSTEM{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"Mode: {Fore.YELLOW}{args.mode.upper()}{Style.RESET_ALL}")
        print(f"Symbols: {Fore.YELLOW}{', '.join(symbols)}{Style.RESET_ALL}")
        print(f"Backtest Period: {Fore.YELLOW}{args.days} days{Style.RESET_ALL}")
        print(f"Initial Capital: {Fore.YELLOW}${args.capital:,.2f}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
        
        # Train models if requested
        if args.mode in ["train", "both"]:
            print(f"{Fore.GREEN}Training models...{Style.RESET_ALL}")
            await system.train_models(symbols)
        
        # Run backtest if requested
        if args.mode in ["backtest", "both"]:
            print(f"{Fore.GREEN}Running backtest...{Style.RESET_ALL}")
            results = await system.run_backtest(symbols)
            
            # Print backtest summary
            system.print_backtest_summary(results, detailed=args.detailed)
        
        print(f"\n{Fore.GREEN}All operations completed successfully!{Style.RESET_ALL}")
        
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        print(f"\n{Fore.RED}ERROR: {str(e)}{Style.RESET_ALL}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Initialize colorama for Windows support
    colorama.init()
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}Critical error: {str(e)}{Style.RESET_ALL}")
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Reset colorama styles
        print(Style.RESET_ALL)
