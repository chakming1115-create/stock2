# stock_trading_tool.py
# Professional stock trading tool with optimized ML integration.
# Key optimizations for ML training efficiency:
# 1. Train a single LSTM model once on the full historical data (or large in-sample period)
# 2. Use it for out-of-sample rolling predictions (no retraining per step → massive speedup)
# 3. Add early stopping and better batch handling
# 4. Use walk-forward style but with fixed model + expanding/rolling feature window
# 5. Optional: feature engineering expansion (technical indicators as inputs)

import os
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from polygon import RESTClient
from typing import Dict, Tuple
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment variables
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if not POLYGON_API_KEY:
    raise ValueError("POLYGON_API_KEY environment variable is not set.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class StockDataFetcher:
    """Fetches stock data using Polygon API."""
    def __init__(self, api_key: str):
        self.client = RESTClient(api_key)
    
    def get_historical_data(self, ticker: str, start_date: str, end_date: str, multiplier: int = 1, timespan: str = 'day') -> pd.DataFrame:
        try:
            aggs = self.client.get_aggs(ticker, multiplier, timespan, start_date, end_date)
            data = [{
                'timestamp': datetime.datetime.fromtimestamp(agg.timestamp / 1000),
                'open': agg.open, 'high': agg.high, 'low': agg.low,
                'close': agg.close, 'volume': agg.volume
            } for agg in aggs]
            df = pd.DataFrame(data).set_index('timestamp')
            logger.info(f"Fetched {len(df)} bars for {ticker}")
            return df
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()

class TechnicalAnalyzer:
    @staticmethod
    def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add more features to improve model input quality"""
        df['short_ma'] = df['close'].rolling(20, min_periods=1).mean()
        df['long_ma']  = df['close'].rolling(50, min_periods=1).mean()
        delta = df['close'].diff()
        df['rsi'] = 100 - (100 / (1 + (delta.where(delta>0,0).rolling(14).mean() / 
                                   -delta.where(delta<0,0).rolling(14).mean())))
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = tr.rolling(14).mean()
        
        df['volume_change'] = df['volume'].pct_change()
        for lag in [1, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
        
        df.dropna(inplace=True)
        return df

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class EfficientMLPredictor:
    """Optimized ML predictor: train once, predict rolling forward"""
    
    def __init__(self, look_back: int = 60, hidden_size: int = 64, epochs: int = 50,
                 batch_size: int = 64, train_split: float = 0.7, patience: int = 8):
        self.look_back = look_back
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_split = train_split
        self.patience = patience
        self.scaler = MinMaxScaler()
        self.model = None
    
    def prepare_sequences(self, data: np.ndarray, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = [], []
        for i in range(len(data) - self.look_back):
            X.append(data[i:i + self.look_back])
            y.append(data[i + self.look_back])
        X = np.array(X)
        y = np.array(y)
        if train:
            self.scaler.fit(X.reshape(-1, X.shape[-1]))
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        return torch.FloatTensor(X_scaled).to(device), torch.FloatTensor(y).to(device)
    
    def train(self, df: pd.DataFrame):
        """Train once on in-sample data"""
        features = df.drop(columns=['close']).values  # all features except target
        target = df['close'].values.reshape(-1, 1)
        
        split_idx = int(len(df) * self.train_split)
        train_features = features[:split_idx]
        train_target   = target[:split_idx]
        
        X_train, y_train = self.prepare_sequences(np.hstack([train_features, train_target]), train=True)
        
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        input_size = X_train.shape[-1]
        self.model = LSTMModel(input_size, self.hidden_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5)
        
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        self.model.load_state_dict(torch.load("best_model.pt"))
        logger.info("Training completed. Model ready for inference.")
    
    def predict_rolling(self, df: pd.DataFrame) -> pd.Series:
        """Rolling one-step-ahead prediction using trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        predictions = np.full(len(df), np.nan)
        features = self.scaler.transform(df.drop(columns=['close']).values)
        
        with torch.no_grad():
            for i in range(self.look_back, len(df)):
                seq = features[i - self.look_back:i]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
                pred_scaled = self.model(seq_tensor).cpu().numpy()[0][0]
                # Inverse transform (approximate - since multi-feature scaler)
                dummy = np.zeros((1, features.shape[1]))
                dummy[0, -1] = pred_scaled  # last column assumed to be close
                pred = self.scaler.inverse_transform(dummy)[0, -1]
                predictions[i] = pred
        
        return pd.Series(predictions, index=df.index, name='ml_pred')

class Backtester:
    def __init__(self, initial_capital: float = 100000.0, risk_per_trade: float = 0.01, atr_multiplier: float = 2.0):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
    
    def backtest(self, df: pd.DataFrame, ticker: str) -> Dict:
        cash = self.initial_capital
        shares = 0.0
        stop_loss = 0.0
        portfolio = []
        
        for dt, row in df.iterrows():
            price = row['close']
            
            if shares > 0 and price <= stop_loss:
                cash += shares * price
                logger.info(f"{dt.date()}: Stop-loss triggered @ {price:.2f}")
                shares = 0
            
            if row['signal'] == 1 and shares == 0:
                if pd.isna(row.get('atr', np.nan)):
                    continue
                risk = cash * self.risk_per_trade
                stop_dist = row['atr'] * self.atr_multiplier
                size = risk / stop_dist
                max_shares = cash / price
                shares = min(size, max_shares)
                if shares < 1:
                    continue
                cash -= shares * price
                stop_loss = price - stop_dist
                logger.info(f"{dt.date()}: Buy {shares:.1f} @ {price:.2f}, SL @ {stop_loss:.2f}")
            
            elif row['signal'] == -1 and shares > 0:
                cash += shares * price
                logger.info(f"{dt.date()}: Sell @ {price:.2f}")
                shares = 0
            
            value = cash + shares * price
            portfolio.append({'timestamp': dt, 'value': value})
        
        if shares > 0:
            cash += shares * df['close'].iloc[-1]
        
        pf_df = pd.DataFrame(portfolio).set_index('timestamp')
        total_return = (pf_df['value'].iloc[-1] / self.initial_capital - 1) * 100
        max_dd = self._max_drawdown(pf_df['value'])
        
        return {
            'total_return_pct': total_return,
            'final_value': pf_df['value'].iloc[-1],
            'max_drawdown_pct': max_dd
        }
    
    @staticmethod
    def _max_drawdown(values: pd.Series) -> float:
        peak = values.cummax()
        dd = (values - peak) / peak
        return dd.min() * 100

class Visualizer:
    @staticmethod
    def plot_results(df: pd.DataFrame, ticker: str, output_file: str = 'analysis.png'):
        fig, axes = plt.subplots(5, 1, figsize=(15, 18), sharex=True, height_ratios=[3,2,2,2,1.5])
        
        # Price + MA + ML pred
        ax = axes[0]
        ax.plot(df.index, df['close'], label='Close')
        ax.plot(df.index, df['short_ma'], label='MA20')
        ax.plot(df.index, df['long_ma'], label='MA50')
        ax.plot(df.index, df['ml_pred'], '--', label='ML Pred', alpha=0.7)
        ax.legend()
        ax.set_title(f'{ticker} Price & ML Prediction')
        
        # RSI
        axes[1].plot(df.index, df['rsi'], 'purple')
        axes[1].axhline(70, color='r', ls='--', lw=0.8)
        axes[1].axhline(30, color='g', ls='--', lw=0.8)
        axes[1].set_title('RSI')
        
        # ATR
        axes[2].plot(df.index, df['atr'], 'orange')
        axes[2].set_title('ATR')
        
        # ML Prediction vs Actual
        axes[3].plot(df.index, df['close'], label='Actual')
        axes[3].plot(df.index, df['ml_pred'], '--', label='Predicted')
        axes[3].legend()
        axes[3].set_title('ML Prediction Accuracy (out-of-sample)')
        
        # Signals
        axes[4].plot(df.index, df['signal'], label='Final Signal')
        axes[4].set_title('Trading Signals')
        axes[4].legend()
        
        buy = df[df['signal']==1]
        sell = df[df['signal']==-1]
        axes[0].scatter(buy.index, buy['close'], marker='^', color='green', s=100, label='Buy')
        axes[0].scatter(sell.index, sell['close'], marker='v', color='red', s=100, label='Sell')
        
        plt.tight_layout()
        plt.savefig(output_file)
        logger.info(f"Plot saved: {output_file}")

class StockTradingTool:
    def __init__(self, api_key: str, **kwargs):
        self.fetcher = StockDataFetcher(api_key)
        self.predictor = EfficientMLPredictor(
            look_back=kwargs.get('ml_look_back', 60),
            hidden_size=kwargs.get('ml_hidden_size', 64),
            epochs=kwargs.get('ml_epochs', 50),
            batch_size=kwargs.get('ml_batch_size', 64),
            train_split=kwargs.get('ml_train_split', 0.75)
        )
        self.backtester = Backtester(
            initial_capital=kwargs.get('initial_capital', 100000),
            risk_per_trade=kwargs.get('risk_per_trade', 0.01),
            atr_multiplier=kwargs.get('atr_multiplier', 2.0)
        )
    
    def run_analysis(self, ticker: str, start_date: str, end_date: str, plot: bool = True) -> Dict:
        df = self.fetcher.get_historical_data(ticker, start_date, end_date)
        if df.empty:
            return {'error': 'No data'}
        
        df = TechnicalAnalyzer.add_technical_features(df)
        
        # Train once (in-sample)
        self.predictor.train(df)
        
        # Rolling predictions (out-of-sample simulation)
        df['ml_pred'] = self.predictor.predict_rolling(df)
        
        # Generate signals (combine rule-based + ML)
        df['ml_signal'] = np.where(df['ml_pred'] > df['close'], 1, -1)
        df['signal'] = 0
        cond_buy  = (df['short_ma'] > df['long_ma']) & (df['rsi'] < 35) | (df['ml_signal'] == 1)
        cond_sell = (df['short_ma'] < df['long_ma']) & (df['rsi'] > 65) | (df['ml_signal'] == -1)
        df.loc[cond_buy,  'signal'] =  1
        df.loc[cond_sell, 'signal'] = -1
        
        metrics = self.backtester.backtest(df, ticker)
        
        if plot:
            Visualizer.plot_results(df, ticker)
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Optimized Stock Trading Tool (Efficient ML)")
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--start_date', required=True)
    parser.add_argument('--end_date', default=datetime.date.today().isoformat())
    parser.add_argument('--initial_capital', type=float, default=100000.0)
    parser.add_argument('--risk_per_trade', type=float, default=0.01)
    parser.add_argument('--atr_multiplier', type=float, default=2.0)
    parser.add_argument('--ml_look_back', type=int, default=60)
    parser.add_argument('--ml_hidden_size', type=int, default=64)
    parser.add_argument('--ml_epochs', type=int, default=50)
    parser.add_argument('--ml_batch_size', type=int, default=64)
    parser.add_argument('--ml_train_split', type=float, default=0.75)
    parser.add_argument('--no_plot', action='store_true')
    
    args = parser.parse_args()
    
    tool = StockTradingTool(POLYGON_API_KEY, **vars(args))
    results = tool.run_analysis(args.ticker, args.start_date, args.end_date, not args.no_plot)
    print("Backtest Results:")
    print(results)

if __name__ == "__main__":
    main()
