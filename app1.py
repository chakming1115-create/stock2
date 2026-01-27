import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier

# 1. 定義篩選函數
def screen_stocks(ticker_list):
    recommendations = []
    
    for ticker in ticker_list:
        try:
            # 下載歷史數據
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if len(df) < 50: continue

            # 2. 特徵工程：計算技術指標
            df['RSI'] = ta.rsi(df['Close'])
            df['EMA_20'] = ta.ema(df['Close'], length=20)
            df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
            
            # 定義標籤：未來 5 天收盤價是否高於當前收盤價
            df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
            
            # 準備訓練數據
            features = ['RSI', 'EMA_20', 'MACD']
            data = df.dropna()
            X = data[features]
            y = data['Target']
            
            # 3. 機器學習模型預測
            # 訓練模型 (僅作範例，實際生產環境需進行滾動窗口回測)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X[:-5], y[:-5])  # 排除最後5天以防標籤洩漏
            
            # 預測最新狀態
            latest_data = X.tail(1)
            prediction_prob = model.predict_proba(latest_data)[0][1] # 上漲機率
            
            if prediction_prob > 0.65:  # 篩選上漲機率大於 65% 的股票
                recommendations.append({
                    "Ticker": ticker,
                    "Bullish_Prob": f"{prediction_prob:.2%}",
                    "Current_Price": df['Close'].iloc[-1]
                })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            
    return pd.DataFrame(recommendations)

# 測試篩選美股清單
mystocks = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"]
results = screen_stocks(mystocks)
print(results)
