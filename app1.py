import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from streamlit_gsheets import GSheetsConnection

# 1. 頁面配置
st.set_page_config(page_title="AI 專業股票決策系統 2026", layout="wide")

# --- 雲端數據庫連接 (Google Sheets) ---
conn = st.connection("gsheets", type=GSheetsConnection)

def load_cloud_history(user_name):
    """從 Google Sheets 獲取該用戶的歷史記錄"""
    try:
        # 讀取現有數據
        df = conn.read(spreadsheet=st.secrets["https://docs.google.com/spreadsheets/d/14LNhM4VyMgTn-OJ4_11vFRMtm1VAlklqfTq-4dvfHt8"], ttl=0)
        user_data = df[df['user_name'] == user_name]
        if not user_data.empty:
            # 取得 history 欄位並轉回 list (假設儲存格式為 "AAPL,TSLA,BTC-USD")
            history_str = str(user_data.iloc[0]['history'])
            return history_str.split(',') if history_str else []
    except Exception:
        pass
    return []

def save_cloud_history(user_name, history_list):
    """將更新後的歷史記錄存回 Google Sheets"""
    try:
        # 讀取全部數據
        df = conn.read(spreadsheet=st.secrets["https://docs.google.com/spreadsheets/d/14LNhM4VyMgTn-OJ4_11vFRMtm1VAlklqfTq-4dvfHt8"], ttl=0)
        history_str = ",".join(history_list)
        
        if user_name in df['user_name'].values:
            # 更新現有用戶
            df.loc[df['user_name'] == user_name, 'history'] = history_str
        else:
            # 新增用戶
            new_row = pd.DataFrame([{"user_name": user_name, "history": history_str}])
            df = pd.concat([df, new_row], ignore_index=True)
        
        # 寫回雲端
        conn.update(spreadsheet=st.secrets["https://docs.google.com/spreadsheets/d/14LNhM4VyMgTn-OJ4_11vFRMtm1VAlklqfTq-4dvfHt8"], data=df)
    except Exception as e:
        st.error(f"雲端存檔失敗: {e}")

# --- 初始化 Session State ---
if 'user_name' not in st.session_state:
    st.session_state.user_name = "訪客"

# 首次運行或切換用戶時載入雲端數據
if 'history' not in st.session_state:
    st.session_state.history = load_cloud_history(st.session_state.user_name)

# --- 側邊欄：用戶系統 ---
st.sidebar.header("👤 用戶帳戶")
user_input = st.sidebar.text_input("請輸入用戶名 (Enter確認)", value=st.session_state.user_name)

if user_input != st.session_state.user_name:
    st.session_state.user_name = user_input
    st.session_state.history = load_cloud_history(user_input)
    st.rerun()

st.sidebar.write(f"當前用戶: *{st.session_state.user_name}*")

# --- 側邊欄：歷史記錄 ---
st.sidebar.markdown("---")
st.sidebar.header("⏳ 雲端同步歷史 (前10次)")
if st.session_state.history:
    for h_ticker in st.session_state.history:
        if st.sidebar.button(f"🔍 {h_ticker}", key=f"btn_{h_ticker}"):
            st.session_state.current_ticker = h_ticker
else:
    st.sidebar.write("尚無查詢記錄")

# --- 主界面輸入 ---
st.title("📈 專家級 AI 股票分析與預測 (雲端版)")
default_ticker = st.session_state.get('current_ticker', 'AAPL')
ticker = st.text_input("輸入股票代號 (如: 2330.TW, TSLA)", value=default_ticker).upper()

def update_history(new_ticker):
    if new_ticker not in st.session_state.history:
        st.session_state.history.insert(0, new_ticker)
        st.session_state.history = st.session_state.history[:10]
        # 同步到雲端
        save_cloud_history(st.session_state.user_name, st.session_state.history)

# --- 核心分析函數 ---
def analyze_stock(symbol):
    update_history(symbol)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400) # 多拿一點數據計算MA
    
    data = yf.download(symbol, start=start_date, progress=False)
    if data.empty:
        st.error("找不到該股票數據，請檢查代號。")
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    data['ATR'] = np.max(ranges, axis=1).rolling(14).mean()

    current_price = float(data['Close'].iloc[-1])
    current_atr = float(data['ATR'].iloc[-1])
    recent_changes = data['Close'].pct_change(5).dropna()
    prob_up = (recent_changes[recent_changes > 0].count() / len(recent_changes)) * 100
    
    return {
        "df": data, "price": current_price, "ma50": float(data['MA50'].iloc[-1]),
        "ma200": float(data['MA200'].iloc[-1]), "buy": current_price - (current_atr * 1.5),
        "sell": current_price + (current_atr * 2.0), "prob": prob_up
    }

# --- 執行分析 ---
if st.button("啟動專業分析"):
    with st.spinner('AI 引擎正在存取雲端數據並運算中...'):
        res = analyze_stock(ticker)
        if res:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("當前股價", f"${res['price']:.2f}")
            col2.metric("預計買入價", f"${res['buy']:.2f}")
            col3.metric("預計目標價", f"${res['sell']:.2f}")
            col4.metric("5日上升機率", f"{res['prob']:.1f}%")

            st.markdown("---")
            st.subheader("💡 投資策略建議")
            c1, c2 = st.columns(2)
            with c1:
                if res['price'] > res['ma50'] > res['ma200']:
                    st.success("🔥 強勢多頭：建議拉回買進。")
                elif res['price'] < res['ma50'] < res['ma200']:
                    st.error("❄️ 強勢空頭：避開或考慮空單。")
                else:
                    st.warning("⚖️ 震盪盤整：建議觀望。")
            with c2:
                st.info(f"根據過去一年經驗，{ticker} 在未來 5 天看漲機率為 *{res['prob']:.1f}%*。")

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(res['df']['Close'], label='股價', color='black', alpha=0.7)
            ax.plot(res['df']['MA50'], label='50MA', color='blue')
            ax.plot(res['df']['MA200'], label='200MA', color='red')
            ax.set_title(f"{ticker} 趨勢分析 (2026)")
            ax.legend()
            st.pyplot(fig)

# --- 頁尾 ---
st.markdown("---")
st.caption(f"當前雲端用戶: {st.session_state.user_name} | 數據已自動同步至 Google Sheets")

