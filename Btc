import yfinance as yf          # 匯入 Yahoo Finance 財經資訊庫
import pandas as pd            # 匯入 pandas 庫，用於數據處理與匯出 CSV/Excel
from datetime import datetime  # 匯入 datetime，用於記錄目前的日期與時間
import os                      # 匯入 os 模組，用於檢查檔案是否存在
import time                    # 匯入 time 模組，用於控制定時執行的等待時間

def fetch_and_save():
    # 定義要抓取的加密貨幣代號列表（比特幣、乙太幣、Solana）
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    # 設定存檔的檔案名稱
    filename = "crypto_history_log.csv"
    
    # 取得系統目前的日期與時間
    now = datetime.now()
    # 將時間格式化為 "年-月-日 時:分:秒" 的字串
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    # 在控制台印出目前的進度
    print(f"[{current_time}] 正在抓取最新價格...")
    
    # 建立一個空的列表，用來暫存這一次抓到的所有數據
    data_list = []
    
    # 開始遍歷代號列表，逐一抓取價格
    for symbol in symbols:
        try:
            # 建立該幣種的 Ticker 物件（Yahoo Finance 的數據接口）
            ticker = yf.Ticker(symbol)
            # 獲取最新成交價（使用 fast_info 屬性能在 2026 年獲得較快的響應速度）
            price = ticker.fast_info['last_price']
            
            # 將抓取到的時間、幣種、價格以字典格式存入列表，價格四捨五入到小數點後兩位
            data_list.append({
                "時間": current_time,
                "幣種": symbol,
                "價格(USD)": round(price, 2)
            })
        except Exception as e:
            # 如果抓取過程中發生錯誤（如網路斷線），印出錯誤訊息並跳過該幣種
            print(f"抓取 {symbol} 失敗: {e}")

    # 如果 data_list 裡面有數據（抓取成功），則進行存檔動作
    if data_list:
        # 將列表轉換為 Pandas 的 DataFrame 表格格式
        df = pd.DataFrame(data_list)
        # 檢查該 CSV 檔案是否已經存在於電腦中
        file_exists = os.path.exists(filename)
        # 寫入檔案：mode='a' 代表追加數據（不覆蓋舊內容）
        # header=not file_exists 代表只有檔案不存在（第一次建立）時才寫入標題欄
        # encoding='utf-8-sig' 確保 Excel 開啟時不會有中文亂碼
        df.to_csv(filename, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
        # 印出成功存檔的通知
        print(f"成功儲存 {len(data_list)} 筆數據至 {filename}\n")

# --- 設定自動執行 ---
# 設定抓取間隔：10 分鐘等於 600 秒
INTERVAL = 600 

# 在控制台顯示程式啟動提示
print("加密貨幣自動監控啟動... (按 Ctrl+C 可停止)")

try:
    # 進入無限循環，除非手動停止程式
    while True:
        # 執行抓取並存檔的函數
        fetch_and_save()
        # 提示下一次抓取的等待時間
        print(f"等待 {INTERVAL} 秒後進行下一次抓取...")
        # 讓程式暫停（睡眠）設定的秒數
        time.sleep(INTERVAL)
except KeyboardInterrupt:
    # 如果使用者按下鍵盤的 Ctrl + C，則優雅地結束程式
    print("\n程式已手動停止。")
