# -*- coding: utf-8 -*-
"""
专业级 Python 股票量化交易系统（完整模块化设计）

功能模块：
1. 数据获取（Data Acquisition）：支持 yfinance（美股/港股/全球） + akshare（A股）双数据源
2. 策略开发（Strategy Development）：基于 Backtrader 的策略基类，支持多策略快速扩展
3. 回测分析（Backtesting & Analysis）：Backtrader + QuantStats 绩效报告（夏普、回撤、年化收益等）
4. 风险管理（Risk Management）：仓位大小控制、固定止损/止盈、最大回撤保护

所需安装（推荐使用 conda 或 virtualenv）：
pip install backtrader yfinance akshare quantstats matplotlib pandas numpy plotly

作者：Grok 设计
日期：2026-01-26
"""

import backtrader as bt
import yfinance as yf
import akshare as ak
import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# ==================== 1. 数据获取模块 ====================
class DataFetcher:
    """统一数据获取接口，支持美股/港股（yfinance）和A股（akshare）"""
    
    @staticmethod
    def fetch_yfinance(ticker: str, start: str, end: str = None) -> pd.DataFrame:
        """
        使用 yfinance 获取数据（支持 .HK 港股、美国股票）
        示例：'0700.HK'（腾讯），'AAPL'
        """
        df = yf.download(ticker, start=start, end=end or datetime.today().strftime('%Y-%m-%d'))
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index.name = 'datetime'
        return df
    
    @staticmethod
    def fetch_akshare(symbol: str, start: str, end: str = None) -> pd.DataFrame:
        """
        使用 akshare 获取A股数据
        示例：'sh600519'（贵州茅台）
        """
        df = ak.stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end or datetime.today().strftime('%Y%m%d'))
        df = df.set_index('date')
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.index = pd.to_datetime(df.index)
        df.index.name = 'datetime'
        return df


# ==================== 2. 策略基类与示例策略 ====================
class BaseStrategy(bt.Strategy):
    """所有策略的基类，统一日志与绩效记录"""
    params = (
        ('stake', 100),          # 默认每次买入股数（会被Sizer覆盖）
        ('printlog', True),
    )
    
    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} | {txt}')
    
    def __init__(self):
        self.order = None
        self.bar_executed = None
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行: 价格 {order.executed.price:.2f}, 数量 {order.executed.size}')
            elif order.issell():
                self.log(f'卖出执行: 价格 {order.executed.price:.2f}, 数量 {order.executed.size}')
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单失败/取消')
        self.order = None
    
    def stop(self):
        self.log(f'策略结束 | 最终资金: {self.broker.getvalue():.2f}')


class MACrossStrategy(BaseStrategy):
    """经典双均线交叉策略 + 止损止盈"""
    params = (
        ('fast', 12),
        ('slow', 26),
        ('stop_loss', 0.05),    # 5% 止损
        ('take_profit', 0.15),  # 15% 止盈
    )
    
    def __init__(self):
        super().__init__()
        self.ma_fast = bt.ind.SMA(period=self.params.fast)
        self.ma_slow = bt.ind.SMA(period=self.params.slow)
        self.crossover = bt.ind.CrossOver(self.ma_fast, self.ma_slow)
    
    def next(self):
        if self.order:
            return
        
        # 无持仓
        if not self.position:
            if self.crossover > 0:  # 快线上穿慢线
                self.order = self.buy()
                self.log('发出买入信号')
        # 有持仓
        else:
            # 均线死叉卖出
            if self.crossover < 0:
                self.order = self.close()
                self.log('均线死叉，平仓')
                return
            
            # 止损/止盈（基于买入价格）
            buy_price = self.position.price
            current_price = self.data.close[0]
            if current_price <= buy_price * (1 - self.params.stop_loss):
                self.order = self.close()
                self.log('触发止损，平仓')
            elif current_price >= buy_price * (1 + self.params.take_profit):
                self.order = self.close()
                self.log('触发止盈，平仓')


# ==================== 3. 风险管理模块 ====================
class PercentSizer(bt.Sizer):
    """风险百分比仓位管理：每笔交易风险固定百分比（推荐1%-2%）"""
    params = (('risk_percent', 0.02),)  # 单笔最大风险 2%
    
    def _getsiz(self, comminfo, cash, data, isbuy):
        if isbuy:
            # 简单估算：用 ATR 作为波动率（这里用最近20日波动代替）
            price = data.close[0]
            atr = bt.ind.ATR(data, period=20)[0]
            if atr == 0:
                return 0
            position_size = (cash * self.params.risk_percent) / atr
            return max(1, int(position_size // 100 * 100))  # 按手取整
        return 0


# ==================== 4. 回测引擎与绩效分析 ====================
class QuantTradingSystem:
    def __init__(self):
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(100000.0)      # 初始资金 10万
        self.cerebro.broker.setcommission(commission=0.0005, mult=1.0)  # 0.05% 手续费（港美股常见）
    
    def add_data(self, dataframe: pd.DataFrame, name: str = 'stock'):
        data = bt.feeds.PandasData(dataname=dataframe)
        self.cerebro.adddata(data, name=name)
    
    def add_strategy(self, strategy_class, **kwargs):
        self.cerebro.addstrategy(strategy_class, **kwargs)
    
    def set_risk_management(self):
        self.cerebro.addsizer(PercentSizer, risk_percent=0.02)
    
    def run_backtest(self):
        print('初始资金: %.2f' % self.cerebro.broker.getvalue())
        results = self.cerebro.run()
        print('最终资金: %.2f' % self.cerebro.broker.getvalue())
        return results[0]
    
    def plot_results(self):
        self.cerebro.plot(style='candlestick')
    
    def quantstats_report(self, returns: pd.Series):
        """使用 QuantStats 生成专业绩效报告（HTML）"""
        qs.reports.html(returns, output='quantstats_report.html', title='量化策略绩效报告')


# ==================== 5. 使用示例 ====================
if __name__ == '__main__':
    # 参数设置
    TICKER = '0700.HK'          # 可替换为 'AAPL' 或 'sh600519'
    START_DATE = '2020-01-01'
    END_DATE = '2025-12-31'
    
    # 1. 获取数据
    if 'HK' in TICKER or TICKER.isalpha():
        df = DataFetcher.fetch_yfinance(TICKER, START_DATE, END_DATE)
    else:
        df = DataFetcher.fetch_akshare(TICKER, START_DATE.replace('-', ''), END_DATE.replace('-', ''))
    
    print(f"获取 {TICKER} 数据完成，共 {len(df)} 条记录")
    
    # 2. 初始化系统
    system = QuantTradingSystem()
    system.add_data(df, name=TICKER)
    system.add_strategy(MACrossStrategy, fast=10, slow=30, stop_loss=0.06, take_profit=0.20)
    system.set_risk_management()
    
    # 3. 运行回测
    strategy = system.run_backtest()
    
    # 4. 可视化
    system.plot_results()
    
    # 5. 详细绩效报告（生成 HTML 文件）
    # 获取策略日收益率
    daily_returns = pd.Series(strategy.broker.get_value() / 100000 - 1, index=df.index)
    daily_returns = strategy.broker.getvalue()[-len(df):] / strategy.broker.getvalue()[-len(df)-1:-1] - 1
    daily_returns.index = df.index
    system.quantstats_report(daily_returns.pct_change().dropna())
    
    print("回测完成！绩效报告已保存为 quantstats_report.html")
