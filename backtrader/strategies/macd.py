
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import backtrader as bt
import backtrader.indicators as btind
class MACDStrategy(bt.Strategy):
    # 定义参数
    params = (
        ('macd_fast', 26),    # MACD 快线周期
        ('macd_slow', 50),    # MACD 慢线周期
        ('macd_signal', 13),  # MACD 信号线周期
    )

    def log(self, txt, dt=None):
        """日志记录函数，便于调试"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def __init__(self):
        # 初始化 MACD 指标
        self.macd = bt.indicators.MACD(
            self.datas[0].close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        # 用于记录账户价值历史（供 CSCVAnalyzer 使用）
        self.value_history = []
        # 跟踪订单
        self.order = None

    def notify_order(self, order):
        """处理订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            # 订单已提交或接受，暂不处理
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}')
            self.position_changed = True

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # 重置订单状态
        self.order = None

    def prenext(self):
        """在指标预热期运行"""
        # 记录账户价值，即使指标未准备好
        self.value_history.append(self.broker.getvalue())

    def next(self):
        """主逻辑，每根K线执行一次"""
        # 记录当前账户价值
        self.value_history.append(self.broker.getvalue())

        # 如果有未完成的订单，跳过
        if self.order:
            return

        # 获取当前现金和价格
        cash = self.broker.get_cash()
        price = self.datas[0].close[0]

        # 检查价格是否有效
        if price <= 0:
            return

        # 计算可购买的数量
        size = int(cash / price)

        # MACD 交易逻辑
        if self.macd.macd[0] > self.macd.signal[0] and not self.position:
            # MACD 上穿信号线，且无持仓，买入
            if size > 0:  # 确保有足够现金
                self.order = self.buy(size=size)
                self.log(f'BUY ORDER CREATED, Size: {size}, Price: {price:.2f}')
        elif self.macd.macd[0] < self.macd.signal[0] and self.position:
            # MACD 下穿信号线，且有持仓，卖出
            self.order = self.sell(size=self.position.size)
            self.log(f'SELL ORDER CREATED, Size: {self.position.size}, Price: {price:.2f}')

# 测试代码
if __name__ == '__main__':
    cerebro = bt.Cerebro()
    data = bt.feeds.YahooFinanceData(
        dataname='AAPL',
        fromdate=datetime(2020, 1, 1),
        todate=datetime(2023, 1, 1)
    )
    cerebro.adddata(data)
    cerebro.addstrategy(MACDStrategy)
    cerebro.broker.setcash(10000.0)
    cerebro.run()