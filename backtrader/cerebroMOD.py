#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2015-2023 Daniel Rodriguez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import collections
import itertools
import multiprocessing

try:  # For new Python versions
    collectionsAbc = collections.abc  # collections.Iterable -> collections.abc.Iterable
except AttributeError:  # For old Python versions
    collectionsAbc = collections  # Используем collections.Iterable

import backtrader as bt
from .utils.py3 import (map, range, zip, with_metaclass, string_types,
                        integer_types)

from . import linebuffer
from . import indicator
from .brokers import BackBroker
from .metabase import MetaParams
from . import observers
from .writer import WriterFile
from .utils import OrderedDict, tzparse, num2date, date2num
from .strategy import Strategy, SignalStrategy
from .tradingcal import (TradingCalendarBase, TradingCalendar,
                         PandasMarketCalendar)
from .timer import Timer
from .cscvAnalyze import CSCVAnalyzer


class cerebroMod(bt.Cerebro):
    def __init__(self):
        super().__init__()
        self.brokers = []  # 存储每个策略的 broker
        self.strat_broker_map = {}  # 映射策略索引到 broker
        self._broker_idx = 0  # 手动跟踪 broker 索引

    def addstrategy(self, strategy, *args, **kwargs):
        # 创建新的 broker
        broker = bt.brokers.BackBroker()
        broker.cerebro = self
        self.brokers.append(broker)
        # 使用手动索引，确保一致性
        idx = self._broker_idx
        self.strat_broker_map[idx] = broker
        self._broker_idx += 1
        # 调用父类的 addstrategy
        super().addstrategy(strategy, *args, **kwargs)
        return idx

    def runstrategies(self, iterstrat, predata=False):
        self._init_stcount()
        self.runningstrats = runstrats = []

        # 调试：打印初始状态
        print("strat_broker_map:", self.strat_broker_map)
        print("Brokers:", len(self.brokers))

        # 启动 stores 和 feeds
        for store in self.stores:
            store.start()
        for feed in self.feeds:
            feed.start()

        # 实例化策略并绑定 broker
        for stratcls, sargs, skwargs in iterstrat:
            sargs = self.datas + list(sargs)
            try:
                strat = stratcls(*sargs, **skwargs)
            except bt.errors.StrategySkipError:
                continue
            idx = len(runstrats)  # 使用运行时索引
            if idx not in self.strat_broker_map:
                raise KeyError(f"No broker mapped for strategy index {idx}")
            strat.broker = self.strat_broker_map[idx]  # 设置策略的 broker
            strat.broker.start()  # 启动 broker
            runstrats.append(strat)
            print(f"Strategy {strat.__class__.__name__} assigned broker for idx {idx}")

        # 添加观察者、分析器等
        if runstrats:
            defaultsizer = self.sizers.get(None, (None, None, None))
            for idx, strat in enumerate(runstrats):
                if self.p.stdstats:
                    strat._addobserver(False, bt.observers.Broker)
                    strat._addobserver(True, bt.observers.BuySell)
                    strat._addobserver(False, bt.observers.Trades)
                for multi, obscls, obsargs, obskwargs in self.observers:
                    strat._addobserver(multi, obscls, *obsargs, **obskwargs)
                for indcls, indargs, indkwargs in self.indicators:
                    strat._addindicator(indcls, *indargs, **indkwargs)
                for ancls, anargs, ankwargs in self.analyzers:
                    strat._addanalyzer(ancls, *anargs, **ankwargs)
                sizer, sargs, skwargs = self.sizers.get(idx, defaultsizer)
                if sizer is not None:
                    strat._addsizer(sizer, *sargs, **skwargs)
                strat._start()

        # 初始化数据
        if not predata:
            for data in self.datas:
                data.reset()
                if self._exactbars < 1:
                    data.extend(size=self.params.lookahead)
                data._start()
                if self._dopreload:
                    data.preload()

        # 运行循环
        if self._dopreload and self._dorunonce:
            self._runonce(runstrats)
        else:
            self._runnext(runstrats)

        # 清理
        for strat in runstrats:
            strat._stop()
            strat.broker.stop()
        if not predata:
            for data in self.datas:
                data.stop()
        for feed in self.feeds:
            feed.stop()
        for store in self.stores:
            store.stop()

        ########################## cscv analysis ################################
        if self.cscv_analyzer:
            self.cscv_analyzer.collect_returns(runstrats)#获取return矩阵,储存在对象内部
            cscv_results = self.cscv_analyzer.get_analysis()
            return {'runstrats': runstrats, 'cscv_results': cscv_results}
          


        ########################## cscv analysis ################################



        return runstrats

    def _runnext(self, runstrats):
        datas = sorted(self.datas, key=lambda x: (x._timeframe, x._compression))
        while True:
            d0ret = any(data.next() for data in datas)
            if not d0ret:
                break
            for strat in runstrats:
                strat.broker.next()
                while True:
                    order = strat.broker.get_notification()
                    if order is None:
                        break
                    strat._addnotification(order, quicknotify=self.p.quicknotify)
                strat._next()
            self._next_writers(runstrats)

    def _runonce(self, runstrats):
        for strat in runstrats:
            strat._once()
            strat.reset()
        datas = sorted(self.datas, key=lambda x: (x._timeframe, x._compression))
        while True:
            dts = [d.advance_peek() for d in datas]
            dt0 = min(dts)
            if dt0 == float('inf'):
                break
            for i, dti in enumerate(dts):
                if dti <= dt0:
                    datas[i].advance()
            for strat in runstrats:
                strat.broker.next()
                strat._oncepost(dt0)
            self._next_writers(runstrats)