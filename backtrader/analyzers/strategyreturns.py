

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
import backtrader as bt
class strategyreturns(bt.Analyzer):
    def __init__(self):
        self.start_value = None
        self.end_value = None
        self.returns = []

    def start(self):
        self.start_value = self.strategy.broker.getvalue()

    def next(self):
        value = self.strategy.broker.getvalue()
        if self.returns:
            ret = (value - self.prev_value) / self.prev_value
            self.returns.append(ret)
        else:
            self.returns.append(0.0)
        self.prev_value = value

    def stop(self):
        self.end_value = self.strategy.broker.getvalue()

    def get_analysis(self):
        total_return = (self.end_value - self.start_value) / self.start_value
        return {
            'start_value': self.start_value,
            'end_value': self.end_value,
            'total_return': total_return,
            'daily_returns': self.returns
        }
