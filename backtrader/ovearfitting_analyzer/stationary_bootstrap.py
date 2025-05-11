#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2025 xAI
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
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import uniform


class StationaryBootstrapAnalyzer:
    """Analyzer for Stationary Bootstrap to assess backtest overfitting."""

    def __init__(self, T1=1000, T2=1000, q=0.1, n_bootstrap=1000, performance_metric='sharpe'):
        """
        Initialize the Stationary Bootstrap Analyzer.

        Parameters:
        - T1: In-sample period length (e.g., 1000 days)
        - T2: Out-of-sample period length (e.g., 1000 days)
        - q: Probability of jumping to a new block (average block length = 1/q)
        - n_bootstrap: Number of bootstrap iterations
        - performance_metric: Performance metric to use ('sharpe' supported)
        """
        self.T1 = T1
        self.T2 = T2
        self.q = q
        self.n_bootstrap = n_bootstrap
        self.performance_metric = performance_metric
        self.returns_matrix = None
        self.results = None

    def collect_returns(self, runstrats):
        """Collect returns matrix from runstrats."""
        returns_dict = {}
        try:
            for i, strat in enumerate(runstrats):
                if hasattr(strat.analyzers, 'timereturn'):
                    returns = strat.analyzers.timereturn.get_analysis()
                    returns_dict[f"Strategy_{i}"] = pd.Series(returns)
                else:
                    print(f"Warning: No TimeReturn analyzer for Strategy_{i}")

            self.returns_matrix = pd.DataFrame(returns_dict).fillna(0)
            print(f"The shape of returns matrix is {self.returns_matrix.shape}")
        except Exception as e:
            print(f"Error collecting returns: {e}")
            self.returns_matrix = pd.DataFrame()

    def get_analysis(self):
        """Run Stationary Bootstrap analysis and return results."""
        if self.returns_matrix is None or self.returns_matrix.empty:
            return {'pbo': None, 'haircut': None, 'loss_prob': None, 'message': 'No valid returns data'}

        try:
            T = self.returns_matrix.shape[0]
            if T < self.T1 + self.T2:
                return {
                    'pbo': None,
                    'haircut': None,
                    'loss_prob': None,
                    'message': f'Insufficient data: T={T} < T1+T2={self.T1 + self.T2}'
                }

            results = self._stationary_bootstrap_analysis(
                self.returns_matrix, self.T1, self.T2, self.q, self.n_bootstrap, self.performance_metric
            )
            self.results = results
            return results
        except Exception as e:
            return {
                'pbo': None,
                'haircut': None,
                'loss_prob': None,
                'message': f'Stationary Bootstrap analysis failed: {e}'
            }

    def plot_performance_degradation(self):
        """Plot in-sample vs out-of-sample performance degradation."""
        if self.results is None:
            print("No results available.")
            return None

        is_perf = np.array(self.results['sharpe_is'])
        oos_perf = np.array(self.results['sharpe_oos'])
        slope = self.results['performance_degradation']['slope']
        intercept = self.results['performance_degradation']['intercept']
        prob_loss = self.results['loss_prob']

        plt.figure(figsize=(10, 6))
        plt.scatter(is_perf, oos_perf, alpha=0.5, label='IS vs OOS')
        plt.plot(is_perf, slope * is_perf + intercept, color='red',
                 label=f'Slope = {slope:.2f}\nProb Loss = {prob_loss:.2f}')
        plt.xlabel('In-Sample Performance (Sharpe)')
        plt.ylabel('Out-of-Sample Performance (Sharpe)')
        plt.title('Performance Degradation Analysis')
        plt.legend()
        plt.grid(True)
        plt.savefig('performance_degradation.png')
        plt.close()

    def _stationary_bootstrap_analysis(self, returns_matrix, T1, T2, q, n_bootstrap, performance_metric):
        """Core Stationary Bootstrap analysis logic."""
        T, K = returns_matrix.shape
        total_length = T1 + T2
        sharpe_is_list = []
        sharpe_oos_list = []
        pbo_list = []
        loss_prob_list = []

        returns_matrix_np = returns_matrix.to_numpy()

        for _ in tqdm(range(n_bootstrap), desc="Processing Stationary Bootstrap iterations"):
                # Generate bootstrap sample
                bootstrap_data = np.zeros((total_length, K))
                t = 0
                i = np.random.randint(0, T)  # Initial index
                while t < total_length:
                    if t == 0 or np.random.random() < q:
                        i = np.random.randint(0, T)
                    else:
                        i = (i + 1) % T
                    bootstrap_data[t] = returns_matrix_np[i]
                    t += 1

                # Split into in-sample (T1) and out-of-sample (T2)
                data_is = bootstrap_data[:T1]
                data_oos = bootstrap_data[T1:total_length]

                # Compute in-sample performance
                if performance_metric == 'sharpe':
                    mean_is = np.mean(data_is, axis=0)
                    std_is = np.std(data_is, axis=0, ddof=1)
                    std_is[std_is == 0] = np.nan
                    sharpe_is = mean_is / std_is * np.sqrt(252)
                    best_strategy = np.nanargmax(sharpe_is)

                    # Compute out-of-sample performance for best strategy
                    mean_oos = np.mean(data_oos[:, best_strategy])
                    std_oos = np.std(data_oos[:, best_strategy], ddof=1)
                    sharpe_oos = mean_oos / std_oos * np.sqrt(252) if std_oos > 0 else 0

                    # Compute out-of-sample ranking
                    mean_oos_all = np.mean(data_oos, axis=0)
                    std_oos_all = np.std(data_oos, axis=0, ddof=1)
                    std_oos_all[std_oos_all == 0] = np.nan
                    sharpe_oos_all = mean_oos_all / std_oos_all * np.sqrt(252)
                    rank_oos = K - rankdata(sharpe_oos_all, method='average')[best_strategy] + 1

                    sharpe_is_list.append(sharpe_is[best_strategy])
                    sharpe_oos_list.append(sharpe_oos)
                    pbo_list.append(1 if rank_oos > K / 2 else 0)
                    loss_prob_list.append(1 if sharpe_oos < 0 else 0)

        # Compute summary statistics
        sharpe_is_mean = np.nanmean(sharpe_is_list)
        sharpe_oos_mean = np.nanmean(sharpe_oos_list)
        haircut = 1 - sharpe_oos_mean / sharpe_is_mean if sharpe_is_mean > 0 else np.nan
        pbo = np.mean(pbo_list)
        loss_prob = np.mean(loss_prob_list)

        # Performance degradation regression
        valid_mask = ~np.isnan(sharpe_is_list) & ~np.isnan(sharpe_oos_list)
        is_perf_clean = np.array(sharpe_is_list)[valid_mask]
        oos_perf_clean = np.array(sharpe_oos_list)[valid_mask]
        reg = LinearRegression().fit(is_perf_clean.reshape(-1, 1), oos_perf_clean)
        slope = reg.coef_[0]
        intercept = reg.intercept_

        return {
            'haircut': haircut,
            'pbo': pbo,
            'loss_prob': loss_prob,
            'sharpe_is': sharpe_is_list,
            'sharpe_oos': sharpe_oos_list,
            'performance_degradation': {'slope': slope, 'intercept': intercept}
        }

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    T, K = 1000, 200
    mean_returns = np.zeros(K)
    mean_returns[0] = 0.0005  # One strategy with positive mean (mimic strategy 7)
    cov_matrix = np.eye(K) * 0.01  # Simplified covariance
    data = np.random.multivariate_normal(mean_returns, cov_matrix, T)

    # Run stationary bootstrap
    sb = StationaryBootstrapAnalyzer(T1=500, T2=500, q=0.1, n_bootstrap=1000)  # Adjusted T1 and T2
    sb.returns_matrix = pd.DataFrame(data)
    results = sb.get_analysis()

    # Print results
    print(f"Haircut (Sharpe Ratio Discount): {results['haircut']:.3f}")
    print(f"PBO (Probability of Backtest Overfitting): {results['pbo']:.3f}")
    print(f"Loss Probability: {results['loss_prob']:.3f}")