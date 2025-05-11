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
import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from random import uniform
import matplotlib.pyplot as plt

class PCSCVAnalyzer():
    def __init__(self, S =10 ,purge_period=0, embargo_period=0, performance_metric='sharpe',optstrats =False):
        """
        Initialize the Purged CSCV estimator.
        
        Parameters:
        - S: int, number of blocks to split the data into
        - performance_metric: str, performance metric to use ('sharpe', 'mean', 'win_rate', 'sortino')
        - purge_period: int, number of periods to purge before/after test set to avoid leakage
        - embargo_period: int, number of periods to embargo after test set
        - optstrats : 
        """
        self.S = S
        self.purge_period = purge_period
        self.embargo_period = embargo_period
        self.performance_metric = performance_metric
        self.optstrats = optstrats
        self.returns_matrix = None
        self.results = None



        


    def collect_returns(self, runstrats):
        """Collect returns matrix from runstrats."""
        returns_dict = {}
        try:
            if self.optstrats or (runstrats and isinstance(runstrats[0], list)):
               
                for i, strat_list in enumerate(runstrats):
                  
                    for j,strat in enumerate(strat_list):
                       
                        if hasattr(strat.analyzers, 'timereturn'):
                            returns = strat.analyzers.timereturn.get_analysis()
                            returns_dict[f"Strat_{i}_{j}"] = pd.Series(returns)
                        else:
                            print(f"Warning: No TimeReturn analyzer for Strat_{i}_{j}")
               
            self.returns_matrix = pd.DataFrame(returns_dict).fillna(0)
            print(f"The shape of returns matrix is {self.returns_matrix.shape}")
        except Exception as e:
            print(f"Error collecting returns: {e}")
            self.returns_matrix = pd.DataFrame()

    def plot_performance_degradation(self):
        if self.results is None:
            print("No results available.")
            return self.results
        is_perf = self.results['is_perf']
        oos_perf = self.results['oos_perf']
        slope = self.results['performance_degradation']['slope']
        intercept = self.results['performance_degradation']['intercept']
        prob_loss = self.results['performance_degradation']['prob_loss']
    
        plt.figure(figsize=(10, 6))
        plt.scatter(is_perf, oos_perf, alpha=0.5, label='IS vs OOS')
        plt.plot(is_perf, slope * is_perf + intercept, color='red', 
             label=f'Slope = {slope:.2f}\nProb Loss = {prob_loss:.2f}')
        plt.xlabel('In-Sample Performance (Sharpe)')
        plt.ylabel('Out-of-Sample Performance (Sharpe)')
        plt.title('Performance Degradation Analysis')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_stochastic_dominance(self):
        if self.results is None:
            print("No results available.")
            return self.results
        x = self.results['stochastic_dominance']['x_range']
        cdf_opt = self.results['stochastic_dominance']['cdf_optimized']
        cdf_rand = self.results['stochastic_dominance']['cdf_random']
        ssd_integral = self.results['stochastic_dominance']['ssd_integral']
        fsd = self.results['stochastic_dominance']['fsd']
        ssd = self.results['stochastic_dominance']['ssd']
    
        plt.figure(figsize=(12, 6))
    
        plt.subplot(1, 2, 1)
        plt.plot(x, cdf_opt, label='Optimized Strategy', color='red')
        plt.plot(x, cdf_rand, label='Random Strategy', color='blue')
        plt.xlabel('Performance (Sharpe)')
        plt.ylabel('CDF')
        plt.title(f'CDF Comparison (FSD-check: {fsd})')
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(x, ssd_integral, color='green', label='SSD Integral')
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel('Performance (Sharpe)')
        plt.ylabel('SSD Integral')
        plt.title(f'Second-Order Dominance (SSD-check: {ssd})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _pcscv_analysis(self, returns_matrix, S,  performance_metric='sharpe'):
        """
        Perform PCSCV analysis on the returns matrix.
        
        Parameters:
        - returns_matrix: pd.DataFrame, matrix of strategy returns (rows: time, columns: strategies)
        - S: int, number of blocks to split the data into
        - performance_metric: str, performance metric to evaluate
        
        Returns:
        - dict, containing PBO, performance metrics, and other analysis results
        """
    
        T = returns_matrix.shape[0]
        N = returns_matrix.shape[1]
        logits = []
        is_perf_list = []
        oos_perf_list = []


        correlation_matrix = returns_matrix.corr()

        block_size = T // S
        #blocks = [returns_matrix.iloc[i * block_size: (i + 1) * block_size] for i in range(S)]
        
        #Generate all combinations of S/2 blocks for training
        all_combinations = list(combinations(range(S), S // 2))
        num_combinations = len(all_combinations)

        for c in tqdm(all_combinations, desc="Processing PCSCV combinations"):
            #Select training and test blocks
            train_indices = []
            test_indices = []
            for i in range(S):
                if i in c:
                    train_indices.extend(range(i * block_size, (i + 1) * block_size))
                else:
                    test_indices.extend(range(i * block_size, (i + 1) * block_size))
            
        
            purge_indices = []
            for idx in test_indices:
                start_purge = max(0, idx - self.purge_period)
                end_purge = min(T, idx + self.purge_period + 1)
                purge_indices.extend(range(start_purge, end_purge))
            

            embargo_indices = []
            for idx in test_indices:
                embargo_start = idx + 1
                embargo_end = min(T, embargo_start + self.embargo_period)
                embargo_indices.extend(range(embargo_start, embargo_end))
            
            #Combine purge and embargo indices
            exclude_indices = set(purge_indices + embargo_indices)
            train_indices = [idx for idx in train_indices if idx not in exclude_indices]
            
            #Create training and test sets
            train_set = returns_matrix.iloc[train_indices]
            test_set = returns_matrix.iloc[test_indices]


            #Compute performance metrics
            if performance_metric == 'sharpe':
                is_std = train_set.std()
                oos_std = test_set.std()
                is_std[is_std == 0] = np.nan
                oos_std[oos_std == 0] = np.nan
           
                is_perf = train_set.mean() / is_std * np.sqrt(252)
                oos_perf = test_set.mean() / oos_std * np.sqrt(252)

                is_perf = is_perf.fillna(uniform(5e-5, 2e-4))
                oos_perf = oos_perf.fillna(uniform(5e-5, 2e-4))
            elif performance_metric == 'mean':
                is_perf = train_set.mean()
                oos_perf = test_set.mean()
            elif performance_metric == 'win_rate':
                is_perf = (train_set > 0).mean()
                oos_perf = (test_set > 0).mean()
            elif performance_metric == 'sortino':
                neg_train = train_set[train_set < 0]
                neg_test = test_set[test_set < 0]
                is_down = neg_train.std() if len(neg_train) > 1 else pd.Series(np.nan, index=train_set.columns)
                oos_down = neg_test.std() if len(neg_test) > 1 else pd.Series(np.nan, index=test_set.columns)
 
                is_perf = train_set.mean() / is_down * np.sqrt(252) if pd.notna(is_down).all() and (is_down > 1e-10).all() else pd.Series(np.nan, index=train_set.columns)
                oos_perf = test_set.mean() / oos_down * np.sqrt(252) if pd.notna(oos_down).all() and (oos_down > 1e-10).all() else pd.Series(np.nan, index=test_set.columns)

            #Select best strategy based on in-sample performance
            best_strategy = is_perf.idxmax()
            oos_perf_best = oos_perf[best_strategy] if not pd.isna(best_strategy) else 0

            #Performance metrics
            oos_perf_list.append(oos_perf_best)
            is_perf_list.append(is_perf.max())

            #Compute PBO logit
            oos_rank = rankdata(-oos_perf)

            best_rank = oos_rank[oos_perf.index.get_loc(best_strategy)]
     
            omega = best_rank / (N + 1)
            logit = np.log(omega / (1 - omega)) if omega != 1 else -np.inf
            logits.append(logit)

        #Compute PBO
        pbo = np.mean(np.array(logits) < 0)
        is_perf_all = np.array(is_perf_list)
        oos_perf_all = np.array(oos_perf_list)

        #Performance degradation
        valid_mask = ~np.isnan(oos_perf_all) & ~np.isnan(is_perf_all)
        is_perf_all_clean = is_perf_all[valid_mask]
        oos_perf_all_clean = oos_perf_all[valid_mask]
        reg = LinearRegression().fit(is_perf_all_clean.reshape(-1, 1), oos_perf_all_clean)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        prob_loss = np.mean(oos_perf_all_clean < 0)

        #Stochastic dominance
        random_perf = [oos_perf[np.random.randint(0, N)] for _ in range(num_combinations)]
        
        def compute_cdf(data, x_range):
            sorted_data = np.sort(data)
            return np.searchsorted(sorted_data, x_range, side='right') / len(data)
        
        x_min = min(np.min(oos_perf_all_clean), np.min(random_perf))
        x_max = max(np.max(oos_perf_all_clean), np.max(random_perf))
        x = np.linspace(x_min, x_max, 1000)
        cdf_optimized = compute_cdf(oos_perf_all_clean, x)
        cdf_random = compute_cdf(random_perf, x)

        fsd = np.all(cdf_optimized <= cdf_random) & np.any(cdf_optimized < cdf_random)
        ssd_integral = np.cumsum(cdf_random - cdf_optimized) * (x[1] - x[0])
        ssd = np.all(ssd_integral >= 0) & np.any(ssd_integral > 0)

        self.results = {
            'pbo': pbo,
            'is_perf': is_perf_all_clean,
            'oos_perf': oos_perf_all_clean,
            'performance_degradation': {'slope': slope, 'intercept': intercept, 'prob_loss': prob_loss},
            'stochastic_dominance': {
                'fsd': fsd,
                'ssd': ssd,
                'cdf_optimized': cdf_optimized,
                'cdf_random': cdf_random,
                'x_range': x,
                'ssd_integral': ssd_integral
            },
            'correlation': correlation_matrix
        }

        return self.results

    def get_analysis(self, returns_matrix=None, S=16 ):
        """
        Fit the PCSCV model to the returns matrix.
        
        Parameters:
        - returns_matrix: pd.DataFrame, matrix of strategy returns
        - S: int, number of blocks to split the data into
        """
        if self.returns_matrix is None:
            self.returns_matrix = returns_matrix
        if self.S is None:
            self.S = S


        self.results = self._pcscv_analysis(self.returns_matrix, self.S,  self.performance_metric)
        return self.results

    
   