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
from scipy.stats import invwishart
import matplotlib.pyplot as plt

class BayesianAnalyzer():
    def __init__(self, T1, T2, n_gibbs=1500, warm_up=500, n_sim=1000,optstrats=False):
        """
        Initialize the Bayesian backtest overfitting analyzer.

        Parameters:
        - returns_matrix: Pandas DataFrame or NumPy array of shape (T, N) with strategy returns
        - T1: Number of in-sample periods
        - T2: Number of out-of-sample periods
        - n_gibbs: Number of Gibbs sampling iterations
        - warm_up: Number of burn-in iterations to discard
        - n_sim: Number of Monte Carlo simulations
        """

        self.T1 = T1
        self.T2 = T2
        self.n_gibbs = n_gibbs
        self.warm_up = warm_up
        self.n_sim = n_sim
        self.optstrats = optstrats
        self.returns_matrix = None
        self.results = {}

    def collect_returns(self, runstrats):
        """Collect returns matrix from runstrats."""
        returns_dict = {}
        try:
            if self.optstrats or (runstrats and isinstance(runstrats[0], list)):
                for i,strat_list in enumerate(runstrats):
                    for j,strat in enumerate(strat_list):
                        if hasattr(strat.analyzers, 'timereturn'):
                            returns = strat.analyzers.timereturn.get_analysis()
                            returns_dict[f"Strat_{i}_{j}"] = pd.Series(returns)
                        else:
                            print(f"Warning: No TimeReturn analyzer ")

            self.returns_matrix = pd.DataFrame(returns_dict).fillna(0).values
            print(f"The shape of returns matrix is {self.returns_matrix.shape}")
        except Exception as e:
            print(f"Error collecting returns: {e}")
            self.returns_matrix = pd.DataFrame()

    def naive_bayes_analyze(self):
        """
        Perform Bayesian analysis using Naive Model (Model 1) assuming multivariate normal returns.
        """
        N = self.returns_matrix.shape[1]

        mu = np.mean(self.returns_matrix, axis=0)
        Sigma = np.cov(self.returns_matrix, rowvar=False)
        mu_samples = []
        Sigma_samples = []

        # Gibbs sampling
        for i in range(self.n_gibbs):
            # Sample mu | Sigma, data
            r_bar = np.mean(self.returns_matrix, axis=0)
            mu_cov = Sigma / self.T1
            mu = np.random.multivariate_normal(r_bar, mu_cov)

            # Sample Sigma | mu, data
            centered = self.returns_matrix - mu
            S = centered.T @ centered
            S = (S + S.T) / 2 + 1e-6 * np.eye(N)  # Ensure positive-definite
            if (self.returns_matrix.std() == 0).any():
                print("警告：某些策略的标准差为零，可能导致逆 Wishart 抽样失败。")
            Sigma = invwishart.rvs(df=self.T1, scale=S)

            if i >= self.warm_up:
                mu_samples.append(mu)
                Sigma_samples.append(Sigma)

        mu_samples = np.array(mu_samples)
        Sigma_samples = np.array(Sigma_samples)

        # Monte Carlo simulation for OOS performance
        sr_is_results = []
        sr_oos_results = []
        pbo_results = []
        ranks_oos = []

        for _ in range(self.n_sim):
            idx = np.random.randint(len(mu_samples))
            mu_sim = mu_samples[idx]
            Sigma_sim = Sigma_samples[idx]

            # Generate simulated returns for T1 + T2 periods
            simulated_returns = np.random.multivariate_normal(mu_sim, Sigma_sim, self.T1 + self.T2)
            returns_is = simulated_returns[:self.T1]
            returns_oos = simulated_returns[self.T1:]

            # Select best IS strategy based on Sharpe ratio
            sr_is = returns_is.mean(axis=0) / returns_is.std(axis=0, ddof=1)
            best_idx = np.argmax(sr_is)
            sr_is_results.append(sr_is[best_idx])

            # Calculate OOS Sharpe ratio for the best IS strategy
            sr_oos = returns_oos[:, best_idx].mean() / returns_oos[:, best_idx].std(ddof=1)
            sr_oos_results.append(sr_oos)

            # Calculate OOS rank and PBO
            sr_oos_all = returns_oos.mean(axis=0) / returns_oos.std(axis=0, ddof=1)
            rank_oos = np.sum(sr_oos_all > sr_oos) + 1  # Rank (1-based)
            ranks_oos.append(rank_oos / N)  # Normalized rank (0 to 1)
            pbo_results.append(rank_oos > N / 2)  # True if rank below median

        sr_is_results = np.array(sr_is_results)
        sr_oos_results = np.array(sr_oos_results)
        pbo_results = np.array(pbo_results)
        ranks_oos = np.array(ranks_oos)

        # Calculate results
        mean_is_sharpe = np.mean(sr_is_results)
        mean_oos_sharpe = np.mean(sr_oos_results)
        sharpe_haircut = (1 - mean_oos_sharpe / mean_is_sharpe if mean_is_sharpe != 0 else np.nan)
        prob_oos_loss = np.mean(sr_oos_results < 0)
        pbo = np.mean(pbo_results)
        mean_oos_rank = np.mean(ranks_oos)
        oos_sharpe_ci = np.percentile(sr_oos_results, [5, 95])
        oos_sharpe_std = np.std(sr_oos_results)

        results = {
            'oos_sharpe_samples': sr_oos_results,
            'mean_is_sharpe': mean_is_sharpe,
            'mean_oos_sharpe': mean_oos_sharpe,
            'sharpe_haircut': sharpe_haircut,
            'prob_oos_loss': prob_oos_loss,
            'pbo': pbo,
            'mean_oos_rank': mean_oos_rank,
            'oos_sharpe_ci': oos_sharpe_ci,
            'oos_sharpe_std': oos_sharpe_std
        }

    
        self.results['naive'] = results
     

    def bimodel_bayes_analyze(self, p0=0.95, m0=0.008, V0=0.0005):
        """
        Perform Bayesian analysis using Bimodal Model (Model 2) with latent indicators.

        Parameters:
        - p0: Prior probability of strategies being noise (gamma = 0)
        - m0: Prior mean of strategies with true alpha
        - V0: Prior variance of strategies with true alpha
        """

        N = self.returns_matrix.shape[1] 
        gamma = np.ones(N)
        mu = np.mean(self.returns_matrix, axis=0)
        Sigma = np.cov(self.returns_matrix, rowvar=False)
        mu_samples = []
        Sigma_samples = []
        gamma_samples = []

        # Gibbs sampling phase
        for i in range(self.n_gibbs):
            # Sample mu | Sigma, gamma, data
            mu = np.zeros(N)
            for j in range(N):
                sigma_jj = Sigma[j, j] + 1e-6
                yj = self.returns_matrix[:, j]
                ybar = yj.mean()

                if gamma[j] == 1:
                    post_var = 1 / (self.T1 / sigma_jj + 1 / V0)
                    post_mean = post_var * (self.T1 * ybar / sigma_jj + m0 / V0)
                    mu[j] = np.random.normal(post_mean, np.sqrt(post_var))
                else:
                    mu[j] = 0.0

            #  Sample gamma | mu, Sigma, data
            for j in range(N):
                sigma_jj = Sigma[j, j] + 1e-6
                yj = self.returns_matrix[:, j]
                ybar = yj.mean()

                # Marginal likelihood for gamma = 1
                var1 = sigma_jj / self.T1 + V0
                ll1 = np.exp(-0.5 * (ybar - m0) ** 2 / var1) / np.sqrt(2 * np.pi * var1)

                # Marginal likelihood for gamma = 0
                var0 = sigma_jj / self.T1
                ll0 = np.exp(-0.5 * ybar ** 2 / var0) / np.sqrt(2 * np.pi * var0)

                posterior_prob = (1 - p0) * ll1 / ((1 - p0) * ll1 + p0 * ll0 + 1e-12)
                gamma[j] = np.random.binomial(1, posterior_prob)

            #  Sample Sigma | mu, data
            centered = self.returns_matrix - mu
            S = centered.T @ centered
            S = (S + S.T) / 2 + 1e-6 * np.eye(N)
            Sigma = invwishart.rvs(df=self.T1, scale=S)

            if i >= self.warm_up:
                mu_samples.append(mu)
                Sigma_samples.append(Sigma)
                gamma_samples.append(gamma.copy())

        mu_samples = np.array(mu_samples)
        Sigma_samples = np.array(Sigma_samples)
        gamma_samples = np.array(gamma_samples)
        prob_discovery = gamma_samples.mean(axis=0)

        # Monte Carlo phase
        sr_is_results = []
        sr_oos_results = []
        pbo_results = []
        ranks_oos = []
        gamma_best = []

        for _ in range(self.n_sim):
            idx = np.random.randint(len(mu_samples))
            mu_sim = mu_samples[idx]
            Sigma_sim = Sigma_samples[idx]
            gamma_sim = gamma_samples[idx]

            # Generate simulated returns
            simulated_returns = np.random.multivariate_normal(mu_sim, Sigma_sim, self.T1 + self.T2)
            returns_is = simulated_returns[:self.T1]
            returns_oos = simulated_returns[self.T1:]

            # Select best IS strategy
            sr_is = returns_is.mean(axis=0) / returns_is.std(axis=0, ddof=1)
            best_idx = np.argmax(sr_is)
            sr_is_results.append(sr_is[best_idx])
            sr_oos = returns_oos[:, best_idx].mean() / returns_oos[:, best_idx].std(ddof=1)
            sr_oos_results.append(sr_oos)

            # Calculate PBO
            sr_oos_all = returns_oos.mean(axis=0) / returns_oos.std(axis=0, ddof=1)
            rank_oos = np.sum(sr_oos_all > sr_oos) + 1
            ranks_oos.append(rank_oos / N)
            pbo_results.append(rank_oos > N / 2)

            gamma_best.append(gamma_sim[best_idx])

        sr_is_results = np.array(sr_is_results)
        sr_oos_results = np.array(sr_oos_results)
        pbo_results = np.array(pbo_results)
        ranks_oos = np.array(ranks_oos)
        gamma_best = np.array(gamma_best)

        mean_is_sharpe = np.mean(sr_is_results)
        mean_oos_sharpe = np.mean(sr_oos_results)
        sharpe_haircut = (1 - mean_oos_sharpe / mean_is_sharpe if mean_is_sharpe != 0 else np.nan)
        prob_oos_loss = np.mean(sr_oos_results < 0)
        pbo = np.mean(pbo_results)
        mean_oos_rank = np.mean(ranks_oos)
        oos_sharpe_ci = np.percentile(sr_oos_results, [5, 95])
        oos_sharpe_std = np.std(sr_oos_results)
        fdr = 1 - np.mean(gamma_best)

        results = {
            'oos_sharpe_samples': sr_oos_results,
            'mean_is_sharpe': mean_is_sharpe,
            'mean_oos_sharpe': mean_oos_sharpe,
            'sharpe_haircut': sharpe_haircut,
            'prob_oos_loss': prob_oos_loss,
            'pbo': pbo,
            'mean_oos_rank': mean_oos_rank,
            'oos_sharpe_ci': oos_sharpe_ci,
            'oos_sharpe_std': oos_sharpe_std,
            'posterior_prob_discovery': prob_discovery,
            'fdr': fdr
        }


        self.results['bimodal'] = results


    def bimodal_bayes_analyze_tristate(self, p0=0.1, m0=0.0003, V0=0.0005):
        """
        Perform Bayesian analysis using Tristate Bimodal Model (gamma in {-1, 0, 1}).

        Parameters:
        - p0: Prior probability of strategies being non-noise (gamma = +/-1)
        - m0: Prior mean of true alpha strategies
        - V0: Prior variance of true alpha strategies
        """
        N = self.returns_matrix.shape[1]
        gamma = np.ones(N)
        mu = np.mean(self.returns_matrix, axis=0)
        Sigma = np.cov(self.returns_matrix, rowvar=False)
        mu_samples = []
        Sigma_samples = []
        gamma_samples = []

        for i in range(self.n_gibbs):
            #  Sample mu | Sigma, gamma, data
            mu = np.zeros(N)
            for j in range(N):
                sigma_jj = Sigma[j, j] + 1e-6
                yj = self.returns_matrix[:, j]
                ybar = yj.mean()

                if gamma[j] != 0:
                    post_var = 1 / (self.T1 / sigma_jj + 1 / V0)
                    post_mean = post_var * (self.T1 * ybar / sigma_jj + gamma[j] * m0 / V0)
                    mu[j] = np.random.normal(post_mean, np.sqrt(post_var))
                else:
                    mu[j] = 0.0

            #  Sample gamma | mu, Sigma, data
            for j in range(N):
                sigma_jj = Sigma[j, j] + 1e-6
                yj = self.returns_matrix[:, j]
                ybar = yj.mean()

                # Marginal likelihoods
                var1 = sigma_jj / self.T1 + V0
                var0 = sigma_jj / self.T1

                # gamma = 1
                ll1 = np.exp(-0.5 * (ybar - m0) ** 2 / var1) / np.sqrt(2 * np.pi * var1)

                #  gamma = -1
                llm1 = np.exp(-0.5 * (ybar + m0) ** 2 / var1) / np.sqrt(2 * np.pi * var1)

                #  gamma = 0
                ll0 = np.exp(-0.5 * (ybar) ** 2 / var0) / np.sqrt(2 * np.pi * var0)

                # Prior probs
                prior1 = p0 / 2
                priorm1 = p0 / 2
                prior0 = 1 - p0

                # Unnormalized posteriors
                w1 = prior1 * ll1
                wm1 = priorm1 * llm1
                w0 = prior0 * ll0
                total = w1 + wm1 + w0 + 1e-12
                probs = np.array([wm1, w0, w1]) / total

                # Sample gamma from categorical 
                gamma[j] = np.random.choice([-1, 0, 1], p=probs)

            #  Sample Sigma | mu, data
            centered = self.returns_matrix - mu
            S = centered.T @ centered
            S = (S + S.T) / 2 + 1e-6 * np.eye(N)
            Sigma = invwishart.rvs(df=self.T1, scale=S)

            if i >= self.warm_up:
                mu_samples.append(mu)
                Sigma_samples.append(Sigma)
                gamma_samples.append(gamma.copy())

        mu_samples = np.array(mu_samples)
        Sigma_samples = np.array(Sigma_samples)
        gamma_samples = np.array(gamma_samples)
        prob_discovery = np.mean(np.abs(gamma_samples), axis=0)

        # Monte Carlo 
        sr_is_results = []
        sr_oos_results = []
        pbo_results = []
        ranks_oos = []
        gamma_best = []

        for _ in range(self.n_sim):
            idx = np.random.randint(len(mu_samples))
            mu_sim = mu_samples[idx]
            Sigma_sim = Sigma_samples[idx]
            gamma_sim = gamma_samples[idx]

            simulated_returns = np.random.multivariate_normal(mu_sim, Sigma_sim, self.T1 + self.T2)
            returns_is = simulated_returns[:self.T1]
            returns_oos = simulated_returns[self.T1:]

            sr_is = returns_is.mean(axis=0) / returns_is.std(axis=0, ddof=1)
            best_idx = np.argmax(sr_is)
            sr_is_results.append(sr_is[best_idx])

            sr_oos = returns_oos[:, best_idx].mean() / returns_oos[:, best_idx].std(ddof=1)
            sr_oos_results.append(sr_oos)

            sr_oos_all = returns_oos.mean(axis=0) / returns_oos.std(axis=0, ddof=1)
            rank_oos = np.sum(sr_oos_all > sr_oos) + 1
            ranks_oos.append(rank_oos / N)
            pbo_results.append(rank_oos > N / 2)

            gamma_best.append(gamma_sim[best_idx])

        sr_is_results = np.array(sr_is_results)
        sr_oos_results = np.array(sr_oos_results)
        pbo_results = np.array(pbo_results)
        ranks_oos = np.array(ranks_oos)
        gamma_best = np.array(gamma_best)

        mean_is_sharpe = np.mean(sr_is_results)
        mean_oos_sharpe = np.mean(sr_oos_results)
        sharpe_haircut = (1 - mean_oos_sharpe / mean_is_sharpe if mean_is_sharpe != 0 else np.nan)
        prob_oos_loss = np.mean(sr_oos_results < 0)
        pbo = np.mean(pbo_results)
        mean_oos_rank = np.mean(ranks_oos)
        oos_sharpe_ci = np.percentile(sr_oos_results, [5, 95])
        oos_sharpe_std = np.std(sr_oos_results)
        fdr = 1 - np.mean(np.abs(gamma_best))

        results = {
            'oos_sharpe_samples': sr_oos_results,
            'mean_is_sharpe': mean_is_sharpe,
            'mean_oos_sharpe': mean_oos_sharpe,
            'sharpe_haircut': sharpe_haircut,
            'prob_oos_loss': prob_oos_loss,
            'pbo': pbo,
            'mean_oos_rank': mean_oos_rank,
            'oos_sharpe_ci': oos_sharpe_ci,
            'oos_sharpe_std': oos_sharpe_std,
            'posterior_prob_discovery': prob_discovery,
            'fdr': fdr
        }

        self.results['bimodal_tristate'] = results


    def visualize_results(self, sr_oos_results, ranks_oos, model_name):
        """
        Visualize OOS Sharpe ratio distribution and OOS rank histogram.

        Parameters:
        - sr_oos_results: Array of OOS Sharpe ratios
        - ranks_oos: Array of normalized OOS ranks
        - model_name: Name of the model for plot titles,naive or bimodal
        """
        plt.figure(figsize=(12, 5))

        # Plot OOS Sharpe ratio distribution
        plt.subplot(1, 2, 1)
        plt.hist(sr_oos_results, bins=50, density=True, alpha=0.7, color='blue', label='OOS Sharpe')
        plt.axvline(0, color='red', linestyle='--', label='Zero')
        plt.title(f'{model_name}: OOS Sharpe Ratio Distribution')
        plt.xlabel('Sharpe Ratio')
        plt.ylabel('Density')
        plt.legend()

        # Plot OOS rank histogram
        plt.subplot(1, 2, 2)
        plt.hist(ranks_oos, bins=20, density=True, alpha=0.7, color='green', label='OOS Rank')
        plt.title(f'{model_name}: OOS Rank Distribution')
        plt.xlabel('Normalized Rank (0 to 1)')
        plt.ylabel('Density')
        plt.legend()

        plt.tight_layout()
        plt.show()


    def get_analysis(self,return_matrix = None):
        if return_matrix is not None:
            self.returns_matrix = return_matrix
        #self.naive_bayes_analyze()
        self.bimodel_bayes_analyze()
        #self.bimodal_bayes_analyze_tristate()
        return self.results

