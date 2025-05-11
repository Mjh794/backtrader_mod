import numpy as np
import pandas as pd
from scipy.stats import invwishart
import matplotlib.pyplot as plt

class BayesAnalyzer():
    def __init__(self, returns_matrix, T1, T2, n_gibbs=1500, warm_up=500, n_sim=1000):
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
        self.returns_matrix = (returns_matrix.values if isinstance(returns_matrix, pd.DataFrame)
                              else returns_matrix)
        self.T1 = T1
        self.T2 = T2
        self.n_gibbs = n_gibbs
        self.warm_up = warm_up
        self.n_sim = n_sim
        self.N = self.returns_matrix.shape[1]  # Number of strategies
        self.results = {}

    def naive_bayes_analyze(self):
        """
        Perform Bayesian analysis using Naive Model (Model 1) assuming multivariate normal returns.
        """
        # Initialize parameters
        mu = np.mean(self.returns_matrix, axis=0)
        Sigma = np.cov(self.returns_matrix, rowvar=False)
        mu_samples = []
        Sigma_samples = []

        # Gibbs sampling
        for i in range(self.n_gibbs):
            
            r_bar = np.mean(self.returns_matrix, axis=0)
            mu_cov = Sigma / self.T1
            mu = np.random.multivariate_normal(r_bar, mu_cov) # Sample mu | Sigma, data

            centered = self.returns_matrix - mu
            S = centered.T @ centered
            S = (S + S.T) / 2 + 1e-6 * np.eye(self.N)  # Ensure positive-definite
            Sigma = invwishart.rvs(df=self.T1, scale=S) # Sample Sigma | mu, data

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
            ranks_oos.append(rank_oos / self.N)  # Normalized rank (0 to 1)
            pbo_results.append(rank_oos > self.N / 2)  # True if rank below median

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

        self.results = {
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

        # Visualize results
        self._visualize_results(sr_oos_results, ranks_oos, model_name='Naive Bayes')

        return self.results

    def bimodel_bayes_analyze(self, p0=0.7, m0=0.0001, V0=0.001):
        """
        Perform Bayesian analysis using Bimodal Model (Model 2) with latent indicators.

        Parameters:
        - p0: Prior probability of strategies being noise (gamma = 0)
        - m0: Prior mean of strategies with true alpha
        - V0: Prior variance of strategies with true alpha
        """
        # Initialize parameters
        gamma = np.ones(self.N)
        mu = np.mean(self.returns_matrix, axis=0)
        Sigma = np.cov(self.returns_matrix, rowvar=False)
        mu_samples = []
        Sigma_samples = []
        gamma_samples = []

        # Gibbs sampling
        for i in range(self.n_gibbs):
            # Step 1: Sample mu | Sigma, gamma, data
            mu = np.zeros(self.N)
            for j in range(self.N):
                sigma_jj = Sigma[j, j] + 1e-6
                yj = self.returns_matrix[:, j]
                ybar = yj.mean()

                if gamma[j] == 1:
                    post_var = 1 / (self.T1 / sigma_jj + 1 / V0)
                    post_mean = post_var * (self.T1 * ybar / sigma_jj + m0 / V0)
                    mu[j] = np.random.normal(post_mean, np.sqrt(post_var))
                else:
                    mu[j] = 0.0

            # Step 2: Sample gamma | mu, Sigma, data
            for j in range(self.N):
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

            # Step 3: Sample Sigma | mu, data
            centered = self.returns_matrix - mu
            S = centered.T @ centered
            S = (S + S.T) / 2 + 1e-6 * np.eye(self.N)
            Sigma = invwishart.rvs(df=self.T1, scale=S)

            if i >= self.warm_up:
                mu_samples.append(mu)
                Sigma_samples.append(Sigma)
                gamma_samples.append(gamma.copy())

        mu_samples = np.array(mu_samples)
        Sigma_samples = np.array(Sigma_samples)
        gamma_samples = np.array(gamma_samples)
        prob_discovery = gamma_samples.mean(axis=0)

        # Monte Carlo simulation for OOS performance
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

            # Calculate OOS Sharpe ratio
            sr_oos = returns_oos[:, best_idx].mean() / returns_oos[:, best_idx].std(ddof=1)
            sr_oos_results.append(sr_oos)

            # Calculate OOS rank and PBO
            sr_oos_all = returns_oos.mean(axis=0) / returns_oos.std(axis=0, ddof=1)
            rank_oos = np.sum(sr_oos_all > sr_oos) + 1
            ranks_oos.append(rank_oos / self.N)
            pbo_results.append(rank_oos > self.N / 2)

            # Record gamma for the best strategy
            gamma_best.append(gamma_sim[best_idx])

        sr_is_results = np.array(sr_is_results)
        sr_oos_results = np.array(sr_oos_results)
        pbo_results = np.array(pbo_results)
        ranks_oos = np.array(ranks_oos)
        gamma_best = np.array(gamma_best)

        # Calculate results
        mean_is_sharpe = np.mean(sr_is_results)
        mean_oos_sharpe = np.mean(sr_oos_results)
        sharpe_haircut = (1 - mean_oos_sharpe / mean_is_sharpe if mean_is_sharpe != 0 else np.nan)
        prob_oos_loss = np.mean(sr_oos_results < 0)
        pbo = np.mean(pbo_results)
        mean_oos_rank = np.mean(ranks_oos)
        oos_sharpe_ci = np.percentile(sr_oos_results, [5, 95])
        oos_sharpe_std = np.std(sr_oos_results)
        fdr = 1 - np.mean(gamma_best)

        self.results = {
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

        # Visualize results
        self._visualize_results(sr_oos_results, ranks_oos, model_name='Bimodal Bayes')

        return self.results

    def _visualize_results(self, sr_oos_results, ranks_oos, model_name):
        """
        Visualize OOS Sharpe ratio distribution and OOS rank histogram.

        Parameters:
        - sr_oos_results: Array of OOS Sharpe ratios
        - ranks_oos: Array of normalized OOS ranks
        - model_name: Name of the model for plot titles
        """
        plt.figure(figsize=(12, 5))

        # Plot OOS Sharpe ratio distribution
        plt.subplot(1, 2, 1)
        plt.hist(sr_oos_results, bins=50, density=True, alpha=0.7, color='blue', label='OOS Sharpe')
        plt.axvline(0, color='red', linestyle='--', label='Zero')
        plt.title(f'{model_name}: OOS Sharpe Ratio Distribution')
        plt.xlabel('Shar lpe Ratio')
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

# Example usage
if __name__ == "__main__":
    # Generate synthetic returns matrix for demonstration (replace with real data)
    np.random.seed(42)
    T, N = 1000, 200  # 1000 periods, 200 strategies
    returns_matrix = np.random.normal(loc=0.0001, scale=0.01, size=(T, N))
    T1, T2 = 500, 500  # 500 IS periods, 500 OOS periods

    # Initialize analyzer
    analyzer = BayesianBacktestOverfitting(returns_matrix, T1, T2, n_gibbs=1500, warm_up=500, n_sim=1000)

    # Run Naive Bayes analysis
    naive_results = analyzer.naive_bayes_analyze()
    print("Naive Bayes Results:")
    for key, value in naive_results.items():
        if key != 'oos_sharpe_samples':
            print(f"{key}: {value}")

    # Run Bimodal Bayes analysis
    bimodal_results = analyzer.bimodel_bayes_analyze(p0=0.7, m0=0.0001, V0=0.001)
    print("\nBimodal Bayes Results:")
    for key, value in bimodal_results.items():
        if key != 'oos_sharpe_samples':
            print(f"{key}: {value}")