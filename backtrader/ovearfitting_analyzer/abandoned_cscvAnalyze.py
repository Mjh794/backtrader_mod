import backtrader as bt
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from random import uniform

class CSCVAnalyzer():
    def __init__(self, S=10, performance_metric='sharpe',optstrats=False):
        self.S = S  # 分块数
        self.performance_metric = performance_metric
        self.optstrats = optstrats
        self.returns_matrix = None
        self.results = None
        #print("成功创建cscv分析")

    def collect_returns(self, runstrats):
        """Collect returns matrix from runstrats."""
        returns_dict = {}
        try:
            if self.optstrats or (runstrats and isinstance(runstrats[0], list)):
                # 优化模式：runstrats 是嵌套列表
                #print("optimization mode on")
                #for strats_list in runstrats:
                    #for strats in strat_list:
                        #time_return = strat.analyzers.timereturn.get_analysis()
                for strat_list in runstrats:
                    #print("这里解析没问题")
                    #print(type(strat_list))
                    for strat in strat_list:
                        #print("这里也没问题")
                        if hasattr(strat.analyzers, 'timereturn'):
                            returns = strat.analyzers.timereturn.get_analysis()
                            returns_dict[f"Strat_{uniform(0,2)}"] = pd.Series(returns)
                        else:
                            print(f"Warning: No TimeReturn analyzer ")
                #print("完成解析")
            else:
                # 非优化模式：runstrats 是平坦列表
                for i, strat in enumerate(runstrats):
                    if hasattr(strat.analyzers, 'timereturn'):
                        print(i)
                        
                        returns = strat.analyzers.timereturn.get_analysis()
                        print(returns)
                        returns_dict[f"Strategy_{i}"] = pd.Series(returns)
                    else:
                        print(f"Warning: No TimeReturn analyzer for Strategy_{i}")

            self.returns_matrix = pd.DataFrame(returns_dict).fillna(0)
            print(f"The shape of returns matrix is {self.returns_matrix.shape}")
        except Exception as e:
            print(f"Error collecting returns: {e}")
            self.returns_matrix = pd.DataFrame()

    def get_analysis(self):
        """Run CSCV analysis and return results."""
        if self.returns_matrix is None or self.returns_matrix.empty:
            return {'pbo': None, 'message': 'No valid returns data'}

        try:
            T = self.returns_matrix.shape[0]
            if T < self.S:
                return {'pbo': None, 'message': f'Insufficient data: T={T} < S={self.S}'}

            results = self._cscv_analysis(self.returns_matrix, self.S, self.performance_metric)
            self.results = results
            return results
        except Exception as e:
            return {'pbo': None, 'message': f'CSCV analysis failed: {e}'}

    def _cscv_analysis(self,returns_matrix, S, performance_metric='sharpe'):
  
    
        T = returns_matrix.shape[0]
        N = returns_matrix.shape[1]
        #print(returns_matrix['Strategy_0'].equals(returns_matrix['Strategy_1']))
        correlation_matrix = returns_matrix.corr()
        #print(correlation_matrix)

        

        block_size = T // S
        blocks = [returns_matrix.iloc[i * block_size: (i + 1) * block_size] for i in range(S)]
    
        all_combinations = list(combinations(range(S), S // 2))
        num_combinations = len(all_combinations)
        logits = []
        is_perf_list = []
        oos_perf_list = []
    
        for c in tqdm(all_combinations):
            train_blocks = [blocks[i] for i in c]
            test_blocks = [blocks[i] for i in range(S) if i not in c]
            train_set = pd.concat(train_blocks, axis=0)
            test_set = pd.concat(test_blocks, axis=0)

            if performance_metric == 'sharpe':
                is_std = train_set.std()
                oos_std = test_set.std()
                is_std[is_std == 0] = np.nan  
                oos_std[oos_std == 0] = np.nan  
                is_perf = train_set.mean() / is_std * np.sqrt(252)
                oos_perf = test_set.mean() / oos_std * np.sqrt(252)

                is_perf = is_perf.fillna(0)
                oos_perf = oos_perf.fillna(0)
            else:
                is_perf = train_set.apply(performance_metric) #这里还有待修改
                oos_perf = test_set.apply(performance_metric)

            best_strategy = is_perf.idxmax()
            oos_perf_best = oos_perf[best_strategy] if not pd.isna(best_strategy) else 0
        
            oos_perf_list.append(oos_perf_best)
            is_perf_list.append(is_perf.max())
        
            oos_rank = rankdata(-oos_perf)
            best_rank = oos_rank[oos_perf.index.get_loc(best_strategy)]
            omega = best_rank / (N + 1)
            logit = np.log(omega / (1 - omega)) if omega != 1 else -np.inf
            logits.append(logit)

        pbo = np.mean(np.array(logits) < 0)
        is_perf_all = np.array(is_perf_list)
        oos_perf_all = np.array(oos_perf_list)
    
        valid_mask = ~np.isnan(oos_perf_all) & ~np.isnan(is_perf_all)
        is_perf_all_clean = is_perf_all[valid_mask]
        oos_perf_all_clean = oos_perf_all[valid_mask]

        reg = LinearRegression().fit(is_perf_all_clean.reshape(-1, 1), oos_perf_all_clean)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        prob_loss = np.mean(oos_perf_all_clean < 0)

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
            #'logits': logits,
            'is_perf': is_perf_all_clean,
            'oos_perf': oos_perf_all_clean,
            'performance_degradation': {'slope': slope, 'intercept': intercept, 'prob_loss': prob_loss},
            'stochastic_dominance': {'fsd': fsd, 'ssd': ssd, 'cdf_optimized': cdf_optimized, 'cdf_random': cdf_random, 'x_range': x, 'ssd_integral': ssd_integral},
            'correlation': correlation_matrix
        }

        return self.results