
import math
from sklearn.base import BaseEstimator
from sympy import preorder_traversal

from rils.rils_ensemble import RILSEnsembleRegressor
from rils.solution import Solution


class RILSPipelineRegressor(BaseEstimator):

    def __init__(self, epochs=100, fit_calls_per_epoch=100000, max_seconds_per_epoch=10000, initial_sample_size=1,parallelism = 10, initial_target_size=20,verbose=False,  random_state=0):
        self.inner_ensemble = RILSEnsembleRegressor(epochs=epochs, fit_calls_per_epoch=fit_calls_per_epoch, max_seconds_per_epoch=max_seconds_per_epoch, initial_sample_size=initial_sample_size, parallelism=parallelism, target_size=initial_target_size, verbose=verbose, random_state=random_state)
        self.fit_calls_per_epoch = fit_calls_per_epoch
        self.max_calls_per_epoch = max_seconds_per_epoch
        self.random_state = random_state
        self.parallelism = parallelism
        self.verbose = verbose
        self.initial_sample_size = initial_sample_size
        self.epochs = epochs
        self.init_target_size = initial_target_size

    def fit(self, X, y, init_sympy_sol_str = "0", dataset_file="", X_test = None, y_test = None):
        self.inner_ensemble.fit(X, y ,init_sympy_sol_str=init_sympy_sol_str, dataset_file=dataset_file, X_test=X_test, y_test=y_test)
        best_tradeoff_sympy_sol = self.select_best_tradeoff_solution()
        print("The best tradeoff model is "+best_tradeoff_sympy_sol)
        self.model_simp = best_tradeoff_sympy_sol
        self.model = Solution([Solution.convert_to_my_nodes(self.model_simp)])
        return (self.model, self.model_simp)

    def select_best_tradeoff_solution(self):
        # reading backwards and finding the last relevant improvement w.r.t. R2
        # relevant improvement is one of at least 0.5% improvement in R2
        r2_with_sols = self.inner_ensemble.best_solutions
        i = len(r2_with_sols)-1
        while i>0:
            if r2_with_sols[i][0]-r2_with_sols[i-1][0]>0.005:
                break
            i-=1
        return r2_with_sols[i][1]
    
    def size(self):
        if self.model_simp is not None:
            return self.complexity(self.model_simp)
        return math.inf
    
    def complexity(self, sm):
        c=0
        for arg in preorder_traversal(sm):
            c += 1
        return c

    def model_string(self):
        if self.model_simp is not None:
            return str(self.model_simp)
        return ""