import math
from random import Random
import time
from sklearn.base import BaseEstimator
import copy
from sympy import *
from .node import Node
from .rils import RILSRegressor, FitnessType
from joblib import Parallel, delayed

import warnings

from .solution import Solution
warnings.filterwarnings("ignore")

class RILSEnsembleRegressor(BaseEstimator):

    def __init__(self, epochs=100, fit_calls_per_epoch=100000, fitness_type=FitnessType.DEFAULT, complexity_penalty=0.001, initial_sample_size=1,target_size=20, parallelism = 8,first_improvement=True,verbose=False, random_state=0):
        self.fit_calls_per_epoch = fit_calls_per_epoch
        self.complexity_penalty = complexity_penalty
        self.random_state = random_state
        self.parallelism = parallelism
        self.verbose = verbose
        self.fitness_type = fitness_type
        self.initial_sample_size = initial_sample_size
        self.first_improvement = first_improvement
        self.target_size = target_size
        self.epochs = epochs
        self.base_regressors = [RILSRegressor(max_fit_calls=fit_calls_per_epoch, fitness_type=fitness_type, 
                                                  complexity_penalty=complexity_penalty, initial_sample_share=initial_sample_size, 
                                                  first_improvement=first_improvement,target_size=target_size,
                                                  verbose=verbose, random_state=i) 
                                                  for i in range(self.parallelism)]

    def fit(self, X, y, init_sympy_sol_str = "0", X_test = None, y_test = None):
        # now run each base regressor (RILSROLSRegressor) as a separate process
        self.start = time.time()
        self.relevant_model = None
        self.relevant_model_simp = None
        epoch = 0
        sympy_sol_str = init_sympy_sol_str
        all_time_best_fit = None
        while epoch < self.epochs:
            results = Parallel(n_jobs=len(self.base_regressors))(delayed(reg.fit)(X, y, init_sympy_sol_str=sympy_sol_str, X_test=X_test, y_test = y_test) for reg in self.base_regressors)
            print("All regressors have finished now")
            best_model, best_model_simp = results[0]
            best_fit = self.base_regressors[0].fitness(best_model, X, y, cache=False)
            i = 0
            for model, model_simp in results:
                model_fit = self.base_regressors[i].fitness(model, X,y, cache=False)
                if self.base_regressors[i].compare_fitness(model_fit, best_fit)<0:
                    if model_fit[0]>=best_fit[0]+0.01:
                        # R2 of new model is at least 1% better than previous -- this is done in order to avoid improvements to very complex models with marginal improvement in R2
                        self.relevant_model = copy.deepcopy(model)
                        self.relevant_model_simp = copy.deepcopy(model_simp)
                    best_fit = model_fit
                    best_model = model
                    best_model_simp = model_simp
                print('Model '+str(model)+'\t'+str(model_fit))
                i+=1
            if all_time_best_fit is None or self.base_regressors[0].compare_fitness(best_fit, all_time_best_fit)<0:
                all_time_best_fit = best_fit
            else:
                self.target_size+=1
                for reg in self.base_regressors:
                    reg.target_size = self.target_size
                print("No global improvement so increasing the target size of all base regressors to "+str(self.target_size))

            self.model = best_model
            self.model_simp = best_model_simp
            sympy_sol_str = str(self.model_simp)
            print('EPOCH '+str(epoch)+'. BEST: '+str(best_fit)+'\t'+sympy_sol_str)
            with open("log.txt", "a") as f:
                output_string =  self.fit_report_string(X, y)
                f.write("epoch "+str(epoch)+" "+output_string+"\n")
            epoch+=1

    def predict(self, X):
        Solution.clearStats()
        Node.reset_node_value_cache()
        return self.model.evaluate_all(X, False)

    def size(self):
        if self.model is not None:
            return self.model.size()
        return math.inf

    def modelString(self):
        if self.model_simp is not None:
            return str(self.model_simp)
        return ""

    def fit_report_string(self, X, y):
        if self.model==None:
            raise Exception("Model is not build yet. First call fit().")
        fitness = self.base_regressors[0].fitness(self.model,X,y, cache=False)
        fitness_relevant = fitness = self.base_regressors[0].fitness(self.relevant_model,X,y, cache=False)
        return "epochs={0}\tsec={1:.1f}\ttarget_size={2}\tR2={3:.7f}\tRMSE={4:.7f}\tsize={5}\tR2Rel={6:.7f}\tRMSERel={7:.7f}\tsizeRel={8}\texprSimp={9}\texprSimpRel={10}\t".format(
            self.epochs,time.time()-self.start,self.target_size, fitness[0], fitness[1], self.complexity(self.model_simp), fitness_relevant[0], fitness_relevant[1], 
            self.complexity(self.relevant_model_simp), self.model_simp, self.relevant_model_simp)

    def complexity(self, model):
        c=0
        for _ in preorder_traversal(model):
            c += 1
        return c
