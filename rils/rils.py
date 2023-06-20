from cmath import inf
import copy
import time
import math
from random import Random, seed, shuffle
from sklearn.base import BaseEstimator
import copy
from sympy import *
from .node import Node, NodeArcCos, NodeArcSin, NodeConstant, NodePow, NodeGauss, NodeTan, NodeTanh, NodeInv, NodeIdentity, NodeVariable, NodePlus, NodeMinus, NodeMultiply, NodeDivide, NodeSqr, NodeSqrt, NodeLn, NodeExp, NodeSin, NodeCos, NodeAbs
from enum import Enum
import warnings
from .solution import Solution
from numpy import imag, real
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score, max_error, mean_absolute_error, mean_squared_log_error, mean_pinball_loss, mean_tweedie_deviance, mean_poisson_deviance, mean_gamma_deviance, d2_tweedie_score, d2_pinball_score, d2_absolute_error_score
from .utils import diff_R2, diff_RMSE, percentile_abs_error, log_cosh, distribution_fit_score
from scipy.stats import normaltest, ks_2samp

warnings.filterwarnings("ignore")


class FitnessType(Enum):
    R2_RMSE = 1
    R2 = 2
    MSE = 3
    EXP_VAR = 4
    MAX_ERR = 5
    MAE = 6
    MSLE = 7
    MPL_95 = 8
    MTD = 9
    MPD = 10
    MGD = 11
    D2_TS = 12
    D2_PS_95 = 13
    D2_AE = 14
    PERC_ABS_95 = 15
    LOG_COSH = 16
    DIFF_R2 = 17
    DIFF_RMSE = 18
    DIST_DIFF = 19
    EXPERIMENTAL = 20

class RILSRegressor(BaseEstimator):

    def __init__(self, max_fit_calls=100000, max_seconds=10000,fitness_type=FitnessType.EXPERIMENTAL, complexity_penalty=0, initial_sample_size=1,simplification=True, first_improvement=True, change_ignoring=False, target_size=20, verbose=False, random_state=0):
        self.max_seconds = max_seconds
        self.max_fit_calls = max_fit_calls
        self.fitness_type = fitness_type
        self.complexity_penalty = complexity_penalty
        self.initial_sample_size = initial_sample_size
        self.verbose = verbose
        self.random_state = random_state
        self.simplification = simplification
        self.first_improvement = first_improvement
        self.change_ignoring = change_ignoring
        self.target_size = target_size

    def __reset(self):
        self.var_cnt = 0
        self.model = None
        self.ls_it = 0
        self.main_it = 0
        self.last_improved_it = 0
        self.last_generalized_tuning_it = 0
        self.last_target_size_increase_it = 0
        self.time_start = 0
        self.time_elapsed = 0
        self.ls_time = 0
        self.ls_cand_time = 0
        self.pert_time = 0
        self.simp_time = 0
        self.dc_time = 0
        seed(self.random_state)
        self.rg = Random(self.random_state)
        self.total_changes_cnt = 0
        self.tried_changes_cnt = 0
        self.change_type_cnts = {}
        self.pareto_set = []
        Solution.clearStats()
        Node.reset_node_value_cache()

    def __setup_nodes(self):
        self.allowed_nodes=[NodeConstant(1), NodeConstant(math.pi), NodeConstant(math.e), NodeConstant(-1), NodeConstant(0), NodeConstant(0.5), NodeConstant(2), NodeConstant(10)]
        for i in range(self.var_cnt):
            self.allowed_nodes.append(NodeVariable(i))
        self.allowed_nodes+=[NodePlus(), NodeMinus(), NodeMultiply(), NodeDivide(), NodeSqr(), NodeSqrt(),NodeLn(), NodeExp(),NodeSin(), NodeCos()]#, NodeTanh(), NodeInv(), NodeGauss()]#, NodeGauss()] #, NodeIdentity()]#, NodeAbs()]#, NodeArcSin(), NodeArcCos()]

    def print_data_stats(self, X, y):
        alpha = 0.05
        for i in range(len(X[0])):
            x_i = X[:, i]
            _, p = normaltest(x_i)
            print(f'Variable {i} normality {p>alpha} pval={p}')
        _, p = normaltest(y)
        print(f"Target normality {p>alpha} pval={p}")

    def fit(self, X, y, init_sympy_sol_str = "0", X_test = None, y_test = None):
        x_all = copy.deepcopy(X)
        y_all = copy.deepcopy(y)
        #self.print_data_stats(x_all, y_all)
        Xy = zip(x_all, y_all)
        Xy_sorted = sorted(Xy, key=lambda p:p[1])
        # take 1% of points or at least 100 points initially 
        self.n = int(self.initial_sample_size*len(x_all))
        if self.n<100:
            self.n=min(100, len(X))
        print("Taking "+str(self.n)+" points initially.")
        skip = round(len(x_all)/self.n)
        i=0
        X = []
        y = []
        while i<self.n:
            X.append(Xy_sorted[i][0])
            y.append(Xy_sorted[i][1])
            i+=skip

        X = np.array(X)
        #X = cp.array(X)
        y = np.array(y)

        self.__reset()
        self.start = time.time()
        if len(X) == 0:
            raise Exception("Input feature data set (X) cannot be empty.")
        if len(X)!=len(y):
            raise Exception("Numbers of feature vectors (X) and target values (y) must be equal.")
        self.var_cnt = len(X[0])
        self.__setup_nodes()
        if init_sympy_sol_str=="0":
            init_sol = NodeConstant(0)
        else:
            init_sympy_sol = sympify(init_sympy_sol_str).evalf()
            init_sol = Solution.convert_to_my_nodes(init_sympy_sol)
        best_solution =  Solution([init_sol])
        best_fitness = self.fitness(best_solution, X, y)
        print("Initial fitness "+str(best_fitness))

        self.main_it = 0
        while self.time_elapsed<self.max_seconds and Solution.fit_calls<self.max_fit_calls: 
            start = time.time()
            new_solution = self.perturb(best_solution) 
            self.pert_time+=(time.time()-start)
            if self.verbose:
                print("Perturbed solution "+str(new_solution))
            
            start = time.time()
            new_solution = self.LS(new_solution, X, y)
            new_fitness = self.fitness(new_solution, X, y)
            self.ls_time+=(time.time()-start)

            fit_comp = self.compare_fitness(new_fitness, best_fitness)
            if fit_comp==0 and self.verbose:
                print("SAME QUALITY")
            if fit_comp<0 and not self.is_dominated(new_fitness):
                if self.verbose:
                    print("perturbation "+str(new_solution)+" produced global improvement.")
                start = time.time()
                best_solution = copy.deepcopy(new_solution)
                self.dc_time+=(time.time()-start)
                best_fitness = new_fitness
                self.last_improved_it = self.main_it

            fitness_test = self.fitness(best_solution, X_test, y_test, cache=False)
            self.time_elapsed = time.time()-self.start
            sympy_sol = sympify(best_solution).evalf()
            sympy_size = RILSRegressor.complexity(sympy_sol)
            print("%d/%d. t=%.1f R2=%.7f R2tst=%.7f RMSE=%.7f RMSEtst=%.7f DiffR2=%.7f DiffRMSE=%.7f DIST=%.7f RES_NORM=%s size=%d symSize=%d trgtSize=%d FIT=%.7f fittype=%s firstImpr=%s simplification=%s changeIgnore=%s factors=%d mathErr=%d fitCalls=%d fitFails=%d fitTime=%d lrTime=%d shTime=%d simpTime=%d lsTime=%d lsCandTime=%d dcTime=%d triedChg=%d totChg=%d cPerc=%.1f cSize=%d\n                                                                          expr=%s"
            %(self.main_it,self.ls_it, self.time_elapsed, best_fitness[0], fitness_test[0], best_fitness[1],fitness_test[1], best_fitness[4], best_fitness[5], best_fitness[6], best_fitness[7], best_solution.size(), sympy_size,self.target_size, best_fitness[3], self.fitness_type, self.first_improvement, self.simplification, self.change_ignoring,  len(best_solution.factors), Solution.math_error_count, Solution.fit_calls, Solution.fit_fails, Solution.fit_time, 
              Solution.lr_solving_time, self.pert_time, self.simp_time, self.ls_time, self.ls_cand_time, self.dc_time, self.tried_changes_cnt, self.total_changes_cnt,  Node.cache_hits*100.0/Node.cache_tries, len(Node.node_value_cache), sympy_sol))
            #self.print_change_type_stats()
        print("Doing sympify of "+str(best_solution))
        self.model_simp = sympify(str(best_solution)).evalf()
        print("Finished with sympy model "+str(self.model_simp))
        self.model = best_solution 
        return (self.model, self.model_simp)
    
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
        fitness = self.fitness(self.model, X,y, cache=False)
        return "maxTime={0}\tmaxFitCalls={1}\tseed={2}\tsizePenalty={3}\tR2={4:.7f}\tRMSE={5:.7f}\tsize={6}\tsec={7:.1f}\tmainIt={8}\tlsIt={9}\tfitCalls={10}\tfitType={11}\tinitSampleSize={12}\tfirstImpr={13}\tsimplification={14}\tchangeIgnore={15}\ttargetSize={16}\texpr={17}\texprSimp={18}".format(
            self.max_seconds,self.max_fit_calls,self.random_state,self.complexity_penalty, fitness[0], fitness[1], RILSRegressor.complexity(self.model_simp), self.time_elapsed,self.main_it, self.ls_it,Solution.fit_calls, self.fitness_type, self.initial_sample_size, 
            self.first_improvement, self.simplification, self.change_ignoring , self.target_size, self.model, self.model_simp)

    def complexity(model_simp):
        c=0
        for arg in preorder_traversal(model_simp):
            c += 1
        return c
    
    def move_to_random_isomorphic(self, solution: Solution):
        assert(len(solution.factors)<=1)
        if len(solution.factors)==0:
            return solution
        solution.factors[0] = solution.factors[0].random_isomorphic_rotation(0.5, self.rg)
        return solution

    def perturb(self, solution: Solution):
        perturbed_solution = solution
        perts = self.all_perturbations(perturbed_solution)
        if self.verbose:
            print("Selecting from "+str(len(perts))+" perturbations") 
        pert_i = self.rg.randrange(0, len(perts))
        perturbed_solution = perts[pert_i]
        return perturbed_solution
    
    def all_perturbations(self, solution: Solution):
        all = []
        perturbed_solution = copy.deepcopy(solution)
        perturbed_solution.simplify_whole(self.var_cnt)
        perturbed_solution.join()
        assert len(perturbed_solution.factors)==1
        j = 0
        all_subtrees = perturbed_solution.factors[0].all_nodes_exact()
        if len(all_subtrees)==0: # this is the case when we have constant or variable, so we just change the root
            for cand in self.perturb_candidates(perturbed_solution.factors[j]):
                perturbed = copy.deepcopy(perturbed_solution)
                perturbed.factors[j] = cand
                #perturbed.simplify_whole(varCnt)
                all.append(perturbed)
        else:
            for i in range(len(all_subtrees)):
                refNode = all_subtrees[i]
                if refNode==perturbed_solution.factors[j]:
                    for cand in self.perturb_candidates(perturbed_solution.factors[j]):
                        perturbed = copy.deepcopy(perturbed_solution)
                        perturbed.factors[j] = cand
                        #perturbed.simplify_whole(varCnt)
                        all.append(perturbed)
                if refNode.arity >= 1:
                    for cand in self.perturb_candidates(refNode.left, refNode, True):
                        perturbed = copy.deepcopy(perturbed_solution)
                        perturbed_subtrees = perturbed.factors[j].all_nodes_exact()
                        perturbed_subtrees[i].left = cand
                        #perturbed.simplify_whole(varCnt)
                        all.append(perturbed)
                if refNode.arity>=2:
                    for cand in self.perturb_candidates(refNode.right, refNode, False):
                        perturbed = copy.deepcopy(perturbed_solution)
                        perturbed_subtrees = perturbed.factors[j].all_nodes_exact()
                        perturbed_subtrees[i].right = cand
                        #perturbed.simplify_whole(varCnt)
                        all.append(perturbed)
        return all
    
    def first_dominates(self, fit1, fit2):
        return (fit1[0]>= fit2[0] and fit1[2]<fit2[2]) or (fit1[0]>fit2[0] and fit1[2]<=fit2[2])
    
    def is_dominated(self, new_fit):
        for _, fit in self.pareto_set:
            if self.first_dominates(fit, new_fit):
                return True
        return False

    def add_to_pareto_set(self, new_sol, new_fit):
        if self.is_dominated(new_fit):
            return False
        # it is not dominated by an of solutions in the non_dominanted_solutions set
        # now we need to check if this one dominates some solution and to remove those
        new_pareto_set = [(new_sol, new_fit)]
        for sol, fit in self.pareto_set:
            if self.first_dominates(new_fit, fit):
                # new_sol dominates sol
                continue
            # otherwise, sol remains in the set
            new_pareto_set.append((sol, fit))
        self.pareto_set = new_pareto_set
        return True
    
    def print_pareto_set(self):
        print("-----------------------------\nPARETO SET\n-----------------------------\n")
        self.pareto_set = sorted(self.pareto_set, key=lambda x : x[1][0], reverse=True)
        print(", ".join(["("+str(f[1][0])+","+str(f[1][2])+")" for f in self.pareto_set]))

    def LS(self, solution: Solution, X, y):
        best_fitness = self.fitness(solution, X, y)
        best_solution = copy.deepcopy(solution)
        impr = True
        if self.verbose:
            print("Starting LS")
        while impr:
            impr = False
            self.ls_it+=1
            self.time_elapsed = time.time()-self.start
            if self.time_elapsed>self.max_seconds or Solution.fit_calls>self.max_fit_calls:
                break
            
            if best_solution.size()>self.target_size:
                increasing = False
            else:
                increasing = None
            new_solution = self.LS_iteration(best_solution, X, y, increasing=increasing)
            new_fitness = self.fitness(new_solution, X, y)

            if self.compare_fitness(new_fitness, best_fitness)<0:
                impr = True
                best_solution = copy.deepcopy(new_solution)
                best_fitness = copy.deepcopy(new_fitness)
                if self.simplification:
                    best_solution_simp = copy.deepcopy(best_solution)
                    best_solution_simp.simplify_whole(len(X[0]))
                    best_fitness_simp = self.fitness(best_solution_simp, X, y)
                    if math.isnan(best_fitness_simp[0]) or math.isnan(best_fitness_simp[1]) or abs(best_fitness_simp[0]-best_fitness[0])>0.0001 or abs(best_fitness_simp[1]-best_fitness[1])>0.0001:
                        if self.verbose:
                            print("WARNING: worse solution after simpliflication OLD "+str(best_solution)+" NEW "+str(best_solution_simp))
                            print("Old fitness "+str(best_fitness)+"  new fitness"+str(best_fitness_simp))
                    else:
                        best_solution = best_solution_simp
                        best_fitness = best_fitness_simp
                if self.verbose:
                    print("IMPROVED to "+str(best_fitness)+"\t"+str(best_solution))
                # put into non_dominated set and recalculate it
                self.add_to_pareto_set(best_solution, best_fitness)
                #if self.verbose:
                #    self.print_pareto_set()
                continue  
        return best_solution

    def all_LS_changes(self, best_solution: Solution, increasing)->set[Solution]:
        solution = copy.deepcopy(best_solution)
        solution_changes = set([])
        if len(solution.factors)>1:
            print("JOINING")
            solution.join()
        assert(len(solution.factors)<=1)
        for i in range(len(solution.factors)):
            factor = solution.factors[i]
            factor_subtrees = factor.all_nodes_exact()
            for j in range(len(factor_subtrees)):
                ref_node = factor_subtrees[j]
                if ref_node==factor: # this subtree is the whole factor
                    candidates = self.change_candidates(ref_node, increasing)
                    for cand in candidates:
                        solution_changes.add((cand, j, "None"))
                if ref_node.arity >= 1:
                    candidates = self.change_candidates(ref_node.left,increasing, parent=ref_node,is_left_from_parent=True)
                    for cand in candidates:
                        solution_changes.add((cand, j, "True"))
                if ref_node.arity>=2:
                    candidates = self.change_candidates(ref_node.right, increasing, parent=ref_node, is_left_from_parent=False)
                    for cand in candidates:
                        solution_changes.add((cand, j, "False"))
        solution_changes_list = list(solution_changes)
        solution_changes_list = sorted(solution_changes_list, key=lambda x: (str(x[0][0]), x[1], x[2]))
        return solution_changes_list   

    def LS_iteration(self, solution: Solution, X, y,increasing=None):
        best_fitness = self.fitness(solution, X, y)
        solution = self.move_to_random_isomorphic(solution)
        test_fitness = self.fitness(solution, X, y)
        if best_fitness[3]!=inf and test_fitness[3]!=inf:
            #assert(abs(best_fitness[3]-test_fitness[3])<0.001)
            if abs(best_fitness[3]-test_fitness[3])>0.001:
                print("Error when moving to isomorphic tree solution.")
        new_solution = copy.deepcopy(solution)
        all_changes = self.all_LS_changes(new_solution, increasing)
        if self.verbose:
            print(str(len(all_changes))+" LS change candidates for mode="+str(increasing))

        self.rg.shuffle(all_changes)
        best_change = None

        for change in all_changes:
            self.total_changes_cnt+=1
            chg, type = change[0]
            if self.change_ignoring==True and type in self.change_type_cnts:
                type_cnts = self.change_type_cnts[type]
                if type_cnts[0]>1000 and type_cnts[2]<0.1:
                    #print("Ignoring change of type "+type)
                    continue    # change was tried at least 1000 times and it was useful in only 0.1% of situations TODO: these two could be parameters
            self.tried_changes_cnt+=1
            if self.tried_changes_cnt%10000==0:
                print("Tried "+str(self.tried_changes_cnt)+"/"+str(self.total_changes_cnt)+"\t"+str(best_fitness))
            start = time.time()
            new_factor_subtrees = new_solution.factors[0].all_nodes_exact()
            chg, type = change[0]
            j = change[1]
            is_left = change[2]
            if is_left == "None":
                old_node = copy.copy(new_solution.factors[0])
                new_solution.factors[0] = chg
            elif is_left == "True":
                old_node = copy.copy(new_factor_subtrees[j].left)
                new_factor_subtrees[j].left=chg  
            else:
                old_node = copy.copy(new_factor_subtrees[j].right)
                new_factor_subtrees[j].right = chg
            self.ls_cand_time+=(time.time()-start)

            new_fitness = self.fitness(new_solution, X, y)

            if type not in self.change_type_cnts:
                self.change_type_cnts[type] = (0, 0, 0)
            self.change_type_cnts[type] = (self.change_type_cnts[type][0]+1, self.change_type_cnts[type][1], self.change_type_cnts[type][1]*100.0/(self.change_type_cnts[type][0]+1)) # coordinate 0 is the number of change types tries

            if self.compare_fitness(new_fitness, best_fitness)<0:
                self.change_type_cnts[type] =(self.change_type_cnts[type][0], self.change_type_cnts[type][1]+1, (self.change_type_cnts[type][1]+1)*100.0/self.change_type_cnts[type][0])
                if self.first_improvement:
                    return new_solution
                else:
                    best_fitness = copy.deepcopy(new_fitness)
                    best_change = change
                    if self.verbose:
                        print("Improving to "+str(best_fitness))
                    #best_solution = copy.deepcopy(new_solution)

            if is_left == "None":
                new_solution.factors[0] = old_node
            elif is_left == "True":
                new_factor_subtrees[j].left=old_node  
            else:
                new_factor_subtrees[j].right = old_node
        
        if best_change is not None:
            new_factor_subtrees = new_solution.factors[0].all_nodes_exact()
            chg, type = best_change[0]
            j = best_change[1]
            is_left = best_change[2]
            if is_left == "None":
                old_node = copy.copy(new_solution.factors[0])
                new_solution.factors[0] = chg
            elif is_left == "True":
                old_node = copy.copy(new_factor_subtrees[j].left)
                new_factor_subtrees[j].left=chg  
            else:
                old_node = copy.copy(new_factor_subtrees[j].right)
                new_factor_subtrees[j].right = chg
            test_fitness = self.fitness(new_solution, X, y)
            assert(abs(test_fitness[3]-best_fitness[3])<0.0001)
            return new_solution
        return solution
    
    def print_change_type_stats(self):
        with open("impr_types_"+str(self.random_state)+".txt", "w") as f:
            for ch_type, stats in {k: v for k, v in sorted(self.change_type_cnts.items(), key=lambda item: item[1][2], reverse=True)}.items():
                f.write(ch_type+"\t"+str(stats[0])+"\t"+str(stats[1])+"\t"+str(stats[2])+"\n")

    # perturb_candidates is a subset of change_candidates
    def perturb_candidates(self, old_node: Node, parent=None, is_left_from_parent=None):
        candidates = set([])
        # change node to one of its subtrees -- reduces the size of expression
        '''
        if old_node.arity>=1:
            all_left_subtrees = old_node.left.all_nodes_exact()
            for ls in all_left_subtrees:
                candidates.add(copy.deepcopy(ls))
            #candidates.append(copy.deepcopy(old_node.left))
        if old_node.arity>=2:
            all_right_subtrees = old_node.right.all_nodes_exact()
            for rs in all_right_subtrees:
                candidates.add(copy.deepcopy(rs))
            #candidates.append(copy.deepcopy(old_node.right))'''
        # change constant or variable to another variable
        if old_node.arity==0:# and type(old_node)==type(NodeConstant(0)):
            for node in filter(lambda x:type(x)==type(NodeVariable(0)) and x!=old_node, self.allowed_nodes):
                new_node = copy.deepcopy(node)
                candidates.add(new_node)
        # change variable to unary operation applied to that variable
        if type(old_node)==type(NodeVariable(0)):
            for node in filter(lambda x:x.arity==1, self.allowed_nodes):
                if not node.is_allowed_left_argument(old_node):
                    continue
                new_node = copy.deepcopy(node)
                new_node.left =copy.deepcopy(old_node)
                new_node.right = None
                candidates.add(new_node)
        # change unary operation to another unary operation
        if old_node.arity == 1:
            for node in filter(lambda x:x.arity==1 and type(x).__name__ !=type(old_node).__name__, self.allowed_nodes):
                new_node = copy.deepcopy(node)
                new_node.left = copy.deepcopy(old_node.left)
                assert old_node.right==None
                candidates.add(new_node)
        # change one binary operation to another
        if old_node.arity==2:
            for nodeOp in filter(lambda x: x.arity==2 and type(x).__name__ !=type(old_node).__name__, self.allowed_nodes):
                if (not nodeOp.is_allowed_left_argument(old_node.left)) or (not nodeOp.is_allowed_right_argument(old_node.right)):
                    continue
                new_node = copy.deepcopy(nodeOp)
                new_node.left = copy.deepcopy(old_node.left)
                new_node.right = copy.deepcopy(old_node.right)
                candidates.add(new_node)
            # swap left and right side if not symmetric op
            if not old_node.symmetric:
                new_node = copy.deepcopy(old_node)
                new_node.left = copy.deepcopy(old_node.right)
                new_node.right = copy.deepcopy(old_node.left)
                candidates.add(new_node)
        # change variable (or constant) to binary operation with some variable  -- increases the model size
        if type(old_node)==type(NodeVariable(0)): #old_node.arity==0:
            node_args = list(filter(lambda x: type(x)==type(NodeVariable(0)), self.allowed_nodes))
            for node_arg in node_args:
                for node_op in filter(lambda x: x.arity==2, self.allowed_nodes):
                    if not node_op.is_allowed_right_argument(node_arg) or not node_op.is_allowed_left_argument(old_node):
                        continue
                    new_node = copy.deepcopy(node_op)
                    new_node.left = copy.deepcopy(old_node)
                    new_node.right = copy.deepcopy(node_arg)
                    candidates.add(new_node)
                    if not node_op.symmetric and node_op.is_allowed_right_argument(old_node) and node_op.is_allowed_left_argument(node_arg):
                        new_node = copy.deepcopy(node_op)
                        new_node.right = copy.deepcopy(old_node)
                        new_node.left = copy.deepcopy(node_arg)
                        candidates.add(new_node)
        # filtering not allowed candidates (because of the parent)

        filtered_candidates = []
        if parent is not None:
            for c in candidates:
                if is_left_from_parent and not parent.is_allowed_left_argument(c):
                    continue
                if not is_left_from_parent and not parent.is_allowed_left_argument(c):
                    continue
                filtered_candidates.append(c)
            candidates = filtered_candidates
        # sort to avoid non-determinism of set structure
        candidates = sorted(candidates, key=lambda x: str(x))
        return candidates

    # superset of perturb_candidates
    def change_candidates(self, old_node:Node, increasing, parent=None, is_left_from_parent=None):
        candidates = set([])
        if increasing is None or increasing == True:
            # change anything to binary operation with some variable or constant -- increases the model size
            node_args = list(filter(lambda x: x.arity==0, self.allowed_nodes))#+[copy.deepcopy(x) for x in old_node.all_nodes_exact()]
            for node_arg in node_args:
                for node_op in filter(lambda x: x.arity==2, self.allowed_nodes):
                    if not node_op.is_allowed_right_argument(node_arg) or not node_op.is_allowed_left_argument(old_node):
                        continue
                    new_node = copy.deepcopy(node_op)
                    new_node.left = copy.deepcopy(old_node)
                    new_node.right = copy.deepcopy(node_arg)
                    candidates.add((new_node, "binary_insert_"+str(node_op)+"_with_"+str(node_arg)))
                    if not node_op.symmetric and node_op.is_allowed_right_argument(old_node) and node_op.is_allowed_left_argument(node_arg):
                        new_node = copy.deepcopy(node_op)
                        new_node.right = copy.deepcopy(old_node)
                        new_node.left = copy.deepcopy(node_arg)
                        candidates.add((new_node, "binary_insert_"+str(node_op)+"_with_"+str(node_arg)))
            # change unary operation to binary operation with its argument and some variable or constant
            if old_node.arity==1:
                for node_arg in node_args:
                    for node_op in filter(lambda x: x.arity==2, self.allowed_nodes):
                        if not node_op.is_allowed_right_argument(node_arg) or not node_op.is_allowed_left_argument(old_node.left):
                            continue
                        new_node = copy.deepcopy(node_op)
                        new_node.left = copy.deepcopy(old_node.left)
                        new_node.right = copy.deepcopy(node_arg)
                        candidates.add((new_node, "unary_to_binary_"+str(node_op)+"_with_"+str(node_arg)))
                        if not node_op.symmetric and node_op.is_allowed_right_argument(old_node.left) and node_op.is_allowed_left_argument(node_arg):
                            new_node = copy.deepcopy(node_op)
                            new_node.right = copy.deepcopy(old_node.left)
                            new_node.left = copy.deepcopy(node_arg)
                            candidates.add((new_node, "unary_to_binary_"+str(node_op)+"_with_"+str(node_arg)))

        if increasing is None or increasing == False:
            # change anything to unary operation applied to that -- slightly increases the model size
            for node in filter(lambda x:x.arity==1, self.allowed_nodes):
                if not node.is_allowed_left_argument(old_node):
                    continue
                new_node = copy.deepcopy(node)
                new_node.left =copy.deepcopy(old_node)
                new_node.right = None
                candidates.add((new_node, "unary_insert_"+str(node)))
            # change constant to something multiplied with it
            if type(old_node)==type(NodeConstant(0)):
                # multiplicative changing
                for mult in [0.01, 0.1, 0.5, 0.8, 0.99, 1, 1.2, 2, 10, 100]: #0.01, 0.1, 0.2, 0.5, 0.8, 0.9,1.1,1.2, 2, 5, 10, 20, 50, 100]:
                    for sign in [-1, 1]:
                        candidates.add((NodeConstant(old_node.value*mult*sign), "constant_mult_"+str(mult*sign)))
                # additive changing as well
                for add in [-100, -10, -1, 1, 10, 100]:
                    candidates.add((NodeConstant(old_node.value+add), "constant_add_"+str(add)))
            # change anything to one of its left subtrees
            if old_node.arity>=1:
                all_left_subtrees = old_node.left.all_nodes_exact()
                for ls in all_left_subtrees:
                    candidates.add((copy.deepcopy(ls), "left_subtree"))
                #candidates.append(copy.deepcopy(old_node.left))
            # change anything to one of its right subtrees
            if old_node.arity>=2:
                all_right_subtrees = old_node.right.all_nodes_exact()
                for rs in all_right_subtrees:
                    candidates.add((copy.deepcopy(rs), "right_subtree"))
                #candidates.append(copy.deepcopy(old_node.right))
            # change anything to a constant or variable
            for node in filter(lambda x:x.arity==0 and x!=old_node, self.allowed_nodes):
                candidates.add((copy.deepcopy(node), "change_to_"+str(node)))
            # change unary operation to another unary operation
            if old_node.arity == 1:
                for node in filter(lambda x:x.arity==1 and type(x).__name__ !=type(old_node).__name__, self.allowed_nodes):
                    new_node = copy.deepcopy(node)
                    new_node.left = copy.deepcopy(old_node.left)
                    assert old_node.right==None
                    candidates.add((new_node, "unary_change_"+str(node)))
            # change one binary operation to another
            if old_node.arity==2:
                for node_op in filter(lambda x: x.arity==2 and type(x).__name__ !=type(old_node).__name__, self.allowed_nodes):
                    if (not node_op.is_allowed_left_argument(old_node.left)) or (not node_op.is_allowed_right_argument(old_node.right)):
                        continue
                    new_node = copy.deepcopy(node_op)
                    new_node.left = copy.deepcopy(old_node.left)
                    new_node.right = copy.deepcopy(old_node.right)
                    candidates.add((new_node, "binary_change_"+str(node_op)))
                # swap left and right side if not symmetric op
                if not old_node.symmetric:
                    new_node = copy.deepcopy(old_node)
                    new_node.left = copy.deepcopy(old_node.right)
                    new_node.right = copy.deepcopy(old_node.left)
                    candidates.add((new_node, "binary_change_"+str(node_op))) 
            # change binary operation to unary with left or right argument (reduces the model)
            if old_node.arity==2:
                for node_op in filter(lambda x: x.arity==1, self.allowed_nodes):
                    if node_op.is_allowed_left_argument(old_node.left):
                        new_node = copy.deepcopy(node_op)
                        new_node.left = copy.deepcopy(old_node.left)
                        candidates.add((new_node, "binary_to_unary_"+str(node_op)))
                    if node_op.is_allowed_left_argument(old_node.right):
                        new_node = copy.deepcopy(node_op)
                        new_node.left = copy.deepcopy(old_node.right)
                        candidates.add((new_node, "binary_to_unary_"+str(node_op)))

        candidates = [x for x in candidates if x is not None]
        # filtering not allowed candidates (because of the parent)
        filtered_candidates = []
        if parent is not None:
            for c, desc in candidates:
                if is_left_from_parent and not parent.is_allowed_left_argument(c):
                    continue
                if not is_left_from_parent and not parent.is_allowed_right_argument(c):
                    continue
                filtered_candidates.append((c, desc))
            candidates = filtered_candidates
        # sort to avoid non-determinism of set structure
        candidates = sorted(candidates, key=lambda x: str(x[0]))
        return candidates


    def fitness(self, solution: Solution, X, y, cache=True):
        try:
            Solution.fit_calls+=1
            start = time.time()
            yp = solution.evaluate_all(X, cache) 
            r2 = r2_score(y, yp)
            diff_rmse = 0
            diff_r2 = 0
            mse = mean_squared_error(y, yp)
            rmse = sqrt(mse)
            size = solution.size()
            # TODO: uncommenting following lines for dist fit score and tests slows down the program ~ 4x
            dist_diff = 0 # distribution_fit_score(y, yp)
            #res = [y[i]-yp[i] for i in range(len(y))]
            _, p_val_norm_res = 0, 0 # normaltest(res)
            _, p_val_same_dist = 0, 0 # ks_2samp(y, yp)
            # 0-hypothesis is that samples of y and yp belong to same distribution
            #same_dist = p_val>0.05 # 0-hypothesis is rejected

            if self.fitness_type == FitnessType.R2_RMSE:
                final = (2-r2)*(1+rmse)*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.R2:
                final = (2-r2)*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.MSE:
                final = mse*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.EXP_VAR:
                final = (1-explained_variance_score(y, yp))*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.MAX_ERR:
                final = max_error(y, yp)*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.MAE:
                final = mean_absolute_error(y, yp)*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.MSLE:
                final = mean_squared_log_error(y, yp)*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.MPL_95:
                final = mean_pinball_loss(y, yp, alpha=0.95)*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.MTD:
                final = mean_tweedie_deviance(y, yp)*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.MPD:
                final = mean_poisson_deviance(y, yp)*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.MGD:
                final = mean_gamma_deviance(y, yp)*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.D2_TS:
                final = 1-d2_tweedie_score(y, yp)*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.D2_PS_95:
                final = 1-d2_pinball_score(y, yp, alpha=0.95)*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.D2_AE:
                final = 1-d2_absolute_error_score(y, yp)*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.PERC_ABS_95:
                final = percentile_abs_error(y, yp, alpha=0.95)*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.LOG_COSH:
                final = log_cosh(y, yp)*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.DIFF_R2:
                diff_r2 = diff_R2(y, yp)
                final = (2-diff_r2)*(1+rmse)*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.DIFF_RMSE:
                diff_rmse = diff_RMSE(y, yp)
                final = (2-r2)*(1+diff_rmse)*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.DIST_DIFF:
                final = (1+0.1*dist_diff)*(2-r2)*(1+self.complexity_penalty*size)
            elif self.fitness_type == FitnessType.EXPERIMENTAL:
                if size<self.target_size:
                    size = self.target_size
                final = (2-r2)*size
            else:
                raise Exception("Unrecognized fitness type "+str(self.fitness_type))
            result = (r2, rmse, size, final, diff_r2, diff_rmse, dist_diff, p_val_norm_res, p_val_same_dist)
            Solution.fit_time+=(time.time()-start)
            return result
        except Exception as e:
            #print(e)
            Solution.math_error_count+=1
            Solution.fit_fails+=1
            return (inf, inf, inf, inf, inf, inf, inf, inf, inf)

    def compare_fitness(self, new_fit, old_fit):
        if math.isnan(new_fit[0]):
            return 1
        # do not accept small improvements (too much time lost), so we can have enough time to perform more shaking steps
        if new_fit[3] < 0.9999*old_fit[3]:
            return -1
        return 0 # 1
