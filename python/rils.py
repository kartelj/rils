import copy
import time
import math
from random import Random
from sklearn.base import BaseEstimator

from node import Node, NodeArcCos, NodeArcSin, NodeConstant, NodeCos, NodeDivide, NodeExp, NodeLn, NodeMax, NodeMin, NodeMinus, NodeMultiply, NodeSin, NodeSqr, NodeSqrt,NodeVariable, NodePlus
from solution import Solution
import warnings
warnings.filterwarnings("ignore")

class RILS(BaseEstimator):

    def __init__(self, max_fit_calls=100000, max_seconds=100, complexity_penalty=0.001, error_tolerance=0.000001,random_state=0):
        self.max_seconds = max_seconds
        self.max_fit_calls = max_fit_calls
        self.complexity_penalty = complexity_penalty
        self.random_state = random_state
        self.error_tolerance = error_tolerance


    def __reset(self):
        self.model = None
        self.ls_it = 0
        self.main_it = 0
        self.last_improved_it = 0
        self.time_start = 0
        self.time_elapsed = 0
        self.rg = Random(self.random_state)
        Solution.clearStats()
        Node.reset_node_value_cache()

    def __setup_nodes(self, variableCount):
        self.allowed_nodes=[NodeConstant(-1), NodeConstant(0), NodeConstant(0.5), NodeConstant(1), NodeConstant(2), NodeConstant(math.pi)]
        for i in range(variableCount):
            self.allowed_nodes.append(NodeVariable(i))
        self.allowed_nodes+=[NodePlus(), NodeMinus(), NodeMultiply(), NodeDivide(), NodeSqr(), NodeSqrt(),NodeLn(), NodeExp(),NodeSin(), NodeCos(), NodeArcSin(), NodeArcCos()]

    def fit(self, X, y):
        x_all = copy.deepcopy(X)
        y_all = copy.deepcopy(y)
        # take 1% of points or at least 100 points initially 
        n = int(0.01*len(x_all))
        if n<100:
            n=100
        print("Taking "+str(n)+" points initially.")
        X = x_all[:n]
        y = y_all[:n]
        size_increased_main_it = 0

        self.__reset()
        self.start = time.time()
        if len(X) == 0:
            raise Exception("Input feature data set (X) cannot be empty.")
        if len(X)!=len(y):
            raise Exception("Numbers of feature vectors (X) and target values (y) must be equal.")
        self.__setup_nodes(len(X[0]))
        best_solution =  Solution([NodeConstant(0)], self.complexity_penalty)
        best_fitness = best_solution.fitness(X, y)
        self.main_it = 0
        while self.time_elapsed<self.max_seconds and Solution.fit_calls<self.max_fit_calls: 
            new_solution = self.preturb(best_solution, len(X[0]))
            new_solution.simplify_whole(len(X[0]))
            new_fitness = new_solution.fitness(X, y)
            new_solution = self.LS_best(new_solution, X, y)
            new_fitness = new_solution.fitness(X, y, False)
            if self.compare_fitness(new_fitness, best_fitness)<0:
                best_solution = copy.deepcopy(new_solution)
                best_fitness = new_fitness
            else:
                if n<len(x_all) and (self.main_it-size_increased_main_it)>=10:
                    n*=2
                    if n>len(x_all):
                        n = len(x_all)
                    print("Increasing data count to "+str(n))
                    X = x_all[:n]
                    y = y_all[:n]
                    size_increased_main_it = self.main_it
                    Node.reset_node_value_cache()

            self.time_elapsed = time.time()-self.start
            print("%d/%d. t=%.1f R2=%.7f RMSE=%.7f size=%d factors=%d mathErr=%d fitCalls=%d fitFails=%d cHits=%d cTries=%d cPerc=%.1f cSize=%d\n                                                                          expr=%s"
            %(self.main_it,self.ls_it, self.time_elapsed, 1-best_fitness[0], best_fitness[1],best_solution.size(), len(best_solution.factors), Solution.math_error_count, Solution.fit_calls, Solution.fit_fails, Node.cache_hits, Node.cache_tries, Node.cache_hits*100.0/Node.cache_tries, len(Node.node_value_cache), best_solution))
            self.main_it+=1
            if best_fitness[0]<self.error_tolerance and best_fitness[1] < self.error_tolerance:
                break
        self.model = best_solution

    def fit_report_string(self, X, y):
        if self.model==None:
            raise Exception("Model is not build yet. First call fit().")
        fitness = self.model.fitness(X,y, False)
        return "maxTime={0}\tmaxFitCalls={1}\tseed={2}\tsizePenalty={3}\tR2={4:.7f}\tRMSE={5:.7f}\tsize={6}\tsec={7:.1f}\tmainIt={8}\tlsIt={9}\tfitCalls={10}\texpr={11}".format(
            self.max_seconds,self.max_fit_calls,self.random_state,self.complexity_penalty, 1-fitness[0], fitness[1], fitness[2], self.time_elapsed,self.main_it, self.ls_it,Solution.fit_calls, self.model)

    def predict(self, X):
        Node.reset_node_value_cache()
        return self.model.evaluate_all(X, False)

    def preturb(self, solution:Solution,varCnt):
        shaked_solution = copy.deepcopy(solution)
        shaked_solution.normalize_constants()
        shaked_solution.simplify_whole(varCnt)
        shaked_solution.join()
        j = self.rg.randrange(len(shaked_solution.factors))
        all_subtrees = list(filter(lambda x: x.arity, shaked_solution.factors[j].all_nodes_exact()))
        if len(all_subtrees)==0: # this is the case when we have constant or variable, so we just change the root
            shaked_solution.factors[j] = self.random_change(shaked_solution.factors[j])
        else:
            i = self.rg.randrange(len(all_subtrees))
            refNode = all_subtrees[i]
            # give chance to root of the tree to get selected as well sometimes -- and not only when it is constant or single variable
            if refNode==shaked_solution.factors[j] and self.rg.random()<=1.0/(1+refNode.arity):
                shaked_solution.factors[j] = self.random_change(shaked_solution.factors[j])
            else:
                if refNode.arity == 1:
                    newNode = self.random_change(refNode.left, refNode, True)
                    refNode.left = newNode
                elif refNode.arity==2:
                    if self.rg.random()<0.5:
                        newNode = self.random_change(refNode.left, refNode, True)
                        refNode.left = newNode
                    else:
                        newNode = self.random_change(refNode.right, refNode, False)
                        refNode.right = newNode
                else:
                    print("WARNING: Preturbation is not performed!")   
        return shaked_solution

    
    def LS_best(self, solution: Solution, X, y):
        best_fitness = solution.fitness(X, y)
        best_solution = copy.deepcopy(solution)
        impr = True
        while impr or impr2:
            impr = False
            impr2 = False
            self.ls_it+=1
            self.time_elapsed = time.time()-self.start
            if self.time_elapsed>self.max_seconds or Solution.fit_calls>self.max_fit_calls:
                break

            old_best_fitness = best_fitness
            old_best_solution = copy.deepcopy(best_solution)
                
            impr, best_solution, best_fitness = self.LS_best_change_iteration(best_solution, X, y, True)
            if not impr:
                best_solution = copy.deepcopy(old_best_solution)
                impr2, best_solution, best_fitness = self.LS_best_change_iteration(best_solution, X, y, True, True)
            if impr or impr2:
                best_solution.simplify_whole(len(X[0]))
                best_fitness = best_solution.fitness(X, y, False)
                if self.compare_fitness(best_fitness, old_best_fitness)>=0:
                    impr = False
                    impr2 = False
                    best_solution = old_best_solution
                    best_fitness = old_best_fitness
                    print("REVERTING back to old best "+str(best_solution))
                else:
                    print("IMPROVED with LS-change impr="+str(impr)+" impr2="+str(impr2)+" "+str(1-best_fitness[0])+"  "+str(best_solution))
                continue  
        return best_solution

   
    def LS_best_change_iteration(self, solution: Solution, X, y, cache, joined=False):
        best_fitness = solution.fitness(X, y, False)
        best_solution = copy.deepcopy(solution)
        if joined:
            print("JOINING SOLUTION IN LS")
            solution.join()
        impr = False
        for i in range(len(solution.factors)):
            factor = solution.factors[i]
            factor_subtrees = factor.all_nodes_exact()
            for j in range(len(factor_subtrees)):
                
                self.time_elapsed = time.time()-self.start
                if self.time_elapsed>self.max_seconds or Solution.fit_calls>self.max_fit_calls:
                    return (impr, best_solution, best_fitness)

                ref_node = factor_subtrees[j]

                if ref_node==factor: # this subtree is the whole factore
                    candidates = self.change_candidates(ref_node)
                    for cand in candidates:
                        new_solution = copy.deepcopy(solution)
                        new_solution.factors[i] = cand
                        if joined:
                            new_solution.expand_fast()
                        new_solution = new_solution.fit_constants_OLS(X, y)
                        new_fitness = new_solution.fitness(X, y, cache)
                        if self.compare_fitness(new_fitness, best_fitness)<0:
                            impr = True
                            best_fitness = new_fitness
                            best_solution = copy.deepcopy(new_solution)
                else:
                    if ref_node.arity >= 1:
                        candidates = self.change_candidates(ref_node.left, ref_node, True)
                        for cand in candidates:
                            new_solution = copy.deepcopy(solution)               
                            new_factor_subtrees = new_solution.factors[i].all_nodes_exact()
                            new_factor_subtrees[j].left=cand
                            if joined:
                                new_solution.expand_fast()
                            new_solution = new_solution.fit_constants_OLS(X, y)
                            new_fitness = new_solution.fitness(X, y, cache)
                            if self.compare_fitness(new_fitness, best_fitness)<0:
                                impr = True
                                best_fitness = new_fitness
                                best_solution = copy.deepcopy(new_solution)

                    if ref_node.arity>=2:
                        candidates = self.change_candidates(ref_node.right, ref_node, False)
                        for cand in candidates:
                            new_solution = copy.deepcopy(solution)               
                            new_factor_subtrees = new_solution.factors[i].all_nodes_exact()
                            new_factor_subtrees[j].right=cand
                            if joined:
                                new_solution.expand_fast()
                            new_solution = new_solution.fit_constants_OLS(X, y)
                            new_fitness = new_solution.fitness(X, y, cache)
                            if self.compare_fitness(new_fitness, best_fitness)<0:
                                impr = True
                                best_fitness = new_fitness
                                best_solution = copy.deepcopy(new_solution)

        return (impr, best_solution, best_fitness)

    def random_change(self, old_node: Node, parent=None, is_left_from_parent=None):
        candidates = self.preturb_candidates(old_node, parent, is_left_from_parent)
        if candidates==[]:
            print("Random preturbation failed.")
            return old_node # Preturbation failed
        i = self.rg.randrange(len(candidates))
        candidate = candidates[i]
        return candidate


    def preturb_candidates(self, old_node: Node, parent=None, is_left_from_parent=None):
        candidates = []
        # change variable or constant to another variable
        if old_node.arity==0 and type(old_node)==type(NodeConstant(0)):
            for node in filter(lambda x:type(x)==type(NodeVariable(0)) and x!=old_node, self.allowed_nodes):
                new_node = copy.deepcopy(node)
                candidates.append(new_node)
        # change anything (except constant) to unary operation applied to that -- increases the model size
        if type(old_node)!=type(NodeConstant(0)):
            for node in filter(lambda x:x.arity==1, self.allowed_nodes):
                if not node.is_allowed_left_argument(old_node):
                    continue
                new_node = copy.deepcopy(node)
                new_node.left =copy.deepcopy(old_node)
                new_node.right = None
                candidates.append(new_node)
        # change variable to unary operation applied to that variable
        if type(old_node)==type(NodeVariable(0)):
            for node in filter(lambda x:x.arity==1, self.allowed_nodes):
                if not node.is_allowed_left_argument(old_node):
                    continue
                new_node = copy.deepcopy(node)
                new_node.left =copy.deepcopy(old_node)
                new_node.right = None
                candidates.append(new_node)
        # change unary operation to another unary operation
        if old_node.arity == 1:
            for node in filter(lambda x:x.arity==1 and type(x).__name__ !=type(old_node).__name__, self.allowed_nodes):
                new_node = copy.deepcopy(node)
                new_node.left = copy.deepcopy(old_node.left)
                assert old_node.right==None
                candidates.append(new_node)
        # change one binary operation to another
        if old_node.arity==2:
            for nodeOp in filter(lambda x: x.arity==2 and type(x).__name__ !=type(old_node).__name__, self.allowed_nodes):
                if (not nodeOp.is_allowed_left_argument(old_node.left)) or (not nodeOp.is_allowed_right_argument(old_node.right)):
                    continue
                new_node = copy.deepcopy(nodeOp)
                new_node.left = copy.deepcopy(old_node.left)
                new_node.right = copy.deepcopy(old_node.right)
                candidates.append(new_node) 
            # swap left and right side if not symmetric op
            if not old_node.symmetric:
                new_node = copy.deepcopy(old_node)
                new_node.left = copy.deepcopy(old_node.right)
                new_node.right = copy.deepcopy(old_node.left)
                candidates.append(new_node)
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
        return candidates

    def change_candidates(self, old_node:Node, parent=None, is_left_from_parent=None):
        candidates = []

        if type(old_node)==type(NodeConstant(0)):
            # change constant to something multiplied with it
            for mult in [0.01, 0.1, 0.2, 0.5, 0.8, 0.9,1.1,1.2, 2, 5, 10, 20, 50, 100]:
                candidates.append(NodeConstant(old_node.value*mult))

        if old_node.arity>=1:
            all_left_subtrees = old_node.left.all_nodes_exact()
            for ls in all_left_subtrees:
                candidates.append(copy.deepcopy(ls))
            #candidates.append(copy.deepcopy(old_node.left))
        if old_node.arity>=2:
            all_right_subtrees = old_node.right.all_nodes_exact()
            for rs in all_right_subtrees:
                candidates.append(copy.deepcopy(rs))
            #candidates.append(copy.deepcopy(old_node.right))
        

        for node in filter(lambda x:x.arity==0 and x!=old_node, self.allowed_nodes):
            candidates.append(copy.deepcopy(node))

        # change anything to unary operation applied to that -- increases the model size
        for node in filter(lambda x:x.arity==1, self.allowed_nodes):
            if not node.is_allowed_left_argument(old_node):
                continue
            new_node = copy.deepcopy(node)
            new_node.left =copy.deepcopy(old_node)
            new_node.right = None
            candidates.append(new_node)
        # change unary operation to another unary operation
        if old_node.arity == 1:
            for node in filter(lambda x:x.arity==1 and type(x).__name__ !=type(old_node).__name__, self.allowed_nodes):
                new_node = copy.deepcopy(node)
                new_node.left = copy.deepcopy(old_node.left)
                assert old_node.right==None
                candidates.append(new_node)
        # change anything to binary operation with some variable or constant -- increases the model size
        # or with some part of itself
        node_args = list(filter(lambda x: x.arity==0, self.allowed_nodes))+[copy.deepcopy(x) for x in old_node.all_nodes_exact()]
        for node_arg in node_args:
            for node_op in filter(lambda x: x.arity==2, self.allowed_nodes):
                if not node_op.is_allowed_right_argument(node_arg) or not node_op.is_allowed_left_argument(old_node):
                    continue
                new_node = copy.deepcopy(node_op)
                new_node.left = copy.deepcopy(old_node)
                new_node.right = copy.deepcopy(node_arg)
                candidates.append(new_node)
                if not node_op.symmetric and node_op.is_allowed_right_argument(old_node) and node_op.is_allowed_left_argument(node_arg):
                    new_node = copy.deepcopy(node_op)
                    new_node.right = copy.deepcopy(old_node)
                    new_node.left = copy.deepcopy(node_arg)
                    candidates.append(new_node)
        # change one binary operation to another
        if old_node.arity==2:
            for node_op in filter(lambda x: x.arity==2 and type(x).__name__ !=type(old_node).__name__, self.allowed_nodes):
                if (not node_op.is_allowed_left_argument(old_node.left)) or (not node_op.is_allowed_right_argument(old_node.right)):
                    continue
                new_node = copy.deepcopy(node_op)
                new_node.left = copy.deepcopy(old_node.left)
                new_node.right = copy.deepcopy(old_node.right)
                candidates.append(new_node) 
            # swap left and right side if not symmetric op
            if not old_node.symmetric:
                new_node = copy.deepcopy(old_node)
                new_node.left = copy.deepcopy(old_node.right)
                new_node.right = copy.deepcopy(old_node.left)
                candidates.append(new_node)

        # filtering not allowed candidates (because of the parent)
        filtered_candidates = []
        if parent is not None:
            for c in candidates:
                if is_left_from_parent and not parent.is_allowed_left_argument(c):
                    continue
                if not is_left_from_parent and not parent.is_allowed_right_argument(c):
                    continue
                filtered_candidates.append(c)
            candidates = filtered_candidates
        return candidates

    def compare_fitness(self, new_fit, old_fit):
        if math.isnan(new_fit[0]):
            return 1
        if self.complexity_penalty is not None:
            new_tot = (1+new_fit[0])*(1+new_fit[2]*self.complexity_penalty) *(1+new_fit[1]) 
            old_tot = (1+old_fit[0])*(1+old_fit[2]*self.complexity_penalty) *(1+old_fit[1]) 
            if new_tot<old_tot-self.error_tolerance:
                return -1
            if new_tot>old_tot+self.error_tolerance:
                return 1
            return 0
        else:
            if new_fit[0]<old_fit[0]:
                return -1
            if new_fit[0]>old_fit[0]:
                return 1
            if new_fit[2]<old_fit[2]:
                return -1
            if new_fit[2]>old_fit[2]:
                return 1
            if new_fit[1]<old_fit[1]:
                return -1
            if new_fit[1]>old_fit[1]:
                return 1
            return 0

