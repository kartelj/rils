import copy
from math import e, inf
import time
from .node import NodeAbs, NodeArcCos, NodeArcSin, NodeArcTan, NodeCeil, NodeConstant, NodeCos, NodeExp, NodeFloor, NodeLn, NodeMax, NodeMin, NodeMultiply, NodePlus, NodePow, NodeSgn, NodeSin, NodeTan, NodeVariable
from sympy import *
from sympy.core.numbers import ImaginaryUnit
from sympy.core.symbol import Symbol
import numpy as np
import statsmodels.api as sma
import hashlib
from scipy.stats import pearsonr
from operator import add

class Solution:

    math_error_count = 0
    fit_calls = 1
    fit_fails = 0
    lr_solving_time = 0
    fit_time = 0
    #factor_value_cache = {}
    #fact_value_calls = 1
    #fact_value_hits = 0
    #factor_correl_cache = {}
    #fact_correl_calls = 1
    #fact_correl_hits = 0

    @classmethod
    def clearStats(cls):
        cls.math_error_count=0
        cls.fit_fails = 0
        cls.fit_calls=1
        cls.lr_solving_time = 0
        cls.fit_time = 0
        #cls.factor_value_cache = {}
        #cls.fact_value_calls = 1
        #cls.fact_value_hits = 0

    def __init__(self, factors):
        # factors are basically subexpressions that enter linear combination, e.g. 2*x+3y*sin(x*y) --> factors = [2x, 3y*sin(x*y)]
        self.factors =  copy.deepcopy(factors)

    def __str__(self) -> str:
        return "+".join([str(x) for x in self.factors])
    
    def __eq__(self, other):
        return str(self) == str(other)
    
    def __hash__(self):
        # Stable hash (same for different runs, unlike standard hash() method)
        h = hashlib.sha1(str(self).encode("ascii"))
        return int(h.hexdigest(), 16)
        #return hash(str(self))
    
    def evaluate_all_new(self, X, cache):
        self.sympy_expr = SymExpr(str(self), X.shape[1])
        yp = self.sympy_expr.eval(X)
        return yp

    def evaluate_all(self,X, cache):
        yp = np.zeros(len(X))
        for fact in self.factors:
            fyp = fact.evaluate_all(X, cache)
            yp = list(map(add, yp, fyp))
        return yp

    def size(self):
        #return self.sympy_expr.count_nodes()
        totSize = len(self.factors) 
        for fact in self.factors:
            totSize+=fact.size()
        return totSize
    
    def size_non_linear(self):
        totSize = 1
        for fact in self.factors:
            totSize+=fact.size_non_linear()
        return totSize
    
    def size_operators_only(self):
        totSize = len(self.factors)
        for fact in self.factors:
            totSize+=fact.size_operators_only()
        return totSize

    def fit_constants_OLS(self, X, y):
        start = time.time()
        new_factors = []
        for fact in self.factors:
            if fact.contains_type(type(NodeVariable(0))):
                new_factors.append(copy.deepcopy(fact))
        Xnew = np.zeros((len(X), len(new_factors)))
        try:
            local_cache = {}
            for i in range(len(new_factors)):
                fiX = new_factors[i].evaluate_all(X, True)
                local_cache[new_factors[i]]=fiX
                Xnew[:, i] = fiX
            X2_new = sma.add_constant(Xnew)
            est = sma.OLS(y, X2_new)
            fit_info = est.fit()
            #print(fitInfo.summary())
            signLevel = 0.05
            params = fit_info.params
            final_factors = []
            p_values = fit_info.pvalues
            if p_values[0]<=signLevel:
                free_element = NodeConstant(params[0])
                final_factors.append(free_element)
                #Solution.factor_value_cache[free_element] = [free_element.value]*len(X)
            #else:
            #    final_factors.append(NodeConstant(0))
            for i in range(1, len(params)):
                if p_values[i]>signLevel:
                    continue
                fi_old = copy.deepcopy(new_factors[i-1])
                new_fact = NodeMultiply()
                coef = params[i]
                new_fact.left = NodeConstant(coef)
                new_fact.right = fi_old
                final_factors.append(new_fact)
                fiX = local_cache[fi_old]
                #new_fiX = list(map(lambda x : new_fact.left.value*x, fiX))
                #Solution.factor_value_cache[new_fact] = new_fiX
            new_sol = Solution(final_factors)
            return new_sol
        except Exception as ex:
            Solution.math_error_count+=1
            print("OLS error "+str(ex))
            return copy.deepcopy(self)
        finally:
            Solution.lr_solving_time+=(time.time()-start)
        
    # https://stackoverflow.com/questions/55367024/fastest-way-of-solving-linear-least-squares
    #custom OLS by LU factorization
    def ord_lu(self, X, y):
        A = X.T @ X
        b = X.T @ y
        beta = solve(A, b, overwrite_a=True, overwrite_b=True,
                    check_finite=False)
        return beta
    
    # normalize linear elements of the factor: c1+c2*f = f  or c2*f=f  or c1+f = f
    def normalize_factor(self, fact):
        fact_norm = copy.deepcopy(fact)
        if type(fact_norm)==type(NodePlus()):
            # remove additive part
            if type(fact_norm.left)==type(NodeConstant(0)):
                fact_norm = fact_norm.right
            elif type(fact_norm.right)==type(NodeConstant(0)):
                fact_norm = fact_norm.left
        if type(fact_norm)==type(NodeMultiply()):
            # remove multiplier
            if type(fact_norm.left)==type(NodeConstant(0)):
                fact_norm = fact_norm.right
            elif type(fact_norm.right)==type(NodeConstant(0)):
                fact_norm = fact_norm.left
        return fact_norm


    def fit_constants_LSTSQ(self, X, y)->'Solution':
        y = np.array(y)
        start = time.time()
        new_factors = []
        fiXall = []
        for fact in self.factors:
            if fact.contains_type(type(NodeVariable(0))):
                try:
                    '''
                    Solution.fact_correl_calls+=1
                    if fact in Solution.factor_correl_cache:
                        #print("FOUND in correl cache "+str(fact) +" normalized "+str(fact_norm))
                        r, pvalue = Solution.factor_correl_cache[fact]
                        Solution.fact_correl_hits+=1
                    else:
                        #print("Not in correl cache "+str(fact) +" normalized "+str(fact_norm))
                        Solution.fact_value_calls+=1
                        if fact in Solution.factor_value_cache:
                            Solution.fact_value_hits+=1
                            fiX = Solution.factor_value_cache[fact]
                        else:
                            fiX = fact.evaluate_all(X, True)
                            Solution.factor_value_cache[fact] = fiX
                            '''
                    fiX = fact.evaluate_all(X, True)
                    r, pvalue = pearsonr(fiX, y)
                    #Solution.factor_correl_cache[fact]=(r, pvalue)
                    #if len(Solution.factor_correl_cache)>100000:
                    #    Solution.factor_correl_cache.clear()
                    if pvalue<0.1:
                        new_factors.append(copy.deepcopy(fact))
                        fiXall.append(fiX)
                    #else:
                        #print("Factor "+str(fact)+" not significantly correlated to y.")
                except:
                    Solution.math_error_count+=1
                    #print("Skipping "+str(fact))
        #if len(new_factors)==0:
        #    return Solution([])
        Xnew = np.zeros((len(X), len(new_factors)+1))
        try:
            for j in range(len(X)):
                Xnew[j, 0] = 1
            for i in range(len(new_factors)):
                Xnew[:, i+1] = fiXall[i]

            final_factors = []
            
            # https://stackoverflow.com/questions/55367024/fastest-way-of-solving-linear-least-squares
            params = self.ord_lu(Xnew, y)
            #params, res, rank, s = lstsq(Xnew, y, lapack_driver='gelsy', check_finite=False)

            min_coef = 0.001
            if abs(params[0])>=min_coef:
                free_element = NodeConstant(params[0])#, ols_fitted=True)
                final_factors.append(free_element)
                # add to factor cache
                #Solution.factor_value_cache[free_element] = [free_element.value]*len(X)
            for i in range(1, len(params)):
                coef = params[i]
                if abs(coef)<min_coef:
                    continue
                fi_old = copy.deepcopy(new_factors[i-1])
                new_fact = NodeMultiply()
                new_fact.left = NodeConstant(coef)#, ols_fitted=True)
                new_fact.right = fi_old
                final_factors.append(new_fact)
                fiX = fiXall[i-1]
                #new_fiX = list(map(lambda x : new_fact.left.value*x, fiX))
                #Solution.factor_value_cache[new_fact] = new_fiX
            new_sol = Solution(final_factors)
            return new_sol
        except Exception as ex:
            Solution.math_error_count+=1
            print("OLS error "+str(ex))
            return copy.deepcopy(self)
        finally:
            Solution.lr_solving_time+=(time.time()-start)

    def normalize_constants(self):
        for fact in self.factors:
            fact.normalize_constants()

    def expand_to_factors(self, expression):
        expr_str_before = str(expression)
        expr_exp = expand(expr_str_before)
        my_factors = []
        try:
            if type(expr_exp)!=Add:
                factors = [copy.deepcopy(expr_exp)]
            else:
                factors = list(expr_exp.args)
            for factor in factors:
                myFactor = Solution.convert_to_my_nodes(factor)
                my_factors.append(myFactor)
        except Exception as ex:
            #print("Expand to factors error: "+str(ex))
            my_factors = None
            Solution.math_error_count+=1
        return my_factors

    def join_factors(self, factors):
        if len(factors)==0:
            return NodeConstant(0)
        if len(factors)==1:
            return copy.deepcopy(factors[0])
        expression = NodePlus()
        expression.left = copy.deepcopy(factors[0])
        for i in range(1, len(factors)):
            expression.right = copy.deepcopy(factors[i])
            if i<len(factors)-1:
                new_expression = NodePlus()
                new_expression.left = expression
                expression = new_expression
        return expression

    def join(self):
        self.factors = [self.join_factors(self.factors)]

    def simplify_whole(self, varCnt):
        vars = ['v'+str(i) for i in range(varCnt)]
        vars_str = ' '.join(vars)
        symbols(vars_str, real=True)
        expression = self.join_factors(self.factors)
        expression_str = str(expression)
        expression_simpy = sympify(expression_str).evalf()
        try:
            new_expression = Solution.convert_to_my_nodes(expression_simpy)
            self.factors = [new_expression]
            #new_factors = self.expand_to_factors(new_expression)
            #if new_factors is not None:
            #    self.factors = new_factors
            #else:
            #    raise Exception("Expansion to factors failed.")
        except Exception as ex:
            print("SimplifyWhole: "+str(ex) +" for sympy_exp "+str(expression_simpy))
            print("Simplifying factors instead.")
            self.simplify_factors(varCnt)

    def simplify_factors(self, varCnt):
        vars = ['v'+str(i) for i in range(varCnt)]
        vars_str = ' '.join(vars)
        symbols(vars_str)
        new_factors = []
        for i in range(len(self.factors)):
            fact = self.factors[i]
            expr_str_before = str(fact)
            expr_simpl = sympify(expr_str_before).evalf()
            try:
                newFact = Solution.convert_to_my_nodes(expr_simpl)
                if type(newFact)==type(NodeConstant(0)) and newFact.value==0:
                    continue
                new_factors.append(newFact)
            except Exception as ex:
                print(ex)
        self.factors = new_factors

    def expand(self):
        new_factors = []
        for fact in self.factors:
            expr_str_before = str(fact)
            expr_exp = expand(expr_str_before)
            if type(expr_exp)!=Add:
                expanded_fact = [expr_exp]
            else:
                expanded_fact = list(expr_exp.args)
            try:
                my_expanded_fact = []
                for f in expanded_fact:
                    new_factor = Solution.convert_to_my_nodes(f)
                    my_expanded_fact.append(new_factor)  
                # when all converted correctly, than add them to the final list
                for f in my_expanded_fact:
                    new_factors.append(f)
            except Exception as ex:
                print(ex) # conversion to my nodes failed, so keeping original fact (non-expanded)
                new_factors.append(fact)
        self.factors = new_factors

    def contains_type(self, type):
        for fact in self.factors:
            if fact.contains_type(type):
                return True
        return False

    @staticmethod
    def convert_to_my_nodes(sympy_node):
        if type(sympy_node)==ImaginaryUnit:
            raise Exception("Not working with imaginary (complex) numbers.")
        sub_nodes = []
        for i in range(len(sympy_node.args)):
            sub_nodes.append(Solution.convert_to_my_nodes(sympy_node.args[i]))

        if len(sympy_node.args)==0:
            if type(sympy_node)==Symbol:
                if str(sympy_node)=="e":
                    return NodeConstant(e)
                try:
                    index = int(str(sympy_node).replace("x",""))
                    return NodeVariable(index)
                except Exception as ex:
                    print(sympy_node)
                    print(ex)
            else:
                if str(sympy_node)=="pi":
                    return NodeConstant(pi)
                elif str(sympy_node)=="1/2":
                    return NodeConstant(0.5)
                elif str(sympy_node)=="-1/2":
                    return NodeConstant(-0.5)
                elif str(sympy_node)=="1/4":
                    return NodeConstant(0.25)
                elif str(sympy_node)=="-1/4":
                    return NodeConstant(-0.25)
                elif str(sympy_node)=="1/10":
                    return NodeConstant(0.1)
                elif str(sympy_node)=="-1/10":
                    return NodeConstant(-0.1)
                return NodeConstant(float(str(sympy_node)))

        if len(sympy_node.args)==1:
            if type(sympy_node)==exp:
                new = NodeExp()
            elif type(sympy_node)==cos:
                new = NodeCos()
            elif type(sympy_node)==sin:
                new = NodeSin()
            elif type(sympy_node)==tan:
                new = NodeTan()
            elif type(sympy_node)==acos:
                new = NodeArcCos()
            elif type(sympy_node)==asin:
                new = NodeArcSin()
            elif type(sympy_node)==atan:
                new = NodeArcTan()
            elif type(sympy_node)==log:
                new = NodeLn()
            elif type(sympy_node).__name__=="sgn":
                new = NodeSgn()
            elif type(sympy_node)==Abs:
                new = NodeAbs()
            elif type(sympy_node)==floor:
                new = NodeFloor()
            elif type(sympy_node)==ceiling:
                new = NodeCeil()
            elif type(sympy_node)==re:
                new = Solution.convert_to_my_nodes(sympy_node.args[0])
            elif type(sympy_node)==im:
                return NodeConstant(0) # not doing with imaginary numbers
            else:
                raise Exception("Non defined node "+str(sympy_node))
            new.left = sub_nodes[0]
            return new
        elif len(sympy_node.args)>=2:
            if type(sympy_node)==Mul:
                new = NodeMultiply()
            elif type(sympy_node)==Add:
                new = NodePlus()
            elif type(sympy_node)==Pow:
                new = NodePow()
            elif type(sympy_node)==Max:
                new = NodeMax()
            elif type(sympy_node)==Min:
                new = NodeMin()
            else:
                raise Exception("Non defined node "+str(sympy_node))
            new.left = sub_nodes[0]
            new.right = sub_nodes[1]
            for i in range(2, len(sympy_node.args)):
                root = copy.deepcopy(new)
                root.left = new
                root.right = sub_nodes[i]
                new = root
            return new
        else:
            raise Exception("Unrecognized node "+str(sympy_node))