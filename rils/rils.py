import copy
import time
import math
from random import Random
from sklearn.base import BaseEstimator
from abc import abstractmethod
import copy
from math import acos, asin, atan, ceil, cos, exp, floor, inf, log, sin, sqrt, tan
from sympy import *
from sympy.core.numbers import ImaginaryUnit
from sympy.core.symbol import Symbol
import numpy as np
import statsmodels.api as sma
from statistics import mean
from math import nan, sqrt
from statistics import mean
from numpy.random import RandomState

import warnings
warnings.filterwarnings("ignore")

class Node:
    tmp = -1
    node_value_cache = {}
    cache_hits = 0
    cache_tries = 1

    VERY_SMALL = 0.0001

    @classmethod
    def reset_node_value_cache(cls):
        cls.node_value_cache = {}
        cls.cache_hits = 0
        cls.cache_tries = 1    

    def __init__(self):
        self.arity = 0
        self.left = None
        self.right = None
        self.symmetric = True

    @abstractmethod
    def evaluate_inner(self,X, a=None, b=None):
        pass

    def evaluate(self, X):
        if self.arity==0:
            return self.evaluate_inner(X, None, None)
        elif self.arity == 1:
            leftVal = self.left.evaluate(X)
            return self.evaluate_inner(X,leftVal, None)
        elif self.arity==2:
            leftVal = self.left.evaluate(X)
            rightVal = self.right.evaluate(X)
            return self.evaluate_inner(X,leftVal,rightVal)
        else:
            raise Exception("Arity > 2 is not allowed.")

    def evaluate_all(self, X, cache):
        key = str(self)
        Node.cache_tries+=1
        yp = []
        if cache and key in Node.node_value_cache:
            Node.cache_hits+=1
            yp = Node.node_value_cache[key]
        else:
            if self.arity==2:
                left_yp = self.left.evaluate_all(X, cache)
                right_yp = self.right.evaluate_all(X, cache)
                for i in range(len(X)):
                    ypi = self.evaluate_inner(X[i], left_yp[i], right_yp[i])
                    yp.append(ypi)
            elif self.arity==1:
                left_yp = self.left.evaluate_all(X, cache)
                for i in range(len(X)):
                    ypi = self.evaluate_inner(X[i], left_yp[i], None)
                    yp.append(ypi)
            elif self.arity==0:
                for i in range(len(X)):
                    ypi = self.evaluate_inner(X[i],None, None)
                    yp.append(ypi)
            if cache:
                Node.node_value_cache[key]=yp
                if len(Node.node_value_cache)==5000:
                    Node.node_value_cache.clear()
        return yp

    def expand_fast(self):
        if type(self)==type(NodePlus()) or type(self)==type(NodeMinus()): # TODO: check if minus is happening
            leftFact = self.left.expand_fast()
            right = self.right
            if type(self)==type(NodeMinus()):
                right = NodeMultiply()
                right.left = NodeConstant(-1)
                right.right = copy.deepcopy(self.right)
            rightFact = right.expand_fast()
            return leftFact+rightFact
        return [copy.deepcopy(self)]

    def is_allowed_left_argument(self, node_arg):
        return True

    def is_allowed_right_argument(self, node_arg):
        return True

    def __eq__(self, object):
        if object is None:
            return False
        return str(self)==str(object)

    def __hash__(self):
        return hash(str(self))

    def all_nodes_exact(self):
        thisList = [self]
        if self.arity==0:
            return thisList
        elif self.arity==1:
            return thisList+self.left.all_nodes_exact()
        elif self.arity==2:
            return thisList+self.left.all_nodes_exact()+self.right.all_nodes_exact()
        else:
            raise Exception("Arity greater than 2 is not allowed.")

    def size(self):
        leftSize = 0
        if self.left!=None:
            leftSize = self.left.size()
        rightSize = 0
        if self.right!=None:
            rightSize = self.right.size()
        return 1+leftSize+rightSize

    def contains_type(self, searchType):
        if type(self)==searchType:
            return True
        if self.left!=None and self.left.contains_type(searchType):
            return True
        if self.right!=None and self.right.contains_type(searchType):
            return True
        return False

    def normalize_constants(self):
        if type(self)==type(NodeConstant(0)):
            self.value = 1
            return
        if self.arity>=1:
            self.left.normalize_constants()
        if self.arity>=2:
            self.right.normalize_constants()

class NodeConstant(Node):
    def __init__(self, value):
        super().__init__()
        self.arity = 0
        self.value = round(value,13)

    def evaluate_inner(self,X, a, b):
        return self.value

    def __str__(self):
        return str(self.value)

class NodeVariable(Node):
    def __init__(self, index):
        super().__init__()
        self.arity = 0
        self.index = index

    def evaluate_inner(self,X, a, b):
        if self.index>=len(X):
            raise Exception("Variable with index "+str(self.index)+" does not exist.")
        return X[self.index]

    def __str__(self):
        return "X_"+str(self.index)

class NodePlus(Node):
    def __init__(self):
        super().__init__()
        self.arity = 2

    def evaluate_inner(self,X, a, b):
        return a+b

    def is_allowed_left_argument(self, node_arg):
        if node_arg == NodeConstant(0):
            return False
        return True

    def is_allowed_right_argument(self, node_arg):
        return self.is_allowed_left_argument(node_arg)

    def __str__(self):
        return "("+str(self.left)+"+"+str(self.right)+")" 

class NodeMinus(Node):
    def __init__(self):
        super().__init__()
        self.arity = 2
        self.symmetric = False

    def evaluate_inner(self,X, a, b):
        return a - b

    def is_allowed_left_argument(self, node_arg):
        if self.right==node_arg:
            return False
        return True

    def is_allowed_right_argument(self, node_arg):
        if node_arg == NodeConstant(0):
            return False
        if self.left == node_arg:
            return False
        return True

    def normalize(self):
        if type(self.right)==type(NodeConstant(0)):
            new_left =  NodeConstant(self.right.value*(-1))
            new_right = copy.deepcopy(self.left)
            self = NodePlus()
            self.left = new_left
            self.right = new_right
            return self.normalize()
        else:
            return super().normalize()

    def __str__(self):
        return "("+str(self.left)+"-"+str(self.right)+")"

class NodeMultiply(Node):
    def __init__(self):
        super().__init__()
        self.arity = 2

    def evaluate_inner(self,X, a, b):
        return a*b

    def is_allowed_left_argument(self, node_arg):
        if node_arg == NodeConstant(1):
            return False
        return True

    def is_allowed_right_argument(self, node_arg):
        return self.is_allowed_left_argument(node_arg)

    def __str__(self):
        return "("+str(self.left)+"*"+str(self.right)+")"

class NodeDivide(Node):
    def __init__(self):
        super().__init__()
        self.arity = 2
        self.symmetric = False

    def evaluate_inner(self,X, a, b):
        if b==0:
            b= Node.VERY_SMALL
        return a/b

    def is_allowed_left_argument(self, node_arg):
        if self.right == node_arg:
            return False
        return True


    def is_allowed_right_argument(self, node_arg):
        if node_arg == NodeConstant(0):
            return False
        if self.left == node_arg:
            return False
        return True

    def __str__(self):
        return "("+str(self.left)+"/"+str(self.right)+")"

class NodeMax(Node):
    def __init__(self):
        super().__init__()
        self.arity = 2
        self.symmetric = True

    def evaluate_inner(self,X, a, b):
        return max(a, b)

    def __str__(self):
        return "max("+str(self.left)+","+str(self.right)+")"

class NodeMin(Node):
    def __init__(self):
        super().__init__()
        self.arity = 2
        self.symmetric = True

    def evaluate_inner(self,X, a, b):
        return min(a, b)

    def __str__(self):
        return "min("+str(self.left)+","+str(self.right)+")"

class NodePow(Node):
    def __init__(self):
        super().__init__()
        self.arity = 2
        self.symmetric = False

    def evaluate_inner(self,X, a, b):
        if a==0 and b<=0:
            a = Node.VERY_SMALL
        return pow(a, b)

    def is_allowed_right_argument(self, node_arg):
        if type(node_arg)!=type(NodeConstant(0)):
            return False
        return True

    def is_allowed_left_argument(self, node_arg):
        if node_arg.contains_type(type(NodePow())) or node_arg.contains_type(type(NodeExp())): # TODO: avoid complicated bases
            return False
        if type(node_arg)==type(NodeConstant(0)) and node_arg.value==0:
            return False
        return True

    def __str__(self):
        return "pow("+str(self.left)+","+str(self.right)+")"

class NodeCos(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        return cos(a)

    def is_allowed_left_argument(self, node_arg): # avoid complicated expression
        if node_arg.contains_type(type(NodeCos())) or node_arg.contains_type(type(NodeSin())) or node_arg.contains_type(type(NodeArcSin())) or node_arg.contains_type(type(NodeArcCos())):
            return False
        return True

    def __str__(self):
        return "cos("+str(self.left)+")"

class NodeArcCos(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        return acos(a)

    def is_allowed_left_argument(self, node_arg):
        if type(node_arg) == type(NodeConstant(0)) and (node_arg.value<-1 or node_arg.value>1):
            return False
        if node_arg.contains_type(type(NodeCos())) or node_arg.contains_type(type(NodeSin())) or node_arg.contains_type(type(NodeArcSin())) or node_arg.contains_type(type(NodeArcCos())):
            return False
        return True

    def __str__(self):
        return "acos("+str(self.left)+")"

class NodeSin(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        return sin(a)
    
    def is_allowed_left_argument(self, node_arg):
        if node_arg.contains_type(type(NodeCos())) or node_arg.contains_type(type(NodeSin())) or node_arg.contains_type(type(NodeArcSin())) or node_arg.contains_type(type(NodeArcCos())):
            return False
        return True

    def __str__(self):
        return "sin("+str(self.left)+")"

class NodeTan(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def is_allowed_left_argument(self, nodeArg):
        if type(nodeArg) == type(NodeConstant(0)) and (nodeArg.value<-1 or nodeArg.value>1):
            return False
        return True

    def evaluate_inner(self,X, a, b):
        return tan(a)

    def __str__(self):
        return "tan("+str(self.left)+")"

class NodeArcSin(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        return asin(a)

    def is_allowed_left_argument(self, node_arg):
        if type(node_arg) == type(NodeConstant(0)) and (node_arg.value<-1 or node_arg.value>1):
            return False
        if node_arg.contains_type(type(NodeCos())) or node_arg.contains_type(type(NodeSin())) or node_arg.contains_type(type(NodeArcSin())) or node_arg.contains_type(type(NodeArcCos())):
            return False
        return True

    def __str__(self):
        return "asin("+str(self.left)+")"

class NodeArcTan(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        return atan(a)

    def __str__(self):
        return "atan("+str(self.left)+")"

class NodeExp(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        return exp(a)

    def is_allowed_left_argument(self, node_arg): # avoid complicated expressions
        if node_arg.contains_type(type(NodeCos())) or node_arg.contains_type(type(NodeSin())) or node_arg.contains_type(type(NodeArcSin())) or node_arg.contains_type(type(NodeArcCos())) or node_arg.contains_type(type(NodeExp())) or node_arg.contains_type(type(NodeLn())):
            return False
        return True

    def __str__(self):
        return "exp("+str(self.left)+")"

class NodeLn(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        if a==0:
            a = Node.VERY_SMALL
        return log(a)#abs(a))

    def is_allowed_left_argument(self, node_arg):
        if type(node_arg) == type(NodeConstant(0)) and node_arg.value<=0:
            return False
        if type(node_arg)==type(NodeLn()) or type(node_arg)==type(NodeExp()):
            return False
        return True

    def __str__(self):
        return "log("+str(self.left)+")"

class NodeInv(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        if a==0:
            a = Node.VERY_SMALL
        return 1.0/a

    def is_allowed_left_argument(self, node_arg):
        if node_arg == NodeConstant(0):
            return False
        return True

    def __str__(self):
        return "1/"+str(self.left)

class NodeSgn(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        if a<0:
            return -1
        elif a==0:
            return 0
        else:
            return 1

    def __str__(self):
        return "sgn("+str(self.left)+")"

class NodeSqr(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        return pow(a, 2) # abs(a)

    def __str__(self):
        return "pow("+str(self.left)+",2)"

class NodeSqrt(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        return sqrt(a) #abs(a)

    def is_allowed_left_argument(self, node_arg):
        if type(node_arg) == type(NodeConstant(0)) and node_arg.value<0:
            return False
        return True

    def __str__(self):
        return "sqrt("+str(self.left)+")"

class NodeUnaryMinus(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        return -a

    def __str__(self):
        return "(-"+str(self.left)+")"

class NodeAbs(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        return abs(a)

    def __str__(self):
        return "abs("+str(self.left)+")"

class NodeTan(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        return tan(a)

    def __str__(self):
        return "tan("+str(self.left)+")"

class NodeFloor(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        return floor(a)

    def __str__(self):
        return "floor("+str(self.left)+")"

class NodeCeil(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        return ceil(a)

    def __str__(self):
        return "ceiling("+str(self.left)+")"

class NodeInc(Node):
    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        return a+1

    def __str__(self):
        return "("+str(self.left)+"+1)"

class NodeDec(Node):

    def __init__(self):
        super().__init__()
        self.arity = 1

    def evaluate_inner(self,X, a, b):
        return a-1

    def __str__(self):
        return "("+str(self.left)+"-1)"

# described in Apendix 4 of paper Contemporary Symbolic Regression Methods and their Relative Performance
def noisefy(y, noise_level, random_state):
    yRMSE = 0
    for i in range(len(y)):
        yRMSE+=(y[i]*y[i])
    yRMSE=sqrt(yRMSE/len(y))
    yRMSE_noise_SD = noise_level*yRMSE
    rg = RandomState(random_state)
    noise = rg.normal(0, yRMSE_noise_SD, len(y))
    y_n = []
    for i in range(len(y)):
        y_n.append(y[i]+noise[i])
    return y_n

def RMSE(yt, yp):
    if len(yp)!=len(yt):
        raise Exception("Vectors of predicted and true y values should be of same size.")
    try:
        rmse = 0.0
        for i in range(len(yp)):
            err = yp[i]-yt[i]
            err*=err
            if err == nan:
                return nan
            rmse+=err
        rmse = sqrt(rmse/len(yp))
        return rmse
    except OverflowError:
       return nan

def percentile_abs_error(yt, yp, percentile):
    if len(yp)!=len(yt):
        raise Exception("Vectors of predicted and true y values should be of same size.")
    try:
        errors = []
        for i in range(len(yp)):
            err = yp[i]-yt[i]
            if err<0:
                err*=-1
            if err == nan:
                return nan
            errors.append(err)
        errors.sort()
        idx = int(percentile*(len(yp)-1)/100)
        return errors[idx]
    except OverflowError:
        return nan

def R2(yt, yp):
    if len(yp)!=len(yt):
        raise Exception("Vectors of predicted and true y values should be of same size.")
    try:
        yt_mean = mean(yt)
        ss_res = 0
        ss_tot = 0
        for i in range(len(yp)):
            err = yp[i]-yt[i]
            err*=err
            if err == nan:
                return nan
            ss_res+=err
            var_err = yt_mean-yt[i]
            var_err*=var_err
            ss_tot+=var_err
        if ss_tot<0.000000000001:
            ss_tot=1
        return 1-ss_res/ss_tot
    except OverflowError:
        return nan

class Solution:

    math_error_count = 0
    fit_calls = 0
    fit_fails = 0

    @classmethod
    def clearStats(cls):
        cls.math_error_count=0
        cls.fit_fails = 0
        cls.fit_calls=0

    def __init__(self, factors, complexity_penalty):
        self.factors =  copy.deepcopy(factors)
        self.complexity_penalty = complexity_penalty

    def __str__(self) -> str:
        return "+".join([str(x) for x in self.factors])

    def evaluate_all(self,X, cache):
        yp = np.zeros(len(X))
        for fact in self.factors:
            fyp = fact.evaluate_all(X, cache)
            for i in range(len(fyp)):
                #try:
                yp[i]+=fyp[i]
                #except:
                #    pass # do nothing
        return yp

    def fitness(self, X, y, cache=True):
        try:
            Solution.fit_calls+=1
            yp = self.evaluate_all(X, cache) 
            return (1-R2(y, yp), RMSE(y, yp), self.size())
        except Exception as e:
            #print(e)
            Solution.math_error_count+=1
            Solution.fit_fails+=1
            return (inf, inf, inf)

    def size(self):
        totSize = len(self.factors) 
        for fact in self.factors:
            totSize+=fact.size()
        return totSize
        

    def fit_constants_OLS(self, X, y):
        new_factors = []
        for fact in self.factors:
            if fact.contains_type(type(NodeVariable(0))):
                new_factors.append(copy.deepcopy(fact))
        Xnew = np.zeros((len(X), len(new_factors)))
        try:
            for i in range(len(new_factors)):
                fiX = new_factors[i].evaluate_all(X, True)
                for j in range(len(fiX)):
                    if math.isnan(fiX[j]):
                        raise Exception("nan happened")
                    if isinstance(fiX[j], complex):
                        raise Exception("complex happened")
                    Xnew[j, i]=fiX[j]
            X2_new = sma.add_constant(Xnew)
            est = sma.OLS(y, X2_new)
            fit_info = est.fit()
            #print(fitInfo.summary())
            signLevel = 0.05
            params = fit_info.params
            final_factors = []
            p_values = fit_info.pvalues
            if p_values[0]<=signLevel:
                final_factors.append(NodeConstant(params[0]))
            else:
                final_factors.append(NodeConstant(0))
            for i in range(1, len(params)):
                if p_values[i]>signLevel:
                    continue
                fi_old = copy.deepcopy(new_factors[i-1])
                new_fact = NodeMultiply()
                coef = params[i]
                new_fact.left = NodeConstant(coef)
                new_fact.right = fi_old
                final_factors.append(new_fact)
            new_sol = Solution(final_factors, self.complexity_penalty)
            return new_sol
        except Exception as ex:
            Solution.math_error_count+=1
            #print("OLS error "+str(ex))
            return copy.deepcopy(self)

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
                myFactor = self.convert_to_my_nodes(factor)
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
            new_expression = self.convert_to_my_nodes(expression_simpy)
            new_factors = self.expand_to_factors(new_expression)
            if new_factors is not None:
                self.factors = new_factors
            else:
                raise Exception("Expansion to factors failed.")
        except Exception as ex:
            print("SimplifyWhole: "+str(ex))
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
                newFact = self.convert_to_my_nodes(expr_simpl)
                if type(newFact)==type(NodeConstant(0)) and newFact.value==0:
                    continue
                new_factors.append(newFact)
            except Exception as ex:
                print(ex)
        self.factors = new_factors

    def expand_fast(self):
        new_factors = []
        for fact in self.factors:
            factFacts = fact.expand_fast()
            new_factors+=factFacts
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
                    new_factor = self.convert_to_my_nodes(f)
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

    def convert_to_my_nodes(self,sympy_node):
        if type(sympy_node)==ImaginaryUnit:
            raise Exception("Not working with imaginary (complex) numbers.")
        sub_nodes = []
        for i in range(len(sympy_node.args)):
            sub_nodes.append(self.convert_to_my_nodes(sympy_node.args[i]))

        if len(sympy_node.args)==0:
            if type(sympy_node)==Symbol:
                if str(sympy_node)=="e":
                    return NodeConstant(e)
                try:
                    index = int(str(sympy_node).replace("X_",""))
                    return NodeVariable(index)
                except Exception as ex:
                    print(sympy_node)
                    print(ex)
            else:
                return NodeConstant(float(sympy_node))

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
                new = self.convert_to_my_nodes(sympy_node.args[0])
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

class RILSRegressor(BaseEstimator):

    def __init__(self, max_fit_calls=100000, max_seconds=100, complexity_penalty=0.001, error_tolerance=0.000001,random_state=0):
        self.max_seconds = max_seconds
        self.max_fit_calls = max_fit_calls
        self.complexity_penalty = complexity_penalty
        self.random_state = random_state
        self.error_tolerance = error_tolerance


    def __reset(self):
        self.model = None
        self.varNames = None
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
    
    def predict(self, X):
        Node.reset_node_value_cache()
        return self.model.evaluate_all(X, False)

    def size(self):
        if self.model is not None:
            return self.model.size()
        return math.inf

    def modelString(self):
        if self.model is not None:
            # replacing v0, v1, ... with real variable names
            return str(self.model)
        return ""

    def fit_report_string(self, X, y):
        if self.model==None:
            raise Exception("Model is not build yet. First call fit().")
        fitness = self.model.fitness(X,y, False)
        return "maxTime={0}\tmaxFitCalls={1}\tseed={2}\tsizePenalty={3}\tR2={4:.7f}\tRMSE={5:.7f}\tsize={6}\tsec={7:.1f}\tmainIt={8}\tlsIt={9}\tfitCalls={10}\texpr={11}".format(
            self.max_seconds,self.max_fit_calls,self.random_state,self.complexity_penalty, 1-fitness[0], fitness[1], fitness[2], self.time_elapsed,self.main_it, self.ls_it,Solution.fit_calls, self.model)

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
