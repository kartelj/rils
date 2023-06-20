from copy import deepcopy
from math import inf, pi

from sklearn.metrics import r2_score
from .solution import Solution
from sympy import preorder_traversal, sympify
from .node import Node, NodeConstant, NodeCos, NodeDivide, NodeLn, NodeMinus, NodeMultiply, NodePlus, NodePow, NodeSin

def reduction_acceptable(r2_old, r2_new, sympy_size_old, sympy_size_new):
    if sympy_size_new>sympy_size_old:
        return False
    if sympy_size_new==sympy_size_old:
        return r2_new>r2_old
    return abs(r2_old-r2_new)/(sympy_size_old-sympy_size_new)<0.001

def find_closest_pi_rational_multiplier(value):
    positive = value>=0
    abs_value = abs(value)
    min_dist = inf
    min_a = None
    min_b = None
    for a in range(1, 11):
        for b in range(1, 11):
            nice_exp = pi*a/b
            if abs(abs_value-nice_exp)/nice_exp<min_dist:
                min_dist = abs(abs_value-a/b*pi)/nice_exp
                min_a = a
                min_b = b
    rational = NodeDivide()
    if positive:
        rational.left = NodeConstant(min_a)
        rational.right = NodeConstant(min_b)
        return rational
    else:
        rational.left = NodeConstant(-min_a)
        rational.right = NodeConstant(min_b)
        return rational
    #return None

def is_trigonometric(node: Node):
    return type(node)==type(NodeSin()) or type(node)==type(NodeCos())

def find_pi_multiplier_candidates(node: Node, candidates, parent=None, is_left_from_parent=None, inside_trig=False):
    # the last criterion is to avoid changing powers inside power function 
    if inside_trig and parent is not None and type(node)==type(NodeConstant(0)) and not (type(parent)==type(NodePow()) and not is_left_from_parent):
        candidates.append((node, parent, is_left_from_parent))
        return
    if node.arity>=1:
        find_pi_multiplier_candidates(node.left, candidates, node, True, inside_trig or is_trigonometric(node))
    if node.arity>=2:
        find_pi_multiplier_candidates(node.right, candidates, node, False, inside_trig or is_trigonometric(node))
        
def log_exp_expand(node: Node, parent = None, is_left_from_parent = None):
    if parent is None and type(node)==type(NodeLn()) and type(node.left)==type(NodePow()):
        new_node = NodeMultiply()
        new_node.left = node.left.right
        new_node.right = NodeLn()
        new_node.right.left = node.left.left
        if is_left_from_parent:
            parent.left = new_node
        else:
            parent.right = new_node
    if node.arity>=1:
        log_exp_expand(node.left, node, True)
    if node.arity>=2:
        log_exp_expand(node.right, node, False)

def try_removals(solution: Solution, node, X, y, sympy_complexity):
    if type(node)==type(NodePlus()) or type(node)==type(NodeMinus()) or type(node)==type(NodeMultiply()) or type(node)==type(NodeDivide()):
        current_r2, current_comp = eval(solution, X, y)
        if sympy_complexity:
            current_comp = eval_sympy_complexity(solution)
        old_left = deepcopy(node.left)
        old_right = deepcopy(node.right)
        if type(node)==type(NodePlus()) or type(node)==type(NodeMinus()):
            constant = 0
        else:
            constant = 1
        node.left = NodeConstant(constant)
        r2, comp = eval(solution, X, y)
        if sympy_complexity:
            comp = eval_sympy_complexity(solution)
        if not reduction_acceptable(current_r2, r2, current_comp, comp):
            node.left = old_left
        else:
            current_r2 = r2
            current_comp = comp
            #print("R2="+str(current_r2)+" after removing "+str(old_left))
        node.right = NodeConstant(constant)
        r2, comp = eval(solution, X, y)
        if sympy_complexity:
            comp = eval_sympy_complexity(solution)
        if not reduction_acceptable(current_r2, r2, current_comp, comp):
            node.right = old_right
        else:
            current_r2 = r2
            current_comp = comp
            #print("R2="+str(current_r2)+" after removing "+str(old_right))
    if node.arity>=1:
        try_removals(solution, node.left, X, y, sympy_complexity)
    if node.arity>=2:
        try_removals(solution, node.right, X, y, sympy_complexity)

class NodePi(Node):
    def __init__(self):
        super().__init__()
        self.arity = 0

    def evaluate_inner(self,X, a, b):
        return pi

    def __str__(self):
        return "pi"

def try_pi_multipliers(solution: Solution, X, y, sympy_complexity):
    current_r2, current_comp = eval(solution, X, y)
    if sympy_complexity:
        current_comp = eval_sympy_complexity(solution)
    for i in range(len(solution.factors)):
        candidates = []
        find_pi_multiplier_candidates(solution.factors[i], candidates)
        for node, parent, is_left_from_parent in candidates:
            mult = find_closest_pi_rational_multiplier(node.value)
            if mult!=None:
                new_node = NodeMultiply()
                new_node.left = mult 
                new_node.right = NodePi()
                if is_left_from_parent:
                    parent.left = new_node
                    r2, comp = eval(solution, X, y)
                    if sympy_complexity:
                        comp = eval_sympy_complexity(solution)
                    if not reduction_acceptable(current_r2, r2, current_comp, comp):
                        parent.left = node
                    else:
                        current_r2 = r2
                        current_comp = comp
                        #print("R2="+str(current_r2)+" after changing "+str(node)+" to "+str(new_node))
                else:
                    parent.right = new_node
                    r2, comp = eval(solution, X, y)
                    if sympy_complexity:
                        comp = eval_sympy_complexity(solution)
                    if not reduction_acceptable(current_r2, r2, current_comp, comp):
                        parent.right = node
                    else:
                        current_r2 = r2
                        current_comp = comp
                        #print("R2="+str(current_r2)+" after changing "+str(node)+" to "+str(new_node))
    r2_test, _ = eval(solution, X, y)
    if r2_test!=current_r2:
        print("Error")    
    return solution

def eval(sol: Solution, X, y):
    try:
        yp = sol.evaluate_all(X, cache=False)
        r2 = r2_score(y, yp)
        return r2, sol.size()
    except:
        return -inf, inf
    
def sympy_complexity(model_simp):
        c=0
        for arg in preorder_traversal(model_simp):
            c += 1
        return c

def eval_sympy_complexity(sol: Solution):
    if str(sol)=="":
        expr = "0"
    else:
        expr = str(sol)
    sympy_expr = sympify(expr).evalf()
    return sympy_complexity(sympy_expr)

def print_info(sol: Solution, X, y, desc):
    r2, _ = eval(sol, X, y)
    sympy_size =eval_sympy_complexity(sol)
    print(desc+"\tR2="+str(r2)+" sympy_size="+str(sympy_size))

def simplify_approximately(sol, X, y, sympy_complexity=True, removals_only=False):
    #print_info(sol, X, y, "BEFORE")
    #assert(len(sol.factors)==1)
    for fact in sol.factors:
        try_removals(sol, fact, X, y, sympy_complexity)
    if removals_only:
        return
    #print_info(sol, X, y, "AFTER REM1")
    sol = try_pi_multipliers(sol, X, y, sympy_complexity)
    #print_info(sol, X, y, "AFTER")
    return

