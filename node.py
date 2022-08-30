from abc import abstractmethod
import copy
from math import acos, asin, atan, ceil, cos, exp, floor, log, sin, sqrt, tan

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
        return "v"+str(self.index)

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
