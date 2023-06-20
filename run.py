from datetime import datetime
from math import inf, sqrt
from random import seed
import sys
import numpy as np
from sympy import sympify
from rils.rils import FitnessType
from rils.rils_ensemble import RILSEnsembleRegressor
from rils.rils import FitnessType
from sklearn.metrics import r2_score, mean_squared_error
from rils.simplify_approx import simplify_approximately

from rils.solution import Solution

def get_data(dataset_dir, dataset_file):
    with open(dataset_dir+"/"+dataset_file) as f:
        lines = f.readlines()
        X = []
        y= []   
        for i in range(len(lines)):
            line = lines[i]       
            tokens = line.split(sep="\t")
            newX = [float(t) for t in tokens[:len(tokens)-1]]
            newY = float(tokens[len(tokens)-1])  
            X.append(newX)
            y.append(newY)
        # making balanced set w.r.t. target variable
        Xy = list(zip(X, y))
        Xy_sorted =  sorted(Xy, key=lambda p:p[1])
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for i in range(len(Xy_sorted)):
            if i%4==0:
                X_test.append(Xy_sorted[i][0])
                y_test.append(Xy_sorted[i][1])
            else:
                X_train.append(Xy_sorted[i][0])
                y_train.append(Xy_sorted[i][1])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
    return X_train, y_train, X_test, y_test

def run_dataset(dataset_dir, dataset_file, random_state, epochs, fit_calls_per_epoch, parallelism, verbose, fitness_type, complexity_penalty, init_sympy_sol_str, init_target_size):
    X_train, y_train, X_test, y_test = get_data(dataset_dir, dataset_file)
    with open("log.txt", "a") as f:
        f.write(f"Running {dataset_file} with fitness_type {fitness_type} and complexity_penalty {complexity_penalty} and init_sympy_sol_str {init_sympy_sol_str}\n")
    rils = RILSEnsembleRegressor(epochs = epochs, fit_calls_per_epoch=fit_calls_per_epoch, random_state = random_state, fitness_type=fitness_type, complexity_penalty=complexity_penalty, parallelism=parallelism, target_size=init_target_size, verbose=verbose)
    rils.fit(X_train, y_train, init_sympy_sol_str= init_sympy_sol_str, dataset_file=dataset_file,  X_test=X_test, y_test=y_test)
    report_string = rils.fit_report_string(X_train, y_train)
    rils_R2 = -1
    rils_RMSE = -1
    try:
        yp = rils.predict(X_test)
        rils_R2 = r2_score(y_test, yp)
        rils_RMSE = sqrt(mean_squared_error(y_test, yp))
        print("R2=%.8f\tRMSE=%.8f\texpr=%s"%(rils_R2, rils_RMSE, rils.model))
    except:
        print("ERROR during test.")
    with open(out_path, "a") as f:
        f.write("{0}\t{1}\tTestR2={2:.8f}\tTestRMSE={3:.8f}\tParallelism={4}\n".format(dataset_file, report_string, rils_R2, rils_RMSE, parallelism))
    return rils.model_simp

def select_best_tradeoff_solution(best_sols_file):
    with open(best_sols_file, "r") as f:
        lines = f.readlines() 
        r2_with_sols = []
        for i in range(len(lines)):
            line = lines[i]       
            tokens = line.split(sep=":")
            r2 = float(tokens[0])
            sol_str = tokens[1]
            r2_with_sols.append((r2, sol_str))
        # reading backwards and finding the last relevant improvement w.r.t. R2
        # relevant improvement is one of at least 0.5% improvement in R2
        i = len(r2_with_sols)-1
        while i>0:
            if r2_with_sols[i][0]-r2_with_sols[i-1][0]>0.005:
                break
            i-=1
        return r2_with_sols[i][1]

def run_simplify(dataset_dir, dataset_file, model_str):
    sympy_expr = sympify(model_str)
    my_expr = Solution.convert_to_my_nodes(sympy_expr)
    sol =  Solution([my_expr])
    X_train, y_train, _, _ = get_data(dataset_dir, dataset_file)
    simplify_approximately(sol, X_train, y_train)
    sympy_expr = sympify(str(sol))
    return sympy_expr

if __name__ == "__main__":

    if len(sys.argv)!=5:
        print("Usage <dataset directory> <dataset file name> <epochs> <init_target_size>")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    dataset_file = sys.argv[2]
    epochs = int(sys.argv[3])
    init_target_size = int(sys.argv[4])

    random_state = 23654
    seed(random_state)
    verbose = False
    
    print(f"Running dataset {dataset_file}")
    out_path = "out.txt" 
    with open(out_path, "a") as f:
        f.write("Running dataset "+dataset_file+" at "+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+"\n")
    fit_calls_per_epoch = 100000
    parallelism = 10
    complexity_penalty = 0
    expr = "0"
    # run dataset with an increasing target sizes
    run_dataset(dataset_dir, dataset_file, random_state, epochs, fit_calls_per_epoch, parallelism, verbose, FitnessType.EXPERIMENTAL, complexity_penalty, expr, init_target_size = init_target_size)
    # once the above call is finished, the file named "best_sols_<dataset_file>" will contain the best solutions through increasing target sizes, so we select the best tradeoff
    best_tradeoff_sympy_sol = select_best_tradeoff_solution("best_sols_"+dataset_file)
    print("The best tradeoff model is "+best_tradeoff_sympy_sol)
    simplifed_sympy_model = run_simplify(dataset_dir, dataset_file, best_tradeoff_sympy_sol)
    print("The final simplified model is "+str(simplifed_sympy_model))
    with open(out_path, "a") as f:
        f.write("The final model for dataset "+dataset_file+" is "+str(simplifed_sympy_model)+"\n")