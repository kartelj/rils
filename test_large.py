from datetime import datetime
from math import sqrt
from random import seed
import numpy as np
from rils.rils_ensemble import RILSEnsembleRegressor
from sklearn.metrics import r2_score, mean_squared_error

instances_dir = "instances/srbench_2023" 
random_state = 23654
seed(random_state)
epochs = 200
fit_calls_per_epoch = 100000
parallelism = 10
verbose = False

instance_files = [("dataset_3.txt", "0"), 
                    ("dataset_3.txt", "0"), 
                    ("dataset_2.txt", "0")] 

out_path = "out.txt" 
with open(out_path, "a") as f:
    f.write("Tests started at "+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+"\n")

for fpath, init_sympy_sol_str in instance_files:
    print("Running instance "+fpath)
    with open(instances_dir+"/"+ fpath) as f:
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

        with open("log.txt", "a") as f:
            f.write(f"Running {fpath} with init_sympy_sol_str {init_sympy_sol_str}, parallelism {parallelism}, epochs {epochs}, fit_calls_per_epoch {fit_calls_per_epoch}, random_state {random_state}\n")
        rils = RILSEnsembleRegressor(epochs = epochs, fit_calls_per_epoch=fit_calls_per_epoch, random_state = random_state, parallelism=parallelism,verbose=verbose)
        rils.fit(X_train, y_train, init_sympy_sol_str= init_sympy_sol_str, X_test=X_test, y_test=y_test)
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
            f.write("{0}\t{1}\tTestR2={2:.8f}\tTestRMSE={3:.8f}\tParallelism={4}\n".format(fpath, report_string, rils_R2, rils_RMSE, parallelism))
