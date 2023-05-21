from datetime import datetime
from math import sqrt
from multiprocessing import freeze_support
from random import Random, seed

import numpy as np
from rils.rils import VINSROLSRegressor, FitnessType
from rils import utils
from os import listdir
from os.path import isfile, join
from rils.rils_ensemble import VINSROLSEnsembleRegressor
from rils.rils import FitnessType
from sklearn.metrics import r2_score, mean_squared_error
import sys

if __name__ == "__main__":

    print(sys.argv)
    simplification = True
    first_improvement = True
    k_max = 1
    change_ignoring = False

    print("Running with simplification={0} first_improvement={1} k_max={2}".format(simplification, first_improvement, k_max))

    instances_dir = "instances/srbench_2023" 
    random_state = 23654
    seed(random_state)
    epochs = 200
    fit_calls_per_epoch = 100000
    fitness_types = [FitnessType.EXPERIMENTAL] # [FitnessType.R2, FitnessType.DEFAULT] #[FitnessType.DEFAULT, FitnessType.R2] #[fit_type for fit_type in FitnessType]
    complexity_penalties = [0.001] #[0.001, 0.0005, 0.0001] #[0.00001, 0.0001] #[0.001, 0.0001, 0.00001] # 0.0001 default
    parallelism = 10
    initial_sample_size = 1
    verbose = True

    instance_files = [("dataset_3.txt", [20], "0"), #"-(0.330696395978521*(0.368302651809364 - 1.0*sin(x6))**4.0*(x0 - 4.81299076890039) - ((0.649946400247015*x7 - 3.038750788742)*sin(x6**(-1.0)) - 1.80555904230299/sin(x0)**1.0 + 6.40226669473056)*sin(1.6846216697853478/x6**1.0 + 3.9955949023809962*x6 - 8.87373718032654) + 0.489290729122628)*(0.702690505080214*sin(2.6910990101744545*x1) + 0.561321842220022*sin(0.640726138384292*x1**2.0 - 0.9999999999999999) + 0.20771866859384*cos(4.704479999999999*x1 + 1.452251253951832))*sin(2.8817880299999996*x0 + 0.5)"),#"((0.445115887159272*x1 + 4.36357921527344*sin(2.98762335*x1 + 10.0))*(x6 + (10.9846952384445 - 4.20089901017446*sin(x6))*sin(2.9577471208588024*x6 + 10.0) + 0.1152) + 0.640010958631211)*(0.0101416693946588*sin(0.4320318029097911/x1**1.0) - 0.00725274802083763)*(x3 + 4.20571002220626/sin(x0)**1.0 - 2.28237611976*sin(x0*(x6 - 3.040339557782314)) - 19.0697715408205)*sin(2.84120008322535*x0 + 0.44343684952463086)"), 
                      ("dataset_3.txt", [20], "0"), 
                      ("dataset_2.txt", [20], "0")] #, "dataset_2.txt"]#, "dataset_4.txt", "dataset_1_medium_abs_y.txt", "dataset_1_small_abs_y.txt", "dataset_4.txt", "dataset_1_reduced.txt", "dataset_1.txt.cleaned"] #[f for f in listdir(instances_dir) if isfile(join(instances_dir, f))]

    out_path = "out.txt" #.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    with open(out_path, "a") as f:
        f.write("Tests started at "+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+"\n")

    for fpath, target_sizes, init_sympy_sol_str in instance_files:
        print("Running instance "+fpath)
        with open(instances_dir+"/"+ fpath) as f:
            lines = f.readlines()
            #train_cnt = int(len(lines)*0.75)
            #rg = Random(random_state)
            #rg.shuffle(lines)
            #Z = np.loadtxt(instances_dir+"/"+ fpath, delimiter="\t")#, skiprows=1)
            #X, y = Z[:, :-1], Z[:, -1]
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

        for fitness_type in fitness_types:
            for complexity_penalty in complexity_penalties:
                for target_size in target_sizes:
                    with open("log.txt", "a") as f:
                        f.write(f"Running {fpath} with fitness_type {fitness_type} complexity_penalty {complexity_penalty} initial_target_size {target_size} and init_sympy_sol_str {init_sympy_sol_str}\n")
                    #if parallelism==1:
                    #    rils = VINSROLSRegressor(max_fit_calls=max_fit, max_seconds=time, random_state = random_state, fitness_type=fitness_type, complexity_penalty=complexity_penalty, initial_sample_size=initial_sample_size, simplification=simplification, first_improvement=first_improvement, k_max=k_max, change_ignoring = change_ignoring, initial_target_size=target_size, verbose=verbose)
                    #elif parallelism>1:
                    rils = VINSROLSEnsembleRegressor(epochs = epochs, fit_calls_per_epoch=fit_calls_per_epoch, random_state = random_state, fitness_type=fitness_type, complexity_penalty=complexity_penalty, parallelism=parallelism, initial_sample_size=initial_sample_size, simplification=simplification, first_improvement=first_improvement, k_max=k_max, change_ignoring=change_ignoring, initial_target_size=target_size, verbose=verbose)
                    #else:
                    #    raise Exception("Parallelism parameter must be >= 1.")
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
