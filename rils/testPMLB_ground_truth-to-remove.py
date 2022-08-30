from tabnanny import check
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
import utils
from rils import RILS
from os import listdir, stat
import sys
from sklearn.utils.estimator_checks import check_estimator

#check_estimator(RILS())

if len(sys.argv)!=8:
    print("Usage: <working part or -1> <parts >=1> <random seed> <max time (s)> <max fit calls> <size penalty> <noise level>")
    print("Passed parameters were:")
    print(sys.argv[1:])
    sys.exit(1)

part = int(sys.argv[1])
parts = int(sys.argv[2])

seed = int(sys.argv[3])
max_seconds = int(sys.argv[4])
max_fit_calls = int(sys.argv[5])
complexity_penalty = float(sys.argv[6])
noise_level = float(sys.argv[7])
test_size = 0.25

pmlb_cache = "../pmlb/datasets"
excluded_feynman = ["feynman_I_26_2", "feynman_I_30_5", "feynman_test_10"] # this are using arcsin or arcos
ground_truth_regr_datasets = list(filter(lambda x: (x.startswith("feynman") or x.startswith("strogatz")) and not x in excluded_feynman, listdir(pmlb_cache)))

tmp_dict = {}
for dataset in ground_truth_regr_datasets:
    size = stat(pmlb_cache+"/"+dataset+"/"+dataset+".tsv.gz").st_size
    tmp_dict[dataset]=size

ground_truth_regr_datasets.clear()
for dataset in sorted(tmp_dict.items(), key=lambda x: x[1]):
    ground_truth_regr_datasets.append(dataset[0])

out_path = "out_sp"+str(complexity_penalty)+"_nl"+str(noise_level)+".txt"
for i in range(len(ground_truth_regr_datasets)):
    if i%parts!=part:
        continue
    dataset = ground_truth_regr_datasets[i]
    print("Doing "+dataset)
    X, y = fetch_data(dataset, return_X_y=True,  local_cache_dir=pmlb_cache)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    y_train = utils.noisefy(y_train, noise_level, seed)
    
    rils = RILS(max_fit_calls,max_seconds, random_state = seed, complexity_penalty=complexity_penalty)
    rils.fit(X_train, y_train)
    report_string = rils.fit_report_string(X_train, y_train)
    rils_R2 = ""
    rils_RMSE = ""
    try:
        yp = rils.predict(X_test)
        rils_R2 = round(utils.R2(y_test, yp),7)
        rils_RMSE = round(utils.RMSE(y_test, yp),7)
        print("%s\tR2=%.7f\tRMSE=%.7f\texpr=%s"%(dataset, rils_R2, rils_RMSE, rils.model))
    except:
        print("ERROR during test.")
    with open(out_path, "a") as f:
        f.write(dataset+"\t"+report_string+"\tTestR2="+str(rils_R2)+"\tTestRMSE="+str(rils_RMSE)+"\n")