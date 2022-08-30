from datetime import datetime
from rils import RILS
from random import Random
import utils
from os import listdir
from os.path import isfile, join


instancesDir = "../toyInstances" 
targetFirst = False 
separator = "\t" 
header = False

print(targetFirst)
print(header)
seed = 12345
# method generalizes very well so very small training set can be used -- like 40% or even less
trainPerc = 0.75
time = 200
sizePenalty =  0.0001 # None
preturbations = 1

#regressors = [make_pipeline(StandardScaler(), svm.SVR(C=1.0, epsilon=0.2)), linear_model.Ridge(alpha=.5)]

instanceFiles = [f for f in listdir(instancesDir) if isfile(join(instancesDir, f))]
#instanceFiles = ["f16.txt"]

outPath = "out_{0}.txt".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
with open(outPath, "w") as f:
    f.write("Tests started\n")

for fpath in instanceFiles:
    print("Running instance "+fpath)
    with open(instancesDir+"/"+ fpath) as f:
        lines = f.readlines()
        trainCnt = int(len(lines)*trainPerc)
        rg = Random(seed)
        if header:
            lines = lines[1:]
        rg.shuffle(lines)
        Xtrain = []
        ytrain= []    
        Xtest = []
        ytest = []
        for i in range(len(lines)):
            line = lines[i]       
            tokens = line.split(sep=separator)
            if targetFirst:
                newY = float(tokens[0])
                newX = [float(t) for t in tokens[1:]]
            else:
                newX = [float(t) for t in tokens[:len(tokens)-1]]
                newY = float(tokens[len(tokens)-1])
            if i<trainCnt:
                Xtrain.append(newX)
                ytrain.append(newY)
            else:
                Xtest.append(newX)
                ytest.append(newY)

    #for regr in regressors:
    #    regr.fit(Xtrain, ytrain)
    #    yp = regr.predict(Xtest)
    #    print("%s\tRMSE=%.2f\tR2=%.2f"%(regr, utils.RMSE(ytest, yp), utils.R2(ytest, yp)))

    vnl = RILS(1000000,time,  random_state = seed, complexity_penalty=sizePenalty, preturbations=preturbations)
    vnl.fit(Xtrain, ytrain)
    reportString = vnl.fit_report_string(Xtrain, ytrain)
    yp = vnl.predict(Xtest)
    print("%s\tRMSE=%.3f\tR2=%.3f"%(vnl, utils.RMSE(ytest, yp), utils.R2(ytest, yp)))
    with open(outPath, "a") as f:
        f.write(fpath+"\tTestRMSE="+str(round(utils.RMSE(ytest, yp),3))+"\tTestR2="+str(round(utils.R2(ytest, yp),3))+"\t"+reportString+"\n")
