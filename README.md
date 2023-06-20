# RILS -- Regression via Iterated Local Search
RILS algorithm for GECCO2023 SR competition

# Installation instructions

0. The program requires Python 3 (tested on version 3.11.3), but it should work on some earlier versions as well. 
We also recommend using pip package manager. 

1. Download the repository and unpack it. 

2. Install the following pip packages:

```console
pip install numpy
pip install sympy
pip install scikit-learn
pip install statsmodels
```

# Execution instructions

1. Position inside the root directory of the unpacked repository. The (sub)directories at this level are: instances/, paper/, results/ and rils/.
Directory instances/ holds datasets that are previously rounded to 9 decimals and saved in the tab separated .txt files.  

2. Call run_all.cmd or run_all.sh script depending on whether you use Windows or Linux. 
If using Linux, you might need to change the accesss rights for this script, i.e. add x (execution) rights. 
Also, you might need to change the name of python executable to python3 on some systems. 
The content of run_all.{cmd|sh} file is as follows:

```console
python run.py "instances/srbench_2023" "dataset_1.txt" 180 20
python run.py "instances/srbench_2023" "dataset_2.txt" 100 10
python run.py "instances/srbench_2023" "dataset_3.txt" 180 20
```

3. The steps in the execution pipeline are explained in paper/rils_gecco2023.pdf. 
Briefly, several output files will occurr during execution:

    1. best_sols_dataset_{1|2|3}.txt
    2. log.txt
    3. out.txt

The model will be written inside out.txt.
Note that there is semi-automated (interactive) simplification (step 4 in the paper) for the dataset_3. 
