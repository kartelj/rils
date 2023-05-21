from math import cosh, log, nan, sqrt
from statistics import mean
from numpy.random import RandomState 
from numpy import imag, linspace, real
from sklearn.metrics import r2_score, mean_squared_error
from numpy import zeros

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

def Imag_Abs_Sum(yp):
    imag_abs_sum = 0
    imag_frac = 0.0
    for i in range(len(yp)):
        imag_abs_sum+=abs(imag(yp[i]))
        if imag(yp[i])!=0:
            imag_frac+=1
    imag_frac/=len(yp)
    #return imag_abs_sum
    return imag_frac

def log_cosh(yt, yp):
    if len(yp)!=len(yt):
        raise Exception("Vectors of predicted and true y values should be of same size.")
    lc = 0.0
    for i in range(len(yp)):
        err = log(cosh(yp[i]-yt[i]))
        if err == nan:
            return nan
        lc+=err
    lc = lc/len(yp)
    return lc

def diff_R2(yt, yp):
    if len(yp)!=len(yt):
        raise Exception("Vectors of predicted and true y values should be of same size.")

    n = len(yp)
    diffs_yp = zeros(n*(n-1)/2)
    diffs_yt = zeros(n*(n-1)/2)
    k=0
    for i in range(n):
        for j in range(i):
            diffs_yp[k]=yp[i]-yp[j]
            diffs_yt[k]=yt[i]-yt[j]
            k+=1

    diffs_r2 = r2_score(diffs_yt, diffs_yp)
    return diffs_r2

def diff_RMSE(yt, yp):
    if len(yp)!=len(yt):
        raise Exception("Vectors of predicted and true y values should be of same size.")

    n = len(yp)
    cnt = round((n*(n-1))/2)
    diffs_yp = zeros(cnt)
    diffs_yt = zeros(cnt)
    #diffs_yp = [yp[i]-yp[j] for (i, j) in [(i, j) for i in range(n) for j in range(i)]]
    #diffs_yp = [yt[i]-yt[j] for (i, j) in [(i, j) for i in range(n) for j in range(i)]]
    k=0
    #mse = sum([abs(yp[i]-yp[j]-yt[i]+yt[j]) for (i, j) in [(i, j) for i in range(n) for j in range(i)]] )
    for i in range(n):
        for j in range(i):
            #err = (yp[i]-yp[j]-yt[i]+yt[j])
            #mse+=(err*err)
            diffs_yp[k]=yp[i]-yp[j]
            diffs_yt[k]=yt[i]-yt[j]
            k+=1
    #rmse = sqrt(mse/cnt)
    rmse = sqrt(mean_squared_error(diffs_yt, diffs_yp))
    return rmse
    
def ResidualVariance(yt, yp, complexity):
    if len(yp)!=len(yt):
        raise Exception("Vectors of predicted and true y values should be of same size.")
    var = 0.0
    for i in range(len(yp)):
        err = yp[i]-yt[i]
        err*=err
        if err == nan:
            return nan
        var+=err
    var = var/(len(yp)-complexity)
    return var

def percentile_abs_error(yt, yp, alpha=0.5):
    if len(yp)!=len(yt):
        raise Exception("Vectors of predicted and true y values should be of same size.")
    errors = []
    for i in range(len(yp)):
        err = yp[i]-yt[i]
        if err<0:
            err*=-1
        if err == nan:
            return nan
        errors.append(err)
    errors.sort()
    idx = int(alpha*(len(yp)-1))
    return errors[idx]

def empirical_cumulative_distribution(vals, bins=None):
    # preparing in advance bins for real target values
    cutoff = 0.01 # 0.05
    tail_size = round(cutoff*len(vals))
    vals_sorted = sorted(vals)[tail_size:len(vals)-tail_size]

    if bins is None: 
        bin_cnt = 30 # 20
        min = vals_sorted[0]
        max = vals_sorted[len(vals_sorted)-1]
        bins = linspace(min, max+0.0001, bin_cnt) # to include it in the the bin

    bin_cnts = zeros(len(bins))
    for v in vals_sorted:
        for b in range(len(bin_cnts)-1):
            if v>=bins[b] and v<bins[b+1]:
                bin_cnts[b]+=1
                break
    return (bins, bin_cnts)

def distribution_fit_score(y, yp):
    score = 0.0
    y_bins, y_cnts = empirical_cumulative_distribution(y)
    _, yp_cnts = empirical_cumulative_distribution(yp, bins=y_bins)
    n = len(yp)
    #Bhattacharyya coefficient
    for b in range(len(y_bins)):
        score+=sqrt(y_cnts[b]*yp_cnts[b]/n/n)
    #for b in range(len(y_bins)):
    #    score+=abs(y_cnts[b]*1.0/n - yp_cnts[b]*1.0/n)
    #return score
    #return 1-score
    return -log(score)
