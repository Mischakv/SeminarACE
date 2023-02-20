import numpy as np
import math
from scipy.stats import multivariate_normal

def read_prob_map(path, w_box_number, h_box_number):
    count = 0
    arr = np.empty((int(w_box_number*h_box_number),3))
    f = open(path, "r")
    for x in f:
        arr[count][1]=math.floor(count/w_box_number)
        arr[count][0]=int(count%h_box_number)
        arr[count][2] = x
        count=count +1
    f.close()
    return arr

def get_pdf(n_components, xy, means, covariances):
    zg = multivariate_normal.pdf(xy, mean=means[0][0:2], cov=covariances[0][0:2,0:2])*round(means[0][2],4)
    for i in range(1,n_components):
        zg += multivariate_normal.pdf(xy, mean=means[i][0:2], cov=covariances[i][0:2,0:2])*round(means[i][2],4)
    return zg
