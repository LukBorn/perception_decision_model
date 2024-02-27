import numpy as np
import pandas as pd

def moving_avg_smooth(data, window_size=10):
    #output a moving average of your data
    #takes both lists and arrays
    if type(data) == list: data = np.array(data)
    smooth = []
    for i in range(data.shape[0] - window_size):
        smooth.append(np.sum(data[i:i + window_size]) / window_size)
    return np.array(smooth)

def log_fun(x, beta): #define a sigmoidal logistic function to model the decision probability
    return(1 / (1 + np.exp(-beta * x)))

def alpha(t, type, constant = 0.4):
    # learning rate:
    # types: "1/n", "constant"
    if type == "1/n":
        if t == 0: return 1
        else: return 1/t
    elif type == "constant":
        return constant