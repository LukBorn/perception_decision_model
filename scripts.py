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

def log_fun(x, beta, **kwargs): #define a sigmoidal logistic function to model the decision probability
    return(1 / (1 + np.exp(-beta * x)))

def decay_alpha(t):
    # learning rate:
    # types: "1/n"
    if t == 0: return 1
    else: return 1/t


def epsilon_greedy(q, epsilon, **kwargs):
    if np.random.random() > epsilon:
        action = np.argmax(q)
    else:
        action = np.random.choice(len(q))
    return action


def greedy(q, **kwargs):
    action = np.argmax(q)
    return action


