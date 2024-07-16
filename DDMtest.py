import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#boundary seperation 1-3
alpha = 1

#staring point 0 to 1
beta = 0.5

#drift rate -2 to 2
delta = 0

#non decision time
tau = 0

#trials
n = 200

data = np.empty((n,2))

for i in range(n):
    #set initial evidence
    evidence = beta
    while np.abs(evidence) < alpha:


    data[i] =



