#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:17:46 2024
@author: lilly
"""
# RL model for DiscrConfid / reward-bias task - perceptual + value-based decision-making

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(50)
V_L = 0.5   # initialize, stored value of action left, 'value function left'
V_R = 0.5  # initialize, stored value of action right, 'value function right'
rho = 1  # reward sensitivity, factor for calculation of reward prediction error
alpha = 0.5  # learning rate, constant
sigma = 0.2  # variance of Gaußian distribution for sensory percept, constant
block = 1    # initialize first block number

stimulus = np.array([])
rewards = np.array([])
decisions = np.array([])
values_left = np.array([])
values_right = np.array([])

for x in range(4):  # 4 blocks: equal_1, L<R/L>R, L>R/L<R, equal_2
    for y in range(200):
        s = np.random.uniform(low=-1,  high=1, size=1)  # auditory stimulus s, chosen randomly from uniform distribution
        stimulus = np.append(stimulus, s)
        s_roof = np.random.normal(s, sigma)   # s_roof = noisy percept, as Gaußian distribution around the stimulus s
        optimal_belief = np.random.normal(s_roof, sigma)  # optimal belief about s is Gaußian with mean = s_roof
        p_L_s_roof = scipy.integrate.quad(lambda s: optimal_belief, a=0, b=np.inf)  # belief state of agent about world --> probability of stimulus being left given noisy estimate
        p_L = abs(p_L_s_roof[0])
        p_R = 1 - p_L   # belief state of agent about world --> probability of stimulus being right given noisy estimate
        if block == 2:       # block with R > L
            R_mag_R = 1     # Reward magnitude right side
            R_mag_L = 0.5   # Reward magnitude left side
        elif block == 3:     # block with L > R
            R_mag_L = 1
            R_mag_R = 0.5
        else:       # blocks 1 & 4 == equal_1 & equal_2
            R_mag_L = 1
            R_mag_R = 1

        Q_L = p_L * V_L   # predictive value of action left, (R_mag influences stored value?)
        Q_R = p_R * V_R   # predictive value of action right

        if Q_L > Q_R:  # if overall value of left (combining past values & percept) is higher than right -> choice left
            if s >= 0.5:  # if stimulus s was really on the left side
                rewards = np.append(rewards, 1)  # correct choice -> reward -> add outcome to reward vector
                V_L = V_L + alpha * (1 * R_mag_L - Q_L)  # update value function V_L of chosen action left for reward
            else:    # if stimulus s was really on the right side
                rewards = np.append(rewards, 0)  # incorrect choice -> no reward
                V_L = V_L + alpha * (0 - Q_L)  # update value function V_L of chosen action left for no reward
            decisions = np.append(decisions, 1)     # decision codes: 1 = left, 0 = right
            values_left = np.append(values_left, V_L)
            values_right = np.append(values_right, V_R)

        else:   # if overall value for right is higher, choice is made for right
            if s < 0.5:   # if stimulus s was really on the right side
                rewards = np.append(rewards, 1)  # correct choice -> reward
                V_R = V_R + alpha * (1 * R_mag_R - Q_R)  # update value function V_R of chosen action right for reward
            else:   # if stimulus s was really on the left side
                rewards = np.append(rewards, 0)  # incorrect choice -> no reward
                V_R = V_R + alpha * (0 - Q_R)  # update value function V_R of chosen action right for no reward
            decisions = np.append(decisions, 0)
            values_right = np.append(values_right, V_R)
            values_left = np.append(values_left, V_L)
    block += 1

# Calculate moving average over 10 trials for better visualization in new vector x_averaged
window_size = 10
i = 0
stimulus_averaged = []
decisions_averaged = []
rewards_averaged = []
values_right_averaged = []
values_left_averaged = []

while i < len(stimulus) - window_size + 1:
    window_average = np.sum(stimulus[i:i + window_size]) / window_size
    stimulus_averaged.append(window_average)
    i += 10
i = 0
while i < len(decisions) - window_size + 1:
    window_average = np.sum(decisions[i:i + window_size]) / window_size
    decisions_averaged.append(window_average)
    i += 10
i = 0
while i < len(rewards) - window_size + 1:
    window_average = np.sum(rewards[i:i + window_size]) / window_size
    rewards_averaged.append(window_average)
    i += 10
i = 0
while i < len(values_right) - window_size + 1:
    window_average = np.sum(values_right[i:i + window_size]) / window_size
    values_right_averaged.append(window_average)
    i += 10
i = 0
while i < len(values_left) - window_size + 1:
    window_average = np.sum(values_left[i:i + window_size]) / window_size
    values_left_averaged.append(window_average)
    i += 10
i = 0


### How is time investment influenced by different factors?
# Visualize model
sns.set_palette("Set2")

fig = plt.figure(figsize=(15,10))
gs = GridSpec(1, 2)


# g = sns.lineplot(data=stimulus_averaged, alpha=1, color="black", label="stimulus")
g = sns.lineplot(data=values_left_averaged, alpha=0.7, color="blue", label="V_L")
g = sns.lineplot(data=values_right_averaged, alpha=0.7, color="purple", label="V_R")
plt.title('Sensory percept RL model')
sns.despine()
plt.xlabel('Trials (averaged over 10)')
plt.ylabel('Fraction left choice')
plt.setp(g, yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# plt.axvspan(0, 20, alpha=0.3, color="gainsboro", label="Equal")
plt.axvspan(20, 40, alpha=0.3, color="pink", label="Rew R > Rew L")
plt.axvspan(40, 60, alpha=0.3, color="cyan", label="Rew L > Rew R")
# plt.axvspan(60, 80, alpha=0.3, color="gainsboro")
plt.legend(fontsize=8, loc='best', frameon=0)   # bbox_to_anchor=(1, 1)

plt.show()

# Plot psychometrics
