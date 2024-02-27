#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 17:17:46 2022

@author: lilly
"""
### RL model for depression learning rate hypothesis
#model contains 2 different learning rates alpha: one for after reward (alpha+), one after no reward (alpha-)
#hypothesis: in healthy agent alpha+ = alpha- ; in depressed agent presuambly alpha- > alpha+
import pandas as pd
#stay_switch_df_2 = pd.DataFrame(columns=['alpha', 'Prop stay', 'Prop switch', 'Block trials'])

import numpy as np
np.random.seed(42)
value_f_left = 0.5
value_f_right = 0.5
beta = 2.5
alpha_plus = 0.5 #alpha learning rate after reward
alpha_minus = 0.5 #alpha learning rate after no reward
rho = 1 #reward sensitivity rho: factor for calculation of prediction error, lower in anhedonia
m = 1 #memory of previous reinforcement: influence by prior r, instead of last trial
P_L_rew = 0.3
P_R_rew = 0.7
rewards = np.array([])
decisions = np.array([])
values_left = np.array([])
values_right = np.array([])
def log_fun(x): #define a sigmoidal logistic function to model the decision probability 
    return(1 / (1 + np.exp(-beta * x)))


for x in range(1000) :
    p_decision_right = log_fun(value_f_right- value_f_left)  #define decision probability which is proportional to value functions
    decision = np.random.binomial(1, p_decision_right)

    if decision == 0 :  #if decision is 0 = left
        if np.random.random() <= P_L_rew :  #P(rew)_L = 0.2 through generation of random float between 0 and 1
            rewards = np.append(rewards, 1)  #add reward value 1 to the reward vector
            value_f_left = m * value_f_left + alpha_plus * (rho * 1-value_f_left)  #update value function for reward
        else :
            rewards = np.append(rewards, 0)      #in the other 0.8 cases no reward is received; add reward 0 to vector
            value_f_left = m * value_f_left + alpha_minus * (0-value_f_left)  #update value function for no reward
        decisions = np.append(decisions, 0)
        values_left = np.append(values_left, value_f_left)
        values_right = np.append(values_right, value_f_right)

    else :  #if decision is 1 = right
        if np.random.random() <= P_R_rew :  # P(rew)_R = 0.5 through generation of random float between 0 and 1
            rewards = np.append(rewards, 1)  #add reward value 1 to the reward vector
            value_f_right = m * value_f_right + alpha_plus * (rho * 1-value_f_right)   #update value function for reward
        else:
            rewards = np.append(rewards, 0)  #add reward value 0 to the reward vector
            value_f_right = m * value_f_right + alpha_minus * (0-value_f_right)   #update value function for no reward
        decisions = np.append(decisions, 1)
        values_right = np.append(values_right, value_f_right)
        values_left = np.append(values_left, value_f_left)

##Calculate moving average over 10 trials for better visualization in new vector x_averaged
window_size = 10
i = 0
decisions_averaged = []
rewards_averaged = []
values_right_averaged = []
values_left_averaged = []

#Moving average decisions
while i < len(decisions) - window_size + 1:
    window_average = np.sum(decisions[i:i+window_size]) / window_size
    decisions_averaged.append(window_average)
    i += 10
i=0


#Moving average rewards
while i < len(rewards) - window_size + 1:
    window_average = np.sum(rewards[i:i+window_size]) / window_size
    rewards_averaged.append(window_average)
    i += 10
i=0
    
#Moving average values right
while i < len(values_right) - window_size + 1:
    window_average = np.sum(values_right[i:i+window_size]) / window_size
    values_right_averaged.append(window_average)
    i += 10
i=0

#Moving average values left
while i < len(values_left) - window_size + 1:
    window_average = np.sum(values_left[i:i+window_size]) / window_size
    values_left_averaged.append(window_average)
    i += 10
i=0

##Visualize model
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("Set2")
#g = sns.lineplot(data=values_right_averaged, alpha = 0.6, color = "c", label = "Value f right")
g = sns.lineplot(data=decisions_averaged, alpha = 0.6, color = "black", label = "Decision (0 = left, 1 = right)")
g = sns.lineplot(data=values_left_averaged, alpha = 0.6, color = "purple", label = "Value f left")
#g = sns.lineplot(data=values_left_averaged, alpha = 0.7, label = "Value f left")
plt.title('Basic RL model')
#plt.title('RL model beta=' + str(beta))
plt.xlabel('Trials (averaged over 10)')
plt.setp(g, yticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.legend(fontsize = 8, loc = 'lower left')
plt.show()


###Quantify when-lose-stay-switch behavior (right = 1, left = 0)
##Stay after reward
#P(d_n = 1 & d_n+1 = 1 | reward = 1)
#P(d_n = 0 & d_n+1 = 0 | reward = 1)

##Switch after omission
#P(d_n = 1 & d_n+1 = 0 | reward = 0)
#P(d_n = 0 & d_n+1 = 1 | reward = 0)

stay_trials = []
switch_trials = []
i = 0

for decisions[i] in decisions:
    if i == len(decisions)-1 :
        break;
    if decisions[i] == decisions[i+1] and rewards[i] == 1 :
            stay_trials.append(1)
            i += 1
    elif decisions[i] != decisions[i+1] and rewards[i] == 0 :
            switch_trials.append(1)
            i += 1
    else :
            i += 1


P_stay = sum(stay_trials) / len(decisions)
print("Proportion of stay trials = " + str(P_stay))

P_switch = sum(switch_trials) / len(decisions)
print("Proportion of switch trials = " + str(P_switch))

#stay_switch_df_2 = stay_switch_df_2.append({'alpha': alpha_plus,
#                                        'Prop stay': P_stay,
#                                        'Prop switch': P_switch,
#                                        'Block trials': 'no'}, ignore_index=True)
#print(stay_switch_df_2)

#Plot stay switch behavior
#stay_switch_df_all = stay_switch_df.append(stay_switch_df_2, ignore_index=True)

#g2 = sns.FacetGrid(stay_switch_df_all, col='Block trials')
#g2.map(sns.lineplot,'alpha', 'Prop stay', label ='P_stay', color ='g')
#g2.map(sns.lineplot,'alpha', 'Prop switch', label ='P_switch', color= 'r')
#g2.fig.suptitle('Stay switch behavior dependent on alpha')
#g2.fig.subplots_adjust(top=0.8)
#g2.set_ylabels('Proportion of trials')
#plt.legend(loc = "lower right")
#plt.show()

