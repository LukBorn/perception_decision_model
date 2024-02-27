# -*- coding: utf-8 -*-

## Reinforcement learning model with changing reward probabilities in block trials

#Block trial structure
#Probabilities change every 100 trials
#P_L(Rew) = {0.2, 0.5}, P_R(Rew) = {0.5, 0.2} ; when P_L(Rew) is high, then P_R(Rew) is low, and vv
import pandas as pd
#stay_switch_df = pd.DataFrame(columns=['alpha', 'Prop stay', 'Prop switch', 'Block trials'])

import numpy as np
np.random.seed(42)
value_f_left = 0.5
value_f_right = 0.5
beta = 9.097
alpha_plus = 0.164 #alpha learning rate after reward
alpha_minus = 0.010 #alpha learning rate after no reward
rho = 1 #reward sensitivity rho: factor for calculation of prediction error, lower in anhedonia
m = 1 #memory of previous reinforcement: influence by prior r, instead of last trial
P_L_rew_1 = 0.4 #reward probability for choice left in block 1
P_R_rew_1 = 0.1 #reward probability for choice right in block 1
P_L_rew_2 = P_R_rew_1 #reward probability for choice left in block 2
P_R_rew_2 = P_L_rew_1 #reward probability for choice right in block 2
rewards = np.array([])
decisions = np.array([])
values_left = np.array([])
values_right = np.array([])
def log_fun(x): #define a sigmoidal logistic function to model the decision probability 
    return(1 / (1 + np.exp(- beta * x)))

for x in range (5) :
    for y in range(100) : #block 1 with P_L(rew) = 0.2 and P_R(rew) = 0.5
        p_decision_right = log_fun(value_f_right- value_f_left)  #define decision probability which is proportional to value functions
        decision = np.random.binomial(1, p_decision_right)

        if decision == 0 :  #if decision is 0 = left
            if np.random.random() <= P_L_rew_1 :  #P(rew)_L = 0.2 through generation of random float between 0 and 1
                rewards = np.append(rewards, 1)  #add reward value 1 to the reward vector
                value_f_left = m * value_f_left + alpha_plus * (rho * 1-value_f_left)  #update value function for reward
            else :
                rewards = np.append(rewards, 0)      #in the other 0.8 cases no reward is received; add reward 0 to vector
                value_f_left = m * value_f_left + alpha_minus * (0-value_f_left)  #update value function for no reward
            decisions = np.append(decisions, 0)
            values_left = np.append(values_left, value_f_left)
            values_right = np.append(values_right, value_f_right)
        else :  #if decision is 1 = right
             if np.random.random() <= P_R_rew_1 :  # P(rew)_R = 0.5 through generation of random float between 0 and 1
                 rewards = np.append(rewards, 1)  #add reward value 1 to the reward vector
                 value_f_right = m * value_f_right + alpha_plus * (rho * 1-value_f_right)   #update value function for reward
             else:
                 rewards = np.append(rewards, 0)  #add reward value 0 to the reward vector
                 value_f_right = m * value_f_right + alpha_minus * (0-value_f_right)  #update value function for no reward
             decisions = np.append(decisions, 1)
             values_right = np.append(values_right, value_f_right)
             values_left = np.append(values_left, value_f_left)
    for z in range(100) : #block 2 with P_L(rew) = 0.5 and P_R(rew) = 0.2
       p_decision_right = log_fun(value_f_right- value_f_left)  #define decision probability which is proportional to value functions
       decision = np.random.binomial(1, p_decision_right)

       if decision == 0 :  #if decision is 0 = left
            if np.random.random() <= P_L_rew_2 :  #P(rew)_L = 0.2 through generation of random float between 0 and 1
                rewards = np.append(rewards, 1)  #add reward value 1 to the reward vector
                value_f_left = m * value_f_left + alpha_plus * (rho * 1-value_f_left)  #update value function for reward
            else :
                rewards = np.append(rewards, 0)      #in the other 0.8 cases no reward is received; add reward 0 to vector
                value_f_left = m * value_f_left + alpha_minus * (0-value_f_left)  #update value function for no reward
            decisions = np.append(decisions, 0)
            values_left = np.append(values_left, value_f_left)
            values_right = np.append(values_right, value_f_right)
       else :  #if decision is 1 = right
             if np.random.random() <= P_R_rew_2 :  # P(rew)_R = 0.5 through generation of random float between 0 and 1
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
g = sns.lineplot(data=decisions_averaged, alpha = 0.6, color = "black", label = "Decision (0 = right, 1 = left)")
#g = sns.lineplot(data=values_right_averaged, alpha = 0.7, label = "Value f right")
#plt.title('RL model alpha=' + str(alpha_plus))
plt.title('RL block model recovered values')
plt.xlabel('Trials (averaged over 10)')
plt.setp(g, yticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.axvspan(0, 10, alpha = 0.3, color = "orange", label = "P(Rew)_R=" + str(P_R_rew_1) + ", P(Rew)_L=" + str(P_L_rew_1))
plt.axvspan(10, 20, alpha=0.3, color= "gainsboro", label = "P(Rew)_R=" + str(P_R_rew_2) + ", P(Rew)_L=" + str(P_L_rew_2))
plt.axvspan(20, 30, alpha = 0.3, color = "orange")
plt.axvspan(30, 40, alpha = 0.3, color = "gainsboro")
plt.axvspan(40, 50, alpha = 0.3, color = "orange")
plt.axvspan(50, 60, alpha = 0.3, color = "gainsboro")
plt.axvspan(60, 70, alpha = 0.3, color = "orange")
plt.axvspan(70, 80, alpha = 0.3, color = "gainsboro")
plt.axvspan(80, 90, alpha = 0.3, color = "orange")
plt.axvspan(90, 100, alpha = 0.3, color = "gainsboro")
plt.legend(fontsize = 7.5, loc = 'lower right')
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

#stay_switch_df = stay_switch_df.append({'alpha': alpha_plus,
#                                        'Prop stay': P_stay,
#                                        'Prop switch': P_switch,
#                                        'Block trials': 'yes'}, ignore_index=True)
#print(stay_switch_df)


### MODEL RECOVERY
#from lmfit import Parameters, report_fit, minimize
#paramsA = Parameters()
#paramsA.add("alpha", min = 0.01, max = 2, value = 0.5) #set initial value and min/max bounds for param search
#paramsA.add("beta", min = 0.01, max = 6, value = 1.5) #set initial value and min/max bounds for param search

#def NLL_functionA(paramsA) :
#    alpha = paramsA['alpha'].value
#    beta = paramsA['beta'].value
#    if alpha < 0.01 or alpha > 2 : #alpha borders that limit parameter search later
#        return(np.inf)
#
#    if beta < 0.01 or beta > 6 :  #beta borders that limit parameter search later
#        return(np.inf)
#    value_right = 0.5
#    value_left = 0.5
#    i_p_right_NLL = np.array([])
#    i = 0
#    for decisions[i] in decisions :
#        if i == 0 :
#            p_right = 0.5
#            i_p_right_NLL = np.append(i_p_right_NLL, 0.5)
#        elif i < len(decisions) :
#            p_right = 1 / (1 + np.exp(- beta * (value_right - value_left)))
#       else :
#          break
#
#        if decisions[i] == 0 :
#            value_left = value_left + alpha * (rewards[i] - value_left)
#            i_p_right_NLL = np.append(i_p_right_NLL, 1-p_right)
#            i += 1
#        elif decisions[i] == 1 :
#            value_right = value_right + alpha * (rewards[i] - value_right)
#            i_p_right_NLL = np.append(i_p_right_NLL, p_right)
#            i += 1
#    NLL = - np.sum(np.log(i_p_right_NLL[1:]))
#    return(NLL)

###MINIMIZATION FUNCTION
#fit_resultA = minimize(NLL_functionA, paramsA, method='nelder', max_nfev= 200)

#report_fit(fit_resultA)


###Compare simulation to rat data
###Import Matlab data from .mat object, tidy it and explore it
#Load .mat file, but only specified variables (with variable_names)
import scipy.io
BC01 = scipy.io.loadmat('/Users/lilly/Desktop/RL_models/BC01/BC01.mat', variable_names = ("ChoiceLeft", "Rewarded")) #mat file becomes dictionary

#Transform items from the BC01 dictionary into numpy arrays
BC01_decisions_all = np.array(BC01["ChoiceLeft"], copy=True, dtype=np.float64)
BC01_rewards_all = np.array(BC01['Rewarded'], copy=True, dtype=np.float64)

#Reduce dimension of arrays
BC01_decisions_all = np.reshape(BC01_decisions_all, (8724,))
BC01_rewards_all = np.reshape(BC01_rewards_all, (8724,))

#identify indices with NaN in decisions array
nan_indices = np.argwhere(np.isnan(BC01_decisions_all))

#delete entries with NaN from decisions and corresponding entries from rewards
BC01_decisions_tidy = np.delete(BC01_decisions_all, nan_indices)
BC01_rewards_tidy = np.delete(BC01_rewards_all, nan_indices)

#Subset behavioral data to the first 300 trials
BC01_decisions = BC01_decisions_tidy[0:1000]
BC01_rewards = BC01_rewards_tidy[0:1000]

##Calculate moving averages
window_size = 10
n = 0
BC01_decisions_averaged = []
BC01_rewards_averaged = []

#Moving average decisions
while n < len(BC01_decisions) - window_size + 1:
     window_average = np.sum(BC01_decisions[n:n+window_size]) / window_size
     BC01_decisions_averaged.append(window_average)
     n += 10
n=0


g2 = sns.lineplot(data=decisions_averaged, alpha = 0.5, color = "red", label = "Decisions model")
g2 = sns.lineplot(data=BC01_decisions_averaged, alpha = 0.6, color = "black", label = "Decisions rat")
plt.title('Comparison model B - rat data')
plt.xlabel('Trials (averaged over 10)')
plt.setp(g, yticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.legend(fontsize = 7.5, loc = 'lower right')
plt.show()

