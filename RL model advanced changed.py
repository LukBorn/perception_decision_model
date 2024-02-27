#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 17:17:46 2022

@author: lilly, lukas now too :)

RL model for depression learning rate hypothesis
model contains 2 different learning rates alpha: one for after reward (alpha+) and one for after no reward (alpha-)
hypothesis: in healthy agent alpha+ = alpha-; in depressed agent presumably alpha+ < alpha-
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scripts as sc

class Params():
    # a class for storing all the paramaters for a certain model
    def __init__(self,
                 seed = 42,
                 verbose = True,
                 value_left = 0.5,
                 value_right = 0.5,
                 beta = 2.5,
                 alpha_plus = 0.5,  # alpha learning rate after reward
                 alpha_minus = 0.5,  # alpha learning rate after no reward
                 rho = 1,  # reward sensitivity rho: factor for calculation of prediction error, lower in anhedonia
                 m = 1,  # memory of previous reinforcement: influence by prior r, instead of last trial
                 P_L_rew = 0.3, # probability of reward for a left trial
                 P_R_rew = 0.7, # probability of reward for a left trial
                 time_steps = 1000,
                 blocks = False,
                 block_n = 5
                 ):
        np.random.seed(seed)
        self.verbose = verbose
        self.value_left_init = value_left
        self.value_right_init = value_right
        self.beta = beta
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.rho = rho
        self.m = m
        self.P_L_rew = P_L_rew
        self.P_R_rew = P_R_rew
        self.time_steps = time_steps
        self.blocks = blocks
        self.block_n = block_n
        self.block_size = time_steps/block_n*2

        def print():
            ##Todo: add comprhensive print function
            print("Not implemented :(")

class Results():
    #a class for storing all the results of a certain experiment
    def __init__(self,
                 params:Params = Params(),
                 ):
        self.params = params
        self.decisions = np.array([])
        self.rewards = np.array([])
        self.values_left = np.array([])
        self.values_right = np.array([])

    def smooth_all(self, window_size = 10):
        self.decisions_smooth = sc.moving_avg_smooth(self.decisions, window_size)
        self.rewards_smooth = sc.moving_avg_smooth(self.rewards, window_size)
        self.values_right_smooth = sc.moving_avg_smooth(self.values_right, window_size)
        self.values_left_smooth = sc.moving_avg_smooth(self.values_left, window_size)

    def plot(self,
             variables = ['decisions','values_left'],
             window_size = 10
             ):

        try:
            self.values_left_smooth
        except AttributeError:
            self.smooth_all()

        sns.set_palette("Set2")
        if 'decisions' in variables:
            g = sns.lineplot(data=self.decisions_smooth, alpha=0.6, color="black",
                             label="Decision (0 = left, 1 = right)")
        if 'values_left' in variables:
            g = sns.lineplot(data=self.values_left_smooth, alpha=0.6, color="purple",
                             label="Value function left")
        if 'values_right' in variables:
            g = sns.lineplot(data=self.values_right_smooth, alpha=0.6, color="c",
                             label="Value f right")
        if 'rewards' in variables:
            g = sns.lineplot(data=self.rewards_smooth, alpha=0.6, color="green",
                             label="Rewards (0 = no reward, 1 = reward)")

        if self.params.blocks:
            plt.title('RL block model recovered values')
            plt.xlabel('Trials (averaged over 10)')
            plt.setp(g, yticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            plt.axvspan(0, 10, alpha=0.3, color="orange",
                        label=f"P(Rew)_R={self.params.P_R_rew}, P(Rew)_L={self.params.P_L_rew}")
            plt.axvspan(10, 20, alpha=0.3, color="gainsboro",
                        label=f"P(Rew)_R={self.params.P_L_rew}, P(Rew)_L={self.params.P_R_rew}")
            plt.axvspan(20, 30, alpha=0.3, color="orange")
            plt.axvspan(30, 40, alpha=0.3, color="gainsboro")
            plt.axvspan(40, 50, alpha=0.3, color="orange")
            plt.axvspan(50, 60, alpha=0.3, color="gainsboro")
            plt.axvspan(60, 70, alpha=0.3, color="orange")
            plt.axvspan(70, 80, alpha=0.3, color="gainsboro")
            plt.axvspan(80, 90, alpha=0.3, color="orange")
            plt.axvspan(90, 100, alpha=0.3, color="gainsboro")
            plt.legend(fontsize=7.5, loc='lower right')
            plt.show()

        else:
            plt.title('Basic RL model')
            plt.xlabel('Trials (averaged over 10)')
            plt.setp(g, yticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            plt.legend(fontsize = 8, loc = 'lower left')
            plt.show()


    def stay_switch(self):
        ###Quantify when-lose-stay-switch behavior (right = 1, left = 0)
        ##Stay after reward
        # P(d_n = 1 & d_n+1 = 1 | reward = 1)
        # P(d_n = 0 & d_n+1 = 0 | reward = 1)

        ##Switch after omission
        # P(d_n = 1 & d_n+1 = 0 | reward = 0)
        # P(d_n = 0 & d_n+1 = 1 | reward = 0)

        self.stay_trials = np.array([])
        self.switch_trials = np.array([])

        for i in range(len(self.decisions)-1):
            if self.decisions[i] == self.decisions[i+1] and self.rewards[i] == 1:
                np.append(self.stay_trials, i)
            elif self.decisions[i] != self.decisions[i+1] and self.rewards[i] == 0:
                np.append(self.switch_trials, i)

        self.stay_proportion = self.stay_trials.shape[0]/self.decisions.shape[0]
        self.switch_proportion =self.switch_trials.shape[0]/self.decisions.shape[0]

        if self.params.verbose:
            print(f"Proportion of stay trials = {self.stay_proportion}")
            print(f"Proportion of switch trials = {self.switch_proportion}")



def plot_stay_switch(alphas:list,blocks = False):
    #alphas = list of tuples (alpha+, alpha-),
    #if were just looking at different alphas, alpha+ == alpha-
    results = pd.DataFrame(columns=['alpha', 'Prop stay', 'Prop switch', 'Block trials'])
    for alpha in alphas:
        params = Params(alpha_plus=alpha[0],
                        alpha_minus=alpha[1],
                        verbose=False,
                        blocks=blocks)
        model_results = run_model(params)
        model_results.stay_switch()
        results.append({'alpha': alpha,
                        'Prop stay': model_results.stay_proportion,
                        'Prop switch': model_results.switch_proportion,
                        'Block trials': blocks}, ignore_index=True)

    g2 = sns.FacetGrid(results, col='Block trials')
    g2.map(sns.lineplot, 'alpha', 'Prop stay', label='P_stay', color='g')
    g2.map(sns.lineplot, 'alpha', 'Prop switch', label='P_switch', color='r')
    g2.fig.suptitle('Stay switch behavior dependent on alpha')
    g2.fig.subplots_adjust(top=0.8)
    g2.set_ylabels('Proportion of trials')
    plt.legend(loc="lower right")
    plt.show()




def run_model(params = Params()):
    results = Results()
    value_left = params.value_left_init
    value_right = params.value_right_init
    block_x = 0


    for x in range(params.time_steps):
        if params.blocks:
            block_x += 1
        # define decision probability which is proportional to value functions)
        decision = np.random.binomial(1, sc.log_fun(value_right - value_left, params.beta))

        if decision == 0:  #if decision is 0 = left
            reward = np.random.binomial(1,params.P_L_rew)
            value_left = (params.m * value_left +
                          (params.alpha_plus if reward == 1 else params.alpha_minus) * (params.rho * 1 - value_left))

        else:  #if decision is 1 = right
            reward = np.random.binomial(1,params.P_R_rew)
            value_right = (params.m * value_right +
                          (params.alpha_plus if reward == 1 else params.alpha_minus) * (params.rho * 1 - value_right))

        results.decisions = np.append(results.decisions, decision)
        results.rewards = np.append(results.rewards, reward)
        results.values_right = np.append(results.values_right, value_right)
        results.values_left = np.append(results.values_left, value_left)

        if block_x < params.block_size:
            _ = params.P_L_rew
            params.P_L_rew = params.P_R_rew
            params.P_R_rew = _
            block_x = 0

    return results


params = Params(verbose=True)
results = run_model(params)
results.plot()
results.stay_switch()

