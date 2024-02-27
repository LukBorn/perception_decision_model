import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import scripts as sc

class Params():
    def __init__(self,
                 seed = 42,
                 verbose = False,
                 init_values = 1,
                 alpha = 0.5,
                 sigma = np.sqrt(0.2),
                 reward_magnitude=(1,1),
                 time_steps = 100000,
                 blocks = None,
                 stimulus_type = "linspace",
                 choice_type = "greedy",
                 beta = 9.097,
                 difficulty_cut = 0.6
                 ):

        np.random.seed = seed
        self.verbose = verbose
    # inital values for value function
        self.init_values = init_values

    # learning constant alpha -> the factor the prediction error is multiplied by
        self.alpha = alpha

    # the standard deviation of the percept gaussian
        self.sigma = sigma

    # # magnitude of reward -> should be in shape [reward_magnitude_left, reward_magnitude_right]
    #     self.reward_magnitude = reward_magnitude

    # time steps -> how many trials a model does
        self.time_steps = time_steps

    # blocks -> None: no blocks with differing reward probabilities (reward bias)
    #        -> int like 500 -> blocks of 500 trials where the reward biases are switched
    #     if type(blocks) is not int or type(blocks) is not None :
    #         raise ValueError("blocks must be either None or dtype int")
        self.blocks = self.get_blocks()


    # type of stimulus: has to be between -1 and 1
    # 'linspace': 10 equally spaced values
    # 'equi': equidistributed --> not fully implemented
        self.stimulus_type = stimulus_type

    # type of choice: how decision is made
    # 'greedy': chooses the maximal of the two Q values to decide the choice
    # 'beta': decides based on delta Q and beta
        self.choice_type = choice_type

    # beta -> standard deviation of
        self.beta = beta

    # the cut for psychometric evaluation between an easy and a hard difficulty stimulus
        self.difficulty_cut = difficulty_cut

    def get_blocks(self):
        blocks = np.empty(self.time_steps)




def decision_model(params = Params(),
                   reward_blocks = None):
    """
    Decision Model based on Lak et al 2019:
    Reinforcement biases subsequent perceptual decisions when confidence is low, a widespread behavioral phenomenon
    https://elifesciences.org/articles/49834

    explained in detail in figure 3a, as well as in these paragraphs:
    Belief-based reinforcement learning models account for choice updating
    Methods: TDRL model with stimulus belief state

    :param params: parameter Class holding all parameters
    :return: Results class holding all results
    """
    def get_stimulus(type):
        if type == "equi":
            # returns stimulus sampled randomly from equidistribution [-1:1]
            return np.random.uniform(1, -1, 1)[0]
        elif type == "linspace":
            # returns stimulus sampled randomly from values equally spaced between -1 and 1
            return np.random.choice(np.linspace(-1, 1, 10), 1).round(3)[0]

    def get_percept(stimulus, sigma):
        # internal estimate of the stimulus is normally distributed
        # with constant variance around true stimulus contrast sigma
        percept = np.random.normal(stimulus, sigma, 1)
        p_percept = 0.5 * (1 + scipy.special.erf(percept / (np.sqrt(2) * sigma)))
        # returns array of shape [p_left, p_right]
        return np.array([1 - p_percept, p_percept]).T[0]

    # set initial values: [value_left, value_right]
    value = np.full(2,params.init_values, dtype=np.float64)

    # instanciate Results class to store all the results
    results = Results(params)

    if reward_blocks is None:
        reward_blocks = np.empty((params.time_steps,2),np.float16)
    elif reward_blocks.shape != (params.time_steps,2):
        raise AttributeError("reward_blocks must be in shape (params.time_steps,2), \n "
                             "with reward_blocks[0] being left reward_magnitudes, and reward_blocks[0] ")

    for i in range(params.time_steps):
        stimulus = get_stimulus(params.stimulus_type)
        percept = get_percept(stimulus, params.sigma)
        Q = percept * value

        if params.choice_type == "greedy":
            choice = np.argmax(Q)  # choice -> left = 0, right = 1
        elif params.choice_type == "beta":
            choice = 1 / (1 - np.exp(params.beta * (Q[1]-Q[0])))

        # right now confidence is not used
        # confidence = np.abs(percept)

        # stimulus < 0 and choice = 0 -> reward = 1
        # stimulus > 0 and choice = 1 -> reward = 0
        if [-1,1][choice] == np.sign(stimulus): # correct choice
            #reward = np.random.binomial(1,params.p_rew[choice],1)[0]
            reward = reward_blocks[i,choice]
        else: reward = 0

        prediction_error = reward-Q[choice]
        #update the value by adding the product of learning rule alpha and reward prediction error
        value[choice] += params.alpha * prediction_error

        # add your results to your Results object
        results.stimuli[i] = stimulus
        results.choices[i] = choice
        results.rewards[i] = reward
        results.prediction_error[i] = prediction_error
        results.values[i] = value

    if params.verbose:
        print(f"finished computing model with {params.time_steps} time steps")

    results.bin_stimuli()
    return results






class Results():
    """
    class for storing all the results of the model
    """
    def __init__(self,
                 params = Params()):
        self.params = params
        self.stimuli = np.empty(params.time_steps,dtype=np.float64)
        self.choices = np.empty(params.time_steps)
        self.rewards = np.empty(params.time_steps)
        self.prediction_error = np.empty(params.time_steps)
        self.values = np.empty((params.time_steps,2))
        self.rewards_smooth = None
        self.values_left_smooth = None
        self.values_right_smooth = None
        self.choices_smooth = None
        self.psychometric = None
        self.updating_matrix = None
        self.updating_function = None

    def smooth_all(self, window_size = 10):
        self.rewards_smooth = sc.moving_avg_smooth(self.rewards, window_size)
        self.values_left_smooth = sc.moving_avg_smooth(self.values[:, 0], window_size)
        self.values_right_smooth = sc.moving_avg_smooth(self.values[:, 1], window_size)
        self.choices_smooth = sc.moving_avg_smooth(self.choices, window_size)

    def bin_stimuli(self):
        if self.params.stimulus_type in ["equi"]:
            #bin the equidistributed variables into num-1 bins
            #with the value assigned corresponding to the max of the bins
            linspace = np.linspace(-1,1,num = 10).round(3)
            self.stimuli = linspace[np.digitize(self.stimuli, linspace)]
            # im not actually this works, honestly just use params.stimulus_type = "linspace"


    def plot(self,
             variables=['choices', 'values_left'],
             start = 0,
             stop = 1):
        # plot the direct results of the model from [start:stop]

        if self.values_left_smooth is None:
            self.smooth_all()

        sns.set_palette("Set2")
        if 'choices' in variables:
            g = sns.lineplot(data=self.choices_smooth[start:stop],
                             alpha=0.6, color="black", label="Decision (0 = left, 1 = right)")
        if 'values_left' in variables:
            g = sns.lineplot(data=self.values_left_smooth[start:stop],
                             alpha=0.6, color="purple", label="Values left")
        if 'values_right' in variables:
            g = sns.lineplot(data=self.values_right_smooth[start:stop],
                             alpha=0.6, color="blue", label="Values right")
        if 'rewards' in variables:
            g = sns.lineplot(data=self.rewards_smooth[start:stop],
                             alpha=0.6, color="green", label="Rewards (0 = no reward, 1 = reward)")

        # if self.params.blocks:
        #     plt.title('RL block model recovered values')
        #     plt.xlabel('Trials (averaged over 10)')
        #     plt.setp(g, yticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        #     plt.axvspan(0, 10, alpha=0.3, color="orange",
        #                 label=f"P(Rew)_R={self.params.P_R_rew}, P(Rew)_L={self.params.P_L_rew}")
        #     plt.axvspan(10, 20, alpha=0.3, color="gainsboro",
        #                 label=f"P(Rew)_R={self.params.P_L_rew}, P(Rew)_L={self.params.P_R_rew}")
        #     plt.axvspan(20, 30, alpha=0.3, color="orange")
        #     plt.axvspan(30, 40, alpha=0.3, color="gainsboro")
        #     plt.axvspan(40, 50, alpha=0.3, color="orange")
        #     plt.axvspan(50, 60, alpha=0.3, color="gainsboro")
        #     plt.axvspan(60, 70, alpha=0.3, color="orange")
        #     plt.axvspan(70, 80, alpha=0.3, color="gainsboro")
        #     plt.axvspan(80, 90, alpha=0.3, color="orange")
        #     plt.axvspan(90, 100, alpha=0.3, color="gainsboro")
        #     plt.legend(fontsize=7.5, loc='lower right')
        #     plt.show()
        #else:

        plt.title('Basic RL model')
        plt.setp(g, yticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.legend(fontsize=8, loc='lower left')
        plt.show()

    def plot_previous_choice(self):
        """
        plots the stimulus vs choice average, seperately for each previous choice
        """
        psychometric = pd.DataFrame(index=["previous_left", "previous_right"],
                                    columns=[i for i in np.unique(self.stimuli)])
        previous_choices = np.concatenate(([0],self.choices[:-1]))
        for stimulus in np.unique(self.stimuli):
            previous_left_idx = np.logical_and(previous_choices == 0, self.stimuli == stimulus).nonzero()[0]
            psychometric.loc["previous_left", stimulus] = np.mean(self.choices[previous_left_idx])
            previous_right_idx = np.logical_and(previous_choices == 1, self.stimuli == stimulus).nonzero()[0]
            psychometric.loc["previous_right", stimulus] = np.mean(self.choices[previous_right_idx])

        g = sns.lineplot(data=psychometric.T, dashes=False, markers=True, palette="Set1")
        g.set_xlabel("Current Stimulus")
        g.set_ylabel("Average Choice")


    # def plot_previous_rewarded(self):
    #     psychometric = pd.DataFrame(index=["current","previous"],
    #                                 columns=[i for i in np.unique(self.stimuli)])
    #
    #     for stimulus in np.unique(self.stimuli):
    #         psychometric.loc["current", stimulus] = np.mean(self.choices[self.stimuli == stimulus])
    #
    #     previous_reward_index = pd.DataFrame(columns = np.arange(len(self.rewards)),
    #                                          data = np.arange(len(self.rewards)))
    #     previous_reward_index[self.rewards == 0] = np.nan
    #     previous_reward_index = previous_reward_index.ffill().values.T.astype(int)[0]
    #     previous_reward_index = np.concatenate([[0], previous_reward_index[:-1]])
    #     # wrong thing to calculate fuck
    #     # yea so this doesnt work

    def get_psychometric(self,
                     difficulty_cut = None,
                     ):
        """
        plotting the psychometrics of the model based on Lak et al 2019:
        Reinforcement biases subsequent perceptual decisions when confidence is low, a widespread behavioral phenomenon
        https://elifesciences.org/articles/49834

        explained in detail in figure 1d
        """

        #plot psychometric curve -> bins for stimulus, corresponding correct choice percentage
        self.bin_stimuli()

        psychometric = pd.DataFrame(index=["current"]+[i for i in np.unique(self.stimuli)],
                                    columns = [i for i in np.unique(self.stimuli)])

        for stimulus in np.unique(self.stimuli):
            psychometric.loc["current", stimulus] = np.mean(self.choices[self.stimuli == stimulus])

        for previous_stimulus in np.unique(self.stimuli):
            previous_stimuli = self.stimuli[np.nonzero(self.stimuli[:-1] == previous_stimulus)[0] + 1]
            previous_choices = self.choices[np.nonzero(self.stimuli[:-1] == previous_stimulus)[0] + 1]

            for stimulus in np.unique(previous_stimuli):
                psychometric.loc[previous_stimulus, stimulus] = np.mean(previous_choices[previous_stimuli == stimulus])

        #updating matrix is the difference of the previous
        updating_matrix = psychometric.loc[np.unique(self.stimuli)] - psychometric.loc["current"]

        if difficulty_cut is not None:
            self.params.difficulty_cut = difficulty_cut

        updating_function = pd.DataFrame(index = updating_matrix.index, columns= ["hard","easy"])
        updating_function["easy"] = updating_matrix.loc[:,np.abs(updating_matrix.index > self.params.difficulty_cut)].mean(axis = 1)
        updating_function["hard"] = updating_matrix.loc[:,np.abs(updating_matrix.index < self.params.difficulty_cut)].mean(axis = 1)


        self.psychometric = psychometric
        self.updating_matrix = updating_matrix
        self.updating_function = updating_function

        return psychometric, updating_matrix, updating_function


    def plot_psychometric(self,
                          previous_psychometric = None, #previous psychometric to plot alongside current
                          ):
        """
        plotting the psychometrics of the model based on Lak et al 2019:
        Reinforcement biases subsequent perceptual decisions when confidence is low, a widespread behavioral phenomenon
        https://elifesciences.org/articles/49834

        explained in detail in figure 1d
        """

        if self.updating_matrix is None:
            self.get_psychometric()

        # plot current psychometric and random previous psychometric
        fig, axes = plt.subplots(nrows=1, ncols=3)
        if previous_psychometric is None:
            # make sure you get a hard choice so the difference is obvious in the plot
            hard_choices = np.unique(self.stimuli)[np.abs(np.unique(np.unique(self.stimuli))) < self.params.difficulty_cut]
            previous_psychometric = np.random.choice(hard_choices)
        data = self.psychometric.loc[["current", previous_psychometric]].T
        data.index.astype(str)
        sns.lineplot(data=data, dashes=False, markers=True, palette="Set1", ax=axes[0])
        axes[0].set_xlabel("Current/Previous Stimulus")
        axes[0].set_ylabel("Average Choice")

        # plot updating matrix

        img = axes[1].imshow(self.updating_matrix.values.astype(np.float64).T * 100,
                             cmap='RdBu', interpolation='nearest', aspect='auto')
        plt.colorbar(img, ax=axes[1], label="Updating %")  # Add color bar
        axes[1].set_xlabel("Current Stimulus")
        axes[1].set_ylabel("Previous Stimulus")
        axes[1].set_xticks(np.arange(len(self.updating_matrix.index)))
        axes[1].set_yticks(np.arange(len(self.updating_matrix.index)))
        axes[1].set_xticklabels(self.updating_matrix.index,rotation=90)
        axes[1].set_yticklabels(self.updating_matrix.index)

        # plot updating function
        sns.lineplot(data=(self.updating_function * 100), dashes=False, markers=True, ax=axes[2])
        axes[2].set_xlabel("Current Stimulus")
        axes[2].set_ylabel("Updating %")




#todo figure out why this doesnt work

def plot_params(alphas = [0.2,0.5,0.7],
                sigmas = [0.2,0.5,0.7],
                n = 10000
):
    alpha_sigma = pd.DataFrame(index = alphas,columns=sigmas)

    for alpha in alphas:
        for sigma in sigmas:
            results = decision_model(Params(alpha=alpha, sigma=sigma,time_steps=n,verbose=True))
            _,matrix,_ = results.get_psychometric()
            alpha_sigma.loc[alpha,sigma] = matrix.values
            ticks = matrix.index

    fig, axes = plt.subplots(nrows=len(alphas),
                             ncols=len(sigmas),
                             figsize=(10, 10))

    for (i, j), ax in np.ndenumerate(axes):
        ax.imshow(alpha_sigma.iloc[i,j].astype(np.float64).T,
                             cmap='RdBu', interpolation='nearest', aspect='auto')

        ax.set_title(f'(alpha = {alpha_sigma.index[i]}, sigma = {alpha_sigma.columns[j]})')
        ax.set_xlabel("Current Stimulus")
        ax.set_ylabel("Previous Stimulus")
        ax.set_xticks(np.arange(len(ticks)))
        ax.set_yticks(np.arange(len(ticks)))
        ax.set_xticklabels(ticks, rotation=90)
        ax.set_yticklabels(ticks)
        # Show y-axis only on the rightmost subplots
        if j > 0:
            ax.yaxis.set_visible(False)
        # Show x-axis only on the bottom subplots
        if i < 2:
            ax.xaxis.set_visible(False)

    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # # Create a single colorbar
    # divider = make_axes_locatable(axes[-1, -1])
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    # cbar = plt.colorbar(im, cax=cax)
    # cbar.set_label('Colorbar Label')
    # plt.show()

    return alpha_sigma


