import matplotlib.colors
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import scripts as sc
from tqdm import tqdm

class Params():
    """
    parameter class holding all model parameters
    """
    def __init__(self,
                 seed = 42,
                 init_values = 1,
                 alpha = 0.5,
                 sigma = np.sqrt(0.2),
                 time_steps = 100000,
                 stimulus_type = "linspace",
                 choice_type = "greedy",
                 beta = 9.097,
                 model_confidence= True,
                 delta = 1,
                 difficulty_cut = 0.6
                 ):

        np.random.seed = seed

    # inital values for value function
        self.init_values = init_values

    # learning constant alpha -> the factor the prediction error is multiplied by
        self.alpha = alpha

    # the standard deviation of the percept gaussian
        self.sigma = sigma

    # time steps -> how many trials a model does
        self.time_steps = time_steps

    # type of stimulus: has to be between -1 and 1
    # 'linspace': 10 equally spaced values
    # 'equi': equidistributed --> not fully implemented
        self.stimulus_type = stimulus_type

    # type of choice: how decision is made
    # 'greedy': chooses the maximal of the two Q values to decide the choice
    # 'beta': decides based on delta Q and beta
        self.choice_type = choice_type

    # beta -> standard deviation of
        # actually im not fully sure i understand this, i think its like epsilon of epsilon greedy k-armed bandit
        self.beta = beta

    # model_confidence -> models confidence if True
        self.model_confidence = model_confidence

    # delta -> factor that the investment sample is multiplied with
        self.delta = delta

    # the cut for psychometric evaluation between an easy and a hard difficulty stimulus
        self.difficulty_cut = difficulty_cut

    # reward magnitude: array with [reward_magnitude_left, reward_magnitude_right] at each time step i
        self.reward_magnitude = np.ones((time_steps, 2))

    # reward probability: array with [reward_probability_left, reward_probability_right] at each time step i
        self.reward_probability = np.ones((time_steps, 2))

    # blocks -> index of magnitude/probability structure
        self.blocks = np.empty(time_steps).astype(int)

        self.block_type = []
        self.magnitude_structure = None
        self.probability_structure = None
        self.block_size = None

    def get_blocks(self,
                   magnitude_structure = None,
                   probability_structure = None,
                   block_size = 400,
                   ):
        """
        changes self.reward_magnitude and self.reward_probabilities to reflect the specified block structure
        keeps repeating the specified block structure over all time steps

        if you don't run this function, reward magnitude and probabilities will remain 1 -> no bias blocks in experiment

        :param magnitude_structure: list of tuples of 2 reward magnitudes
            can also be an array of shape (number_of_blocks, 2)
            eg: [(m_left1,m_right1),(m_left2,m_right2),...]
        :param probability_structure: list of tuples of 2 reward magnitudes
            can also be an array of shape (number_of_blocks, 2)
            eg: [(m_left1,m_right1),(m_left2,m_right2),...]
        :param block_size: size of each block
        :return:
        """
        # make sure the magnitude and reward structure have the same shape
        if magnitude_structure is not None and probability_structure is not None:
            if magnitude_structure.shape != probability_structure.shape:
                raise ValueError("magnitude and reward structure must be same shape")


        if probability_structure is not None:
            # make sure it is an array
            probability_structure = np.array(probability_structure)
            #add probability to block_type
            self.block_type.append("probability")
            if probability_structure.shape[1] != 2:
                raise ValueError("magnitude structure must have shape (number_of_blocks, 2)with index [i,0] being left reward magnitude in block i, [i,1] for right")
            # define the length of all blocks combined
            total_blocks_length = probability_structure.shape[0] * block_size
            if total_blocks_length > self.time_steps:
                raise Warning("the time steps necessary for the specified block structure/size are more than the total time steps of the model. block structure will be truncated. Please specify a smaller block size or more total time steps")


        if magnitude_structure is not None:
            magnitude_structure = np.array(magnitude_structure)
            self.block_type.append("magnitude")
            if magnitude_structure.shape[1] != 2:
                raise ValueError("magnitude structure must have shape (number_of_blocks, 2) with index [i,0] being left reward magnitude in block i, [i,1] for right")
            # define the length of all blocks combined
            total_blocks_length = magnitude_structure.shape[0] * block_size
            if total_blocks_length > self.time_steps:
                raise Warning("the time steps necessary for the specified block structure/size are more than the total time steps of the model. block structure will be truncated. Please specify a smaller block size or more total time steps")

        print("Calculating block structure")
        for i in tqdm(range(self.time_steps)):
            self.blocks[i] = int((i%total_blocks_length)/block_size)
            if "magnitude" in self.block_type:
                self.reward_magnitude[i] = magnitude_structure[self.blocks[i]]
            if "probability" in self.block_type:
                self.reward_probability[i] = probability_structure[self.blocks[i]]

        self.magnitude_structure = magnitude_structure
        self.probability_structure = probability_structure
        self.block_size = block_size


class Model():
    """
    Decision Model based on Lak et al 2019:
    Reinforcement biases subsequent perceptual decisions when confidence is low, a widespread behavioral phenomenon
    https://elifesciences.org/articles/49834

    explained in detail in figure 3a, as well as in these paragraphs:
    Belief-based reinforcement learning models account for choice updating
    Methods: TDRL model with stimulus belief state

    run model using run_model()

    includes plotting functions:
    plot_psychometric()
        plots psychometric, updating matrix, and updating function
    plot([variables],start,stop)
        plots variables like stimuli, choices, rewards, or left and right values averaged over window_size
        in a line graph
    plot_previous_choice()
        plots psychometric for each previous choice separately
    plot_prediction_error()
        plots prediction error against stimulus
    plot_block_psychometric()
        plots the psychometric for each blocktype seperately
    plot_previous_bias()
        plots psychometric for if your previous choice was biased for or against
    """
    def __init__(self,
                 params = Params()):
        self.params = params

        self.stimuli = np.empty(params.time_steps,dtype=np.float64)
        self.percept = np.empty(params.time_steps,dtype=np.float64)
        self.choices = np.empty(params.time_steps)
        self.rewards = np.empty(params.time_steps)
        self.prediction_error = np.empty(params.time_steps)
        self.values = np.empty((params.time_steps,2))
        self.confidence = np.empty(params.time_steps,dtype=np.float64)

        self.rewards_smooth = None
        self.values_left_smooth = None
        self.values_right_smooth = None
        self.choices_smooth = None

        self.psychometric = None
        self.updating_matrix = None
        self.updating_function = None

    def run_model(self,
                  params=None):
        """
        runs the model, saving all results into itself

        get stimulus
        add perceptual uncertainty
        calculate p_left and p_right
        multiply p_left with value_left, same for right
        choose left or right based on values
        recieve reward or not
        update values based on reward prediction error
        """
        #generate params class if none is specified
        if params is not None:
            self.params = params

        # set initial values: [value_left, value_right]
        value = np.full(2, self.params.init_values, dtype=np.float64)

        print("Model Running:")
        for i in tqdm(range(self.params.time_steps)):
            # generate stimulus: random sample from one of 10 equally spaced stimuli between -1 and 1
            stimulus = np.random.choice(np.linspace(-1, 1, 10), 1).round(3)[0]

            # generate internal estimate of the stimulus
            percept = np.random.normal(stimulus, self.params.sigma, 1)[0]
            self.percept[i] = percept

            # calculate value-adjusted percept probability Q
            p_percept = 0.5 * (1 + scipy.special.erf(percept / (np.sqrt(2) * self.params.sigma)))
            p_percept = np.array([1 - p_percept, p_percept]).T[0]
            Q = p_percept * value

            # calculate choice based on Q
            if self.params.choice_type == "greedy":
                choice = np.argmax(Q)  # choice -> left = 0, right = 1
            elif self.params.choice_type == "beta":
                choice = 1 / (1 - np.exp(self.params.beta * (Q[1] - Q[0])))

            # calculate reward
            # stimulus < 0 and choice = 0 -> reward = 1
            # stimulus > 0 and choice = 1 -> reward = 0
            if [-1, 1][choice] != np.sign(stimulus): # incorrect choice
                reward = 0
            else:
                # reward is reward magnitude * 1 or 0 depending on reward probability
                reward = self.params.reward_magnitude[i, choice] * \
                         np.random.binomial(1, self.params.reward_probability[i, choice], 1)[0]

            if self.params.model_confidence:
                confidence = 0.5 * np.abs(percept) + 0.5
                if confidence < self.params.delta * np.random.uniform(0.5,1,1)[0]: # investment bigger than confidence -> model "waited out the time investment"
                    reward = 0
                self.confidence[i] = confidence

            # update the value by adding the product of learning rule alpha and reward prediction error
            prediction_error = reward - Q[choice]
            value[choice] += self.params.alpha * prediction_error

            # add your results to your Results object
            self.stimuli[i] = stimulus
            self.choices[i] = choice
            self.rewards[i] = reward
            self.prediction_error[i] = prediction_error
            self.values[i] = value

    def plot(self,
             variables=['choices', 'values_left'],
             start = 0,
             stop = 2000,
             window_size = 10,
             block_colors = None,
             ):
        """
        directly plots selection of variables of the model in a line graph

        :param variables: list of variables to plot in the line graph
            must be in ["choices","rewards","values_left","values_right"]
        :param start:
            where to start the line plot
        :param stop:
            where to end the line plot
        :param window_size:
            window size for smoothing function
        """
        sns.set_palette("Set2")
        # define the lines for each variable in variables
        if 'choices' in variables:
            g = sns.lineplot(data=sc.moving_avg_smooth(self.choices, window_size),
                             alpha=0.6, color="black", label="Decision (0 = left, 1 = right)")
        if 'values_left' in variables:
            g = sns.lineplot(data=self.values[:, 0],
                             alpha=0.6, color="purple", label="Values left")
        if 'values_right' in variables:
            g = sns.lineplot(data=self.values[:, 1],
                             alpha=0.6, color="blue", label="Values right")
        if 'rewards' in variables:
            g = sns.lineplot(data=sc.moving_avg_smooth(self.rewards, window_size),
                             alpha=0.6, color="green", label="Rewards (0 = no reward, 1 = reward)")

        # if any type of bias block was used during the experiment
        if np.unique(self.params.magnitude_structure).shape[0] > 1 or np.unique(self.params.probability_structure).shape[0] > 1:
            #find indexes where the blocks change, including start and stop index
            blocks = np.concatenate((np.arange(0,self.params.time_steps,self.params.block_size),np.array([start,stop])))
            blocks = np.unique(blocks[np.logical_and(blocks>=start,blocks<=stop)])

            #define the colors for the blocks
            block_color = pd.DataFrame(index=["values", "color"],
                                       columns=range(self.params.magnitude_structure.shape[0]))
            if block_colors is None:
                block_colors = ["gainsboro","orange","gold","coral","firebrick","sienna"]
            block_color.loc["color"] = block_colors[:self.params.magnitude_structure.shape[0]]

            if self.params.magnitude_structure.shape[0] > 1:
                block_color.loc["values"] = list(self.params.magnitude_structure)
                block_type = "magnitude"
            elif self.params.probability_structure.shape[0] > 1:
                block_color.loc["values"] = list(self.params.probability_structure)
                block_type = "probability"

            for i in range(blocks.shape[0]-1):
                plt.axvspan(blocks[i], #starts at this index
                            blocks[i+1], #ends at this index
                            alpha=0.3,
                            # im sorry the color selection code is so gross and incoherent
                            # it just takes the color from color_block where "values" = the current reward_magnitude vector
                            color=block_color.loc["color",block_color.loc["values"].apply(
                                                      lambda x: np.array_equal(x, self.params.magnitude_structure[self.params.blocks[blocks[i]]]))].values[0],
                            label=f"reward {block_type} left: {self.params.magnitude_structure[self.params.blocks[blocks[i]]][0] if "magnitude" in self.params.block_type else self.params.probability_structure[self.params.blocks[blocks[i]]][0]}\n"
                                  f"reward {block_type} right: {self.params.magnitude_structure[self.params.blocks[blocks[i]]][1] if "magnitude" in self.params.block_type else self.params.probability_structure[self.params.blocks[blocks[i]]][1]}" if i < self.params.magnitude_structure.shape[0] else None
                )

            plt.title('RL model with reward bias blocks')

        else: plt.title('Basic RL model')
        plt.xlim(start,stop)
        plt.setp(g, yticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.legend(fontsize=8, loc='lower left')
        plt.show()

    def plot_previous_choice(self):
        """
        plots the spychometric (stimulus vs choice average) seperately for each previous choice
        """
        data = pd.DataFrame(columns=["previous_left", "previous_right"],
                            index=[i for i in np.unique(self.stimuli)])
        previous_choices = np.concatenate(([0],self.choices[:-1]))
        for stimulus in np.unique(self.stimuli):
            previous_left_idx = np.logical_and(previous_choices == 0, self.stimuli == stimulus).nonzero()[0]
            data.loc[stimulus, "previous_left"] = np.mean(self.choices[previous_left_idx])
            previous_right_idx = np.logical_and(previous_choices == 1, self.stimuli == stimulus).nonzero()[0]
            data.loc[stimulus, "previous_right"] = np.mean(self.choices[previous_right_idx])

        g = sns.lineplot(data=data, dashes=False, markers=True, palette="Set1")
        g.set_xlabel("Current Stimulus")
        g.set_ylabel("Average Choice")

    def plot_prediction_error(self):
        """
        plots the reward prediction error against the stimulus
        """
        data = pd.DataFrame(columns = ["mean", "sd"],index = [i for i in np.unique(self.stimuli)])
        for stimulus in np.unique(self.stimuli):
            data.loc[stimulus, "mean"] = np.mean(self.prediction_error[self.stimuli == stimulus])
            data.loc[stimulus, "sd"] = np.std(self.prediction_error[self.stimuli == stimulus])

        sns.lineplot(data=data["mean"], marker='o')
        # Add error bars
        plt.errorbar(data.index, data['mean'], yerr=data['sd'], fmt='none', capsize=5)
        plt.xlabel('Stimulus')
        plt.ylabel('Reward Prediction Error')

    def get_psychometric(self):
        """
        calculates the psychometrics of the model -> see plot_psychometric() description

        explained in detail in figure 1 of Lak et al 2019
        """
        psychometric = pd.DataFrame(index=["current"]+list(np.unique(self.stimuli)),
                                    columns = np.unique(self.stimuli))

        for stimulus in np.unique(self.stimuli):
            psychometric.loc["current", stimulus] = np.mean(self.choices[self.stimuli == stimulus])

        for previous_stimulus in np.unique(self.stimuli):
            previous_stimuli = self.stimuli[np.nonzero(self.stimuli[:-1] == previous_stimulus)[0] + 1]
            previous_choices = self.choices[np.nonzero(self.stimuli[:-1] == previous_stimulus)[0] + 1]

            for stimulus in np.unique(previous_stimuli):
                psychometric.loc[previous_stimulus, stimulus] = np.mean(previous_choices[previous_stimuli == stimulus])

        #updating matrix is the difference of the previous
        updating_matrix = psychometric.loc[np.unique(self.stimuli)] - psychometric.loc["current"]

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
        plots the psychometrics of the model based on Lak et al 2019:
        Reinforcement biases subsequent perceptual decisions when confidence is low, a widespread behavioral phenomenon
        https://elifesciences.org/articles/49834

        explained in detail in figure 1d

        psychometric function (choice average vs stimulus) of all current stimuli
            along with psychometric of a subset where the previous stimulus was a hard choice

        updating percentage is the difference between all current psychometric and previous stimulus psychometrics
            for each stimulus and previous stimulus

        updating function is the updating percentage for each previous stimulus,
            seperated between hard and easy choices based on current stimulus difficulty
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
        axes[2].set_xlabel("Previous Stimulus")
        axes[2].set_ylabel("Updating %")

    def plot_block_psychometric(self):
        data = pd.DataFrame(index = np.unique(self.params.blocks),
                            columns = np.unique(self.stimuli))
        for block in np.unique(self.params.blocks):
            for stimulus in np.unique(self.stimuli):
                data.loc[block,stimulus] = np.mean(self.choices[np.logical_and(self.params.blocks == block,
                                                                               self.stimuli == stimulus)])
            sns.lineplot(data = data.loc[block],
                         dashes=False,
                         markers=True,
                         label = f"Reward magnitudes: {self.params.magnitude_structure[block]} \n"
                                     f"Reward probabilities: {self.params.probability_structure[block]}" if len(self.params.block_type) > 1
                                     else f"Reward {self.params.block_type}: "
                                     f"{self.params.magnitude_structure[block] if "magnitude" in self.params.block_type 
                                        else self.params.probability_structure[block]}")
        plt.title("Psychometrics seperated by block")
        plt.xlabel("Current Stimulus")
        plt.ylabel("Average Choice")
        plt.legend(fontsize=8, loc='lower right')
        plt.show()


    def plot_previous_bias(self,
                           metric = "correct",
                           plot = "metric"):
        '''
        plot are based on previous lean or previous rewarded block
        psychometric
        :return:
        '''
        # get lean and rich blocks
        # 0 -> lean, 1 -> rich, 2 -> unbiased
        if "magnitude" in self.params.block_type:
            bias = np.argmax(self.params.reward_magnitude, axis = 1)
            # 0 -> lean (biased against), 1 -> rich (biased for)
            bias = (bias == self.choices.astype(int)).astype(int)
            # set unbiased trials to 2
            bias[self.params.reward_magnitude[:, 0] == self.params.reward_magnitude[:, 1]] = 2
        elif "probability" in self.params.block_type:
            bias = np.argmax(self.params.reward_probability, axis=1)
            bias = (bias == self.choices.astype(int)).astype(int)
            # set unbiased trials to 2
            bias[self.params.reward_probability[:, 0] == self.params.reward_probability[:, 1]] = 2

        previous_bias = np.concatenate(([2],bias[:-1]))

        # get previous metric
        if metric == "correct":
            # 1 -> previous choice matches stimulus -> correct, 0 -> incorrect
            previous_reward = (np.sign(np.concatenate(([1],self.stimuli[:-1]))) == np.concatenate(([0], self.choices[:-1]))).astype(int)
            # todo get this to work
        elif metric == "reward":
            # 1 -> previous choice was rewarded, 0 -> previous choice was not rewarded
            previous_reward = np.sign(np.concatenate(([0], self.rewards[:-1])))

        # define dataframe for psychometrics to be stored in
        data = pd.DataFrame(index = ["correct_rich","correct_lean","correct_unbiased",
                                     "incorrect_rich","incorrect_lean","incorrect_unbiased"],
                            columns=np.unique(self.stimuli))
        for stimulus in np.unique(self.stimuli):
            # previous correct/rewarded and rich
            data.loc["correct_lean", stimulus] = np.mean(self.choices[np.logical_and(np.logical_and(previous_reward == 1,
                                                                                                    previous_bias == 0),
                                                                                                    self.stimuli==stimulus)])
            # previous correct/rewarded and lean
            data.loc["correct_rich", stimulus] = np.mean(self.choices[np.logical_and(np.logical_and(previous_reward == 1,
                                                                                                    previous_bias == 1),
                                                                                                    self.stimuli==stimulus)])
            # previous correct/rewarded and unbiased
            data.loc["correct_unbiased", stimulus] = np.mean(self.choices[np.logical_and(np.logical_and(previous_reward == 1,
                                                                                                        previous_bias == 2),
                                                                                                        self.stimuli==stimulus)])
            # previous incorrect/unrewarded and rich
            data.loc["incorrect_lean", stimulus] = np.mean(self.choices[np.logical_and(np.logical_and(previous_reward == 0,
                                                                                                      previous_bias == 0),
                                                                                                      self.stimuli==stimulus)])
            # previous incorrect/unrewarded and rich
            data.loc["incorrect_rich", stimulus] = np.mean(self.choices[np.logical_and(np.logical_and(previous_reward == 0,
                                                                                                      previous_bias == 1),
                                                                                                      self.stimuli==stimulus)])
            # previous incorrect/unrewarded and rich
            data.loc["incorrect_unbiased", stimulus] = np.mean(self.choices[np.logical_and(np.logical_and(previous_reward == 0,
                                                                                                          previous_bias == 2),
                                                                                                          self.stimuli==stimulus)])
        # use the right metric in the index
        if metric == "reward":
            data.index = ["rewarded_lean", "rewarded_rich", "rewarded_unbiased",
                          "unrewarded_lean", "unrewarded_rich", "unrewarded_unbiased"]


        # calculate the difference between the biases for each metric
        data.loc["previous_lean"] = data.iloc[0] - data.iloc[3]
        data.loc["previous_rich"] = data.iloc[1] - data.iloc[4]
        data.loc["previous_unbiased"] = data.iloc[2] - data.iloc[5]



        # plot the
        if "bias" in plot:
            fig, axes = plt.subplots(nrows=1, ncols=2)
            sns.lineplot(data=data.iloc[0:3].T, dashes=False, markers=True,ax=axes[0])
            axes[0].set_title(f"Psychometric for previously {data.index[0].rstrip("_lean")} choices")
            axes[0].set_xlabel("Current Stimulus")
            axes[0].set_ylabel("Average Choice")
            sns.lineplot(data=data.iloc[3:6].T, dashes=False, markers=True, ax=axes[1])
            axes[1].set_title(f"Psychometric for previously {data.index[3].rstrip("_lean")} choices")
            axes[1].set_xlabel("Current Stimulus")
            axes[1].set_ylabel("Average Choice")
            plt.show()

        #
        if "metric" in plot:
            fig, axes = plt.subplots(nrows=1,ncols=4)
            sns.lineplot(data=data.iloc[[0,3]].T, dashes=False, markers=True, ax=axes[0])
            axes[0].set_title(f"Psychometric for previously lean choices")
            axes[0].set_xlabel("Current Stimulus")
            axes[0].set_ylabel("Average Choice")
            sns.lineplot(data=data.iloc[[1,4]].T, dashes=False, markers=True, ax=axes[1])
            axes[1].set_title(f"Psychometric for previously rich choices")
            axes[1].set_xlabel("Current Stimulus")
            axes[1].set_ylabel("Average Choice")
            sns.lineplot(data=data.iloc[[2,5]].T, dashes=False, markers=True, ax=axes[2])
            axes[2].set_title(f"Psychometric for previously unbiased choices")
            axes[2].set_xlabel("Current Stimulus")
            axes[2].set_ylabel("Average Choice")
            sns.lineplot(data=data.iloc[6:9].T, dashes=False, markers=True, ax=axes[3])
            axes[3].set_title("difference rewarded vs unrewarded")
            axes[3].set_xlabel("Current Stimulus")
            axes[3].set_ylabel("Difference")

        return data

    def plot_bias_prediction_error(self):
        # get lean and rich blocks
        # 0 -> lean, 1 -> rich, 2 -> unbiased
        if "magnitude" in self.params.block_type:
            bias = np.argmax(self.params.reward_magnitude, axis=1)
            # 0 -> lean (biased against), 1 -> rich (biased for)
            bias = (bias == self.choices.astype(int)).astype(int)
            # set unbiased trials to 2
            bias[self.params.reward_magnitude[:, 0] == self.params.reward_magnitude[:, 1]] = 2
        elif "probability" in self.params.block_type:
            bias = np.argmax(self.params.reward_probability, axis=1)
            bias = (bias == self.choices.astype(int)).astype(int)
            # set unbiased trials to 2
            bias[self.params.reward_probability[:, 0] == self.params.reward_probability[:, 1]] = 2

        previous_bias = np.concatenate(([2], bias[:-1]))

        data = pd.DataFrame(columns = np.unique(previous_bias),
                            index = [i for i in np.unique(self.stimuli)])

        for bias in np.unique(previous_bias):
            for stimulus in np.unique(self.stimuli):
                data.loc[stimulus,bias] = np.mean(self.prediction_error[np.logical_and(self.stimuli == stimulus, previous_bias == bias)])

        data.columns = ["lean","rich","unbiased"]

        sns.lineplot(data=data, dashes = False)

        plt.xlabel('Stimulus')
        plt.ylabel('Average Reward Prediction Error')

#todo plot average reward based on confidence


def plot_params(params_a = [0.2,0.5,0.7],
                params_b = [0.2,0.5,0.7],
                n = 50000,
                params_ab = None
                ):

    if params_ab is None:
        params_ab = pd.DataFrame(index = params_a,columns=params_b)

        for param_a in params_a:
            for param_b in params_b:
                #change the parameters in this line to make sure that your parameters are the corect ones in th
                model = Model(Params(alpha=param_a, delta=param_b,time_steps=n))
                model.run_model()
                #change what exactly you want to plot here
                _,matrix,function = model.get_psychometric()
                params_ab.loc[param_a,param_b] = matrix.values
                ticks = matrix.index

    fig, axes = plt.subplots(nrows=len(params_a),
                             ncols=len(params_b),
                             figsize=(10, 10))

    max_abs = np.max(np.abs(np.concatenate(params_ab.values.flatten())))

    for (i, j), ax in np.ndenumerate(axes):
        im = ax.imshow(params_ab.iloc[i,j].astype(np.float64).T,
                       cmap='RdBu', norm = matplotlib.colors.Normalize(vmin=-max_abs, vmax=max_abs),
                       interpolation='nearest', aspect='auto')

        ax.set_title(f'(alpha = {params_ab.index[i]}, sigma = {params_ab.columns[j]})')
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

    fig.colorbar(im, ax=axes.ravel().tolist()).set_label("updating %")
    plt.show()

    return params_ab


# self = Model(Params(time_steps= 500000))
# self.params.get_blocks([(1,1),(1,0.5),(0.5,1)])
# self.run_model()


