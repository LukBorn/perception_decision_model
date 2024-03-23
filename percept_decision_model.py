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
                 policy=sc.greedy,
                 beta=9.097,
                 epsilon=0.5,
                 TDRL_init = 1,
                 TDRL_steps = 20,
                 TDRL_bins = 0.5,
                 TDRL_omega = -0.008,
                 TDRL_policy=sc.epsilon_greedy,
                 TDRL_epsilon = 0.1,
                 TDRL_beta = None,
                 TDRL_alpha = 0.1,
                 TDRL_gamma = 1,
                 wait_time = 2,
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

    # choose_action: function for choosing action based given array q
        self.policy = policy
    # in case of sc.greedy its just argmax
    # epsilon: kwarg for choose_action = sc.epsilon_greedy
        self.epsilon = epsilon
    # beta: kwarg for choose_action = sc.log_fun -> stolen from lilys code so im not sure what the math behind that is
        self.beta = beta


    # discrete confidence states that are modeled (confidence needs to discrete)
        self.confidences = np.linspace(0.5, 1, 10).round(3)

    # alpha_confidence -> learning rate for the confidence updating
        self.TDRL_alpha = TDRL_alpha

    # TDRL_steps -> the maximum amount of time steps the TDRL model for determining time investment can take
        self.TDRL_steps = TDRL_steps

    # TDRL_bins -> size of the time bin (in seconds)
        self.TDRL_bins = TDRL_bins

    # omega -> punishment factor that the time investment is multiplied by
        self.TDRL_omega = TDRL_omega

    # TDRL_init -> initial values for the value matrix in TDRL
    # should be either an array of shape [value_stay, value_leave]
    # or 2d array in the correct shape of previously learned [value_stay, value_leave]
        self.TDRL_init =TDRL_init

    # TDRL_policy: function for choosing action based given array q
        self.TDRL_policy = TDRL_policy
    # in case of sc.greedy its just argmax
    # epsilon: kwarg for choose_action = sc.epsilon_greedy
        self.TDRL_epsilon = TDRL_epsilon
    # beta: kwarg for choose_action = sc.log_fun -> stolen from lilys code so im not sure what the math behind that is
        self.TDRL_beta = TDRL_beta

    #TDRL_gamma: discount factor for the max(next q) of the TDRL
        self.TDRL_gamma = TDRL_gamma

    # wait_time -> time that agent waits between trials
        self.wait_time = wait_time


    # reward magnitude: array with [reward_magnitude_left, reward_magnitude_right] at each time step i
        self.reward_magnitude = np.ones((time_steps, 2))
    # reward probability: array with [reward_probability_left, reward_probability_right] at each time step i
        self.reward_probability = np.ones((time_steps, 2))
    # blocks -> index of magnitude/probability structure
        self.blocks = np.ones(time_steps).astype(int)

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
            values for probability between 0,1
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

        for i in range(self.time_steps):
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
        self.percepts = np.empty(params.time_steps,dtype=np.float64)
        self.choices = np.empty(params.time_steps)
        self.rewards = np.empty(params.time_steps)
        self.prediction_error = np.empty(params.time_steps)
        self.values = np.empty((params.time_steps,2))
        self.confidence = np.empty(params.time_steps,dtype=np.float64)

        # TDRL_values -> values for each action at each time point and decision confidence
        self.TDRL_values = None
        self.time_investment = np.zeros(params.time_steps)

        self.psychometric = None
        self.updating_matrix = None
        self.updating_function = None

    def run_model(self,
                  params = None,
                  reset_after_blocks = False):

        if params is not None:
            self.params = params

        self.model_type = "standard"
        if reset_after_blocks:
            trial_start = np.arange(0, self.params.time_steps, self.params.block_size*self.params.magnitude_structure.shape[0])

        # set initial values: [value_left, value_right]
        value = np.full(2, self.params.init_values, dtype=np.float64)

        print("Model Running:")
        for i in tqdm(range(self.params.time_steps)):
            # generate stimulus: random sample from one of 10 equally spaced stimuli between -1 and 1
            stimulus = np.random.choice(np.linspace(-1, 1, 10), 1).round(3)[0]

            # internal estimate of the stimulus is normally distributed
            # with constant variance around true stimulus contrast sigma
            percept = np.random.normal(stimulus, self.params.sigma, 1)

            p_percept = 0.5 * (1 + scipy.special.erf(percept / (np.sqrt(2) * self.params.sigma)))
            p_percept = np.array([1 - p_percept, p_percept]).T[0]

            Q = p_percept * value

            # calculate choice based on Q -> left = 0, right = 1
            # greedy choice
            choice = self.params.policy(Q, epsilon = self.params.epsilon, beta=self.params.beta)

            # stimulus < 0 and choice = 0 -> reward = 1
            # stimulus > 0 and choice = 1 -> reward = 0
            if [-1, 1][choice] != np.sign(stimulus):  # incorrect choice
                reward = 0
            else:
                # reward is reward magnitude * 1 or 0 depending on reward probability
                reward = self.params.reward_magnitude[i, choice] * \
                         np.random.binomial(1, self.params.reward_probability[i, choice], 1)[0]

            prediction_error = reward - Q[choice]
            # update the value by adding the product of learning rule alpha and reward prediction error
            value[choice] += self.params.alpha * prediction_error

            # add your results to your Results object
            self.stimuli[i] = stimulus
            self.choices[i] = choice
            self.rewards[i] = reward
            self.prediction_error[i] = prediction_error
            self.values[i] = value

            if (reset_after_blocks):
                if (i in trial_start):
                    value = np.full(2, self.params.init_values, dtype=np.float64)


    def run_investment_model_old_old(self,
                                     params = None):
        """
        runs the model, saving all results into itself
        simulate time investment as a "function" of confidence
        just weights for each confidence level that get adjusted after every decision

        get stimulus
        add perceptual uncertainty
        calculate p_left and p_right
        multiply p_left with value_left, same for right
        choose left or right based on values
        calculate time investment based on decision confidence based on percept
        generate reward_time
        recieve reward based on choice and time investment
        update values based on reward prediction error
        """
        #generate params class if none is specified
        if params is not None:
            self.params = params

        self.model_type = "time_investment_old_old"

        # set initial values: [value_left, value_right]
        value = np.full(2, self.params.init_values, dtype=np.float64)

        # set the investment values for each confidence
        investment_function = self.params.TDRL_init

        self.time = np.zeros(self.params.time_steps)

        print("Model Running:")
        for i in tqdm(range(self.params.time_steps)):
            # generate stimulus: random sample from one of 10 equally spaced stimuli between -1 and 1
            stimulus = np.random.choice(np.linspace(-1, 1, 10), 1).round(3)[0]

            # generate internal estimate of the stimulus
            percept = np.random.normal(stimulus, self.params.sigma, 1)[0]

            # calculate decision confidence
            confidence = 0.5 * np.abs(percept) + 0.5
            # discretize by setting it to the closest one in our confidence statespace
            confidence = self.params.confidences[np.abs(self.params.confidences - confidence).argmin()]

            #calculate time investment
            time_investment = investment_function[np.nonzero(self.params.confidences == 1)[0][0]]

            # calculate value-adjusted percept probability Q
            p_percept = 0.5 * (1 + scipy.special.erf(percept / (np.sqrt(2) * self.params.sigma)))
            p_percept = np.array([1 - p_percept, p_percept]).T[0]
            Q = p_percept * value

            # calculate choice based on Q -> left = 0, right = 1
            choice = self.params.policy(Q, epsilon = self.params.epsilon, beta=self.params.beta)

            # generate reward time
            reward_time = np.random.randint(5,10)/10

            # calculate reward
            if [-1, 1][choice] != np.sign(stimulus): # incorrect choice
                reward = self.params.omega * time_investment
            elif time_investment < reward_time: # early withdrawal
                reward = self.params.omega * time_investment
            else:
                reward = self.params.reward_magnitude[i, choice] * \
                         np.random.binomial(1, self.params.reward_probability[i, choice], 1)[0] + self.params.omega * time_investment

            # update the value by adding the product of learning rule alpha and reward prediction error
            prediction_error = reward - Q[choice]
            value[choice] += self.params.alpha * prediction_error

            # calculate the time_point of the trial
            self.time[i] = self.time[i - 1] + time_investment + self.params.wait_time

            # update confidence value
            # its this learning rule thats fucked up and idk how to fix it
            time_investment += self.params.TDRL_alpha * prediction_error
            investment_function[np.nonzero(self.params.confidences == 1)[0][0]] = time_investment


            self.stimuli[i] = stimulus
            self.percepts[i] = percept
            self.confidence[i] = confidence
            self.choices[i] = choice
            self.rewards[i] = reward
            self.prediction_error[i] = prediction_error
            self.values[i] = value

    def run_investment_model_old(self,
                                 params = None):
        """
        runs the model, saving all results into itself
        model all time points and confidences as one big state space,
        with values for the actions stay or leave saved in a multiindex

        get stimulus
        add perceptual uncertainty
        calculate p_left and p_right
        multiply p_left with value_left, same for right
        choose left or right based on values
        generate reward_time
        Temporal Difference Learning of time investment:
            run through your state space with stay/leave options
            updating values based on expected reward
        recieve reward based on choice and time investment
        update values based on reward prediction error
        """

        # set params if different ones specified
        if params is not None:
            self.params = params

        self.model_type = "time_investment_TDRL_old"

        print(f"Total reward for non-rewarded, non-terminated trial is {self.params.TDRL_steps*self.params.TDRL_omega}")


        # set initial values: [value_left, value_right]
        value = np.full(2, self.params.init_values, dtype=np.float64)

        if self.TDRL_values is None:
            TDRL_values = pd.DataFrame(index = self.params.confidences,
                                       columns = pd.MultiIndex.from_product((np.arange(self.params.TDRL_steps),[0,1]),
                                                                            names = ["n", "action"]),
                                       data = self.params.TDRL_init,
                                       dtype = np.float64
                                       )
        else:
            TDRL_values = self.TDRL_values

        self.time = np.zeros(self.params.time_steps)

        print("Model Running:")
        for i in tqdm(range(self.params.time_steps)):
            # generate stimulus: random sample from one of 10 equally spaced stimuli between -1 and 1
            stimulus = np.random.choice(np.linspace(-1, 1, 10), 1).round(3)[0]

            # generate internal estimate of the stimulus
            percept = np.random.normal(stimulus, self.params.sigma, 1)[0]

            # calculate decision confidence
            confidence = 0.5 * np.abs(percept) + 0.5
            # discretize by setting it to the closest one in our confidence statespace
            confidence = self.params.confidences[np.abs(self.params.confidences - confidence).argmin()]

            # calculate value-adjusted percept probability Q
            p_percept = 0.5 * (1 + scipy.special.erf(percept / (np.sqrt(2) * self.params.sigma)))
            p_percept = np.array([p_percept, 1-p_percept]).T
            Q = p_percept * value

            # calculate choice based on Q -> left = 0, right = 1
            choice = self.params.policy(Q, epsilon = self.params.epsilon, beta=self.params.beta)

            # generate reward_time
            if [-1, 1][choice] != np.sign(stimulus):  # incorrect choice
                reward_time = self.params.TDRL_steps+10
            elif np.random.binomial(1, self.params.reward_probability[i, choice], 1)[0] == 0: #clutch trial
                reward_time = self.params.TDRL_steps + 10
            else:
                reward_time = sc.trunc_expon()
                #todo add a way for the model to learn by giving the reward progressively later
                # either by passing a function in here, or making a seperate function,
                # but then the parameter for randint would have to be different bc the statespae has to stay the same

            for time_step in range(self.params.TDRL_steps):
                # states tuples (confidence, time_step) that can be used to index into
                # investment_values is a dataframe with confidence as index and n as columns,
                # each value is a 1d array of shape [value_stay, value_leave]

                # choose the action
                action = self.params.TDRL_policy(TDRL_values.loc[confidence, time_step], epsilon = self.params.TDRL_epsilon)
                q = TDRL_values.loc[confidence, (time_step, action)]

                # calculate the reward for this action, and the next step
                if action == 1: #leave
                    reward = self.params.TDRL_omega * time_step
                    max_next_q = 0
                elif time_step == self.params.TDRL_steps - 1 : #trial time limit has been reached
                    reward = self.params.TDRL_omega * time_step
                    max_next_q = 0
                elif time_step == reward_time: # reward time babyyyy
                    reward =  self.params.TDRL_omega * time_step + self.params.reward_magnitude[i, choice]
                    max_next_q = 0
                else: # stay
                    reward = self.params.TDRL_omega * time_step
                    max_next_q = TDRL_values.loc[confidence, time_step + 1].max()

                TD_error = reward + self.params.TDRL_gamma * max_next_q - q
                TDRL_values.loc[confidence, (time_step, action)] = q + self.params.TDRL_alpha * TD_error

                if max_next_q == 0:
                    break


            # update the value by adding the product of learning rule alpha and reward prediction error
            prediction_error = reward - Q[choice]
            value[choice] += self.params.alpha * prediction_error

            self.time[i] = time_step
            self.stimuli[i] = stimulus
            self.percepts[i] = percept
            self.confidence[i] = confidence
            self.choices[i] = choice
            self.rewards[i] = reward
            self.prediction_error[i] = prediction_error
            self.values[i] = value

        self.TDRL_values = TDRL_values

    def run_investment_model(self,
                             params = None):
        """
        change to old model:
        only model time -> multiply confidence into the value function deciding stay or leave

        """
        # set params if different ones specified
        if params is not None:
            self.params = params

        self.model_type = "time_investment_TDRL"

        print(f"Total reward for non-rewarded, non-terminated trial is {self.params.TDRL_steps * self.params.TDRL_omega}")

        # set initial values: [value_left, value_right]
        value = np.full(2, self.params.init_values, dtype=np.float64)

        if self.TDRL_values is None:
            TDRL_values = pd.DataFrame(index=["stay","leave"],
                                       columns=np.arange(self.params.TDRL_steps),
                                       data=self.params.TDRL_init,
                                       dtype=np.float64
                                       )
        else:
            TDRL_values = self.TDRL_values

        self.time = np.zeros(self.params.time_steps)

        print("Model Running:")
        for i in tqdm(range(self.params.time_steps)):
            # generate stimulus: random sample from one of 10 equally spaced stimuli between -1 and 1
            stimulus = np.random.choice(np.linspace(-1, 1, 10), 1).round(3)[0]

            # generate internal estimate of the stimulus
            percept = np.random.normal(stimulus, self.params.sigma, 1)[0]

            # calculate decision confidence
            confidence = 0.5 * np.abs(percept) + 0.5
            # discretize by setting it to the closest one in our confidences
            confidence = self.params.confidences[np.abs(self.params.confidences - confidence).argmin()]

            # calculate value-adjusted percept probability Q
            p_percept = 0.5 * (1 + scipy.special.erf(percept / (np.sqrt(2) * self.params.sigma)))
            p_percept = np.array([p_percept, 1 - p_percept]).T
            Q = p_percept * value

            # calculate choice based on Q -> left = 0, right = 1
            choice = self.params.policy(Q, epsilon=self.params.epsilon, beta=self.params.beta)

            # generate reward_time
            if [-1, 1][choice] != np.sign(stimulus):  # incorrect choice
                reward_time = self.params.TDRL_steps + 10
            elif np.random.binomial(1, self.params.reward_probability[i, choice], 1)[0] == 0:  # clutch trial
                reward_time = self.params.TDRL_steps + 10
            else:
                reward_time = sc.trunc_expon() / self.params.TDRL_bins
                # todo add a way for the model to learn by giving the reward progressively later
                # either by passing a function in here, or making a seperate function,
                # but then the parameter for randint would have to be different bc the statespae has to stay the same

            for time_step in TDRL_values.columns.values:
                # choose the action
                action = TDRL_values.index[self.params.TDRL_policy(TDRL_values[time_step],
                                                                   epsilon=self.params.TDRL_epsilon)]
                q = TDRL_values.loc[action, time_step] * confidence

                # calculate the reward for this action, and the next step
                if action == 1:  # leave
                    reward = self.params.TDRL_omega * time_step
                    max_next_q = 0
                elif time_step == self.params.TDRL_steps - 1:  # trial time limit has been reached
                    reward = self.params.TDRL_omega * time_step
                    max_next_q = 0
                elif time_step == reward_time:  # reward time babyyyy
                    reward = self.params.TDRL_omega * time_step + self.params.reward_magnitude[i, choice]
                    max_next_q = 0
                else:  # stay
                    reward = self.params.TDRL_omega * time_step
                    max_next_q = TDRL_values[time_step + 1].max() * confidence

                TD_error = reward + self.params.TDRL_gamma * max_next_q - q
                TDRL_values.loc[action, time_step] = q + self.params.TDRL_alpha * TD_error

                if max_next_q == 0:
                    break

            # update the value by adding the product of learning rule alpha and reward prediction error
            prediction_error = reward - Q[choice]
            value[choice] += self.params.alpha * prediction_error

            self.time[i] = time_step
            self.stimuli[i] = stimulus
            self.percepts[i] = percept
            self.confidence[i] = confidence
            self.choices[i] = choice
            self.rewards[i] = reward
            self.prediction_error[i] = prediction_error
            self.values[i] = value

        self.TDRL_values = TDRL_values

    def plot(self,
             variables=None,
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
        if variables is None:
            variables = ['choices', 'values_left']
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
        #plt.setp(g, yticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.legend(fontsize=8, loc='lower left')
        plt.show()

    def plot_simple_psychometric(self):
        self.get_psychometric()
        data = self.psychometric.loc["current"].T
        data.index.astype(str)
        sns.lineplot(data=data, dashes=False, marker="o", palette="Set1")
        plt.xlabel("Current Stimulus")
        plt.ylabel("Average Choice")

    def plot_previous_choice(self):
        """
        plots the pychometric (stimulus vs choice average) seperately for each previous choice
        """

        data = pd.DataFrame(columns=["previous_left", "previous_right", "total"],
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

    def plot_previous_rewarded(self):
        """
        plots the average psychometric seperately for each previously rewarded choice
        :return:
        """

        data = pd.DataFrame(columns=["previous_left", "previous_right", "total"],
                            index=[i for i in np.unique(self.stimuli)])
        previous_choices = np.concatenate(([0], self.choices[:-1]))
        previous_rewards = np.concatenate(([0], self.rewards[:-1]))
        for stimulus in np.unique(self.stimuli):
            data.loc[stimulus, "previous_left"] = self.choices[(previous_choices == 0) & (self.stimuli == stimulus) & (previous_rewards == 1)].mean()
            data.loc[stimulus, "previous_right"] = self.choices[(previous_choices == 1) & (self.stimuli == stimulus) & (previous_rewards == 1)].mean()
            data.loc[stimulus, "total"] = self.choices[(self.stimuli == stimulus) & (previous_rewards == 1)].mean()

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

    def get_psychometric(self,
                         difficulty_cut = 0.6):
        """
        calculates the psychometrics of the model -> see plot_psychometric() description

        explained in detail in figure 1 of Lak et al 2019
        """
        self.params.difficulty_cut = difficulty_cut

        psychometric = pd.DataFrame(index=["current"]+list(np.unique(self.stimuli)),
                                    columns = np.unique(self.stimuli))

        previous_stimuli = np.concatenate(([0], self.stimuli[:-1]))
        previous_rewards = np.concatenate(([0], self.rewards[:-1]))

        for stimulus in np.unique(self.stimuli):
            psychometric.loc["current", stimulus] = self.choices[(self.stimuli == stimulus)].mean()
            for previous_stimulus in np.unique(self.stimuli):
                psychometric.loc[previous_stimulus, stimulus] = self.choices[(self.stimuli==stimulus)&(previous_stimuli == previous_stimulus)].mean()

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
                          previous_psychometric = None):
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
        self.get_psychometric()

        fig, axes = plt.subplots(nrows=1, ncols=3)

        # plot current psychometric and random previous psychometric
        # get random psychometric to plot
        if previous_psychometric is None:
            # make sure you get a hard choice so the difference is obvious in the plot
            hard_choices = np.unique(self.stimuli)[np.abs(np.unique(np.unique(self.stimuli))) < self.params.difficulty_cut]
            previous_psychometric = np.random.choice(hard_choices)
        # plot psychometric
        data = self.psychometric.loc[["current", previous_psychometric]].T
        data.index.astype(str)
        sns.lineplot(data=data, dashes=False, markers=True, palette="Set1", ax=axes[0])
        axes[0].set_xlabel("Current/Previous Stimulus")
        axes[0].set_ylabel("Average Choice")

        # plot updating matrix
        img = axes[1].imshow(self.updating_matrix.values.astype(np.float64).T * 100,
                             cmap='RdBu', interpolation='nearest', aspect='auto')
        plt.colorbar(img, ax=axes[1], label="Updating %", fraction=0.05, pad=0.04)  # Add color bar
        axes[1].set_xlabel("Current Stimulus")
        axes[1].set_ylabel("Previous Stimulus")
        axes[1].set_xticks(np.arange(len(self.updating_matrix.index)))
        axes[1].set_yticks(np.arange(len(self.updating_matrix.index)))
        axes[1].set_xticklabels(self.updating_matrix.index,rotation=90)
        axes[1].set_yticklabels(self.updating_matrix.index)


        # plot updating function
        sns.lineplot(data=(self.updating_function * 100), palette="Set1", dashes=False, markers=True, ax=axes[2])
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


    def plot_block_transition(self,
                              variables = None,
                              before = 50,
                              after = 50):
        if variables is None:
            variables = ["values_left", "values_right"]
        plot = []
        if 'values_left' in variables:
            plot.append(self.values[:,0])
        if "values_right" in variables:
            plot.append(self.values[:,1])
        if "choice" in variables:
            plot.append(self.choices[:])
        if "reward" in variables:
            plot.append(self.rewards)

        colors = ["black", "purple","blue","red", "gainsboro", "orange"]

        all_transitions = np.arange(start=0,stop=self.params.time_steps,step= self.params.block_size)
        n_transition_types = self.params.magnitude_structure.shape[0]
        index = np.arange(before + after) - before

        fig, axes = plt.subplots(nrows = n_transition_types, ncols=1)
        data = pd.DataFrame(index = index,
                            columns = pd.MultiIndex.from_product(iterables = (range(n_transition_types),range(len(plot))),
                                                                 names = ("transition","plot")))
        for type in range(n_transition_types):
            transitions = all_transitions[type+1::n_transition_types]
            for j in range(len(plot)):
                variable = plot[j]
                for i in range(before+after):
                    data.loc[index[i],(type, j)] = np.mean(variable[transitions+index[i]-1]) #-1 because the values saved for each time_step are the new ones

                sns.lineplot(data=data[(type,j)], alpha=0.6, color=colors[j],
                             ax=axes[type], label = f"{variables[j]}")

            block_before, block_after= self.params.reward_magnitude[transitions[0]-1], self.params.reward_magnitude[transitions[0]]


            axes[type].axvspan(-before, 0, alpha = 0.3, color = colors[-1],
                               label = f"Reward Magnitude left: {block_before[0]} \n Reward Magnitude right: {block_before[1]}")
            axes[type].axvspan(0, after, alpha = 0.3, color = colors[-2],
                               label = f"Reward Magnitude left: {block_after[0]} \n Reward Magnitude right: {block_after[1]}")

        return data

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
            # todo get this to work so far only reward works
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


    def plot_TDRL_values(self):
        fig, axes = plt.subplots(nrows=3, ncols=1)

        # plot the stay values
        im = axes[0].imshow(self.TDRL_values.loc[:,(slice(None),0)])
        axes[0].set_title("stay values")
        axes[0].set_yticks(np.arange(len(self.params.confidences)), self.params.confidences)
        axes[0].set_ylabel("confidence")
        fig.colorbar(im)

        # plot the leave values
        im = axes[1].imshow(self.TDRL_values.loc[:, (slice(None), 1)])
        axes[1].set_title("leave values")
        axes[1].set_yticks(np.arange(len(self.params.confidences)), self.params.confidences)
        axes[1].set_ylabel("confidence")
        fig.colorbar(im)

        # plot the difference between stay and leave values
        im = axes[2].imshow(self.TDRL_values.loc[:,(slice(None),0)].values - self.TDRL_values.loc[:,(slice(None),1)].values)
        axes[2].set_title("difference")
        axes[2].set_xlabel("time steps")
        axes[2].set_yticks(np.arange(len(self.params.confidences)), self.params.confidences)
        axes[2].set_ylabel("confidence")
        fig.colorbar(im)


def plot_investment(self, x = "confidence"):
    # plot average time investment for each confidence level/stimulus
    # calculate the average time investment for each confidence level
    if x =="confidence":
        data = pd.DataFrame(index = self.params.confidences, columns= ["mean", "sd"])
        for confidence in self.params.confidences:
            data.loc[confidence,"mean"] = np.mean(self.time[self.confidence == confidence])
            data.loc[confidence,"sd"] = np.std(self.time[self.confidence == confidence])
    elif x == "stimulus":
        data = pd.DataFrame(index=np.unique(self.stimuli), columns=["mean", "sd"])
        for stimulus in np.unique(self.stimuli):
            data.loc[stimulus, "mean"] = np.mean(self.time[self.confidence == stimulus])
            data.loc[stimulus, "sd"] = np.std(self.time[self.confidence == stimulus])
            # todo this doesnt work for some reason
    else: print("you suck lol")
    sns.lineplot(data=data["mean"], marker='o')
    # Add error bars
    plt.errorbar(data.index, data['mean'], yerr=data['sd'], fmt='none', capsize=5)
    plt.xlabel(x)
    plt.ylabel('Average time investement')

def plot_reward_real_time(self):
    real_time = np.zeros(self.params.time_steps)
    # todo plot reward per time
    ...


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
        # todo add other things you can plot here
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


def simple_model(steps = 100000):
    model = Model(Params(time_steps = steps))
    model.run_model()
    return model

def block_model(steps = 100000):
    model = Model(Params(time_steps=steps))
    model.params.get_blocks(magnitude_structure=[(1, 1), (1, 0.5), (0.5, 1)], block_size=400)
    model.run_model()
    return model

def investment_model(steps = 100000):
    model = Model(Params(time_steps=steps))
    model.run_investment_model()
    return model



