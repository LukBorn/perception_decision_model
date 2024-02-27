import scripts as sc
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import matplotlib.pyplot as plt

class Params():
    def __init__(self,
                 init_values = 5,
                 epsilon = 0.1,
                 time_steps = 1000,
                 k = 5,
                 alpha = "1/n",
                 ):
        self.init_values = init_values  # inital values for value function
        self.epsilon = epsilon          # probablity to make an exploratory choice
        self.time_steps = time_steps    #
        self.k = k
        self.alpha = alpha

class Results():
    def __init__(self,
                 params = Params()):
        self.params = params
        self.choice = np.empty(params.time_steps)
        self.reward = np.empty(params.time_steps)
        self.value = np.empty(params.time_steps)



def k_arm_bandit(params = Params()):
    states = pd.DataFrame(index=["value", "n", "true_value"],
                          data=np.array([np.full(params.k,params.init_values),
                                         np.zeros(params.k),
                                         np.array([1,1.05,1.5,0.7,0.4,0.9,1.2,1.7,0.5,1,1.4,1.6])[0:params.k]]))

    results = Results(params)

    for i in range(params.time_steps):
        #exploratory or exploitative?
        if np.random.rand(1) > params.epsilon:
            #greedy choice
            choice = states.idxmax(axis = "columns")["value"]
        else:
            #exploratory
            choice = np.random.choice(states.columns)

        #update choice n
        results.choice[i] = choice
        states.loc["n",choice] += 1

        #reward
        reward = np.random.normal(states.loc["true_value",choice],0.2,1)[0]
        results.reward[i] = reward

        #update value
        states.loc["value", choice] += (sc.alpha(states.loc["n", choice],"1/n")*
                                        (reward - states.loc["value", choice]))

        results.value[i] = states.loc["value", choice]

    return results

epsilons = [0.025, 0.05, 0.1, 0.2]
colors = ["black","purple","green","red","orange","yellow"]

# params1 = Params(epsilon=0.025)
# results1 = k_arm_bandit(params1)
# g = sns.lineplot(data=results1.reward_smooth,
#                      alpha=0.6,
#                      color = "blue",
#                      label= f"epsilon = 0.025")

for i in range(len(epsilons)):
    params = Params(epsilon = epsilons[i], init_values=1, k = 9)
    results = k_arm_bandit(params)
    results.smooth_reward()
    g = sns.lineplot(data=results.reward_smooth[:200],
                     alpha=0.6,
                     color = colors[i],
                     label= f"epsilon = {epsilons[i]}")
plt.title('Basic epsilon-greedy k-armed bandit \n rewards (moving average)')
plt.setp(g)
    # , yticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
plt.legend(fontsize=8, loc='lower right')
plt.show()


