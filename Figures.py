from percept_decision_model import Params, Model
import matplotlib.colors
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import scripts as sc
from tqdm import tqdm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statannot import add_stat_annotation



def Figure_1(self,
             previous_psychometric = -0.111
             ):
    fig, axes = plt.subplots(nrows=2,ncols=3, figsize=(13,8))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.35, hspace=0.25)
    self.get_psychometric()

    # Fig 1A
    data = self.psychometric.loc["current"].T
    data.index.astype(str)
    sns.lineplot(data=data, dashes=False, marker="o", palette="Set1", ax=axes[0,0])
    axes[0,0].set_xlabel("Current Stimulus")
    axes[0,0].set_ylabel("Average Choice")
    axes[0, 0].text(-0.2,  0.94, 'A', fontsize=15, transform=axes[0, 0].transAxes)

    #Fig 1B
    data = pd.DataFrame(columns=["previous_left", "previous_right", "total"],
                        index=[i for i in np.unique(self.stimuli)])
    previous_choices = np.concatenate(([0], self.choices[:-1]))
    previous_rewards = np.concatenate(([0], self.rewards[:-1]))
    for stimulus in np.unique(self.stimuli):
        data.loc[stimulus, "previous_left"] = self.choices[
            (previous_choices == 0) & (self.stimuli == stimulus) & (previous_rewards == 1)].mean()
        data.loc[stimulus, "previous_right"] = self.choices[
            (previous_choices == 1) & (self.stimuli == stimulus) & (previous_rewards == 1)].mean()
        data.loc[stimulus, "total"] = self.choices[(self.stimuli == stimulus) & (previous_rewards == 1)].mean()

    sns.lineplot(data=data, dashes=False, markers=True, palette="Set1", ax=axes[0,1])
    axes[0, 1].set_xlabel("Current Stimulus")
    axes[0, 1].set_ylabel("Average Choice")
    #axes[0, 1].set_title("Previous Rewarded")
    axes[0, 1].text(-0.2, 0.94, 'B', fontsize=15, transform=axes[0, 1].transAxes)

    data = pd.DataFrame(columns=["previous_left", "previous_right", "total"],
                        index=[i for i in np.unique(self.stimuli)])
    previous_choices = np.concatenate(([0], self.choices[:-1]))
    previous_rewards = np.concatenate(([0], self.rewards[:-1]))
    for stimulus in np.unique(self.stimuli):
        data.loc[stimulus, "previous_left"] = self.choices[
            (previous_choices == 0) & (self.stimuli == stimulus) & (previous_rewards == 0)].mean()
        data.loc[stimulus, "previous_right"] = self.choices[
            (previous_choices == 1) & (self.stimuli == stimulus) & (previous_rewards == 0)].mean()
        data.loc[stimulus, "total"] = self.choices[(self.stimuli == stimulus) & (previous_rewards == 1)].mean()

    sns.lineplot(data=data, dashes=False, markers=True, palette="Set1", ax=axes[0, 2])
    axes[0, 2].set_xlabel("Current Stimulus")
    axes[0, 2].set_ylabel("Average Choice")
    #axes[0, 2].set_title("Previous Unrewarded")


    #Figure 1 C
    data = self.psychometric.loc[["current", previous_psychometric]].T
    data.index.astype(str)
    sns.lineplot(data=data, dashes=False, markers=True, palette="Set1", ax=axes[1,0])
    axes[1,0].set_xlabel("Current/Previous Stimulus")
    axes[1,0].set_ylabel("Average Choice")
    axes[1,0].text(-0.2, 0.94, 'C', fontsize=15, transform=axes[1,0].transAxes)

    # plot updating matrix
    img = axes[1,1].imshow(self.updating_matrix.values.astype(np.float64).T * 100,
                         cmap='RdBu', interpolation='nearest', aspect='auto')
    plt.colorbar(img, ax=axes[1,1], fraction=0.05, pad=0.04)  # Add color bar
    axes[1,1].set_xlabel("Current Stimulus")
    axes[1,1].set_ylabel("Previous Stimulus")
    axes[1,1].set_xticks(np.arange(len(self.updating_matrix.index)))
    axes[1,1].set_yticks(np.arange(len(self.updating_matrix.index)))
    axes[1,1].set_xticklabels(self.updating_matrix.index, rotation=90)
    axes[1,1].set_yticklabels(self.updating_matrix.index)

    # plot updating function
    sns.lineplot(data=(self.updating_function * 100), palette="Set1", dashes=False, markers=True, ax=axes[1,2])
    axes[1,2].set_xlabel("Previous Stimulus")
    axes[1,2].set_ylabel("Updating %")


def Figure_2(self):
    fig = plt.figure(layout="constrained", figsize=(11,10))

    gs = matplotlib.gridspec.GridSpec(10, 9, figure=fig)
    A = fig.add_subplot(gs[0:3, :])
    B1 = fig.add_subplot(gs[3:5, :-3])
    B2 = fig.add_subplot(gs[5:7, :-3])
    B3 = fig.add_subplot(gs[7:9, :-3])
    C1 = fig.add_subplot(gs[3:6, -3:])
    #C2 = fig.add_subplot(gs[6:9, -3:])

    #Figure 2A
    start = 0
    stop = 2000
    A.set_xlim(start, stop)
    sns.lineplot(data=self.values[:, 0],alpha=0.6, color="black", label="Values left", ax = A)
    sns.lineplot(data=self.values[:, 1],alpha=0.6, color="purple", label="Values right", ax=A)

    if np.unique(self.params.magnitude_structure).shape[0] > 1 or np.unique(self.params.probability_structure).shape[
        0] > 1:
        # find indexes where the blocks change, including start and stop index
        blocks = np.concatenate((np.arange(0, self.params.time_steps, self.params.block_size), np.array([start, stop])))
        blocks = np.unique(blocks[np.logical_and(blocks >= start, blocks <= stop)])

        # define the colors for the blocks
        block_color = pd.DataFrame(index=["values", "color"],
                                   columns=range(self.params.magnitude_structure.shape[0]))
        block_colors = ["gainsboro", "orange", "gold", "coral", "firebrick", "sienna"]
        block_color.loc["color"] = block_colors[:self.params.magnitude_structure.shape[0]]

        if self.params.magnitude_structure.shape[0] > 1:
            block_color.loc["values"] = list(self.params.magnitude_structure)
            block_type = "magnitude"
        elif self.params.probability_structure.shape[0] > 1:
            block_color.loc["values"] = list(self.params.probability_structure)
            block_type = "probability"

        for i in range(blocks.shape[0] - 1):
            A.axvspan(blocks[i],  # starts at this index
                        blocks[i + 1],  # ends at this index
                        alpha=0.3,
                        # im sorry the color selection code is so gross and incoherent
                        # it just takes the color from color_block where "values" = the current reward_magnitude vector
                        color=block_color.loc["color", block_color.loc["values"].apply(
                            lambda x: np.array_equal(x, self.params.magnitude_structure[
                                self.params.blocks[blocks[i]]]))].values[0],
                        label=f"Reward left: {self.params.magnitude_structure[self.params.blocks[blocks[i]]][0] if "magnitude" in self.params.block_type else self.params.probability_structure[self.params.blocks[blocks[i]]][0]}\n"
                              f"Reward right: {self.params.magnitude_structure[self.params.blocks[blocks[i]]][1] if "magnitude" in self.params.block_type else self.params.probability_structure[self.params.blocks[blocks[i]]][1]}" if i <
                                                                                                                                                                                                                                                     self.params.magnitude_structure.shape[
                                                                                                                                                                                                                                                         0] else None
                        )
    A.legend(fontsize=8, loc='lower left')
    A.text(-0.05, 0.94, 'A', fontsize=15, transform=A.transAxes)



    #Fig 2B
    before = 100
    after = 100
    variables = ["Values left","Values right"]
    plot = []
    plot.append(self.values[:, 0])
    plot.append(self.values[:, 1])
    line_colors = ["black", "purple"]

    all_transitions = np.arange(start=0, stop=self.params.time_steps, step=self.params.block_size)
    n_transition_types = self.params.magnitude_structure.shape[0]
    index = np.arange(before + after) - before

    data = pd.DataFrame(index=index,
                        columns=pd.MultiIndex.from_product(iterables=(range(n_transition_types), range(len(plot))),
                                                           names=("transition", "plot")))
    for type in range(n_transition_types):
        match type:
            case 0:
                ax = B1
                colors = ["gainsboro", "orange"]
            case 1:
                ax = B2
                colors = ["orange", "gold"]
            case 2:
                ax = B3
                colors = ["gold", "gainsboro"]

        transitions = all_transitions[type + 1::n_transition_types]
        for j in range(len(plot)):
            variable = plot[j]
            for i in range(before + after):
                data.loc[index[i], (type, j)] = np.mean(variable[transitions + index[
                    i] - 1])  # -1 because the values saved for each time_step are the new ones

            sns.lineplot(data=data[(type, j)], alpha=0.6, color=line_colors[j],
                         ax=ax, label=f"{variables[j]}")

        block_before, block_after = self.params.reward_magnitude[transitions[0] - 1], self.params.reward_magnitude[
            transitions[0]]

        ax.axvspan(-before, 0, alpha=0.3, color=colors[0],
                           label=f"Reward Magnitude left: {block_before[0]} \n Reward Magnitude right: {block_before[1]}")
        ax.axvspan(0, after, alpha=0.3, color=colors[1],
                           label=f"Reward Magnitude left: {block_after[0]} \n Reward Magnitude right: {block_after[1]}")

        ax.set_ylabel("")
        if type == 0:
            ax.text(-0.08, 0.9, 'B', fontsize=15, transform=ax.transAxes)




    #Fig 2C
    data = pd.DataFrame(index=np.unique(self.params.blocks),
                        columns=np.unique(self.stimuli))
    for block in np.unique(self.params.blocks):
        for stimulus in np.unique(self.stimuli):
            data.loc[block, stimulus] = np.mean(self.choices[np.logical_and(self.params.blocks == block,
                                                                            self.stimuli == stimulus)])
        sns.lineplot(data=data.loc[block],
                     dashes=False,
                     marker="o",
                     ax = C1,
                     label=f"Reward left: {self.params.magnitude_structure[block][0]} \n"
                          f"Reward right: {self.params.magnitude_structure[block][1]}"
                          )
    C1.set_xlabel("Current Stimulus")
    C1.set_ylabel("Average Choice")
    C1.legend(fontsize=8, loc='lower right')
    C1.text(-0.2, 0.94, 'C', fontsize=15, transform=C1.transAxes)

    #Fig C2

    # todo ask lily for the mouse data



def Figure_3(steps,
             alpha_list = [0.1,0.5,0.9]):
    param_list = []
    for alpha in alpha_list:
        param = Params(alpha = alpha, time_steps=steps)
        param.get_blocks(magnitude_structure=[(1, 1), (1, 0.5), (0.5, 1)], block_size=400)
        param_list.append(param)

    model0 = Model(param_list[0])
    model0.run_model()
    model1 = Model(param_list[1])
    model1.run_model()
    model2 = Model(param_list[2])
    model2.run_model()

    fig = plt.figure(layout="constrained", figsize=(11, 10))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    gs = matplotlib.gridspec.GridSpec(9, 9, figure=fig)
    A1 = fig.add_subplot(gs[0:2, :-3])
    A2 = fig.add_subplot(gs[2:4, :-3])
    A3 = fig.add_subplot(gs[4:6, :-3])
    B = fig.add_subplot(gs[0:3, -3:])
    C = fig.add_subplot(gs[3:6, -3:])
    D1 = fig.add_subplot(gs[-3:, 0:3])
    D2 = fig.add_subplot(gs[-3:, 3:6])
    D3 = fig.add_subplot(gs[-3:, 6:9])


    #fig 3 a1
    before = 100
    after = 100
    variables = ["Values left", "Values right"]
    line_colors = ["black", "purple"]
    colors = ["orange", "gold"]
    all_transitions = np.arange(start=0, stop=model0.params.time_steps, step=model0.params.block_size)
    index = np.arange(before + after) - before

    plot = []
    plot.append(model0.values[:, 0])
    plot.append(model0.values[:, 1])
    data = pd.DataFrame(index=index, columns=range(len(plot)))
    transitions = all_transitions[2::3]
    for j in range(len(plot)):
        variable = plot[j]
        for i in range(before + after):
            data.loc[index[i], j] = np.mean(variable[transitions + index[i] - 1])

        sns.lineplot(data=data[j], alpha=0.6, color=line_colors[j],
                     ax=A1, label=f"{variables[j]}")

    A1.axvspan(-before, 0, alpha=0.3, color=colors[0])
    A1.axvspan(0, after, alpha=0.3, color=colors[1])
    A1.set_ylabel("")
    A1.text(-0.08, 0.9, 'A', fontsize=15, transform=A1.transAxes)

    plot = []
    plot.append(model1.values[:, 0])
    plot.append(model1.values[:, 1])
    data = pd.DataFrame(index=index, columns=range(len(plot)))
    transitions = all_transitions[2::3]
    for j in range(len(plot)):
        variable = plot[j]
        for i in range(before + after):
            data.loc[index[i], j] = np.mean(variable[transitions + index[i] - 1])

        sns.lineplot(data=data[j], alpha=0.6, color=line_colors[j],
                     ax=A2, label=f"{variables[j]}")

    A2.axvspan(-before, 0, alpha=0.3, color=colors[0])
    A2.axvspan(0, after, alpha=0.3, color=colors[1])
    A2.set_ylabel("")

    plot = []
    plot.append(model2.values[:, 0])
    plot.append(model2.values[:, 1])
    data = pd.DataFrame(index=index, columns=range(len(plot)))
    transitions = all_transitions[2::3]
    for j in range(len(plot)):
        variable = plot[j]
        for i in range(before + after):
            data.loc[index[i],  j] = np.mean(variable[transitions + index[i] - 1])

        sns.lineplot(data=data[j], alpha=0.6, color=line_colors[j],
                     ax=A3, label=f"{variables[j]}")

    A3.axvspan(-before, 0, alpha=0.3, color=colors[0])
    A3.axvspan(0, after, alpha=0.3, color=colors[1])
    A3.set_ylabel("")


    #figure 3B
    data = pd.DataFrame(index=alpha_list)
    data[alpha_list[0]] = (np.std(model0.values[0]) + np.std(model0.values[1]))/2
    data[alpha_list[1]] = (np.std(model1.values[0]) + np.std(model1.values[1]))/2
    data[alpha_list[2]] = (np.std(model2.values[0]) + np.std(model2.values[1]))/2
    data = data.iloc[0]
    data.index = [f"alpha: {alpha}" for alpha in alpha_list]
    sns.barplot(x=data.index, y=data.values, ax = B)

    #figure 3C -> average rewards
    data = pd.DataFrame(index=alpha_list)
    data[alpha_list[0]] = np.mean(model0.rewards)
    data[alpha_list[1]] = np.mean(model1.rewards)
    data[alpha_list[2]] = np.mean(model2.rewards)
    data = data.iloc[0]
    data.index = [f"alpha: {alpha}" for alpha in alpha_list]
    plot = sns.barplot(x=data.index, y=data.values, ax=C)

    # calculate and add significance levels
    #add_stat_annotation(ax = plot,)
    #https://stackoverflow.com/questions/36578458/how-does-one-insert-statistical-annotations-stars-or-p-values?noredirect=1&lq=1









    #figure 3D
    _,matrix,_ = model0.get_psychometric()
    # plot updating matrix
    img = D1.imshow(matrix.values.astype(np.float64).T * 100,
                         cmap='RdBu', interpolation='nearest', aspect='auto')
    plt.colorbar(img, ax=D1, fraction=0.05, pad=0.04)  # Add color bar
    D1.set_xlabel("Current Stimulus")
    D1.set_ylabel("Previous Stimulus")
    D1.set_xticks(np.arange(len(matrix.index)))
    D1.set_yticks(np.arange(len(matrix.index)))
    D1.set_xticklabels(matrix.index, rotation=90)
    D1.set_yticklabels(matrix.index)

    _, matrix, _ = model1.get_psychometric()
    # plot updating matrix
    img = D2.imshow(matrix.values.astype(np.float64).T * 100,
                    cmap='RdBu', interpolation='nearest', aspect='auto')
    plt.colorbar(img, ax=D2, fraction=0.05, pad=0.04)  # Add color bar
    D2.set_xlabel("Current Stimulus")
    D2.set_ylabel("Previous Stimulus")
    D2.set_xticks(np.arange(len(matrix.index)))
    D2.set_yticks(np.arange(len(matrix.index)))
    D2.set_xticklabels(matrix.index, rotation=90)
    D2.set_yticklabels(matrix.index)

    _, matrix, _ = model2.get_psychometric()
    # plot updating matrix
    img = D3.imshow(matrix.values.astype(np.float64).T * 100,
                    cmap='RdBu', interpolation='nearest', aspect='auto')
    plt.colorbar(img, ax=D3, fraction=0.05, pad=0.04)  # Add color bar
    D3.set_xlabel("Current Stimulus")
    D3.set_ylabel("Previous Stimulus")
    D3.set_xticks(np.arange(len(matrix.index)))
    D3.set_yticks(np.arange(len(matrix.index)))
    D3.set_xticklabels(matrix.index, rotation=90)
    D3.set_yticklabels(matrix.index)


