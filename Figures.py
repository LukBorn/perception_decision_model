from percept_decision_model import Params, Model
import matplotlib.colors
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import scripts as sc
import pingouin
from tqdm import tqdm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statannot import add_stat_annotation



def Figure_1(self,
             previous_psychometric = -0.111
             ):
    fig, axes = plt.subplots(nrows=2,ncols=3, figsize=(13,9))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.35, hspace=0.25)

    # Fig 1A
    data = self.psychometric.loc["current"].T
    data.index.astype(str)
    sns.lineplot(data=data, dashes=False, marker="o", palette="Set1", ax=axes[0,0])
    axes[0,0].set_xlabel("Current Stimulus")
    axes[0,0].set_ylabel("Average Choice")
    axes[0, 0].text(-0.2,  0.94, 'A', fontsize=15, transform=axes[0, 0].transAxes)

    #Figure 1 B
    self.get_psychometric(subset = "total")
    data = self.psychometric.loc[["current", previous_psychometric]].T
    data.index.astype(str)
    sns.lineplot(data=data, dashes=False, markers=True, palette="Set1", ax=axes[1,0])
    axes[1,0].set_xlabel("Current/Previous Stimulus")
    axes[1,0].set_ylabel("Average Choice")
    axes[1,0].text(-0.2, 0.94, 'B', fontsize=15, transform=axes[1,0].transAxes)

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
    axes[1,1].set_xlabel("Previous Stimulus")
    axes[1,1].set_ylabel("Updating %")

    # Fig 1C
    data = pd.DataFrame(columns=["Previous Left", "Previous Right", "Total"],
                        index=[i for i in np.unique(self.stimuli)])
    previous_choices = np.concatenate(([0], self.choices[:-1]))
    previous_rewards = np.concatenate(([0], self.rewards[:-1]))
    for stimulus in np.unique(self.stimuli):
        data.loc[stimulus, "Previous Left"] = self.choices[
            (previous_choices == 0) & (self.stimuli == stimulus) & (previous_rewards == 1)].mean()
        data.loc[stimulus, "Previous Right"] = self.choices[
            (previous_choices == 1) & (self.stimuli == stimulus) & (previous_rewards == 1)].mean()
        data.loc[stimulus, "Total"] = self.choices[(self.stimuli == stimulus) & (previous_rewards == 1)].mean()

    sns.lineplot(data=data, dashes=False, markers=True, palette="Set1", ax=axes[0,1])
    axes[0,1].set_xlabel("Current Stimulus")
    axes[0,1].set_ylabel("Average Choice")
    axes[0,1].legend(title="Previous Rewarded")
    axes[0,1].text(-0.2, 0.94, 'C', fontsize=15, transform=axes[0,1].transAxes)

    data = pd.DataFrame(columns=["Previous Left", "Previous Right", "Total"],
                        index=[i for i in np.unique(self.stimuli)])
    previous_choices = np.concatenate(([0], self.choices[:-1]))
    previous_rewards = np.concatenate(([0], self.rewards[:-1]))
    for stimulus in np.unique(self.stimuli):
        data.loc[stimulus, "Previous Left"] = self.choices[
            (previous_choices == 0) & (self.stimuli == stimulus) & (previous_rewards == 0)].mean()
        data.loc[stimulus, "Previous Right"] = self.choices[
            (previous_choices == 1) & (self.stimuli == stimulus) & (previous_rewards == 0)].mean()
        data.loc[stimulus, "Total"] = self.choices[(self.stimuli == stimulus) & (previous_rewards == 1)].mean()

    sns.lineplot(data=data, dashes=False, markers=True, palette="Set1", ax=axes[0,2])
    axes[0,2].set_xlabel("Current Stimulus")
    axes[0,2].set_ylabel("Average Choice")
    axes[0,2].legend(title="Previous Unrewarded")




# self.get_psychometric(rewarded = False)
    #
    # # Figure 1 D
    # data = self.psychometric.loc[["current", previous_psychometric]].T
    # data.index.astype(str)
    # sns.lineplot(data=data, dashes=False, markers=True, palette="Set1", ax=axes[2, 0])
    # axes[2, 0].set_xlabel("Current/Previous Stimulus")
    # axes[2, 0].set_ylabel("Average Choice")
    # axes[2, 0].text(-0.2, 0.94, 'D', fontsize=15, transform=axes[2, 0].transAxes)
    #

    # # plot updating function
    # sns.lineplot(data=(self.updating_function * 100), palette="Set1", dashes=False, markers=True, ax=axes[2, 2])
    # axes[2, 2].set_xlabel("Previous Stimulus")
    # axes[2, 2].set_ylabel("Updating %")


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



def Figure_3(models = None,
             steps = 50000,
             alpha_list = [0.1,0.5,0.9],
             ):
    param_list = []
    for alpha in alpha_list:
        param = Params(alpha = alpha, time_steps=steps)
        param.get_blocks(magnitude_structure=[(1, 1), (1, 0.5), (0.5, 1)], block_size=200)
        param_list.append(param)
    if models is None:
        model0 = Model(param_list[0])
        model0.run_model()
        model1 = Model(param_list[1])
        model1.run_model()
        model2 = Model(param_list[2])
        model2.run_model()
    else:
        model0, model1, model2 = models

    fig = plt.figure(figsize=(12.5, 10))
    gs = matplotlib.gridspec.GridSpec(9, 9, figure=fig,wspace=3.3, hspace=0.5)
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
    A1.set_ylim(0.5, 1)
    for j in range(len(plot)):
        variable = plot[j]
        for i in range(before + after):
            data.loc[index[i], j] = np.mean(variable[transitions + index[i] - 1])

        g = data[j]
        sns.lineplot(data=g, alpha=0.6, color=line_colors[j],
                     ax=A1, label=f"{variables[j]}")

    A1.axvspan(-before, 0, alpha=0.3, color=colors[0])
    A1.axvspan(0, after, alpha=0.3, color=colors[1])
    A1.set_ylabel(f"alpha: {alpha_list[0]}")

    A1.text(-0.1, 0.9, 'A', fontsize=15, transform=A1.transAxes)

    plot = []
    plot.append(model1.values[:, 0])
    plot.append(model1.values[:, 1])
    data = pd.DataFrame(index=index, columns=range(len(plot)))
    transitions = all_transitions[2::3]
    for j in range(len(plot)):
        variable = plot[j]
        for i in range(before + after):
            data.loc[index[i], j] = np.mean(variable[transitions + index[i] - 1])

        g = data[j]
        sns.lineplot(data=g, alpha=0.6, color=line_colors[j],
                     ax=A2, label=f"{variables[j]}")

    A2.axvspan(-before, 0, alpha=0.3, color=colors[0])
    A2.axvspan(0, after, alpha=0.3, color=colors[1])
    A2.set_ylabel(f"alpha: {alpha_list[1]}")

    plot = []
    plot.append(model2.values[:, 0])
    plot.append(model2.values[:, 1])
    data = pd.DataFrame(index=index, columns=range(len(plot)))
    transitions = all_transitions[2::3]
    for j in range(len(plot)):
        variable = plot[j]
        for i in range(before + after):
            data.loc[index[i],  j] = np.mean(variable[transitions + index[i] - 1])

        g = data[j]
        sns.lineplot(data=g, alpha=0.6, color=line_colors[j],
                     ax=A3, label=f"{variables[j]}")

    A3.axvspan(-before, 0, alpha=0.3, color=colors[0])
    A3.axvspan(0, after, alpha=0.3, color=colors[1])
    A3.set_ylabel(f"alpha: {alpha_list[2]}")

    data = pd.DataFrame(index=alpha_list)
    data[alpha_list[0]] = (np.std(model0.values[0]) + np.std(model0.values[1])) / 2
    data[alpha_list[1]] = (np.std(model1.values[0]) + np.std(model1.values[1])) / 2
    data[alpha_list[2]] = (np.std(model2.values[0]) + np.std(model2.values[1])) / 2
    data = data.iloc[0]
    data.index = [f"sigma: {alpha}" for alpha in alpha_list]
    sns.barplot(x=data.index, y=data.values, ax=B)
    B.set_xlabel("")
    B.set_ylabel("Value Standard Deviation")
    B.text(-0.3, 0.94, 'B', fontsize=15, transform=B.transAxes)


    #figure 3C -> average rewards
    data = pd.DataFrame()
    data["Average Reward"] = np.concatenate((model0.rewards, model1.rewards, model2.rewards))
    data["alpha"] = [f"alpha: {alpha_list[0]}"] * model0.rewards.shape[0] + [f"alpha: {alpha_list[1]}"] * model1.rewards.shape[
        0] + [f"alpha: {alpha_list[2]}"] * model2.rewards.shape[0]
    plot = sns.barplot(data = data, x = "alpha", y= "Average Reward", order = [f"alpha: {alpha}" for alpha in alpha_list], ax = C,
                       bottom = 0)
    plot.set_ylim(0.6, 0.8)
    plot.set_xlabel("")
    C.text(-0.3, 0.94, 'C', fontsize=15, transform=C.transAxes)

    t_test_results = pingouin.pairwise_tests(data=data, dv='Average Reward', between='alpha')

    print(t_test_results)


    #figure 3D
    _,matrix,_ = model0.get_psychometric()
    # plot updating matrix
    img = D1.imshow(matrix.values.astype(np.float64).T * 100,
                         cmap='RdBu', interpolation='nearest', aspect='auto')
    plt.colorbar(img, ax=D1, fraction=0.05, pad=0.04)  # Add color bar
    D1.set_xlabel(f"Current Stimulus \n alpha: {alpha_list[1]}")
    D1.set_ylabel("Previous Stimulus")
    D1.set_xticks(np.arange(len(matrix.index)))
    D1.set_yticks(np.arange(len(matrix.index)))
    D1.set_xticklabels(matrix.index, rotation=90)
    D1.set_yticklabels(matrix.index)
    D1.text(-0.3, 0.94, 'D', fontsize=15, transform=D1.transAxes)

    _, matrix, _ = model1.get_psychometric()
    # plot updating matrix
    img = D2.imshow(matrix.values.astype(np.float64).T * 100,
                    cmap='RdBu', interpolation='nearest', aspect='auto')
    plt.colorbar(img, ax=D2, fraction=0.05, pad=0.04)  # Add color bar
    D2.set_xlabel(f"Current Stimulus \n alpha: {alpha_list[1]}")
    D2.set_xticks(np.arange(len(matrix.index)))
    D2.set_yticks(np.arange(len(matrix.index)))
    D2.set_xticklabels(matrix.index, rotation=90)
    D2.set_yticklabels(matrix.index)

    _, matrix, _ = model2.get_psychometric()
    # plot updating matrix
    img = D3.imshow(matrix.values.astype(np.float64).T * 100,
                    cmap='RdBu', interpolation='nearest', aspect='auto')
    plt.colorbar(img, ax=D3, fraction=0.05, pad=0.04)  # Add color bar
    D3.set_xlabel(f"Current Stimulus \n alpha: {alpha_list[2]}")
    D3.set_xticks(np.arange(len(matrix.index)))
    D3.set_yticks(np.arange(len(matrix.index)))
    D3.set_xticklabels(matrix.index, rotation=90)
    D3.set_yticklabels(matrix.index)

    return (model0, model1 , model2), t_test_results


def Figure_4(models = None,
             steps = 50000,
             sigma_list = [0.1,0.5,0.9],
             ):
    param_list = []
    for sigma in sigma_list:
        param = Params(sigma = sigma, time_steps=steps)
        param.get_blocks(magnitude_structure=[(1, 1), (1, 0.5), (0.5, 1)], block_size=400)
        param_list.append(param)
    if models is None:
        model0 = Model(param_list[0])
        model0.run_model()
        model1 = Model(param_list[1])
        model1.run_model()
        model2 = Model(param_list[2])
        model2.run_model()
    else:
        model0, model1, model2 = models

    fig = plt.figure(figsize=(12.5, 10))
    gs = matplotlib.gridspec.GridSpec(9, 9, figure=fig,wspace=3.3, hspace=0.5)
    A1 = fig.add_subplot(gs[0:2, :-3])
    A2 = fig.add_subplot(gs[2:4, :-3])
    A3 = fig.add_subplot(gs[4:6, :-3])
    B = fig.add_subplot(gs[0:3, -3:])
    C = fig.add_subplot(gs[3:6, -3:])
    D1 = fig.add_subplot(gs[-3:, 0:3])
    D2 = fig.add_subplot(gs[-3:, 3:6])
    D3 = fig.add_subplot(gs[-3:, 6:9])


    #fig 4 a1
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

        g = data[j]
        sns.lineplot(data=g, alpha=0.6, color=line_colors[j],
                     ax=A1, label=f"{variables[j]}")

    A1.axvspan(-before, 0, alpha=0.3, color=colors[0])
    A1.axvspan(0, after, alpha=0.3, color=colors[1])
    A1.set_ylabel(f"sigma: {sigma_list[0]}")
    A1.text(-0.1, 0.9, 'A', fontsize=15, transform=A1.transAxes)

    plot = []
    plot.append(model1.values[:, 0])
    plot.append(model1.values[:, 1])
    data = pd.DataFrame(index=index, columns=range(len(plot)))
    transitions = all_transitions[2::3]
    for j in range(len(plot)):
        variable = plot[j]
        for i in range(before + after):
            data.loc[index[i], j] = np.mean(variable[transitions + index[i] - 1])

        g = data[j]
        sns.lineplot(data=g, alpha=0.6, color=line_colors[j],
                     ax=A2, label=f"{variables[j]}")

    A2.axvspan(-before, 0, alpha=0.3, color=colors[0])
    A2.axvspan(0, after, alpha=0.3, color=colors[1])
    A2.set_ylabel(f"sigma: {sigma_list[1]}")

    plot = []
    plot.append(model2.values[:, 0])
    plot.append(model2.values[:, 1])
    data = pd.DataFrame(index=index, columns=range(len(plot)))
    transitions = all_transitions[2::3]
    for j in range(len(plot)):
        variable = plot[j]
        for i in range(before + after):
            data.loc[index[i],  j] = np.mean(variable[transitions + index[i] - 1])

        g = data[j]
        sns.lineplot(data=g, alpha=0.6, color=line_colors[j],
                     ax=A3, label=f"{variables[j]}")

    A3.axvspan(-before, 0, alpha=0.3, color=colors[0])
    A3.axvspan(0, after, alpha=0.3, color=colors[1])
    A3.set_ylabel(f"sigma: {sigma_list[2]}")


    #figure 4B
    data = pd.DataFrame(index=sigma_list)
    data[sigma_list[0]] = (np.std(model0.values[0]) + np.std(model0.values[1]))/2
    data[sigma_list[1]] = (np.std(model1.values[0]) + np.std(model1.values[1]))/2
    data[sigma_list[2]] = (np.std(model2.values[0]) + np.std(model2.values[1]))/2
    data = data.iloc[0]
    data.index = [f"sigma: {sigma}" for sigma in sigma_list]
    sns.barplot(x=data.index, y=data.values, ax = B)
    B.set_xlabel("")
    B.set_ylabel("Value Standard Deviation")
    B.text(-0.3, 0.94, 'B', fontsize=15, transform=B.transAxes)

    #figure 4C -> average rewards
    data = pd.DataFrame()
    data["Average Reward"] = np.concatenate((model0.rewards, model1.rewards, model2.rewards))
    data["sigma"] = [f"sigma: {sigma_list[0]}"] * model0.rewards.shape[0] + [f"sigma: {sigma_list[1]}"] * model1.rewards.shape[
        0] + [f"sigma: {sigma_list[2]}"] * model2.rewards.shape[0]
    plot = sns.barplot(data = data, x = "sigma", y= "Average Reward", order = [f"sigma: {sigma}" for sigma in sigma_list], ax = C,
                       bottom = 0)
    plot.set_xlabel("")
    C.text(-0.3, 0.94, 'C', fontsize=15, transform=C.transAxes)

    t_test_results = pingouin.pairwise_ttests(data=data, dv='Average Reward', between='sigma')

    print(t_test_results)


    #figure 4D
    _,matrix,_ = model0.get_psychometric()
    # plot updating matrix
    img = D1.imshow(matrix.values.astype(np.float64).T * 100,
                         cmap='RdBu', interpolation='nearest', aspect='auto')
    plt.colorbar(img, ax=D1, fraction=0.05, pad=0.04)  # Add color bar
    D1.set_xlabel(f"Current Stimulus \n sigma: {sigma_list[1]}")
    D1.set_ylabel("Previous Stimulus")
    D1.set_xticks(np.arange(len(matrix.index)))
    D1.set_yticks(np.arange(len(matrix.index)))
    D1.set_xticklabels(matrix.index, rotation=90)
    D1.set_yticklabels(matrix.index)
    D1.text(-0.3, 0.94, 'D', fontsize=15, transform=D1.transAxes)

    _, matrix, _ = model1.get_psychometric()
    # plot updating matrix
    img = D2.imshow(matrix.values.astype(np.float64).T * 100,
                    cmap='RdBu', interpolation='nearest', aspect='auto')
    plt.colorbar(img, ax=D2, fraction=0.05, pad=0.04)  # Add color bar
    D2.set_xlabel(f"Current Stimulus \n sigma: {sigma_list[1]}")
    D2.set_xticks(np.arange(len(matrix.index)))
    D2.set_yticks(np.arange(len(matrix.index)))
    D2.set_xticklabels(matrix.index, rotation=90)
    D2.set_yticklabels(matrix.index)

    _, matrix, _ = model2.get_psychometric()
    # plot updating matrix
    img = D3.imshow(matrix.values.astype(np.float64).T * 100,
                    cmap='RdBu', interpolation='nearest', aspect='auto')
    plt.colorbar(img, ax=D3, fraction=0.05, pad=0.04)  # Add color bar
    D3.set_xlabel(f"Current Stimulus \n sigma: {sigma_list[2]}")
    D3.set_xticks(np.arange(len(matrix.index)))
    D3.set_yticks(np.arange(len(matrix.index)))
    D3.set_xticklabels(matrix.index, rotation=90)
    D3.set_yticklabels(matrix.index)

    return (model0, model1 , model2), t_test_results




def Figure_confidence(self = None):
    if self is None:
        self = Model()
        self.run_model()

    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(8,5))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.35, hspace=0.25)

    A1 = ax[0]
    A1.text(-0.2, 0.94, 'A', fontsize=15, transform=A1.transAxes)
    B1 = ax[1]
    B1.text(-0.2, 0.94, 'B', fontsize=15, transform=B1.transAxes)

    #plot accuracy = average reward for each confidence
    data = pd.DataFrame(index= self.params.confidences, columns = ["mean", "sd"])
    for confidence in self.params.confidences:
        data.loc[confidence, "mean"] = self.rewards[self.confidence == confidence].mean()
        data.loc[confidence, "sd"] = self.rewards[self.confidence == confidence].std()
    sns.lineplot(data = data["mean"], label = "Accuracy", marker = "o", ax = A1)

    # Fit linear regression line
    slope, intercept, r_value, _, _ = scipy.stats.linregress(data.index.astype(np.float64), data["mean"].values.astype(np.float64))
    line = slope * data.index + intercept

    # Plot the linear regression line
    A1.plot(data.index, line, color='red',
             label=f'Regression:\ny = {slope:.2f}x + {intercept:.2f}\nR\u00b2 = {r_value ** 2:.2f}')
    A1.legend()
    A1.set_xlabel("Confidence")
    A1.set_ylabel("Average Reward")



    #plot confidence for correct/ incorrect choices against absolute stimulus
    absolute_stimuli = np.abs(self.stimuli)
    data = pd.DataFrame(index=np.unique(absolute_stimuli),
                        columns= ["Correct", "Incorrect"])
    for abs_stim in np.unique(absolute_stimuli):
        data.loc[abs_stim, "Correct"] = self.confidence[(absolute_stimuli == abs_stim)&(self.rewards == 1)].mean()
        data.loc[abs_stim, "Incorrect"] = self.confidence[(absolute_stimuli == abs_stim)&(self.rewards == 0)].mean()
    sns.lineplot(data, ax = B1)
    B1.set_ylabel("Average Confidence")
    B1.set_xlabel("Absolute Stimulus")

    #
    confidence_cut = 0.75
    data = pd.DataFrame(index=np.linspace(0.4,1.5,100),
                        columns=["Minimum Difference"])
    for confidence_cut in data.index:
        difference = []
        for abs_stim in np.unique(absolute_stimuli):
            high_confidence = self.confidence[(absolute_stimuli == abs_stim) & (self.confidence > confidence_cut)].mean()
            low_confidence = self.confidence[(absolute_stimuli == abs_stim) & (self.confidence < confidence_cut)].mean()
            difference.append(high_confidence - low_confidence)
        data.loc[confidence_cut, "Minimum Difference"] = min(difference)
    print(str(data.min())+ "over all confidence cuts")


    #calculate mean and standard deviation for stimulus == 0:
    print(f"avg conf for stim == 0: {self.confidence[self.stimuli == 0].mean()} +- {self.confidence[self.stimuli == 0].std()}")
    # ttest differnce from 0.75
    t_stat, p_value = scipy.stats.ttest_1samp(self.confidence[self.stimuli == 0], 0.75)
    print(f"pvalue for significant difference between "
          f"average confidence for neutral stimulus and 0.75: {p_value}")




def Figure_TDRL_models(models = None):
    if models is not None:
        old2,old,new = models
    else:
        new = Model()
        new.run_investment_model()

        old = Model()
        old.run_investment_model_old()

        old2 = Model()
        old2.run_investment_model_old_old()

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize = (9,5))

    A1 = ax[0,0]
    A1.text(-0.2, 0.94, 'A', fontsize=15, transform=A1.transAxes)
    A2 = ax[1,0]
    B1 = ax[0,1]
    B1.text(-0.2, 0.94, 'B', fontsize=15, transform=B1.transAxes)
    B2 = ax[1,1]
    C1 = ax[0,2]
    C1.text(-0.2, 0.94, 'C', fontsize=15, transform=C1.transAxes)
    C2 = ax[1,2]

    old2.get_psychometric()
    data = old2.psychometric.loc["current"].T
    data.index.astype(str)
    sns.lineplot(data=data, dashes=False, marker="o", palette="Set1", ax = A1)
    A1.set_xlabel("Stimulus")
    A1.ylabel("Average Choice")

    data = pd.DataFrame(index=np.unique(old2.stimuli),
                        columns=["Correct", "Incorrect"])
    choices = old2.choices
    choices[choices == 0] = -1
    for stimulus in np.unique(old2.stimuli):
        data.loc[stimulus, "Correct"] = old2.time_investment[
            (np.sign(old2.stimuli) == choices) & (old2.stimuli == stimulus)].mean()
        data.loc[stimulus, "Incorrect"] = old2.time_investment[
            ~(np.sign(old2.stimuli) == choices) & (old2.stimuli == stimulus)].mean()

    sns.lineplot(data=data, dashes=False, marker="o", palette="Set1", ax = A2)
    A2.set_xlabel("Stimulus")
    A2.set_ylabel("Average Time Investment")

    #old
    old.get_psychometric()
    data = old.psychometric.loc["current"].T
    data.index.astype(str)
    sns.lineplot(data=data, dashes=False, marker="o", palette="Set1", ax=B1)
    B1.set_xlabel("Stimulus")
    B1.ylabel("Average Choice")

    data = pd.DataFrame(index=np.unique(old.stimuli),
                        columns=["Correct", "Incorrect"])
    choices = old.choices
    choices[choices == 0] = -1
    for stimulus in np.unique(old.stimuli):
        data.loc[stimulus, "Correct"] = old.time_investment[
            (np.sign(old.stimuli) == choices) & (old.stimuli == stimulus)].mean()
        data.loc[stimulus, "Incorrect"] = old.time_investment[
            ~(np.sign(old.stimuli) == choices) & (old.stimuli == stimulus)].mean()

    sns.lineplot(data=data, dashes=False, marker="o", palette="Set1", ax=B2)
    B2.set_xlabel("Stimulus")
    B2.set_ylabel("Average Time Investment")

    new.get_psychometric()
    data = new.psychometric.loc["current"].T
    data.index.astype(str)
    sns.lineplot(data=data, dashes=False, marker="o", palette="Set1", ax=C1)
    C1.set_xlabel("Stimulus")
    C1.ylabel("Average Choice")

    data = pd.DataFrame(index=np.unique(new.stimuli),
                        columns=["Correct", "Incorrect"])
    choices = new.choices
    choices[choices == 0] = -1
    for stimulus in np.unique(new.stimuli):
        data.loc[stimulus, "Correct"] = new.time_investment[
            (np.sign(new.stimuli) == choices) & (new.stimuli == stimulus)].mean()
        data.loc[stimulus, "Incorrect"] = old2.time_investment[
            ~(np.sign(new.stimuli) == choices) & (new.stimuli == stimulus)].mean()

    sns.lineplot(data=data, dashes=False, marker="o", palette="Set1", ax=C2)
    C2.set_xlabel("Stimulus")
    C2.set_ylabel("Average Time Investment")

    return new, old, old2