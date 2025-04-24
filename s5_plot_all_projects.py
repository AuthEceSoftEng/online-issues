import os
import json
import numpy as np
import matplotlib.pyplot as plt
from properties import data_folder, results_folder, graphs_folder
from online_models import NaiveBayesOnlineModel, NaiveBayesWithADWINOnlineModel, AdaboostOnlineModel, AdaboostWithADWINOnlineModel

project_names = [projectfile[:-4] for projectfile in os.listdir(data_folder) if projectfile.endswith(".csv")]
allmodels = [NaiveBayesOnlineModel, NaiveBayesWithADWINOnlineModel, AdaboostOnlineModel, AdaboostWithADWINOnlineModel]

accuracy = {}
allresults = {}
for model in allmodels:
    model_name = model().name
    alldrifts = {}
    allresults[model_name] = {}
    for project_name in project_names:
        with open(os.path.join(results_folder, model_name, project_name + ".json")) as infile:
            filecontents = json.load(infile)
            allresults[model_name][project_name] = filecontents["accuracies"]
            alldrifts[project_name] = len(filecontents["drifts"])
    accuracy[model_name] = [allresults[model_name][project_name][-1] for project_name in project_names]
print(accuracy)

#project_names = [project_name + "\n(%d drifts)" % alldrifts[project_name] for project_name in project_names]

fig, ax = plt.subplots(figsize=(10, 3))
for c, model in enumerate(allmodels):
    model_name = model().name
    accuracies = accuracy[model_name]

    # Number of bars (12 in this case)
    n = len(accuracies)

    # X-axis values
    x = np.arange(n)

    width = 0.15

    avg = sum(accuracies) / len(accuracies)

    bars1 = ax.bar(x + (c - 1.5) * width, accuracies, width, label=model_name + " - ACC: %.3f" %avg, zorder=2)
#    bars1 = ax.bar(x + (c - 1.5) * width, accuracies, width, label=model_name, zorder=2)
    #ax.axhline(y=avg, linestyle='--', color=bars1.patches[0].get_facecolor(), linewidth=2, zorder=1)

    # Adding labels and title
    ax.set_xlabel('Project')
    ax.set_ylabel('Moving Average Accuracy')
    # Set y-ticks (to make the labels more readable)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    # Customize font size or tick size to make them stand out more
    ax.set_xticks(x)
    # Set x-axis label size and rotate the labels for better readability
    ax.set_xticklabels(project_names)
    
    ax.set_ylim(0, 1.12)

    # Manually add a single legend entry for the average line
    #avg_line = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1)

    # Combine all handles for the legend (bars and the average line)
    handles, labels = ax.get_legend_handles_labels()
    #handles.append(avg_line)
    labels.append('Avg value')

    # Set the legend with all required labels
    ax.legend(handles=handles, labels=labels, loc='upper left', ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(graphs_folder, "ALL_naivebayes_vs_adaboost.pdf"))
plt.show()

