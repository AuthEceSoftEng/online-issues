import os
import json
import matplotlib.pyplot as plt
from properties import results_folder, graphs_folder
from online_models import NaiveBayesOnlineModel, NaiveBayesWithADWINOnlineModel, AdaboostOnlineModel, AdaboostWithADWINOnlineModel

project_name = "DATALAB"
allmodels = [NaiveBayesOnlineModel, NaiveBayesWithADWINOnlineModel, AdaboostOnlineModel, AdaboostWithADWINOnlineModel]

fig, ax = plt.subplots(figsize=(5, 2.8))
plt.xlabel("Number of Instances")
plt.ylabel("Moving Average Accuracy")
for m, model in enumerate(allmodels):
    model_name = model().name
    filename = project_name + "_" + model_name
    with open(os.path.join(results_folder, model_name, project_name + ".json")) as infile:
        modelresults = json.load(infile)
    
    # Plot the final metric plot
#    if m == 2: plt.plot(modelresults["accuracies"], label=model_name, color='black')
#    elif m == 4: plt.plot(modelresults["accuracies"], label=model_name, color='gray')
#    else: 
    plt.plot(modelresults["accuracies"], label=model_name)
for drift_detected in modelresults["drifts"]: # Place a line exactly where the drift was found
    ax.axvline(drift_detected, color="black", linestyle="--")
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(graphs_folder, "_".join(filename.split()) + ".pdf"))
plt.show()
    
