import os
import json
import pandas as pd
from itertools import repeat
from helpers import preprocess_data
from properties import data_folder, results_folder
from online_models import NaiveBayesOnlineModel, NaiveBayesWithADWINOnlineModel, AdaboostOnlineModel, AdaboostWithADWINOnlineModel,\
    EnhancedOnlineModel

project_names = [projectfile[:-4] for projectfile in os.listdir(data_folder) if projectfile.endswith(".csv")]
allmodels = [NaiveBayesOnlineModel, NaiveBayesWithADWINOnlineModel, AdaboostOnlineModel, AdaboostWithADWINOnlineModel]
allmodels = [EnhancedOnlineModel]


def execute_model_on_project(model_name, model, project_name):
    print("Processing project ", project_name)

    # Read data for a project
    data = pd.read_csv(os.path.join(data_folder, project_name + ".csv"))
    dataX, dataY = preprocess_data(data)

    my_model = model()
    printthreshold = 0                                                 ## USED FOR PROGRESS
    for row, y in zip(dataX.iterrows(), dataY):
        # Iterate row wise through the dataset
        i, x = row
        # Apply the model
        my_model.apply_model(i, x, y)

        if int(100 * (i + 1) / len(dataX)) > printthreshold + 10 - 1:  ## USED
            printthreshold += 10                                       ## FOR
            print(project_name + " %d%%" %printthreshold)              ## PROGRESS

    # Write the results to disk
    with open(os.path.join(results_folder, model_name, project_name + ".json"), 'w') as outfile:
        json.dump(my_model.results(), outfile, indent = 3)

if __name__ == '__main__':
    for model in allmodels:
        model_name = model().name
        if not os.path.exists(os.path.join(results_folder, model_name)):
            os.makedirs(os.path.join(results_folder, model_name))
        import multiprocessing as mp
        with mp.Pool(8) as pool:
            pool.starmap(execute_model_on_project, zip(repeat(model_name), repeat(model), project_names))
#            for project_name in project_names:                
