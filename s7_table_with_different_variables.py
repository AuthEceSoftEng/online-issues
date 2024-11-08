import os
import json
import numpy as np
from properties import data_folder, results_folder
from online_models import AdaboostWithADWINOnlineModel

project_names = [projectfile[:-4] for projectfile in os.listdir(data_folder) if projectfile.endswith(".csv")]
allmodels = [AdaboostWithADWINOnlineModel]
allvariables = [("summary"), ("summary", "description"), ("summary", "description", "labels"), ("summary", "description", "labels", "components_name"), ("summary", "description", "labels", "components_name", "issue_type_name"), ("summary", "description", "labels", "components_name", "priority_name", "issue_type_name")]

results = {}
for project_name in project_names:
    results[project_name] = {}
    for model in allmodels:
        for variables in allvariables:
            model_name = model(variables=variables).name
            with open(os.path.join(results_folder, model_name, project_name + ".json")) as infile:
                results[project_name][" \& ".join(variables) if type(variables) != str else variables] = json.load(infile)["accuracies"][-1]
#print(results)

print("\\toprule")
header = ""
for setting in results[list(results.keys())[0]]:
    header += "\\rotatebox[origin = lb]{60}{\\parbox{3.1cm}{" + setting.title() + "}} & "
header = header[:-3]
header = header.replace("Issue_", "").replace("_Name", "")
print("Project & " + " ".join(header.split("_")) + " \\\\")
print("\\midrule")

for project in results:
    print(project, end='')
    indexmax = np.argmax([results[project][setting] for setting in results[project]])
    for i, setting in enumerate(results[project]):
        if i == indexmax:
            print(" & \\textbf{%.2f\%%}" %(results[project][setting] * 100), end='')
        else:
            print(" & %.2f\%%" %(results[project][setting] * 100), end='')
    print(" \\\\")
print("\\bottomrule")

