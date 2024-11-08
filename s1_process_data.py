import os
import json
import codecs
import pandas as pd
from properties import data_folder

projectfiles = [projectfile for projectfile in os.listdir(data_folder) if projectfile.endswith(".json")]
for projectfile in projectfiles:
	# Processing the output files of script 1
	with codecs.open(os.path.join(data_folder, projectfile), 'r', 'utf-8') as file:
		issues_data = json.load(file)

	# Retrieve the project's name
	projectname = issues_data[0]["projectname"]
	print("Processing project: ", projectname)

	res = {}
	df = pd.DataFrame()

	for d in issues_data:
		res["id"] = d["_id"]
		res["summary"] = d["summary"]
		res["description"] = d["description"]
		# An issue can be characterized by multiple labels store them all together
		res["labels"] = " ".join(d["labels"])

		components_name = []
		components_name.clear()
		# An issue can belong to multiple components that's why we store all of them
		for idx in d["components"]:
			components_name.append(idx["name"])

		res["components_name"] = [components_name]

		res["project_name"] = d["projectname"]
		res["issue_type_name"] = d["issuetype"]["name"]
		res["priority_name"] = d["priority"]["name"]
		res["created_date"] = d["created"]
		res["assignee"] = d["assignee"]

		# Creating a dataframe of the desired fields-values for easier handling of the data
		df = pd.concat([df, pd.DataFrame.from_dict(res)], axis=0)

	# Assign a unique id to each assignee. This is crucial as this field will be the target class for the classifier
	df["assignee"] = pd.factorize(df["assignee"])[0]

	df.to_csv(os.path.join(data_folder, projectname + ".csv"), index=False)

