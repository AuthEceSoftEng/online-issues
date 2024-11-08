import os
import json
import codecs
import pymongo
from properties import mongo_URL, data_folder, results_folder, graphs_folder, min_issues_per_assignee

# Create folders
if not os.path.exists(data_folder):
	os.makedirs(data_folder)
if not os.path.exists(results_folder):
	os.makedirs(results_folder)
if not os.path.exists(graphs_folder):
	os.makedirs(graphs_folder)

# Connect to database
client = pymongo.MongoClient(mongo_URL)
db = client["jidata"]


# Filters the issues, keeping only those with summary, description, and labels
# Also allows setting a specific filter in the assignee field
def issue_filter(project_name, assignee):
	return {
				'projectname': project_name,
				'summary': {'$exists': True, '$not': {'$size': 0}},
				'description': {'$exists': True, '$not': {'$size': 0}},
				'labels': {'$exists': True, '$not': {'$size': 0}},
				'assignee': assignee
			}


# Get all project names
for project in db["projects"].find(projection={"projectname": 1, "_id": 0}):
	projectname = project["projectname"]

	# For each project find the assignees that have been assigned at least 80 issues
	qpipeline = [
		{'$match': issue_filter(projectname, {'$exists': True, '$not': {'$size': 0}})},
		# Gather and count the issues for every unique assignee inside the project
		{'$group': {'_id': '$assignee', 'count': {'$sum': 1}}},
		# if the number of the issues is equal or greater than the minimum issues per assignee keep these issues
		{'$match': {'$and': [
			{'count': {'$gte': min_issues_per_assignee}},
			# There is an assigned developer
			{'_id': {'$ne': None}}
		]}},
		# Sort the number of issues per assignee by descending order
		{'$sort': {'count': -1}}
	]
	projectassignees = [assignee["_id"] for assignee in db["issues"].aggregate(qpipeline)]

	# If there are at least 5 such assignees
	if len(projectassignees) >= 5:
		print("Processing project " + projectname)

		# Download all the issues of these assignees for the project
		mongo_filter = issue_filter(projectname, {'$in': projectassignees})

		# Download all the issues of the project
		mongo_filter_all = issue_filter(projectname, {'$exists': True, '$not': {'$size': 0}})

		mongo_projection = {"summary": 1, "description": 1, "labels": 1, "components": 1, "priority": 1, "issuetype": 1,
							"projectname": 1, "created": 1, "assignee": 1}

		issues = [issue for issue in db["issues"].find(filter=mongo_filter, projection=mongo_projection)]
		with codecs.open(os.path.join(data_folder, projectname + ".json"), 'w', 'utf-8') as outfile:
			json.dump(issues, outfile, indent=3, default=str)
