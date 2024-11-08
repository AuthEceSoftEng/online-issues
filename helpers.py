import pandas as pd

def preprocess_data(data, columns_to_drop=[]):
    # Reverse the order of the rows in the dataframe
    data = data.iloc[::-1]
    data = data.reset_index()
    # Convert the created_date variable of every issue into it's toordinal form
    data["created_date"] = pd.to_datetime(data["created_date"])
    data["created_date"] = data["created_date"].map(lambda x: x.toordinal())
    # Sort the data by ascending chronological order
    data = data.sort_values(by='created_date')
    # Remove all the keys/value pairs that won't be features in our model
    dataY = data["assignee"]
    dataX = data.drop(columns=["index", "id", "project_name", "created_date", "assignee"])
    dataX = dataX.drop(columns=columns_to_drop)
    return dataX, dataY
