import joblib
import pandas as pd

# load model
model = joblib.load("churn_model.pkl")

# load column names
columns = joblib.load("model_columns.pkl")

# create empty input data
input_data = pd.DataFrame(columns=columns)

# add sample values
input_data.loc[0] = 0

input_data["tenure"] = 2
input_data["MonthlyCharges"] = 95
input_data["TotalCharges"] = 190

# prediction
prediction = model.predict(input_data)

if prediction[0] == 1:
    print("Customer likely to churn")
else:
    print("Customer likely to stay")