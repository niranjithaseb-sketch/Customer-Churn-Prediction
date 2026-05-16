import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd


df = pd.read_csv("churndataset.csv")

df = df.drop("customerID", axis=1)


df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# remove missing val
df = df.dropna()


df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


df = pd.get_dummies(df, drop_first=True)

print(df.head())
print(df.shape)

X = df.drop("Churn", axis=1)
y = df["Churn"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
df["Churn"].value_counts().plot(kind="bar")

plt.title("Customer Churn Count")
plt.xlabel("Churn")
plt.ylabel("Count")

plt.show()


joblib.dump(model, "churn_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.head(10))
# top 10 features
top_features = feature_importance.head(10)

# create graph
plt.figure(figsize=(10,6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features for Churn Prediction")

plt.gca().invert_yaxis()

plt.show()
print("Model loaded successfully")
