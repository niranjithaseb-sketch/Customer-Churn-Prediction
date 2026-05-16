import streamlit as st
import pandas as pd
import joblib

# page settings
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# load model and columns
model = joblib.load("churn_model.pkl")
columns = joblib.load("model_columns.pkl")

# sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This Machine Learning app predicts whether a customer is likely to churn."
)

# title
st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer details below to predict whether the customer may churn."
)

# inputs
tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    value=1
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=100.0
)

tech_tickets = st.number_input(
    "Number of Tech Tickets",
    min_value=0,
    value=0
)

# prediction button
if st.button("Predict Churn"):

    # create dataframe
    input_data = pd.DataFrame(columns=columns)

    # fill all columns with 0
    input_data.loc[0] = 0

    # important features
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges
    input_data["numTechTickets"] = tech_tickets

    # prediction
    prediction = model.predict(input_data)

    # probability
    probability = model.predict_proba(input_data)[0][1]

    # result
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")

    st.write(f"Churn Probability: {probability:.2%}")