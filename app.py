import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

model = joblib.load("churn_model.pkl")
columns = joblib.load("model_columns.pkl")


st.sidebar.title("About")
st.sidebar.info(
    "This Machine Learning app predicts whether a customer is likely to churn."
)


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


if st.button("Predict Churn"):

    input_data = pd.DataFrame(columns=columns)

    
    input_data.loc[0] = 0

    # important features
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges
    input_data["numTechTickets"] = tech_tickets

    
    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")

    st.write(f"Churn Probability: {probability:.2%}")
