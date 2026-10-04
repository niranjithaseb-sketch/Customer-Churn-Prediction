import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #171a24 100%);
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .block-container {
        padding-top: 3rem;
        padding-left: 5rem;
        padding-right: 5rem;
        max-width: 1200px;
    }

    /* Title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
        background: linear-gradient(90deg, #ff4b91, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: #a9adba;
        font-size: 17px;
        margin-bottom: 35px;
    }

    /* Cards */
    .card {
        background: #1e212b;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #303441;
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 20px;
        font-weight: 700;
        color: white;
        margin-bottom: 15px;
    }

    /* Input labels */
    label {
        font-weight: 600 !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg, #ff4b91, #a855f7);
        color: white;
        font-size: 17px;
        font-weight: 700;
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 75, 145, 0.25);
    }

    /* Result cards */
    .result-card {
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        background: #1e212b;
        border: 1px solid #303441;
        margin-top: 25px;
    }

    .result-title {
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .probability {
        font-size: 36px;
        font-weight: 800;
        color: #ff4b91;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #15171f;
        border-right: 1px solid #2b2e39;
    }

    .sidebar-title {
        font-size: 25px;
        font-weight: 800;
        color: white;
    }

    .sidebar-text {
        color: #b4b7c2;
        line-height: 1.7;
    }

</style>
""", unsafe_allow_html=True)

model = joblib.load("churn_model.pkl")
columns = joblib.load("model_columns.pkl")


st.sidebar.markdown(
    '<div class="sidebar-title">📊 Churn Predictor</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div class="sidebar-text">
    <b>About this project</b><br><br>
    This Machine Learning application predicts whether a customer
    is likely to churn based on their account information.
    <br><br>
    <b>Model Features</b><br>
    • Customer Tenure<br>
    • Monthly Charges<br>
    • Total Charges<br>
    • Technical Support Tickets
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.caption("Built with Python • Pandas • Scikit-learn • Streamlit")

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict whether a customer is likely to leave the service</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="card"><div class="card-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    tenure = st.number_input(
        "📅 Tenure (Months)",
        min_value=0,
        value=1,
        help="How long the customer has been with the company."
    )

    monthly_charges = st.number_input(
        "💳 Monthly Charges",
        min_value=0.0,
        value=50.0,
        step=1.0,
        help="The customer's monthly service charge."
    )


with col2:

    total_charges = st.number_input(
        "💰 Total Charges",
        min_value=0.0,
        value=100.0,
        step=10.0,
        help="Total amount charged to the customer."
    )

    tech_tickets = st.number_input(
        "🛠️ Technical Support Tickets",
        min_value=0,
        value=0,
        step=1,
        help="Number of technical support requests raised by the customer."
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Predict Customer Churn"):

    input_data = pd.DataFrame(columns=columns)

    input_data.loc[0] = 0

    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges
    input_data["numTechTickets"] = tech_tickets

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">⚠️ Customer Likely to Churn</div>
                <p style="color:#a9adba;">
                The model predicts that this customer has a higher
                likelihood of leaving the service.
                </p>
                <div class="probability">
                    {probability:.1%}
                </div>
                <p style="color:#a9adba;">
                Estimated Churn Probability
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">✅ Customer Likely to Stay</div>
                <p style="color:#a9adba;">
                The model predicts that this customer is likely
                to continue using the service.
                </p>
                <div class="probability">
                    {probability:.1%}
                </div>
                <p style="color:#a9adba;">
                Estimated Churn Probability
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
st.markdown(
    """
    <br><br>
    <div style="text-align:center; color:#777; font-size:13px;">
        Customer Churn Prediction • Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
