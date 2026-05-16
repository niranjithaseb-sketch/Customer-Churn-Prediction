# Customer Churn Prediction

A Machine Learning web app that predicts whether a customer is likely to churn or stay.

## Features
- Customer churn prediction
- Feature importance analysis
- Interactive Streamlit web app
- Machine learning model training
- Data visualization

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Machine Learning Concepts Used
- Classification
- Data preprocessing
- One-hot encoding
- Feature importance
- Model evaluation

## How to Run

Install required libraries:

pip install -r requirements.txt

Run the Streamlit app:

python -m streamlit run app.py

## Project Files

- app.py → Streamlit web app
- churn.py → ML model training
- churn_model.pkl → saved ML model
- model_columns.pkl → saved feature columns
- churndataset.csv → dataset
- requirements.txt → required libraries

## Output

The app predicts whether a customer is likely to churn based on:
- tenure
- monthly charges
- total charges
- support tickets