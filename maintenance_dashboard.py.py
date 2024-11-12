import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("rul_predictor_model.joblib")  # Replace with your model's path


# Define function to predict RUL
def predict_rul(input_data):
    # Process input data here if needed
    predictions = model.predict(input_data)
    return predictions


# Streamlit App Layout
st.title("Predictive Maintenance Dashboard")

# File uploader to load data
uploaded_file = st.file_uploader("Upload your equipment data", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Display uploaded data
    st.subheader("Uploaded Data")
    st.write(data.head())

    # Predict RUL
    st.subheader("Predicted Remaining Useful Life (RUL)")
    predictions = predict_rul(data)
    data["Predicted_RUL"] = predictions
    st.write(data[["unit_number", "cycle", "Predicted_RUL"]])  # Display results

    # Plot RUL predictions
    st.subheader("RUL Distribution")
    plt.hist(predictions, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Predicted RUL")
    plt.ylabel("Frequency")
    st.pyplot(plt)

    # Add maintenance recommendations based on RUL values
    def maintenance_recommendation(rul):
        if rul < 20:
            return "Immediate Maintenance"
        elif rul < 50:
            return "Scheduled Maintenance"
        else:
            return "In Good Condition"

    data["Maintenance Recommendation"] = data["Predicted_RUL"].apply(
        maintenance_recommendation
    )
    st.subheader("Maintenance Recommendations")
    st.write(data[["unit_number", "Predicted_RUL", "Maintenance Recommendation"]])
