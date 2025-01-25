import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder # For encoding cube states

st.title("Rubik's Cube Solver Prototype")

# File uploader
data_file = st.file_uploader("Upload your Rubik's Cube CSV file", type=["csv"])

if data_file:
    # Load dataset
    data = pd.read_csv(data_file)
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Assuming your dataset has columns like 'cube_state' and 'solution_steps'
    # Adapt column names to match your actual dataset
    if 'cube_state' not in data.columns or 'solution_steps' not in data.columns:
        st.error("Dataset must contain 'cube_state' and 'solution_steps' columns.")
    else:
        # Preprocess data:  Encode cube states (this is crucial)
        le = LabelEncoder()
        data['encoded_state'] = le.fit_transform(data['cube_state'])

        # Feature and target preparation
        X = data[['encoded_state']].values
        y = data['solution_steps'].values  # Assuming 'solution_steps' is the target

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a model (RandomForestClassifier might not be ideal here)
        model = RandomForestClassifier(n_estimators=100, random_state=42) #Increased n_estimators
        model.fit(X_train, y_train)

        # Predictions and accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader("Model Performance")
        st.write(f"Prediction Accuracy: {accuracy * 100:.2f}%")


        #Prototype solver (very basic example)
        st.subheader("Prototype Solver")
        cube_state_input = st.text_input("Enter cube state (as in your dataset):")
        if st.button("Solve"):
            try:
                encoded_input = le.transform([cube_state_input])
                predicted_solution = model.predict(encoded_input)
                st.write(f"Predicted solution steps: {predicted_solution[0]}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")