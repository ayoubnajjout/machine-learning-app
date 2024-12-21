import streamlit as st
import pandas as pd
import numpy as np
from inter.training import encode_data
from sklearn.preprocessing import LabelEncoder

def show():
    st.title("Prédiction avec le modèle entraîné")

    # Check if the model is trained and available
    if 'trained_model' not in st.session_state or st.session_state.trained_model is None:
        st.warning("Veuillez entraîner un modèle dans l'onglet Entraînement avant de faire des prédictions.")
        return

    # Check if the dataset is available
    if 'dataset' not in st.session_state or st.session_state.dataset is None:
        st.warning("Veuillez importer et préparer vos données avant de faire des prédictions.")
        return

    # Get the trained model and problem type
    model = st.session_state.trained_model
    problem_type = st.session_state.problem_type
    target_column = st.session_state.target_column

    # Input data for prediction
    st.subheader("Entrée des données pour la prédiction")
    input_data = {}
    for col in st.session_state.dataset.columns:
        if col != target_column:
            if st.session_state.dataset[col].dtype == 'object':
                unique_values = st.session_state.dataset[col].unique()
                st.write(f"Valeurs possibles pour {col} : {unique_values}")
            input_data[col] = st.text_input(f"Valeur pour {col}")

    if st.button("Faire une prédiction"):
        try:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])

            # Encode categorical variables
            input_df = encode_data(input_df)

            # Make prediction
            prediction = model.predict(input_df)

            if problem_type == "classification":
                # Decode the predicted class
                le = LabelEncoder()
                le.fit(st.session_state.dataset[target_column])
                predicted_class = le.inverse_transform(prediction)
                st.success(f"Classe prédite : {predicted_class[0]}")
            else:
                st.success(f"Valeur prédite : {prediction[0]}")

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)}")