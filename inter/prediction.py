import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from .training import preprocess_data


def show():
    st.title("Prédiction du modèle")

    if 'trained_model' not in st.session_state or 'preprocessed_dataset' not in st.session_state:
        st.error("Veuillez d'abord entraîner un modèle dans l'onglet 'Entraînement'")
        return

    model = st.session_state.trained_model
    target_column = st.session_state.target_column
    columns = [col for col in st.session_state.preprocessed_dataset.columns if col != target_column]

    st.subheader("Entrer les données pour la prédiction")
    input_data = {}
    for col in columns:
        input_data[col] = st.text_input(f"Valeur pour {col}")

    if st.button("Prédire"):
        input_df = pd.DataFrame([input_data])
        input_df = preprocess_data(input_df)
        prediction = model.predict(input_df)
        st.success(f"Prédiction: {prediction[0]}")

