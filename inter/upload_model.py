from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import numpy as np
import pickle

from inter.training import encode_data

def show():
    st.title("Utiliser un modèle existant")

    st.subheader("Charger un modèle existant")
    uploaded_model = st.file_uploader("Charger un fichier de modèle (.pkl)", type="pkl")
    if uploaded_model is not None:
        try:
            data = pickle.load(uploaded_model)
            model = data['model']
            columns = data['columns']
            target_column = data['target_column']
            problem_type = data['problem_type']
            dataset = pd.DataFrame(data['dataset'])
            st.success("Modèle chargé avec succès.")
            
            st.write("Modèle prêt à être utilisé pour les prédictions.")
            
            st.subheader("Entrée des données pour la prédiction")
            input_data = {}
            for col in columns:
                if col != target_column:
                    if dataset[col].dtype == 'object':
                        unique_values = dataset[col].unique()
                        st.write(f"Valeurs possibles pour {col} : {unique_values}")
                    input_data[col] = st.text_input(f"Valeur pour {col}")

            if st.button("Faire une prédiction"):
                try:
                    input_df = pd.DataFrame([input_data])
                    input_df = encode_data(input_df)
                    prediction = model.predict(input_df)

                    if problem_type == "classification":
                        le = LabelEncoder()
                        le.fit(dataset[target_column])
                        predicted_class = le.inverse_transform(prediction)
                        st.success(f"Classe prédite : {predicted_class[0]}")
                    else:
                        st.success(f"Valeur prédite : {prediction[0]}")
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {str(e)}")
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle : {str(e)}")
    else:
        st.warning("Veuillez charger un modèle pour continuer.")
