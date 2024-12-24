import streamlit as st
import pandas as pd
import numpy as np
from inter.training import encode_data
from sklearn.preprocessing import LabelEncoder
import pickle
import base64

def encode_prediction_data(df, encoders):
    """Encode prediction data using stored encoders"""
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])
    return df

def show():
    st.title("Prédiction avec le modèle entraîné")

    if 'trained_model' not in st.session_state or st.session_state.trained_model is None:
        st.warning("Veuillez entraîner un modèle dans l'onglet Entraînement avant de faire des prédictions.")
        return

    if 'dataset' not in st.session_state or st.session_state.dataset is None:
        st.warning("Veuillez importer et préparer vos données avant de faire des prédictions.")
        return

    model = st.session_state.trained_model
    problem_type = st.session_state.problem_type

    st.subheader("Entrée des données pour la prédiction")
    input_data = {}

    if problem_type == "clustering":
        # For clustering, we need all features except the Cluster column
        columns_to_use = [col for col in st.session_state.dataset.columns if col != 'Cluster']
    else:
        # For supervised learning, exclude target column
        target_column = st.session_state.target_column
        columns_to_use = [col for col in st.session_state.dataset.columns if col != target_column]

    for col in columns_to_use:
        unique_values = st.session_state.dataset[col].unique()
        if st.session_state.dataset[col].dtype == 'object' or len(unique_values) <= 10:
            st.write(f"Valeurs possibles pour {col} : {unique_values}")
        input_data[col] = st.text_input(f"Valeur pour {col}")

    if st.button("Faire une prédiction"):
        try:
            input_df = pd.DataFrame([input_data])
            st.write("Données d'entrée pour la prédiction:", input_df)

            input_df = encode_prediction_data(input_df, st.session_state.feature_encoders)
            st.write("Données encodées pour la prédiction:", input_df)

            if problem_type == "clustering":
                cluster = model.predict(input_df)[0]
                st.success(f"Cluster prédit : {cluster}")
                
                # Calculate distance to cluster center
                distances = model.transform(input_df)[0]
                st.write("Distance aux centres des clusters:")
                for i, distance in enumerate(distances):
                    st.write(f"Cluster {i}: {distance:.2f}")
            else:
                # Existing classification/regression prediction code
                prediction = model.predict(input_df)
                if problem_type == "classification":
                    if 'label_encoder' in st.session_state:
                        predicted_class = st.session_state.label_encoder.inverse_transform(prediction)
                        st.success(f"Classe prédite : {predicted_class[0]}")
                    else:
                        st.success(f"Classe prédite : {prediction[0]}")
                else:
                    st.success(f"Valeur prédite : {prediction[0]}")

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)}")

    st.subheader("Exporter le modèle")
    model_name = st.text_input("Nom du fichier du modèle (sans extension)", "model")
    if st.button("Exporter le modèle"):
        try:
            data = {
                'model': model,
                'columns': st.session_state.dataset.columns.tolist(),
                'problem_type': problem_type,
                'feature_encoders': st.session_state.feature_encoders,
            }
            
            if problem_type == "clustering":
                data.update({
                    'num_clusters': st.session_state.num_clusters,
                    'cluster_centers': model.cluster_centers_
                })
            else:
                data.update({
                    'target_column': st.session_state.target_column,
                    'label_encoder': st.session_state.label_encoder if 'label_encoder' in st.session_state else None,
                })
            
            data['dataset_sample'] = st.session_state.dataset.head().to_dict()
            
            with open(f"{model_name}.pkl", "wb") as f:
                pickle.dump(data, f)
            st.success(f"Modèle exporté sous le nom : {model_name}.pkl")

            with open(f"{model_name}.pkl", "rb") as f:
                bytes_data = f.read()
                b64 = base64.b64encode(bytes_data).decode()
                st.download_button(
                    label="Télécharger le modèle",
                    data=bytes_data,
                    file_name=f"{model_name}.pkl",
                    mime="application/octet-stream"
                )
        except Exception as e:
            st.error(f"Erreur lors de l'exportation du modèle : {str(e)}")