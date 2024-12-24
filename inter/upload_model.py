from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import pickle

def encode_prediction_data(df, encoders):
    """Encode prediction data using stored encoders"""
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])
    return df

def show():
    st.title("Utiliser un modèle existant")

    st.subheader("Charger un modèle existant")
    uploaded_model = st.file_uploader("Charger un fichier de modèle (.pkl)", type="pkl")
    if uploaded_model is not None:
        try:
            data = pickle.load(uploaded_model)
            model = data['model']
            columns = data['columns']
            problem_type = data['problem_type']
            feature_encoders = data.get('feature_encoders', {})
            
            if problem_type == "clustering":
                num_clusters = data.get('num_clusters')
                cluster_centers = data.get('cluster_centers')
                st.success(f"Modèle de clustering chargé avec succès. ({num_clusters} clusters)")
                
                st.subheader("Centres des clusters")
                centers_df = pd.DataFrame(cluster_centers, 
                                        columns=columns,
                                        index=[f"Cluster {i}" for i in range(num_clusters)])
                st.dataframe(centers_df)
            else:
                target_column = data['target_column']
                label_encoder = data.get('label_encoder')
                st.success("Modèle supervisé chargé avec succès.")
            
            dataset_sample = pd.DataFrame(data['dataset_sample'])
            st.write("Modèle prêt à être utilisé pour les prédictions.")
            
            st.subheader("Entrée des données pour la prédiction")
            input_data = {}
            
            # Determine which columns to use based on problem type
            if problem_type == "clustering":
                columns_to_use = columns
            else:
                columns_to_use = [col for col in columns if col != target_column]
            
            for col in columns_to_use:
                if col in dataset_sample and dataset_sample[col].dtype == 'object':
                    unique_values = dataset_sample[col].unique()
                    st.write(f"Valeurs possibles pour {col} : {unique_values}")
                input_data[col] = st.text_input(f"Valeur pour {col}")

            if st.button("Faire une prédiction"):
                try:
                    input_df = pd.DataFrame([input_data])
                    input_df = encode_prediction_data(input_df, feature_encoders)
                    
                    if problem_type == "clustering":
                        cluster = model.predict(input_df)[0]
                        st.success(f"Cluster prédit : {cluster}")
                        
                    else:
                        prediction = model.predict(input_df)
                        if problem_type == "classification" and label_encoder:
                            predicted_class = label_encoder.inverse_transform(prediction)
                            st.success(f"Classe prédite : {predicted_class[0]}")
                        else:
                            st.success(f"Valeur prédite : {prediction[0]}")
                            
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {str(e)}")
                    st.write("Données d'entrée :", input_df)
                    
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle : {str(e)}")
    else:
        st.warning("Veuillez charger un modèle pour continuer.")
