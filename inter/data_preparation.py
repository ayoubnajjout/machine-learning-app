import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from io import BytesIO
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

from inter.training import encode_data


def handle_missing_values(data: pd.DataFrame):
    st.subheader("Analyse des valeurs manquantes")


    missing_data = data.isnull().sum()
    total_missing = missing_data.sum()
    st.write(f"**Total des valeurs manquantes : {total_missing}**")

    if total_missing == 0:
        st.success("Aucune valeur manquante détectée.")
        return


    missing_cols = missing_data[missing_data > 0]
    st.dataframe(missing_cols, use_container_width=True, key="missing_cols_df")

    st.subheader("Options pour gérer les valeurs manquantes")
    

    numeric_cols = data.select_dtypes(include=["number"]).columns
    categorical_cols = data.select_dtypes(exclude=["number"]).columns


    if len(numeric_cols) > 0:
        st.write("Colonnes numériques avec des valeurs manquantes :", numeric_cols.tolist())
        numeric_option = st.radio(
            "Méthode pour colonnes numériques :",
            ["Remplir avec la moyenne", "Remplir avec la médiane", "Supprimer les lignes", "Ne rien faire"],
            key="numeric_option"
        )
    else:
        st.info("Aucune colonne numérique avec des valeurs manquantes.")

    if len(categorical_cols) > 0:
        st.write("Colonnes catégoriques avec des valeurs manquantes :", categorical_cols.tolist())
        categorical_option = st.radio(
            "Méthode pour colonnes catégoriques :",
            ["Remplir avec le mode", "Supprimer les lignes", "Ne rien faire"],
            key="categorical_option"
        )
    else:
        st.info("Aucune colonne catégorique avec des valeurs manquantes.")


    if st.button("Appliquer les méthodes choisies"):

        if len(numeric_cols) > 0:
            if numeric_option == "Remplir avec la moyenne":
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
                st.success("Colonnes numériques remplies avec la moyenne.")
            elif numeric_option == "Remplir avec la médiane":
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
                st.success("Colonnes numériques remplies avec la médiane.")
            elif numeric_option == "Supprimer les lignes":
                data.dropna(subset=numeric_cols, inplace=True)
                st.success("Lignes avec des valeurs manquantes dans les colonnes numériques supprimées.")


        if len(categorical_cols) > 0:
            if categorical_option == "Remplir avec le mode":
                data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
                st.success("Colonnes catégoriques remplies avec le mode.")
            elif categorical_option == "Supprimer les lignes":
                data.dropna(subset=categorical_cols, inplace=True)
                st.success("Lignes avec des valeurs manquantes dans les colonnes catégoriques supprimées.")


        st.subheader("Aperçu des données après traitement")
        st.dataframe(data.head(), key="missing_values_preview")



def normalize_or_standardize(data: pd.DataFrame, target_col: str = None):
    st.subheader("Normalisation ou Standardisation des Colonnes Numériques")


    numeric_cols = data.select_dtypes(include=["number"]).columns

    if len(numeric_cols) > 0:
        st.write("Colonnes numériques disponibles :", numeric_cols.tolist())
        

        exclude_cols = st.multiselect(
            "Choisir les colonnes à exclure de la transformation (ex. colonne cible, colonnes discrètes) :", 
            options=numeric_cols,
            default=[target_col] if target_col else []
        )
        

        cols_to_transform = [col for col in numeric_cols if col not in exclude_cols]


        if len(cols_to_transform) == 0:
            st.warning("Aucune colonne à transformer après exclusion des colonnes sélectionnées.")
            return
        
        st.write("Colonnes sélectionnées pour la transformation :", cols_to_transform)
        

        method = st.radio(
            "Méthode de transformation :",
            ["Normaliser (Min-Max)", "Standardiser (Z-score)"]
        )
        

        exclude_discrete = st.checkbox("Exclure les colonnes avec des valeurs discrètes (par exemple, 0, 1, 2) ?", value=True)
        
        if exclude_discrete:

            cols_to_exclude = []
            for col in cols_to_transform:
                if data[col].nunique() <= 3: 
                    cols_to_exclude.append(col)
            

            cols_to_transform = [col for col in cols_to_transform if col not in cols_to_exclude]
            if len(cols_to_transform) == 0:
                st.warning("Toutes les colonnes sélectionnées ont été exclues en raison de leurs valeurs discrètes.")
                return
            
            st.write(f"Colonnes après exclusion des colonnes discrètes : {cols_to_transform}")
        

        if st.button("Appliquer la transformation"):
            st.write(f"Transformation appliquée aux colonnes : {cols_to_transform}")
            

            if method == "Normaliser (Min-Max)":
                scaler = MinMaxScaler()
                data[cols_to_transform] = scaler.fit_transform(data[cols_to_transform])
                st.success(f"Colonnes sélectionnées normalisées avec Min-Max.")

            elif method == "Standardiser (Z-score)":
                scaler = StandardScaler()
                data[cols_to_transform] = scaler.fit_transform(data[cols_to_transform])
                st.success(f"Colonnes sélectionnées standardisées avec Z-score.")
            

            st.subheader("Aperçu des données après transformation")
            st.dataframe(data[cols_to_transform].head(), key="normalized_preview")
        else:
            st.info("Cliquez sur 'Appliquer la transformation' pour effectuer l'opération.")
        
    else:
        st.info("Aucune colonne numérique disponible pour la transformation.")


def encode_categorical_columns(data: pd.DataFrame):
    st.subheader("Encodage des Colonnes Catégoriques")


    categorical_cols = data.select_dtypes(exclude=["number"]).columns

    if len(categorical_cols) > 0:
        st.write("Colonnes catégoriques disponibles :", categorical_cols.tolist())
        

        encode_all = st.checkbox("Encoder toutes les colonnes catégoriques")
        
        if encode_all:
            selected_cols = categorical_cols.tolist()
        else:

            selected_cols = st.multiselect(
                "Sélectionner les colonnes à encoder :", 
                categorical_cols.tolist()
            )
        

        if st.button("Encoder les colonnes sélectionnées"):
            if selected_cols:
                encoder = LabelEncoder()
                for col in selected_cols:
                    data[col] = encoder.fit_transform(data[col])
                st.success(f"Colonnes encodées : {', '.join(selected_cols)}")
                

                st.subheader("Aperçu des données après encodage")
                st.dataframe(data[selected_cols].head(), key="encoded_preview")
            else:
                st.warning("Veuillez sélectionner au moins une colonne pour l'encodage.")
    else:
        st.info("Aucune colonne catégorique disponible pour l'encodage.")



def delete_columns(data: pd.DataFrame):
    st.subheader("Suppression de Colonnes")


    all_columns = data.columns.tolist()
    st.write("Colonnes disponibles dans le dataset :", all_columns)


    columns_to_delete = st.multiselect(
        "Sélectionner les colonnes à supprimer :",
        options=all_columns,
    )


    if columns_to_delete:
        if st.button("Supprimer les colonnes sélectionnées"):

            data.drop(columns=columns_to_delete, inplace=True)
            st.success(f"Les colonnes suivantes ont été supprimées : {columns_to_delete}")
            

            st.subheader("Aperçu des données après suppression des colonnes")
            st.dataframe(data.head(), key="deleted_cols_preview")
    else:
        st.info("Aucune colonne sélectionnée pour suppression.")


def remove_outliers_iqr(data: pd.DataFrame, column: str) -> pd.DataFrame:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR


    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)].copy()


def remove_outliers_zscore(data: pd.DataFrame, column: str, threshold: float = 3) -> pd.DataFrame:
    from scipy.stats import zscore

    z_scores = zscore(data[column])

    return data[np.abs(z_scores) <= threshold].copy()



def remove_outliers_ui(data: pd.DataFrame):
    st.subheader("Suppression des Outliers")

    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_columns:
        st.warning("Aucune colonne numérique trouvée pour gérer les outliers.")
        return


    method = st.radio("Choisissez une méthode pour supprimer les outliers :", ["IQR", "Z-Score"])
    mode = st.radio("Appliquer sur :", ["Une colonne spécifique", "Toutes les colonnes numériques"])

    if mode == "Une colonne spécifique":
        selected_column = st.selectbox("Choisissez une colonne :", numeric_columns)

        if method == "IQR":
            if st.button("Supprimer les outliers avec la méthode IQR"):
                updated_data = remove_outliers_iqr(data, selected_column)
                st.session_state["temp_dataset"] = updated_data
                st.success(f"Outliers supprimés avec la méthode IQR pour la colonne {selected_column}.")
        elif method == "Z-Score":
            threshold = st.slider("Seuil de Z-Score :", 1, 5, 3)
            if st.button("Supprimer les outliers avec la méthode Z-Score"):
                updated_data = remove_outliers_zscore(data, selected_column, threshold)
                st.session_state["temp_dataset"] = updated_data
                st.success(f"Outliers supprimés avec la méthode Z-Score pour la colonne {selected_column}.")

    elif mode == "Toutes les colonnes numériques":
        if method == "IQR":
            if st.button("Supprimer les outliers avec la méthode IQR pour toutes les colonnes numériques"):
                for col in numeric_columns:
                    data = remove_outliers_iqr(data, col)
                st.session_state["temp_dataset"] = data
                st.success("Outliers supprimés avec la méthode IQR pour toutes les colonnes numériques.")
        elif method == "Z-Score":
            threshold = st.slider("Seuil de Z-Score :", 1, 5, 3)
            if st.button("Supprimer les outliers avec la méthode Z-Score pour toutes les colonnes numériques"):
                for col in numeric_columns:
                    data = remove_outliers_zscore(data, col, threshold)
                st.session_state["temp_dataset"] = data
                st.success("Outliers supprimés avec la méthode Z-Score pour toutes les colonnes numériques.")




def manage_duplicates(data: pd.DataFrame):
    st.subheader("Gestion des Doublons")
    

    num_duplicates = data.duplicated().sum()


    st.write(f"Nombre de doublons dans le dataset : {num_duplicates}")

    if num_duplicates > 0:

        if st.button("Supprimer les doublons"):

            data.drop_duplicates(inplace=True)


            st.session_state["temp_dataset"] = data


            st.success(f"{num_duplicates} doublons ont été supprimés.")



def balance_data(data: pd.DataFrame, target_column: str, method='auto'):
    """
    Équilibre les classes dans le dataset en utilisant oversampling ou undersampling.
    """
    if target_column not in data.columns:
        st.error(f"La colonne cible '{target_column}' n'existe pas dans le dataset.")
        return data

    try:

        X = data.drop(columns=[target_column])
        y = data[target_column]


        X = X.apply(pd.to_numeric, errors="coerce")


        if X.isnull().any().any():
            st.error("Certaines colonnes contiennent des valeurs non numériques ou NaN. Veuillez les traiter avant d'appliquer l'équilibrage.")
            return data


        if method == 'auto':
            ratio = min(Counter(y).values()) / max(Counter(y).values())
            if ratio < 0.2:  
                sampler = RandomOverSampler(random_state=42)
                message = "Sur-échantillonnage aléatoire automatique effectué"
            else:
                sampler = RandomUnderSampler(random_state=42)
                message = "Sous-échantillonnage aléatoire automatique effectué"
        elif method == 'random_over':
            sampler = RandomOverSampler(random_state=42)
            message = "Sur-échantillonnage aléatoire effectué"
        elif method == 'random_under':
            sampler = RandomUnderSampler(random_state=42)
            message = "Sous-échantillonnage aléatoire effectué"

        X_resampled, y_resampled = sampler.fit_resample(X, y)


        balanced_data = pd.concat(
            [pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=target_column)], axis=1
        )

        st.success(f"Le dataset a été équilibré avec succès. Nombre de lignes : {len(balanced_data)}. {message}")


        st.subheader("Distribution des classes après équilibrage")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x=y_resampled, ax=ax)
        plt.title('Distribution des classes après équilibrage')
        st.pyplot(fig)

        return balanced_data

    except Exception as e:
        st.error(f"Erreur lors de l'application de l'équilibrage : {str(e)}")
        return data



def add_column_or_row(data: pd.DataFrame):
    st.subheader("Ajouter une Colonne ou une Ligne")
    
    option = st.radio("Choisissez une option :", ["Ajouter une colonne", "Ajouter une ligne"])
    
    if option == "Ajouter une colonne":
        col_name = st.text_input("Nom de la nouvelle colonne")
        if col_name and st.button("Ajouter la colonne"):
            data[col_name] = None
            st.session_state["temp_dataset"] = data
            st.success(f"Colonne '{col_name}' ajoutée avec succès.")
            st.dataframe(data.head(), key="added_col_row_preview")
    
    elif option == "Ajouter une ligne":

        num_columns = len(data.columns)
        num_rows = (num_columns + 2) // 3  
        
        new_row = {}
        

        for row in range(num_rows):

            cols = st.columns(3)
            

            for col_idx in range(3):

                current_idx = row * 3 + col_idx
                

                if current_idx >= num_columns:
                    break
                    

                column_name = data.columns[current_idx]
                if column_name == 'selected':
                    new_row[column_name] = False
                    continue


                col_dtype = data[column_name].dtype
                

                try:
                    if np.issubdtype(col_dtype, np.number):
                        if np.issubdtype(col_dtype, np.integer):
                            value = cols[col_idx].number_input(
                                f"Valeur pour {column_name} (entier)",
                                step=1,
                                value=0
                            )
                        else:
                            value = cols[col_idx].number_input(
                                f"Valeur pour {column_name} (décimal)",
                                step=0.1,
                                value=0.0
                            )
                    elif col_dtype == 'bool':
                        value = cols[col_idx].checkbox(f"Valeur pour {column_name}")
                    elif col_dtype == 'datetime64[ns]':
                        value = cols[col_idx].date_input(f"Valeur pour {column_name}")
                    else:
                        value = cols[col_idx].text_input(f"Valeur pour {column_name}")
                    
                    new_row[column_name] = value
                except Exception as e:
                    st.error(f"Erreur pour la colonne {column_name}: {str(e)}")
                    return
        
        if st.button("Ajouter la ligne"):
            try:

                for col in data.columns:
                    if col != 'selected':
                        new_row[col] = data[col].dtype.type(new_row[col])
                
                # Set selected to False by default
                if 'selected' in data.columns:
                    new_row['selected'] = False
                
                data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
                st.session_state["temp_dataset"] = data
                st.success("Nouvelle ligne ajoutée avec succès.")
                st.dataframe(data.head(), key="added_col_row_preview")
            except Exception as e:
                st.error(f"Erreur lors de l'ajout de la ligne: {str(e)}")

def export_data(data: pd.DataFrame):
    st.subheader("Exporter les données")
    

    data_to_export = data.drop(columns=['selected']) if 'selected' in data.columns else data
    

    export_format = st.selectbox(
        "Format d'export",
        ["CSV", "Excel", "JSON"]
    )
    
    try:
        if export_format == "CSV":

            csv = data_to_export.to_csv(index=False)
            b64 = BytesIO()
            b64.write(csv.encode())
            st.download_button(
                label="📥 Télécharger CSV",
                data=b64.getvalue(),
                file_name="data_prepared.csv",
                mime='text/csv',
                use_container_width=True
            )
        
        elif export_format == "Excel":

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                data_to_export.to_excel(writer, index=False)
            st.download_button(
                label="📥 Télécharger Excel",
                data=output.getvalue(),
                file_name="data_prepared.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                use_container_width=True
            )
        
        else:  

            json_str = data_to_export.to_json(orient='records')
            st.download_button(
                label="📥 Télécharger JSON",
                data=json_str,
                file_name="data_prepared.json",
                mime='application/json',
                use_container_width=True
            )
    
    except Exception as e:
        st.error(f"Une erreur s'est produite lors de l'export : {str(e)}")


def show():
    st.title("Préparation des Données : Gestion des Valeurs Manquantes, Normalisation, Encodage et Suppression des Outliers")


    if "original_dataset" not in st.session_state or st.session_state["original_dataset"] is None:
        st.warning("Veuillez importer un dataset avant de continuer.")
        return


    if "temp_dataset" not in st.session_state:
        st.session_state["temp_dataset"] = st.session_state["original_dataset"].copy()
    

    data = st.session_state["temp_dataset"]

    
    with st.expander("📝 Modification et suppression des lignes"):
        data['selected'] = False
        unique_key = f"main_editor_{hash(str(data.shape))}"
        edited_df = st.data_editor(
            data,
            key=unique_key,
            column_config={
                "selected": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select rows to delete",
                    default=False,
                )
            }
        )

        if st.button("Supprimer les lignes sélectionnées", key="delete_button"):
            filtered_data = edited_df[~edited_df['selected']]
            st.session_state["temp_dataset"] = filtered_data
            st.success("Les lignes sélectionnées ont été supprimées.")
            filtered_key = f"filtered_editor_{hash(str(filtered_data.shape))}"
            st.write("Données après suppression:")
            st.data_editor(
                filtered_data, 
                key=filtered_key,
            )


    with st.expander("🗑️ Suppression de colonnes"):
        delete_columns(data)


    with st.expander("➕ Ajout de colonnes ou lignes"):
        add_column_or_row(data)


    with st.expander("🔄 Gestion des doublons"):
        manage_duplicates(data)


    with st.expander("❓ Gestion des valeurs manquantes"):
        handle_missing_values(data)


    with st.expander("📊 Normalisation et standardisation"):
        normalize_or_standardize(data)


    with st.expander("🔠 Encodage des colonnes catégoriques"):
        encode_categorical_columns(data)


    with st.expander("📉 Gestion des outliers"):
        remove_outliers_ui(data)


    with st.expander("⚖️ Équilibrage des classes"):
        target_column = st.selectbox("Sélectionnez la colonne cible pour l'équilibrage (classification)", data.columns)
        resampling_method = st.selectbox(
            "Choisissez une méthode de rééquilibrage:",
            ['auto', 'random_over', 'random_under'],
            format_func=lambda x: {
                'auto': 'Automatique',
                'random_over': 'Sur-échantillonnage aléatoire',
                'random_under': 'Sous-échantillonnage aléatoire'
            }[x]
        )
        if st.button("Appliquer l'équilibrage", key="apply_balance_button"):
            st.session_state["temp_dataset"] = balance_data(data, target_column, resampling_method)


    st.write("Aperçu du dataset mis à jour :")
    preview_data = data.drop(columns=['selected']) if 'selected' in data.columns else data
    st.dataframe(preview_data, key="updated_dataset_preview")
    export_data(data)


    col1, col2 = st.columns(2)
    with col1:
        if st.button("❌ Annuler les modifications", key="cancel_bottom", type="secondary", use_container_width=True):
            st.session_state["temp_dataset"] = st.session_state["original_dataset"].copy()
            st.success("Modifications annulées!")
    with col2:
        if st.button("💾 Sauvegarder les modifications", key="save_bottom", type="primary", use_container_width=True):

            if 'selected' in st.session_state["temp_dataset"].columns:
                st.session_state["temp_dataset"] = st.session_state["temp_dataset"].drop(columns=['selected'])
            
            st.session_state["original_dataset"] = st.session_state["temp_dataset"].copy()
            st.session_state["dataset"] = st.session_state["temp_dataset"].copy()  
            st.success("Modifications sauvegardées avec succès!")
            st.session_state["encoded_dataset"] = encode_data(st.session_state["original_dataset"].copy())

