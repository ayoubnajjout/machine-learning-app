import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO

# Fonction pour gérer les valeurs manquantes
def handle_missing_values(data: pd.DataFrame):
    st.subheader("Analyse des valeurs manquantes")

    # Afficher un résumé des valeurs manquantes
    missing_data = data.isnull().sum()
    total_missing = missing_data.sum()
    st.write(f"**Total des valeurs manquantes : {total_missing}**")

    if total_missing == 0:
        st.success("Aucune valeur manquante détectée.")
        return

    # Afficher uniquement les colonnes avec des valeurs manquantes
    missing_cols = missing_data[missing_data > 0]
    st.dataframe(missing_cols, use_container_width=True, key="missing_cols_df")

    st.subheader("Options pour gérer les valeurs manquantes")
    
    # Séparer les colonnes en types (numériques vs catégoriques)
    numeric_cols = data.select_dtypes(include=["number"]).columns
    categorical_cols = data.select_dtypes(exclude=["number"]).columns

    # Options par type de colonne
    with st.expander("Colonnes Numériques"):
        if len(numeric_cols) > 0:
            st.write("Colonnes numériques avec des valeurs manquantes :", numeric_cols.tolist())
            numeric_option = st.radio(
                "Méthode pour colonnes numériques :",
                ["Remplir avec la moyenne", "Remplir avec la médiane", "Supprimer les lignes", "Ne rien faire"],
                key="numeric_option"
            )
        else:
            st.info("Aucune colonne numérique avec des valeurs manquantes.")

    with st.expander("Colonnes Catégoriques"):
        if len(categorical_cols) > 0:
            st.write("Colonnes catégoriques avec des valeurs manquantes :", categorical_cols.tolist())
            categorical_option = st.radio(
                "Méthode pour colonnes catégoriques :",
                ["Remplir avec le mode", "Supprimer les lignes", "Ne rien faire"],
                key="categorical_option"
            )
        else:
            st.info("Aucune colonne catégorique avec des valeurs manquantes.")

    # Appliquer les méthodes choisies
    if st.button("Appliquer les méthodes choisies"):
        # Gestion des colonnes numériques
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

        # Gestion des colonnes catégoriques
        if len(categorical_cols) > 0:
            if categorical_option == "Remplir avec le mode":
                data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
                st.success("Colonnes catégoriques remplies avec le mode.")
            elif categorical_option == "Supprimer les lignes":
                data.dropna(subset=categorical_cols, inplace=True)
                st.success("Lignes avec des valeurs manquantes dans les colonnes catégoriques supprimées.")

        # Afficher un aperçu des données après le traitement
        st.subheader("Aperçu des données après traitement")
        st.dataframe(data.head(), key="missing_values_preview")


# Fonction pour normaliser ou standardiser les données, avec exclusions de certaines colonnes
def normalize_or_standardize(data: pd.DataFrame, target_col: str = None):
    st.subheader("Normalisation ou Standardisation des Colonnes Numériques")

    # Séparer les colonnes numériques
    numeric_cols = data.select_dtypes(include=["number"]).columns

    if len(numeric_cols) > 0:
        st.write("Colonnes numériques disponibles :", numeric_cols.tolist())
        
        # Option pour exclure certaines colonnes de la transformation (ex: colonne cible ou colonnes discrètes)
        exclude_cols = st.multiselect(
            "Choisir les colonnes à exclure de la transformation (ex. colonne cible, colonnes discrètes) :", 
            options=numeric_cols,
            default=[target_col] if target_col else []
        )
        
        # Filtrer les colonnes numériques à transformer (en excluant celles choisies)
        cols_to_transform = [col for col in numeric_cols if col not in exclude_cols]

        # Vérifier si après exclusion, il y a encore des colonnes à transformer
        if len(cols_to_transform) == 0:
            st.warning("Aucune colonne à transformer après exclusion des colonnes sélectionnées.")
            return
        
        st.write("Colonnes sélectionnées pour la transformation :", cols_to_transform)
        
        # Demander à l'utilisateur de choisir la méthode de transformation
        method = st.radio(
            "Méthode de transformation :",
            ["Normaliser (Min-Max)", "Standardiser (Z-score)"]
        )
        
        # Demander si on veut exclure les colonnes discrètes (comme 0, 1, 2)
        exclude_discrete = st.checkbox("Exclure les colonnes avec des valeurs discrètes (par exemple, 0, 1, 2) ?", value=True)
        
        if exclude_discrete:
            # Exclure les colonnes qui contiennent des valeurs discrètes (0, 1, 2)
            cols_to_exclude = []
            for col in cols_to_transform:
                if data[col].nunique() <= 3:  # Si le nombre de valeurs uniques <= 3, considérer comme discrètes
                    cols_to_exclude.append(col)
            
            # Filtrer les colonnes à transformer en excluant celles discrètes
            cols_to_transform = [col for col in cols_to_transform if col not in cols_to_exclude]
            if len(cols_to_transform) == 0:
                st.warning("Toutes les colonnes sélectionnées ont été exclues en raison de leurs valeurs discrètes.")
                return
            
            st.write(f"Colonnes après exclusion des colonnes discrètes : {cols_to_transform}")
        
        # Créer un bouton pour appliquer la transformation
        if st.button("Appliquer la transformation"):
            st.write(f"Transformation appliquée aux colonnes : {cols_to_transform}")
            
            # Appliquer la transformation choisie
            if method == "Normaliser (Min-Max)":
                scaler = MinMaxScaler()
                data[cols_to_transform] = scaler.fit_transform(data[cols_to_transform])
                st.success(f"Colonnes sélectionnées normalisées avec Min-Max.")

            elif method == "Standardiser (Z-score)":
                scaler = StandardScaler()
                data[cols_to_transform] = scaler.fit_transform(data[cols_to_transform])
                st.success(f"Colonnes sélectionnées standardisées avec Z-score.")
            
            # Afficher un aperçu des données après transformation
            st.subheader("Aperçu des données après transformation")
            st.dataframe(data[cols_to_transform].head(), key="normalized_preview")
        else:
            st.info("Cliquez sur 'Appliquer la transformation' pour effectuer l'opération.")
        
    else:
        st.info("Aucune colonne numérique disponible pour la transformation.")

# Fonction pour encoder les colonnes catégoriques
def encode_categorical_columns(data: pd.DataFrame):
    st.subheader("Encodage des Colonnes Catégoriques")

    # Séparer les colonnes catégoriques
    categorical_cols = data.select_dtypes(exclude=["number"]).columns

    if len(categorical_cols) > 0:
        st.write("Colonnes catégoriques disponibles :", categorical_cols.tolist())
        
        # Checkbox pour encoder toutes les colonnes
        encode_all = st.checkbox("Encoder toutes les colonnes catégoriques")
        
        if encode_all:
            selected_cols = categorical_cols.tolist()
        else:
            # Demander à l'utilisateur de sélectionner les colonnes à encoder
            selected_cols = st.multiselect(
                "Sélectionner les colonnes à encoder :", 
                categorical_cols.tolist()
            )
        
        # Bouton pour déclencher l'encodage
        if st.button("Encoder les colonnes sélectionnées"):
            if selected_cols:
                encoder = LabelEncoder()
                for col in selected_cols:
                    data[col] = encoder.fit_transform(data[col])
                st.success(f"Colonnes encodées : {', '.join(selected_cols)}")
                
                # Aperçu des données après encodage
                st.subheader("Aperçu des données après encodage")
                st.dataframe(data[selected_cols].head(), key="encoded_preview")
            else:
                st.warning("Veuillez sélectionner au moins une colonne pour l'encodage.")
    else:
        st.info("Aucune colonne catégorique disponible pour l'encodage.")


# Fonction pour supprimer des colonnes
def delete_columns(data: pd.DataFrame):
    st.subheader("Suppression de Colonnes")

    # Lister toutes les colonnes du dataset
    all_columns = data.columns.tolist()
    st.write("Colonnes disponibles dans le dataset :", all_columns)

    # Demander à l'utilisateur de sélectionner les colonnes à supprimer
    columns_to_delete = st.multiselect(
        "Sélectionner les colonnes à supprimer :",
        options=all_columns,
    )

    # Vérifier si l'utilisateur a sélectionné des colonnes
    if columns_to_delete:
        if st.button("Supprimer les colonnes sélectionnées"):
            # Supprimer les colonnes sélectionnées
            data.drop(columns=columns_to_delete, inplace=True)
            st.success(f"Les colonnes suivantes ont été supprimées : {columns_to_delete}")
            
            # Aperçu du dataset après suppression
            st.subheader("Aperçu des données après suppression des colonnes")
            st.dataframe(data.head(), key="deleted_cols_preview")
    else:
        st.info("Aucune colonne sélectionnée pour suppression.")

# # Fonction pour détecter et supprimer les outliers
# def remove_outliers(data: pd.DataFrame):
#     st.subheader("Gestion des Outliers")
    
#     # Filtrer uniquement les colonnes numériques
#     numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()

#     if not numeric_columns:
#         st.warning("Aucune colonne numérique trouvée dans le dataset.")
#         return

#     # Afficher les colonnes numériques
#     st.write("Colonnes numériques disponibles :", numeric_columns)

#     # Choisir une option : supprimer les outliers d'une seule colonne ou de toutes les colonnes
#     option = st.radio(
#         "Choisir une méthode de suppression des outliers :",
#         ("Supprimer les outliers d'une colonne spécifique", "Supprimer les outliers de tout le dataset")
#     )

#     # Choisir une méthode de détection des outliers
#     method = st.selectbox(
#         "Méthode de détection des outliers :",
#         ["IQR (Interquartile Range)", "Z-Score"]
#     )

#     # Appliquer la suppression des outliers
#     if option == "Supprimer les outliers d'une colonne spécifique":
#         column = st.selectbox("Sélectionnez la colonne :", numeric_columns)

#         if st.button("Supprimer les outliers de cette colonne"):
#             initial_size = data.shape[0]

#             if method == "IQR (Interquartile Range)":
#                 data = remove_outliers_iqr(data, column)
#             elif method == "Z-Score":
#                 data = remove_outliers_zscore(data, column)

#             final_size = data.shape[0]
#             st.success(f"Outliers supprimés de la colonne '{column}'. {initial_size - final_size} lignes ont été supprimées.")
#             st.dataframe(data)

#     elif option == "Supprimer les outliers de tout le dataset":
#         if st.button("Supprimer les outliers de tout le dataset"):
#             initial_size = data.shape[0]

#             if method == "IQR (Interquartile Range)":
#                 for column in numeric_columns:
#                     data = remove_outliers_iqr(data, column)
#             elif method == "Z-Score":
#                 for column in numeric_columns:
#                     data = remove_outliers_zscore(data, column)

#             final_size = data.shape[0]
#             st.success(f"Outliers supprimés du dataset entier. {initial_size - final_size} lignes ont été supprimées.")
#             st.dataframe(data)

#     return data


# Méthode 1 : Suppression des outliers avec l'IQR
def remove_outliers_iqr(data: pd.DataFrame, column: str) -> pd.DataFrame:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Retourner les données sans les outliers
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)].copy()

# Méthode 2 : Suppression des outliers avec le Z-Score
def remove_outliers_zscore(data: pd.DataFrame, column: str, threshold: float = 3) -> pd.DataFrame:
    from scipy.stats import zscore

    z_scores = zscore(data[column])
    # Retourner les données sans les outliers
    return data[np.abs(z_scores) <= threshold].copy()


# user interface outliers ...
def remove_outliers_ui(data: pd.DataFrame):
    st.subheader("Suppression des Outliers")

    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_columns:
        st.warning("Aucune colonne numérique trouvée pour gérer les outliers.")
        return

    # Choix de la méthode (IQR ou Z-Score)
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



# Fonction pour gérer les doublons
def manage_duplicates(data: pd.DataFrame):
    st.subheader("Gestion des Doublons")
    
    # Calculer le nombre de doublons
    num_duplicates = data.duplicated().sum()

    # Afficher le nombre de doublons
    st.write(f"Nombre de doublons dans le dataset : {num_duplicates}")

    if num_duplicates > 0:
        # Ajouter un bouton pour supprimer les doublons
        if st.button("Supprimer les doublons"):
            # Supprimer les doublons du dataset
            data.drop_duplicates(inplace=True)

            # Mettre à jour le dataset dans la session
            st.session_state["temp_dataset"] = data

            # Afficher un message de succès
            st.success(f"{num_duplicates} doublons ont été supprimés.")


# balance data
def balance_data_with_smote(data: pd.DataFrame, target_column: str):
    """
    Équilibre les classes dans le dataset en utilisant SMOTE.
    """
    if target_column not in data.columns:
        st.error(f"La colonne cible '{target_column}' n'existe pas dans le dataset.")
        return data

    try:
        # Vérifier que la colonne cible existe et extraire X (caractéristiques) et y (cible)
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Forcer les colonnes à être numériques si nécessaire
        X = X.apply(pd.to_numeric, errors="coerce")

        # Vérifier si des colonnes contiennent des valeurs NaN après conversion
        if X.isnull().any().any():
            st.error("Certaines colonnes contiennent des valeurs non numériques ou NaN. Veuillez les traiter avant d'appliquer SMOTE.")
            return data

        # Appliquer SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Reconstruire le dataset équilibré
        balanced_data = pd.concat(
            [pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=target_column)], axis=1
        )

        st.success(f"Le dataset a été équilibré avec succès. Nombre de lignes : {len(balanced_data)}.")
        return balanced_data

    except Exception as e:
        st.error(f"Erreur lors de l'application de SMOTE : {str(e)}")
        return data


# Fonction pour ajouter une colonne ou une ligne
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
        # Calculate rows needed
        num_columns = len(data.columns)
        num_rows = (num_columns + 2) // 3  # Ceiling division to handle non-divisible numbers
        
        new_row = {}
        
        # Create buttons row by row
        for row in range(num_rows):
            # Create 3 columns for each row
            cols = st.columns(3)
            
            # Fill each column in current row
            for col_idx in range(3):
                # Calculate current column index
                current_idx = row * 3 + col_idx
                
                # Break if we've handled all columns
                if current_idx >= num_columns:
                    break
                    
                # Get column name and create input in appropriate column
                column_name = data.columns[current_idx]
                if column_name == 'selected':
                    new_row[column_name] = False
                    continue

                # Get column data type
                col_dtype = data[column_name].dtype
                
                # Create appropriate input field based on data type
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
                # Convert the values to appropriate types
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
    
    # Drop the 'selected' column before export
    data_to_export = data.drop(columns=['selected']) if 'selected' in data.columns else data
    
    # Sélection du format d'export
    export_format = st.selectbox(
        "Format d'export",
        ["CSV", "Excel", "JSON"]
    )
    
    try:
        if export_format == "CSV":
            # Export CSV
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
            # Export Excel
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
        
        else:  # JSON
            # Export JSON
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

# Fonction principale pour afficher la section
def show():
    st.title("Préparation des Données : Gestion des Valeurs Manquantes, Normalisation, Encodage et Suppression des Outliers")

    # Vérifier si le dataset est défini dans la session
    if "dataset" not in st.session_state or st.session_state["dataset"] is None:
        st.warning("Veuillez importer un dataset avant de continuer.")
        return

    # Créer une copie de travail des données
    if "temp_dataset" not in st.session_state:
        st.session_state["temp_dataset"] = st.session_state["dataset"].copy()
    
    # Utiliser temp_dataset pour toutes les modifications
    data = st.session_state["temp_dataset"]

    # Ajouter les boutons Save/Cancel en haut de la page

    # Modification et suppression des lignes
    st.subheader("Modification et suppression des lignes")
    data['selected'] = False
    
    # ... rest of your existing code, but use 'data' instead of accessing st.session_state["dataset"] directly ...
    # Replace all instances of st.session_state["dataset"] = ... with:
    # st.session_state["temp_dataset"] = ...

    # For example, when deleting rows:
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

    # ... continue with other functions, but make sure they modify temp_dataset instead of dataset ...
    delete_columns(data)
    add_column_or_row(data)
    manage_duplicates(data)
    handle_missing_values(data)
    normalize_or_standardize(data)
    encode_categorical_columns(data)
    remove_outliers_ui(data)

    # SMOTE section
    st.subheader("Équilibrage des Classes avec SMOTE")
    target_column = st.selectbox("Sélectionnez la colonne cible pour l'équilibrage (classification)", data.columns)

    if st.button("Appliquer SMOTE", key="apply_smote_button"):
        st.session_state["temp_dataset"] = balance_data_with_smote(data, target_column)

    
    # Afficher un aperçu final
    st.write("Aperçu du dataset mis à jour :")
    preview_data = data.drop(columns=['selected']) if 'selected' in data.columns else data
    st.dataframe(preview_data, key="updated_dataset_preview")

    # Ajouter les boutons Save/Cancel en bas de la page aussi
    col1, col2 = st.columns(2)
    with col1:
        if st.button("❌ Annuler les modifications", key="cancel_bottom", type="secondary", use_container_width=True):
            st.session_state["temp_dataset"] = st.session_state["dataset"].copy()
            st.success("Modifications annulées!")
            st.rerun()
    with col2:
        if st.button("💾 Sauvegarder les modifications", key="save_bottom", type="primary", use_container_width=True):
            # Drop the 'selected' column before saving
            if 'selected' in st.session_state["temp_dataset"].columns:
                st.session_state["temp_dataset"] = st.session_state["temp_dataset"].drop(columns=['selected'])
            
            st.session_state["dataset"] = st.session_state["temp_dataset"].copy()
            st.success("Modifications sauvegardées avec succès!")

    export_data(data)

