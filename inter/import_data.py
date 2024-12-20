import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show():
    # Titre de la section
    st.title("Importer le Dataset")
    st.markdown("""
    Cette section vous permet d'importer votre dataset. Veuillez télécharger un fichier au format **CSV** ou **Excel**.
    Une fois le fichier chargé, son contenu sera affiché pour confirmation.
    """)

    # Initialisation de la session
    if "dataset" not in st.session_state:
        st.session_state["dataset"] = None

    # Zone de téléchargement
    uploaded_file = st.file_uploader(
        label="Téléchargez votre fichier (CSV ou Excel)",
        type=["csv", "xlsx"],
        help="Assurez-vous que le fichier est correctement formaté avant de l'importer."
    )

    # Gestion du téléchargement et affichage
    if uploaded_file is not None:
        try:
            # Lecture du fichier en fonction du type
            if uploaded_file.name.endswith(".csv"):
                st.session_state["dataset"] = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                st.session_state["dataset"] = pd.read_excel(uploaded_file)

            st.session_state["temp_dataset"] = st.session_state["dataset"].copy()
            st.session_state["preprocessed_dataset"] = None  # Reset preprocessed dataset
            st.success("Fichier téléchargé avec succès !")

        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
            st.session_state["dataset"] = None

    # Affichage des informations sur le dataset (si disponible)
    if st.session_state["dataset"] is not None:
        display_dataset_info()

        # Bouton pour réinitialiser
        if st.button("Réinitialiser le dataset"):
            st.session_state["dataset"] = None
            st.session_state["temp_dataset"] = None
            st.session_state["preprocessed_dataset"] = None  # Reset preprocessed dataset
            st.warning("Dataset réinitialisé. Veuillez recharger un fichier.")

def display_dataset_info():
    """Affiche les informations détaillées du dataset."""
    dataset = st.session_state["dataset"]

    # Aperçu des données
    st.subheader("Aperçu des données")
    st.dataframe(dataset.head(10))  # Afficher les 10 premières lignes
    st.write(f"**Dimensions du dataset** : {dataset.shape[0]} lignes, {dataset.shape[1]} colonnes")
    st.markdown("---")
    # Informations générales sur les colonnes
    st.subheader("Informations générales")
    dtype_counts = dataset.dtypes.value_counts()
    st.write("**Types de colonnes :**")
    for dtype, count in dtype_counts.items():
        st.write(f"- `{dtype}` : {count} colonnes")

    # Colonnes catégoriques et numériques
    categorical_columns = list(dataset.select_dtypes(include=["object", "category"]).columns)
    numeric_columns = list(dataset.select_dtypes(include=["number"]).columns)
    st.write(f"**Colonnes catégoriques :** {categorical_columns}")
    st.write(f"**Colonnes numériques :** {numeric_columns}")

    # Statistiques descriptives
    st.subheader("Statistiques descriptives")
    st.write("Résumé statistique des colonnes numériques :")
    st.dataframe(dataset.describe())

    # Vérification des valeurs manquantes
    st.subheader("Valeurs manquantes")
    missing_values = dataset.isnull().sum()
    if missing_values.any():
        st.write("**Colonnes avec des valeurs manquantes :**")
        st.dataframe(missing_values[missing_values > 0])
    else:
        st.success("Aucune valeur manquante détectée dans le dataset.")

    # Choix des colonnes à afficher
    st.subheader("Visualisation des données")
    st.write("Sélectionnez les colonnes que vous souhaitez visualiser :")
    selected_columns = st.multiselect("Colonnes disponibles", options=categorical_columns + numeric_columns)

    # Affichage des graphiques pour chaque colonne choisie
    for col in selected_columns:
        if col in numeric_columns:
            st.write(f"Distribution des valeurs pour **{col}** (numérique) :")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(dataset[col].dropna(), bins=20, color="skyblue", edgecolor="black")
            ax.set_title(f"Histogramme de {col}")
            ax.set_xlabel("Valeurs")
            ax.set_ylabel("Fréquence")
            st.pyplot(fig)
        elif col in categorical_columns:
            st.write(f"Distribution des valeurs pour **{col}** (catégorielle) :")
            fig, ax = plt.subplots(figsize=(6, 4))
            dataset[col].value_counts().plot(kind="bar", color="lightcoral", edgecolor="black", ax=ax)
            ax.set_title(f"Graphique en barres pour {col}")
            ax.set_xlabel("Catégories")
            ax.set_ylabel("Fréquence")
            st.pyplot(fig)

        # PARTIE 1 : Visualisation 2D - Nuage de points
    st.subheader("Visualisation 2D - Nuage de points")
    # Sélectionner les colonnes pour la partie 2D
    selected_columns1 = st.multiselect("Colonnes disponibles pour le nuage de points 2D", options=numeric_columns)

    if len(selected_columns1) >= 2:
        x_col1 = st.selectbox("Choisissez la première colonne pour l'axe X", selected_columns1)
        y_col1 = st.selectbox("Choisissez la deuxième colonne pour l'axe Y", selected_columns1)

        # Affichage du nuage de points 2D
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(dataset[x_col1], dataset[y_col1], alpha=0.6)
            ax.set_title(f"Nuage de points 2D : {x_col1} vs {y_col1}")
            ax.set_xlabel(x_col1)
            ax.set_ylabel(y_col1)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur lors de la création du nuage de points 2D : {e}")


    # PARTIE 2 : Visualisation 3D - Nuage de points
    st.subheader("Visualisation 3D - Nuage de points")
    # Sélectionner les colonnes pour la partie 3D
    selected_columns2 = st.multiselect("Colonnes disponibles pour le nuage de points 3D", options=numeric_columns)

    if len(selected_columns2) >= 3:
        x_col2 = st.selectbox("Choisissez la première colonne pour l'axe X", selected_columns2)
        y_col2 = st.selectbox("Choisissez la deuxième colonne pour l'axe Y", selected_columns2)
        z_col2 = st.selectbox("Choisissez la troisième colonne pour l'axe Z", selected_columns2)

        # Affichage du nuage de points 3D
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(dataset[x_col2], dataset[y_col2], dataset[z_col2], alpha=0.6)
            ax.set_title(f"Nuage de points 3D : {x_col2}, {y_col2}, {z_col2}")
            ax.set_xlabel(x_col2)
            ax.set_ylabel(y_col2)
            ax.set_zlabel(z_col2)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur lors de la création du nuage de points 3D : {e}")
    # Détection des outliers
    st.subheader("Détection des Outliers")
    outlier_column = st.selectbox(
        "Sélectionnez une colonne pour vérifier les outliers :",
        options=["Choisissez une option"] + numeric_columns,  # Ajouter une option par défaut
        index=0
    )

    if outlier_column != "Choisissez une option":
        try:
            # Calcul des outliers à l'aide de la méthode des quartiles
            Q1 = dataset[outlier_column].quantile(0.25)
            Q3 = dataset[outlier_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = dataset[(dataset[outlier_column] < lower_bound) | (dataset[outlier_column] > upper_bound)]

            # Afficher les résultats
            st.write(f"**Nombre d'outliers détectés dans la colonne {outlier_column} :** {len(outliers)}")
            if not outliers.empty:
                st.dataframe(outliers)
                # Visualisation des outliers
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.boxplot(dataset[outlier_column].dropna(), vert=False)
                ax.set_title(f"Boxplot pour détecter les outliers dans la colonne {outlier_column}")
                st.pyplot(fig)
            else:
                st.success(f"Aucun outlier détecté dans la colonne {outlier_column}.")
        except Exception as e:
            st.error(f"Erreur lors de la détection des outliers : {e}")


    # Option pour détecter les outliers dans toutes les colonnes numériques
    if 1 == 1:
        st.subheader("Détection des Outliers dans toutes les colonnes numériques")
        outliers_info = {}

        try:
            for column in numeric_columns:
                # Calcul des outliers pour chaque colonne
                Q1 = dataset[column].quantile(0.25)
                Q3 = dataset[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = dataset[(dataset[column] < lower_bound) | (dataset[column] > upper_bound)]

                # Sauvegarder les résultats dans un dictionnaire
                outliers_info[column] = len(outliers)

            # Afficher les résultats dans un tableau
            st.write("**Résumé des outliers détectés pour chaque colonne :**")
            outliers_summary = pd.DataFrame({
                "Colonne": numeric_columns,
                "Nombre d'outliers": [outliers_info[col] for col in numeric_columns]
            })
            st.dataframe(outliers_summary)

        except Exception as e:
            st.error(f"Erreur lors de la détection des outliers : {e}")


    # Vérification de l'équilibre
    st.subheader("Vérification de l'équilibre des colonnes catégoriques")
    balance_column = st.selectbox(
        "Sélectionnez une colonne pour vérifier l'équilibre :",
        options=["Choisissez une option"] + categorical_columns + numeric_columns,  # Ajouter une option par défaut
        index=0
    )

    if balance_column != "Choisissez une option":
        try:
            # Comptage des valeurs uniques
            value_counts = dataset[balance_column].value_counts(normalize=True)
            st.write(f"**Répartition des catégories dans la colonne {balance_column} :**")
            st.bar_chart(value_counts)

            # Vérification de l'équilibre
            imbalance_threshold = 0.6  # Définir un seuil arbitraire pour détecter un déséquilibre
            if any(value_counts > imbalance_threshold):
                st.warning(f"La colonne {balance_column} semble déséquilibrée.")
            else:
                st.success(f"La colonne {balance_column} est équilibrée.")
        except Exception as e:
            st.error(f"Erreur lors de la vérification de l'équilibre : {e}")
