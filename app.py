import streamlit as st


# Importation des fichiers de pages
from inter import welcome, import_data, data_preparation, training, prediction, export_model

def main():
    # Configuration de la barre latérale
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller à", ("Bienvenue", "Importer les données", "Préparation des données", "Entraînement", "Prédiction", "Exporter le modèle"))
    
    # Afficher le contenu en fonction de la page sélectionnée
    if page == "Bienvenue":
        welcome.show()  # Affiche la page de bienvenue
    elif page == "Importer les données":
        import_data.show()  # Affiche la page d'importation des données
    elif page == "Préparation des données":
        data_preparation.show()  # Affiche la page de préparation des données
    elif page == "Entraînement":
        training.show()  # Affiche la page d'entraînement
    elif page == "Prédiction":
        prediction.show()  # Affiche la page de prédiction
    elif page == "Exporter le modèle":
        export_model.show()  # Affiche la page d'exportation du modèle

# Lancer l'application
if __name__ == "__main__":
    main()