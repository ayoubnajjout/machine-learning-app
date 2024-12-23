import streamlit as st

def show():

    st.title("Bienvenue dans votre plateforme de Machine Learning facile à utiliser")


    st.markdown("""
    Bienvenue sur cette plateforme interactive dédiée à l'apprentissage automatique (Machine Learning). Cette application permet 
    à toute personne, qu'elle soit novice ou ayant une expérience en data science, de développer et d'appliquer des modèles de 
    machine learning sur leurs propres données, sans nécessiter de compétences en programmation.

    **Que pouvez-vous faire ici ?**
    - Importer vos propres jeux de données.
    - Préparer les données pour l'entraînement des modèles.
    - Entraîner différents types de modèles de Machine Learning.
    - Tester et évaluer la performance de vos modèles.
    - Sauvegarder et exporter vos modèles pour une utilisation future.
    
    Cette plateforme simplifie le processus de machine learning et vous guide à chaque étape.
    """)


    st.header("Objectifs de la plateforme")
    st.write("""
    L'objectif de cette plateforme est de rendre le machine learning accessible à tous. Que vous soyez un étudiant, un professionnel 
    ou simplement curieux, vous pouvez créer et tester des modèles de machine learning facilement. Voici ce que vous pouvez accomplir :
    - **Importer des données** : Téléchargez vos propres fichiers CSV ou Excel pour commencer à travailler.
    - **Préparer vos données** : Appliquez des techniques de nettoyage et de transformation sur vos données pour les rendre prêtes à l'emploi.
    - **Entraîner des modèles** : Choisissez parmi plusieurs modèles populaires de machine learning pour entraîner votre propre modèle.
    - **Évaluer les résultats** : Testez les performances de votre modèle sur un jeu de données de test et comprenez ses résultats.
    - **Exporter votre modèle** : Une fois satisfait des résultats, exportez votre modèle pour une utilisation dans vos applications ou projets.
    """)


    st.header("Les étapes pour démarrer")
    st.markdown("""
    Suivez ces étapes simples pour utiliser la plateforme et créer votre propre modèle de machine learning :
    1. **Bienvenue** : Découvrez l'interface et les fonctionnalités de la plateforme.
    2. **Importer les données** : Téléchargez votre jeu de données et préparez-le pour l'analyse.
    3. **Préparation des données** : Nettoyez et transformez vos données pour les rendre prêtes à l’entraînement.
    4. **Entraînement** : Sélectionnez un modèle de machine learning et entraînez-le sur vos données.
    5. **Prédiction et évaluation** : Utilisez le modèle entraîné pour effectuer des prédictions et évaluer ses performances.
    6. **Exporter le modèle** : Sauvegardez le modèle pour l’utiliser ailleurs ou pour d’autres analyses.
    """)


    st.header("Vidéo d'introduction")
    st.write("""
    Découvrez comment utiliser la plateforme en visionnant notre vidéo d'introduction. 
    Cette vidéo vous guide à travers toutes les étapes, de l'importation des données à l'exportation du modèle.
    """)
    

    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 
    st.video(video_url)


    st.header("Téléchargez la documentation complète")
    st.write("""
    Pour en savoir plus, téléchargez la documentation PDF détaillée de la plateforme. Ce guide contient des informations sur 
    l'utilisation de chaque fonctionnalité, les modèles de machine learning disponibles, et des conseils pour obtenir les meilleurs 
    résultats.
    """)
    

    with open("assets/documentation.pdf", "rb") as pdf_file:
        st.download_button(
            label="Télécharger la documentation PDF",
            data=pdf_file,
            file_name="documentation_machine_learning.pdf",
            mime="application/pdf"
        )


    st.header("Contact et support")
    st.write("""
    Si vous avez des questions ou des problèmes, notre équipe est là pour vous aider. N'hésitez pas à nous contacter :
    **support@mlplatform.com**.
    Nous nous ferons un plaisir de vous accompagner dans vos projets de machine learning.
    """)
