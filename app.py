import streamlit as st


from inter import create_dataset, welcome, import_data, data_preparation, training, prediction, upload_model

def main():

    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            padding: 2rem 1rem;
        }
        div.stButton > button {
            width: 100%;
            margin-bottom: 10px;
            border: 1px solid #e0e0e0;
            padding: 10px;
            text-align: left;
            font-weight: bold;
            display: flex;
            justify-content: flex-start;
        }
        div.stButton > button:hover {
            background-color: #e0e2e6;
            border-color: #d0d0d0;
        }
        </style>
    """, unsafe_allow_html=True)


    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Bienvenue"
    

    st.sidebar.title("Navigation")
    

    if st.sidebar.button("📝 Bienvenue"):
        st.session_state.current_page = "Bienvenue"
    if st.sidebar.button("📊 Création de dataset"):
        st.session_state.current_page = "Création de dataset"
    if st.sidebar.button("📥 Importer les données"):
        st.session_state.current_page = "Importer les données"
    if st.sidebar.button("🔧 Préparation des données"):
        st.session_state.current_page = "Préparation des données"
    if st.sidebar.button("🎯 Entraînement"):
        st.session_state.current_page = "Entraînement"
    if st.sidebar.button("🔮 Prédiction"):
        st.session_state.current_page = "Prédiction"
    if st.sidebar.button("🔮 Utiliser un modèle existant"):
        st.session_state.current_page = "Utiliser un modèle existant"

    

    if st.session_state.current_page == "Bienvenue":
        welcome.show()
    elif st.session_state.current_page == "Création de dataset":
        create_dataset.show()
    elif st.session_state.current_page == "Importer les données":
        import_data.show()
    elif st.session_state.current_page == "Préparation des données":
        data_preparation.show()
    elif st.session_state.current_page == "Entraînement":
        training.show()
    elif st.session_state.current_page == "Prédiction":
        prediction.show()
    elif st.session_state.current_page == "Utiliser un modèle existant":
        upload_model.show()



if __name__ == "__main__":
    main()