import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.utils.multiclass import unique_labels


if 'target_column' not in st.session_state:
    st.session_state.target_column = None

def preprocess_data(df):

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    

    if df[numeric_cols].isnull().any().any():
        num_imputer = SimpleImputer(strategy='mean')
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    
    if df[categorical_cols].isnull().any().any():
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    

    if df[numeric_cols].std().mean() > 1:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def encode_data(df):
    """Encode categorical variables"""
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    encoders = {}
    le = LabelEncoder()
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])
    return df, encoders

def check_class_distribution(y):
    value_counts = y.value_counts()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.title('Distribution des classes')
    plt.xticks(rotation=45)  
    return plt.gcf(), value_counts

def balance_data(X, y, method='auto'):
    """Balance dataset using specified method"""
    try:
        if method == 'auto':

            ratio = min(Counter(y).values()) / max(Counter(y).values())
            if ratio < 0.2: 
                sampler = RandomOverSampler(random_state=42)
                message = "Sur-échantillonnage aléatoire automatique effectué"
            else:
                sampler = RandomUnderSampler(random_state=42)
                message = "Sous-échantillonnage aléatoire automatique effectué"
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            return X_resampled, y_resampled, message
            
        elif method == 'random_over':
            sampler = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            return X_resampled, y_resampled, "Sur-échantillonnage aléatoire effectué"
        
        elif method == 'random_under':
            sampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            return X_resampled, y_resampled, "Sous-échantillonnage aléatoire effectué"
            
    except Exception as e:
        return None, None, str(e)



def detect_problem_type(y):
    """Detect if we have a classification or regression problem"""
    

    if y.dtype.name == 'category' or y.nunique() <= 10:
        return "classification"
    

    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() > len(y) * 0.1:
            return "regression"
    

    return "classification"

def get_models(problem_type):
    """Return appropriate models based on problem type"""
    if problem_type == "classification":
        return {
            "Régression Logistique": LogisticRegression(),
            "SVM": SVC(),
            "Arbre de Décision": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Naive Bayes": GaussianNB()
        }
    else:  
        return {
            "Régression Linéaire": LinearRegression(),
            "SVR": SVR(),
            "Arbre de Décision": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "AdaBoost": AdaBoostRegressor(),
            "Gradient Boosting": GradientBoostingRegressor()
        }

def show():
    st.title("Entraînement du modèle")


    if 'dataset' not in st.session_state or st.session_state.dataset is None:
        st.warning("Veuillez d'abord charger vos données dans l'onglet Importer les données.")
        return
    

    if 'encoded_dataset' not in st.session_state or st.session_state.dataset is not st.session_state.encoded_dataset:
        st.session_state.encoded_dataset, st.session_state.encoders = encode_data(st.session_state.dataset.copy())


    if st.button("Prétraitement automatique des données"):
        with st.spinner('Prétraitement automatique des données...'):
            st.session_state.preprocessed_dataset = preprocess_data(st.session_state.encoded_dataset.copy())
        st.success('Données prétraitées avec succès!')


    st.subheader("Aperçu des données prétraitées")
    if st.session_state.encoded_dataset is not None:
        st.dataframe(st.session_state.encoded_dataset)
    else:
        st.error("Les données encodées sont introuvables.")

    selected2 = option_menu(None, ["Supervised Learning", "Unsupervised Learning"], 
        icons=['book-open', 'chart-line'], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    
    if selected2 == "Supervised Learning":

        st.subheader("Sélection de la variable cible")
        if st.session_state.encoded_dataset is not None:
            target_column = st.selectbox(
                "Choisissez la colonne cible",
                st.session_state.encoded_dataset.columns
            )
            st.session_state.target_column = target_column 
        else:
            st.error("Les données encodées sont introuvables.")
            return

        if target_column:

            X = st.session_state.encoded_dataset.drop(columns=[target_column])
            y = st.session_state.encoded_dataset[target_column]


            X = preprocess_data(X)
            y = preprocess_data(pd.DataFrame(y)).iloc[:, 0]


            problem_type = detect_problem_type(y)
            st.session_state.problem_type = problem_type  
            st.info(f"Type de problème détecté: {problem_type.title()}")

            if problem_type == "classification":
                
                fig, class_counts = check_class_distribution(y)
                st.pyplot(fig)


                class_distribution = Counter(y)
                min_samples = min(class_distribution.values())
                max_samples = max(class_distribution.values())
                
                if max_samples / min_samples > 1.5:
                    st.warning("Données déséquilibrées détectées...")
                    resampling_method = st.selectbox(
                        "Choisissez une méthode de rééquilibrage:",
                        ['auto', 'random_over', 'random_under'],
                        format_func=lambda x: {
                            'auto': 'Automatique',
                            'random_over': 'Sur-échantillonnage aléatoire',
                            'random_under': 'Sous-échantillonnage aléatoire'
                        }[x]
                    )
                    X_resampled, y_resampled, message = balance_data(X, y, resampling_method)
                    if X_resampled is not None:
                        X, y = X_resampled, y_resampled
                        fig_new, _ = check_class_distribution(pd.Series(y_resampled))
                        st.pyplot(fig_new)
                        st.success(f"Données rééquilibrées avec succès! {message}")


            test_size = st.slider(
                "Taille de l'ensemble de test (%)", 
                min_value=10, 
                max_value=40, 
                value=20, 
                step=5
            ) / 100


            models = get_models(problem_type)
            selected_model = st.selectbox(
                "Sélectionnez un modèle",
                list(models.keys())
            )
            
            st.session_state.selected_model = models[selected_model]


            if st.button("Entraîner le modèle"):

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    random_state=42
                )


                model = st.session_state.selected_model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)


                st.session_state.trained_model = model
                st.session_state.target_column = target_column

                if problem_type == "classification":
                    le = LabelEncoder()
                    le.fit(st.session_state.dataset[target_column])
                    st.session_state.label_encoder = le

                # Store encoders in session state
                st.session_state.feature_encoders = st.session_state.encoders

                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test


                st.info(f"Taille des ensembles - Entraînement: {len(X_train)} échantillons, Test: {len(X_test)} échantillons")


                st.subheader("Évaluation du modèle")
                if problem_type == "classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    cm = confusion_matrix(y_test, y_pred)
                    display_labels = unique_labels(y_test, y_pred)
                    cmd = ConfusionMatrixDisplay(cm, display_labels=display_labels)

                    st.write(f"**Exactitude:** {accuracy:.2f}")
                    st.write(f"**Précision:** {precision:.2f}")
                    st.write(f"**Rappel:** {recall:.2f}")
                    st.subheader("Matrice de confusion")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    cmd.plot(ax=ax)
                    st.pyplot(fig)


                    st.subheader("Rapport de classification")
                    target_names = [str(cls) for cls in model.classes_]
                    report = classification_report(y_test, y_pred, target_names=target_names, labels=model.classes_, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
                else:  
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    st.write(f"**Erreur quadratique moyenne (MSE):** {mse:.2f}")
                    st.write(f"**Coefficient de détermination (R²):** {r2:.2f}")


                    plt.figure(figsize=(10, 5))
                    plt.scatter(y_test, y_pred, alpha=0.5)
                    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                    plt.xlabel('Valeurs réelles')
                    plt.ylabel('Valeurs prédites')
                    plt.title('Valeurs réelles vs. Valeurs prédites')
                    st.pyplot(plt.gcf())


                    st.subheader("Rapport de régression")
                    regression_report = {
                        "MSE": mse,
                        "R²": r2
                    }
                    report_df = pd.DataFrame(regression_report, index=[0])
                    st.dataframe(report_df)

    elif selected2 == "Unsupervised Learning":
        st.subheader("KMeans Clustering")
        num_clusters = st.slider("Nombre de clusters", min_value=2, max_value=10, value=3, step=1)
        
        if st.button("Appliquer KMeans"):
            X = st.session_state.encoded_dataset
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(X)
            st.session_state.encoded_dataset['Cluster'] = kmeans.labels_
            st.success(f"KMeans clustering appliqué avec {num_clusters} clusters.")
            
            st.subheader("Aperçu des clusters")
            st.dataframe(st.session_state.encoded_dataset)
            
            st.subheader("Visualisation des clusters")
            plt.figure(figsize=(10, 5))
            sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=kmeans.labels_, palette="viridis")
            plt.title("Clusters KMeans")
            st.pyplot(plt.gcf())
