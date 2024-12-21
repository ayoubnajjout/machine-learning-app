import streamlit as st
import pandas as pd
from io import BytesIO

def create_dataset():
    st.title("Création de Dataset")
    
    # User input for dataset shape
    rows = st.number_input("Nombre de lignes", min_value=1, value=5)
    cols = st.number_input("Nombre de colonnes", min_value=1, value=3)
    
    # User input for column names
    column_names = []
    for i in range(cols):
        column_name = st.text_input(f"Nom de la colonne {i+1}", value=f"Colonne {i+1}")
        column_names.append(column_name)
    
    # Create an empty DataFrame with the specified shape and columns
    df = pd.DataFrame(columns=column_names, index=range(rows))
    
    # Use data_editor to allow the user to edit the DataFrame
    edited_df = st.data_editor(df)
    
    # Display the edited DataFrame
    st.write("Dataset édité:")
    st.write(edited_df)
    
    # Export data
    export_data(edited_df)
    

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def export_data(data: pd.DataFrame):
    st.subheader("Exporter les données")
    
    # Sélection du format d'export
    export_format = st.selectbox(
        "Format d'export",
        ["CSV", "Excel", "JSON"]
    )
    
    try:
        if export_format == "CSV":
            # Export CSV
            csv = data.to_csv(index=False)
            b64 = BytesIO()
            b64.write(csv.encode())
            st.download_button(
                label="📥 Télécharger CSV",
                data=b64.getvalue(),
                file_name="dataset.csv",
                mime='text/csv',
                use_container_width=True
            )
        
        elif export_format == "Excel":
            # Export Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                data.to_excel(writer, index=False)
            st.download_button(
                label="📥 Télécharger Excel",
                data=output.getvalue(),
                file_name="dataset.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                use_container_width=True
            )
        
        else:  # JSON
            # Export JSON
            json_str = data.to_json(orient='records')
            st.download_button(
                label="📥 Télécharger JSON",
                data=json_str,
                file_name="dataset.json",
                mime='application/json',
                use_container_width=True
            )
    
    except Exception as e:
        st.error(f"Une erreur s'est produite lors de l'export : {str(e)}")

def show():
    create_dataset()
