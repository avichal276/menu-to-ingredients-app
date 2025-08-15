import streamlit as st
import pandas as pd
import requests
import json
import datetime
import io
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import tempfile
import os

# Load API keys & Google Drive settings
HF_API_KEY = st.secrets["HF_API_KEY"]
GDRIVE_CREDENTIALS_JSON = st.secrets["gdrive"]["credentials_json"]
GDRIVE_FOLDER_NAME = st.secrets["gdrive"]["folder_name"]

# Hugging Face Model
HF_MODEL = "tiiuae/falcon-7b-instruct"  # You can change this to another free model

st.set_page_config(page_title="Menu → Ingredients App", layout="centered")
st.title("🍽️ Menu → Ingredients + QC Generator")

st.write("Upload your restaurant menu and get a detailed list of ingredients, quantities, and quality check parameters.")

# File uploader
uploaded_file = st.file_uploader("Upload menu file (TXT, CSV, XLSX)", type=["txt", "csv", "xlsx"])

# Hugging Face API call
def query_hf(prompt):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt}
    response = requests.post(f"https://api-inference.huggingface.co/models/{HF_MODEL}", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        st.error(f"Hugging Face API error: {response.text}")
        return None

# Google Drive authentication
def init_gdrive():
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as temp_json:
        temp_json.write(GDRIVE_CREDENTIALS_JSON)
        temp_json_path = temp_json.name

    gauth = GoogleAuth()
    gauth.LoadCredentialsFile(temp_json_path)
    if not gauth.credentials:
        gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    # Ensure folder exists
    folder_list = drive.ListFile({'q': f"title='{GDRIVE_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
    if folder_list:
        folder_id = folder_list[0]['id']
    else:
        folder = drive.CreateFile({'title': GDRIVE_FOLDER_NAME, 'mimeType': 'application/vnd.google-apps.folder'})
        folder.Upload()
        folder_id = folder['id']
    return drive, folder_id

# Save file to Google Drive
def save_to_gdrive(file_content, filename):
    drive, folder_id = init_gdrive()
    file_drive = drive.CreateFile({'title': filename, 'parents': [{'id': folder_id}]})
    file_drive.SetContentString(file_content)
    file_drive.Upload()
    return f"https://drive.google.com/file/d/{file_drive['id']}/view?usp=sharing"

if uploaded_file:
    # Read uploaded file
    if uploaded_file.name.endswith(".txt"):
        menu_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".csv"):
        df_menu = pd.read_csv(uploaded_file)
        menu_text = "\n".join(df_menu.iloc[:, 0].astype(str).tolist())
    else:
        df_menu = pd.read_excel(uploaded_file)
        menu_text = "\n".join(df_menu.iloc[:, 0].astype(str).tolist())

    st.subheader("📋 Menu Preview")
    st.text(menu_text)

    if st.button("Generate Ingredients & QC Parameters"):
        with st.spinner("Processing with AI..."):
            prompt = f"""
            You are a chef and food safety expert. For the following restaurant menu, return a table with:
            - Ingredient Name
            - Quantity per serving (approximate)
            - Quality Check Parameters (freshness, size, smell, etc.)
            Menu:
            {menu_text}
            """
            ai_response = query_hf(prompt)
            if ai_response:
                try:
                    # Try to convert AI output into DataFrame
                    df_result = pd.read_csv(io.StringIO(ai_response))
                except:
                    # If AI output is not CSV, just display text
                    st.text(ai_response)
                    df_result = pd.DataFrame({"AI Output": [ai_response]})

                st.subheader("✅ Generated Ingredients List")
                st.dataframe(df_result)

                # Save to CSV
                csv_data = df_result.to_csv(index=False)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ingredients_output_{timestamp}.csv"

                # Download button
                st.download_button("⬇ Download CSV", csv_data, file_name=filename, mime="text/csv")

                # Save to Google Drive
                drive_link = save_to_gdrive(csv_data, filename)
                st.markdown(f"✅ **Saved to Google Drive:** [Open File]({drive_link})", unsafe_allow_html=True)
