import streamlit as st
import requests
import pandas as pd
import re
import io
import time
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# ==============================
# CONFIG
# ==============================
HF_API_KEY = st.secrets["HF_API_KEY"]
HF_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "tiiuae/falcon-7b-instruct"
]
API_BASE = "https://api-inference.huggingface.co/models"

# ==============================
# GOOGLE DRIVE AUTH
# ==============================
@st.cache_resource
def gdrive_auth():
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile("mycreds.txt")
    return GoogleDrive(gauth)

drive = gdrive_auth()

# ==============================
# HUGGING FACE QUERY
# ==============================
def query_hf_model(model, prompt):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        resp = requests.post(f"{API_BASE}/{model}", headers=headers, json={"inputs": prompt})
        if resp.status_code == 200:
            out = resp.json()
            if isinstance(out, list) and "generated_text" in out[0]:
                return out[0]["generated_text"]
            return str(out)
        else:
            return f"Error from Hugging Face: {resp.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# ==============================
# CSV PARSER
# ==============================
def parse_to_csv(output_text):
    try:
        df = pd.read_csv(io.StringIO(output_text))
        if all(col in df.columns for col in ["Ingredient", "Quantity", "Quality Parameter"]):
            return df
        return None
    except:
        return None

# ==============================
# STREAMLIT UI
# ==============================
st.title("📋 Menu → Ingredients & Quality Parameters")
st.write("Upload your menu file, choose model, and get structured ingredient data with quality checks.")

model_choice = st.selectbox("Choose AI Model", HF_MODELS)
temperature = st.slider("Creativity Level (Temperature)", 0.0, 1.0, 0.3, 0.1)

uploaded_file = st.file_uploader("Upload Menu File (TXT or CSV)", type=["txt", "csv"])

if uploaded_file:
    if uploaded_file.type == "text/plain":
        menu_text = uploaded_file.read().decode("utf-8")
    else:
        df = pd.read_csv(uploaded_file)
        menu_text = "\n".join(df.iloc[:, 0].astype(str).tolist())

    st.subheader("Menu Preview")
    st.text(menu_text)

    if st.button("Generate"):
        prompt = f"""
        You are a data extraction engine.
        From the following restaurant menu, create a CSV with EXACTLY 3 columns:
        Ingredient, Quantity, Quality Parameter.
        - Do NOT add any explanations, notes, or text outside the CSV.
        - First row must be the headers: Ingredient,Quantity,Quality Parameter
        - Every ingredient must have a precise quantity with units (e.g., 200g, 50ml)
        - Every ingredient must have a measurable quality parameter (e.g., Fresh, Organic, Grade A, 95% purity)
        - Output must be valid CSV format only.

        Menu:
        {menu_text}
        """

        output = query_hf_model(model_choice, prompt)

        st.subheader("Raw AI Output")
        st.code(output, language="csv")

        df = parse_to_csv(output)
        if df is not None:
            st.subheader("Structured Table")
            st.dataframe(df)

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv_data, file_name="ingredients_output.csv")

            # ==============================
            # SAVE TO GOOGLE DRIVE
            # ==============================
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"ingredients_output_{timestamp}.csv"

            file_drive = drive.CreateFile({"title": filename})
            file_drive.SetContentString(df.to_csv(index=False))
            file_drive.Upload()
            st.success(f"Saved to Google Drive as {filename}")
        else:
            st.error("Model output was not valid CSV. Please retry.")
