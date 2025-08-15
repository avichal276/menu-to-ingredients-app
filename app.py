import streamlit as st
import requests
import pandas as pd
import os
from io import StringIO

# Load Hugging Face API key
HF_API_KEY = st.secrets["HF_API_KEY"]

# Hugging Face model endpoint
HF_MODEL = "gpt2"  # You can change to another model if needed
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def query_hf_model(prompt):
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            return f"Error from Hugging Face: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

st.title("📋 Menu → Ingredients & Quality Parameters")

# File uploader
uploaded_file = st.file_uploader("Upload your restaurant menu (TXT or CSV)", type=["txt", "csv"])

if uploaded_file:
    # Read the uploaded file content
    if uploaded_file.type == "text/plain":
        menu_text = uploaded_file.read().decode("utf-8")
    else:
        df = pd.read_csv(uploaded_file)
        menu_text = "\n".join(df.iloc[:, 0].astype(str).tolist())  # Assuming first column has menu items

    st.subheader("Menu Uploaded:")
    st.text(menu_text)

    # Generate prompt for Hugging Face model
    prompt = f"""
    For the following restaurant menu items, list the ingredients with quantity and quality check parameters for each:
    {menu_text}
    """

    if st.button("Generate Ingredients & Quality Parameters"):
        output = query_hf_model(prompt)

        st.subheader("Generated Output:")
        st.write(output)

        # Save output to CSV
        output_file = "output.csv"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)

        st.download_button("Download as CSV", data=output, file_name="ingredients_output.csv")

else:
    st.info("Please upload your menu file to continue.")
