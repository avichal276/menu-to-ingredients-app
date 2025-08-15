import streamlit as st
import pandas as pd
import requests
import csv
import io

# Hugging Face API key from Streamlit secrets
HF_API_KEY = st.secrets["HF_API_KEY"]

# Public, free model
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"

# Function to query Hugging Face
def query_hf(prompt, temperature):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": 1024,
            "return_full_text": False
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return None, f"Error from Hugging Face: {response.status_code} - {response.text}"
    try:
        return response.json()[0]["generated_text"], None
    except Exception as e:
        return None, f"Error parsing response: {e}"

# Function to parse AI output into DataFrame
def parse_to_csv(ai_text):
    try:
        reader = csv.reader(io.StringIO(ai_text))
        rows = list(reader)
        if len(rows) < 2:
            return None
        df = pd.DataFrame(rows[1:], columns=rows[0])
        return df
    except Exception:
        return None

# Streamlit UI
st.title("🍽️ Menu → Ingredients Converter")

uploaded_file = st.file_uploader("Upload your menu (TXT or CSV)", type=["txt", "csv"])
temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.7)

if uploaded_file is not None:
    # Read file content
    if uploaded_file.name.endswith(".txt"):
        menu_text = uploaded_file.read().decode("utf-8")
    else:
        df_menu = pd.read_csv(uploaded_file)
        menu_text = "\n".join(df_menu.iloc[:, 0].astype(str))

    if st.button("Process"):
        prompt = f"""
        You are a helpful assistant. Convert the following menu items into a CSV table with columns:
        Item, Ingredients, Quantity, Quality Parameter.
        Be concise and accurate. Only return CSV content.

        Menu:
        {menu_text}
        """

        ai_output, error = query_hf(prompt, temperature)

        if error:
            st.error(error)
        elif ai_output:
            df = parse_to_csv(ai_output)
            if df is not None:
                st.success("✅ Parsed AI output into table")
                st.dataframe(df)
                # Download button
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)
                st.download_button("📥 Download CSV", csv_buf.getvalue(), "ingredients.csv", "text/csv")
            else:
                st.warning("⚠️ Could not parse into CSV. Showing raw output:")
                st.text(ai_output)
