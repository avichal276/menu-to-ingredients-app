import streamlit as st
import pandas as pd
import requests
import csv
import io

# Hugging Face API key from Streamlit secrets
HF_API_KEY = st.secrets["HF_API_KEY"]

# Hugging Face Model API URLs
API_URLS = {
    "Zephyr-7B": "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha",  # public, no approval
    "Mistral-7B": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",  # needs approval
    "Falcon-7B": "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"  # needs approval
}

# Query Hugging Face Inference API
def query_hf(payload, model_choice):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    API_URL = API_URLS[model_choice]
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return None, f"Error from Hugging Face: {response.status_code} - {response.text}"
    try:
        return response.json()[0]["generated_text"], None
    except Exception as e:
        return None, f"Error parsing response: {e}"

# Convert AI output to DataFrame
def parse_to_csv(ai_text):
    try:
        reader = csv.reader(io.StringIO(ai_text), delimiter=",")
        rows = list(reader)
        if len(rows) < 2:
            return None
        df = pd.DataFrame(rows[1:], columns=rows[0])
        return df
    except Exception:
        return None

# Streamlit UI
st.title("Menu → Ingredients Converter")

uploaded_file = st.file_uploader("Upload your menu (TXT or CSV)", type=["txt", "csv"])
model_choice = st.selectbox("Choose AI Model", ["Zephyr-7B", "Mistral-7B", "Falcon-7B"], index=0)
temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.7)

if uploaded_file is not None:
    file_content = uploaded_file.read().decode("utf-8")

    if st.button("Process"):
        prompt = f"""
        You are a helpful assistant. Convert the following menu items into a CSV table with columns: Item, Ingredients.
        Be concise and accurate.

        Menu:
        {file_content}
        """
        ai_output, error = query_hf({"inputs": prompt, "parameters": {"temperature": temperature}}, model_choice)

        if error:
            st.error(error)
        elif ai_output:
            df = parse_to_csv(ai_output)
            if df is not None:
                st.success("Here’s your parsed table:")
                st.dataframe(df)
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)
                st.download_button("Download CSV", csv_buf.getvalue(), "ingredients.csv", "text/csv")
            else:
                st.warning("Couldn’t parse into table. Showing raw output instead:")
                st.text(ai_output)
