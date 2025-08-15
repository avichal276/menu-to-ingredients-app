import streamlit as st
import requests
import pandas as pd
import io

# Load Hugging Face API key
HF_API_KEY = st.secrets["HF_API_KEY"]

# Use a free, public model
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def query_hf_model(prompt, temperature):
    try:
        response = requests.post(API_URL, headers=headers, json={
            "inputs": prompt,
            "parameters": {"temperature": temperature}
        })
        if response.status_code == 200:
            generated = response.json()
            if isinstance(generated, list) and "generated_text" in generated[0]:
                return generated[0]["generated_text"]
            else:
                return str(generated)
        else:
            return f"Error from Hugging Face: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

st.title("📋 Menu → Ingredients & Quality Parameters")

uploaded_file = st.file_uploader("Upload your restaurant menu (TXT or CSV)", type=["txt", "csv"])

# Slider for temperature
temperature = st.slider(
    "AI Creativity (Lower = More Structured CSV, Higher = More Creative)",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.1
)

if uploaded_file:
    if uploaded_file.type == "text/plain":
        menu_text = uploaded_file.read().decode("utf-8")
    else:
        df = pd.read_csv(uploaded_file)
        menu_text = "\n".join(df.iloc[:, 0].astype(str).tolist())

    st.subheader("Menu Uploaded:")
    st.text(menu_text)

    prompt = f"""
    For the following restaurant menu items, create a table in CSV format with these columns:
    Menu Item, Ingredient, Quantity, Quality Check Parameters.
    Make sure each menu item has its own set of rows, one ingredient per line.
    Do NOT include extra commentary, only CSV rows.

    Menu:
    {menu_text}
    """

    if st.button("Generate Ingredients & Quality Parameters"):
        output_text = query_hf_model(prompt, temperature)

        try:
            df_output = pd.read_csv(io.StringIO(output_text))
            st.subheader("Generated Table:")
            st.dataframe(df_output)

            csv_data = df_output.to_csv(index=False)
            st.download_button("Download as CSV", data=csv_data, file_name="ingredients_output.csv")
        except Exception:
            st.subheader("Raw Output (model could not produce clean CSV):")
            st.write(output_text)
            st.warning("The AI output wasn't perfectly structured. You may need to clean it manually.")
else:
    st.info("Please upload your menu file to continue.")
