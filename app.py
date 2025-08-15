import streamlit as st
import pandas as pd
import requests
import io

# ---------------------------
# STREAMLIT APP CONFIG
# ---------------------------
st.set_page_config(page_title="Menu to Ingredients", layout="centered")
st.title("🍽️ Menu to Ingredients List")

# ---------------------------
# SETTINGS
# ---------------------------
HF_API_KEY = st.secrets["HF_API_KEY"]  # Add your key in Streamlit secrets

MODEL_OPTIONS = {
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
    "Falcon-7B": "tiiuae/falcon-7b-instruct"
}

# ---------------------------
# FUNCTION TO CALL HUGGING FACE
# ---------------------------
def query_hf(model_id, prompt, temperature):
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": 1024,
            "return_full_text": False
        }
    }

    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code != 200:
        return f"Error from Hugging Face: {response.text}"
    
    try:
        return response.json()[0]["generated_text"]
    except Exception:
        return "Error: Could not parse model response"

# ---------------------------
# USER INPUTS
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload your menu file (.txt or .csv)", type=["txt", "csv"])
model_choice = st.selectbox("🤖 Choose AI Model", list(MODEL_OPTIONS.keys()))
temperature = st.slider("🎨 Creativity (Temperature)", 0.0, 1.0, 0.3, 0.1)

# ---------------------------
# PROCESSING
# ---------------------------
if uploaded_file is not None:
    if st.button("Generate Ingredients List"):
        # Read menu content
        if uploaded_file.name.endswith(".txt"):
            menu_text = uploaded_file.read().decode("utf-8")
        else:
            df_menu = pd.read_csv(uploaded_file)
            menu_text = "\n".join(df_menu.iloc[:, 0].astype(str))

        # Create prompt
        prompt = f"""
        You are a chef's assistant.
        For each menu item listed below, output a CSV table with:
        Item Name, Ingredients, Quantity (with units), Quality Check Parameters.

        Menu:
        {menu_text}

        Output format:
        Item Name, Ingredients, Quantity, Quality Check Parameters
        """

        # Get AI output
        model_id = MODEL_OPTIONS[model_choice]
        ai_output = query_hf(model_id, prompt, temperature)

        # Try parsing into CSV
        try:
            df = pd.read_csv(io.StringIO(ai_output))
            st.success("✅ Parsed AI output into table")
            st.dataframe(df)

            # Download button
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Ingredients CSV",
                data=csv_bytes,
                file_name="ingredients_list.csv",
                mime="text/csv"
            )
        except Exception:
            st.warning("⚠️ Could not parse AI output into table. Showing raw text:")
            st.text(ai_output)
