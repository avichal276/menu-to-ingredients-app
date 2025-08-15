import streamlit as st
import requests

# Primary and fallback models
PRIMARY_MODEL = "google/flan-t5-base"
FALLBACK_MODEL = "bigscience/bloomz-560m"

# Hugging Face API key from Streamlit secrets
HF_API_KEY = st.secrets.get("HF_API_KEY", None)
if HF_API_KEY is None:
    st.error("Please set your Hugging Face API key in Streamlit secrets as HF_API_KEY")
    st.stop()

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def query_huggingface(model, payload):
    """Query the Hugging Face Inference API."""
    url = f"https://api-inference.huggingface.co/models/{model}"
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.warning(f"Model {model} failed: {response.text}")
        return None

def get_model_output(prompt):
    """Try primary model, then fallback model."""
    # Try primary model
    output = query_huggingface(PRIMARY_MODEL, {"inputs": prompt})
    if output:
        return output

    # If failed, try fallback model
    output = query_huggingface(FALLBACK_MODEL, {"inputs": prompt})
    if output:
        return output

    # If both fail
    st.error("Both primary and fallback models failed. Please check your API key and internet connection.")
    return None

# Streamlit UI
st.title("Menu → Ingredients with AI")

menu_text = st.text_area("Paste your restaurant menu here:")
if st.button("Generate Ingredients"):
    if not menu_text.strip():
        st.warning("Please paste a menu first.")
    else:
        with st.spinner("Processing menu..."):
            prompt = f"Given this restaurant menu:\n{menu_text}\nList ingredients with quantities and quality checks for each dish."
            result = get_model_output(prompt)
            if result:
                st.subheader("Generated Output")
                st.write(result)
