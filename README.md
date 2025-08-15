# 🍽️ Menu → Ingredients + QC Generator

This app takes a restaurant menu and generates:
- Ingredients list
- Quantity per serving
- Quality check parameters

## 🚀 How to Deploy on Streamlit Cloud

1. **Fork this repo** to your GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io/) → "New app" → Select your repo.
3. Set **Python version** to 3.9+.
4. Paste the contents of `requirements.txt` into your app's settings.
5. Go to "Secrets" and paste:

```toml
HF_API_KEY = "your_huggingface_api_key"

[gdrive]
credentials_json = """paste entire credentials.json here"""
folder_name = "Menu_Ingredients_Outputs"
