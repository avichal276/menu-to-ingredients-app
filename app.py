import streamlit as st
import pandas as pd
import openai
import csv
import io

# Hugging Face token equivalent: OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to query OpenAI GPT
def query_gpt(prompt, temperature):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024
    )
    return response['choices'][0]['message']['content']

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
st.title("🍽️ Menu → Ingredients Converter (OpenAI GPT)")

uploaded_file = st.file_uploader("Upload your menu (TXT or CSV)", type=["txt", "csv"])
temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.7)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".txt"):
        menu_text = uploaded_file.read().decode("utf-8")
    else:
        df_menu = pd.read_csv(uploaded_file)
        menu_text = "\n".join(df_menu.iloc[:,0].astype(str))

    if st.button("Process"):
        prompt = f"""
You are a helpful assistant specialized in restaurant menus. 

Task:
Convert the following menu into a CSV table with these exact columns:
Item, Ingredients, Quantity, Quality Parameter

Instructions:
- Only return CSV content, do NOT include any extra text or explanations.
- Separate multiple ingredients in a single cell with semicolons (;)
- Specify quantity for each ingredient if possible
- Include quality parameters like "Fresh", "Organic", "Good quality" if known
- If some information is missing, leave the cell blank but keep the CSV structure intact
- Make sure the first row is the header

Menu:
{menu_text}
"""

        try:
            ai_output = query_gpt(prompt, temperature)
            df = parse_to_csv(ai_output)
            if df is not None:
                st.success("✅ Parsed AI output into table")
                st.dataframe(df)
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)
                st.download_button("📥 Download CSV", csv_buf.getvalue(), "ingredients.csv", "text/csv")
            else:
                st.warning("⚠️ Could not parse into CSV. Showing raw output:")
                st.text(ai_output)
        except Exception as e:
            st.error(f"Error: {e}")
