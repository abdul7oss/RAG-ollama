import streamlit as st
import pandas as pd
import json
import os

params = st.experimental_get_query_params()
idx = str(params.get("idx", [0])[0])  # ensure string key

db_path = "attributions/db.json"
if not os.path.exists(db_path):
    st.error("⚠️ db.json not found.")
    st.stop()

with open(db_path, "r") as f:
    db = json.load(f)

if idx not in db:
    st.error("⚠️ Attribution entry not found.")
    st.stop()

entry = db[idx]
df = pd.DataFrame(entry["attribution_df"])

st.title("🔎 Citation for Sentence")
st.markdown(f"**Sentence:** {entry['sentence']}")
st.markdown(f"**Start:** `{entry['start']}`, **End:** `{entry['end']}`")
st.dataframe(df)

for i, row in df.iterrows():
    st.markdown(f"""
    <div style="padding:10px; border:1px solid #ccc; border-radius:6px; margin-bottom:10px;">
        <strong>Source {i+1}</strong> | <em>Confidence:</em> {row['confidence']:.2%}<br><br>
        <div style="background-color:#f9f9f9; padding:10px; border-radius:5px;">
            {row['text']}
        </div>
    </div>
    """, unsafe_allow_html=True)
