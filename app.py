import os
import re
import torch
import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import nest_asyncio

nest_asyncio.apply()

# ------------------------
# Utility: Clean Input
# ------------------------
def preprocess_input(raw_text):
    clean = re.sub(r'[^\x00-\x7F]+', ' ', raw_text)  # Remove non-ASCII
    return re.sub(r'\s+', ' ', clean).strip()

# ------------------------
# Load Model & Tokenizer
# ------------------------
@st.cache_resource(show_spinner="Initializing NLP Engine...")
def initialize_model(model_path):
    if not os.path.exists(model_path):
        st.error("Model path not found. Please check the directory.")
        return None, None, None

    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return tokenizer, model, device

# ------------------------
# Generate Summary
# ------------------------
def summarize_text(text, tokenizer, model, device, length=200):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=length,
        min_length=int(length * 0.6),
        num_beams=6,
        repetition_penalty=2.5,
        length_penalty=1.0,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ------------------------
# App UI
# ------------------------
def run_ui():
    st.set_page_config(page_title="SmartSummarizer", layout="centered")
    st.title("🧠 SmartSummarizer")
    st.markdown("Transform lengthy text into precise summaries with a fine-tuned BART model.")

    with st.sidebar:
        st.header("📦 Load Your Model")
        model_dir = st.text_input("Model Directory", value=r"C:\Users\Mohit\OneDrive\Desktop\codeTech\task1\bart-xsum-finetuned")
        length_slider = st.slider("Summary Length", 50, 512, 200, step=10)

    with st.expander("📘 Instructions", expanded=False):
        st.markdown("""
        1. Paste your text in the input box below.
        2. Adjust summary length using the sidebar.
        3. Click "Summarize" to generate output.
        """)

    raw_input = st.text_area("✏️ Enter text here:", height=200)

    if st.button("🔍 Summarize"):
        if raw_input.strip():
            tokenizer, model, device = initialize_model(model_dir)
            if tokenizer is None or model is None:
                return

            st.info("Processing your input...")
            clean_input = preprocess_input(raw_input)

            with st.spinner("Generating summary..."):
                summary = summarize_text(clean_input, tokenizer, model, device, length_slider)

            st.success("✅ Summary Generated")
            st.subheader("📝 Summary Output")
            st.write(summary)
        else:
            st.warning("Please enter some text first.")

# ------------------------
# Launch
# ------------------------
if __name__ == "__main__":
    run_ui()
