import streamlit as st
import torch
import nltk
from nltk.tokenize import sent_tokenize
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ---------------------------
# Page config (renders first)
# ---------------------------
st.set_page_config(page_title="Aspect-Based Sentiment Analysis")
st.title("Aspect-Based Sentiment Analysis")
st.write("Analyze product reviews and get sentiment per aspect.")

# ---------------------------
# Ensure NLTK tokenizer exists
# ---------------------------
@st.cache_resource
def download_nltk():
    nltk.download("punkt")

download_nltk()

# ---------------------------
# Load model (cached + spinner)
# ---------------------------
@st.cache_resource
def load_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, device

with st.spinner("Loading sentiment model (first run may take ~1 minute)..."):
    model, tokenizer, device = load_model()

# ---------------------------
# Aspect keywords
# ---------------------------
ASPECTS = {
    "delivery": ["delivery", "shipping", "late", "delay"],
    "price": ["price", "cost", "expensive", "cheap"],
    "quality": ["quality", "build", "durable", "defective"],
    "service": ["service", "support", "customer"],
    "packaging": ["package", "packaging"]
}

LABELS = ["Negative", "Positive"]

# ---------------------------
# Sentiment prediction
# ---------------------------
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    return LABELS[torch.argmax(logits).item()]

# ---------------------------
# Aspect-based sentiment
# ---------------------------
def aspect_sentiment(review):
    sentences = sent_tokenize(review)
    results = {}

    for aspect, keywords in ASPECTS.items():
        for sentence in sentences:
            if any(k in sentence.lower() for k in keywords):
                results[aspect] = predict_sentiment(sentence)

    return results

# ---------------------------
# Streamlit UI
# ---------------------------
review_input = st.text_area(
    "Enter a product review:",
    placeholder="Example: The delivery was late but the product quality is excellent."
)

if st.button("Analyze"):
    if review_input.strip():
        with st.spinner("Analyzing review..."):
            results = aspect_sentiment(review_input)

        if results:
            st.success("Analysis Complete!")
            for aspect, sentiment in results.items():
                st.write(f"**{aspect.capitalize()}**: {sentiment}")
        else:
            st.warning("No predefined aspects detected in the review.")
    else:
        st.error("Please enter a review.")
