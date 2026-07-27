import re
import joblib
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Page Configuration


st.set_page_config(
    page_title="Emotion Detection using DistilBERT",
    page_icon="😊",
    layout="centered"
)


# Load Model


@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained("model")
    model = AutoModelForSequenceClassification.from_pretrained("model")
    label_encoder = joblib.load("label_encoder.pkl")

    model.eval()

    return tokenizer, model, label_encoder

tokenizer, model, label_encoder = load_model()


# Text Cleaning


def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"#", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text



# Prediction


def predict(text):

    text = clean_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)

        prediction = torch.argmax(probabilities, dim=1).item()

    emotion = label_encoder.inverse_transform([prediction])[0]

    confidence = probabilities[0][prediction].item()

    return emotion, confidence, probabilities.squeeze()



# Streamlit UI


st.title("😊 Emotion Detection using DistilBERT")

st.write("Enter a sentence and the model will predict the emotion.")

text = st.text_area(
    "Enter your text here",
    height=180
)

if st.button("Predict Emotion"):

    if text.strip() == "":

        st.warning("Please enter some text.")

    else:

        emotion, confidence, probabilities = predict(text)

        st.success(f"Predicted Emotion : {emotion}")

        st.info(f"Confidence : {confidence*100:.2f}%")

        st.subheader("Prediction Probabilities")

        prob_dict = {}

        for label, prob in zip(label_encoder.classes_, probabilities):

            prob_dict[label] = float(prob)

        st.bar_chart(prob_dict)