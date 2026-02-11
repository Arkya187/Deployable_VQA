import streamlit as st
import tensorflow as tf
import pickle
import cv2
import numpy as np
import gdown
import os

MODEL_URL = "https://drive.google.com/uc?id=1FVlrEQdI8qCRZFPF3a4La5KHsrHF0E1v"

MODEL_PATH = "vqa_model.h5"

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... please wait")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
tokenizer = pickle.load(open("tokenizer.pkl","rb"))

MAX_LEN = 20

st.title("Medical VQA")

uploaded = st.file_uploader("Upload Image")

question = st.text_input("Enter Question")

if uploaded and question:

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.resize(image, (256,256))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    seq = tokenizer.texts_to_sequences([question])
    seq = tf.keras.utils.pad_sequences(seq, maxlen=MAX_LEN)

    pred = model.predict([image, seq])[0][0]

    answer = "YES" if pred > 0.5 else "NO"

    st.write("Prediction:", answer)


