import os
import time
import numpy as np
from PIL import Image
import streamlit as st
import custom_model as cm

st.set_page_config(
    page_title="Medical Report Generation using CXR Images", layout="wide")

_, col2, _ = st.beta_columns([1, 2, 1])
with col2:
    st.title("Medical Report Generation using Chest X-Ray Images")
    st.header("Capstone Project (17BCE0136)")


st.markdown("---")

col1, col2 = st.beta_columns(2)
image_1 = col1.file_uploader("Frontal X-Ray", type=['png', 'jpg', 'jpeg'])
image_2 = col1.file_uploader("Lateral X-Ray", type=['png', 'jpg', 'jpeg'])

predict_button = col2.button('Generate Report')

col2.markdown("***")
col2.text("")


@st.cache
def create_model():
    model_tokenizer = cm.create_model()
    return model_tokenizer


def predict(image_1, image_2, model_tokenizer, predict_button=predict_button):
    start = time.process_time()
    if predict_button:
        if (image_1 is not None):
            start = time.process_time()
            image_1 = Image.open(image_1).convert("RGB")
            image_1 = np.array(image_1)/255
            if image_2 is None:
                image_2 = image_1
            else:
                image_2 = Image.open(image_2).convert("RGB")
                image_2 = np.array(image_2)/255

            col2.image([image_1, image_2], width=300)
            caption = cm.function1([image_1], [image_2], model_tokenizer)
            col2.markdown(" # **Impression:**")
            impression = col2.empty()
            impression.write(caption[0])
            time_taken = "Time Taken : %i seconds" % (
                time.process_time()-start)
            col2.write(time_taken)
            del image_1, image_2
        else:
            st.markdown("## Please upload an Image")


model_tokenizer = create_model()

predict(image_1, image_2, model_tokenizer)
