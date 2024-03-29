import streamlit as st
import cv2 as cv
import numpy as np
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from tensorflow.keras.applications import MobileNetV2
from PIL import Image 

st.title("Phân loại ảnh")
st.header("Nguyễn Đức Tuệ - 21521644")

feature = "HOG"
model = "model1"

img = st.file_uploader("Ảnh", type = ["png", "jpg"])

if img is not None:
  st.image(img)
  if st.button("Classify"):

    image = Image.open(img)
    image = image.resize((224, 224))
    x = np.array(image)
    x = np.expand_dims(x, axis= 0)
    x = preprocess_input(x)

    # load model
    model = MobileNetV2(weights='imagenet')

    # processing image
    y = model.predict(x)

    # result
    print(decode_predictions(y, top= 5))
    st.write(decode_predictions(y, top= 5))

with st.sidebar:
  feature = st.selectbox("Feature", ["HOG", "HIST"])

  model = st.selectbox("Feature", ["model1", "model2"])