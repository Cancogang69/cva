import streamlit as st
import cv2
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras.applications import MobileNetV2
from PIL import Image 

st.title("Phân loại ảnh")
st.header("Nguyễn Đức Tuệ - 21521644")

upload_image = st.file_uploader("Ảnh", type = ["png", "jpg"])

model = MobileNetV2(weights='imagenet')

red = (0, 0, 255)
green = (0, 255, 0)
work = st.sidebar.selectbox("Yêu cầu", ["1", "2"])

def get_image_label(image):
  image = image.resize((224, 224))
  x = np.array(image)
  x = np.expand_dims(x, axis= 0)
  x = preprocess_input(x)
  y = model.predict(x)
  label = decode_predictions(y, top= 1)[0][0][1]
  return label

def sliding_windows(image, bbox_h, bbox_w, h_step, w_step, label):
  bbox_list = []
  height_limit = bbox_h + 1 - h_step
  width_limit = bbox_w + 1 - w_step
  for y in range(0, height_limit, h_step):
    for x in range(0, width_limit, w_step):
      bbox = (x, y, x + bbox_w, y + bbox_h)
      bbox_img = image.crop(bbox)
      bbox_label = get_image_label(bbox_img)
      if label == bbox_label:
        bbox_list.append(bbox)
  return bbox_list

if upload_image is not None:
  st.image(upload_image)
  bbox_height = st.number_input("bbox height", min_value= 1, value= "min")
  bbox_width = st.number_input("bbox width", min_value= 1, value= "min")
  height_step = st.number_input("height step", min_value= 1, value= "min")
  width_step = st.number_input("width steap", min_value= 1, value= "min")
  if st.button("Detect"):
    if work == "1":
      image = Image.open(upload_image)
      label = get_image_label(image)
      st.write(label)
      bbox_list = sliding_windows(image, bbox_height, bbox_width, height_step, width_step, label)

      result_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)   
      for bbox in bbox_list:  
        cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), red, 2)

      st.image(Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)))
    elif work == "2":
      image = Image.open(upload_image)
      dog_bbox_list = sliding_windows(image, bbox_height, bbox_width, height_step, width_step, "Cardigan")
      cat_bbox_list = sliding_windows(image, bbox_height, bbox_width, height_step, width_step, "papillon")

      result_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)   
      for bbox in dog_bbox_list:
        cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), red, 2)

      for bbox in cat_bbox_list:
        cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), green, 2)

      st.image(Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)))
    
