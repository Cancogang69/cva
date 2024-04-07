import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
import torchvision
import time

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
model.eval()

st.title("Object detection")
st.header("Nguyễn Đức Tuệ - 21521644")

upload_image = st.file_uploader("Ảnh", type = ["png", "jpg"])

red = (0, 0, 255)
green = (0, 255, 0)
work = st.sidebar.selectbox("Yêu cầu", ["1", "2"])

CLASS = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def parse_int(box):
  return (int(box[0]), int(box[1]))

def get_prediction(img, threshold=0.5):
  img = np.array(img.convert("RGB"))
  img = torchvision.transforms.transforms.ToTensor()(torchvision.transforms.ToPILImage()(img))
  pred = model([img])
  pred_class = [CLASS[i] for i in list(pred[0]['labels'].numpy())]
  pred_boxes = [[parse_int((i[0], i[1])), parse_int((i[2], i[3]))] for i in list(pred[0]['boxes'].detach().numpy())]
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x>threshold]
  pred_t = pred_t[-1] if len(pred_t) is not 0 else 0
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return pred_boxes, pred_class


def object_detection_api(img, label_count, threshold=0.5, rect_th=3): 
  boxes, pred_cls = get_prediction(img, threshold) 
  img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)   
  count = 0
  for box, label in zip(boxes, pred_cls):
    if label_count == "Nothing":
      cv2.rectangle(img, box[0], box[1], color=green, thickness= rect_th) 
    if label == label_count:
      count+=1
      cv2.rectangle(img, box[0], box[1], color=green, thickness= rect_th) 
  
  return img, count

if upload_image is not None:
  st.image(upload_image)
  if st.button("Detect"):
    if work == "1":
      image = Image.open(upload_image)
      start = time.time()
      result_image, count = object_detection_api(image, "person")
      end = time.time()
      st.write(f"Work time: {end - start}s")
      st.write(f"Number of person: {count}")
      st.image(Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)))
    elif work == "2":
      image = Image.open(upload_image)
      start = time.time()
      result_image, count = object_detection_api(image, "Nothing", threshold= 0.3)
      end = time.time()
      st.write(f"Work time: {end - start}s")
      st.image(Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)))
    
