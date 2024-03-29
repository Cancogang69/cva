import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import euclidean
from PIL import Image 
import glob

def lbp_hist_cal(gray_img):
  #resize grayscale image to 224*224
  gray_img = gray_img.resize((300, 300))

  #compute lbp feature
  x = np.array(gray_img)
  lbp_feat=local_binary_pattern(x, 8, 2, method='uniform')

  #compute hist of lbp
  hist, _ = np.histogram(lbp_feat, bins=np.arange(2**8 + 2), density=True)
  return hist

def cal_hist_db(database_path):
  hist_db = []
  for filename in glob.glob(f"{database_path}\*.jpg"):
    gray_img = Image.open(filename).convert("L")
    hist_db.append((filename, lbp_hist_cal(gray_img)))
  return hist_db

def retrieve_img(img):
  #open image and convert to grayscale
  gray_img = Image.open(img).convert("L")
  img_hist = lbp_hist_cal(gray_img)

  hist_db = cal_hist_db("D:\\HocTap\\CVA\\TH\\bai1\\database")

  euclid_dis = [(hist[0], euclidean(hist[1], img_hist)) for hist in hist_db]
  return sorted(euclid_dis, key=lambda x: x[1])

def label(filepath):
  return filepath.split("\\")[-1].split("_")[0]

def precision_at_k(k, query_name, arr):
  query_label = label(query_name)
  true_guest = 0
  for i, img in enumerate(arr):
    if i == k:
      break
    if label(img[0]) == query_label:
      true_guest += 1
    
  return true_guest/k

def recal_at_k(k, query_name, arr):
  query_label = label(query_name)
  true_guest = 0
  total_instance = 0
  for i, img in enumerate(arr):
    if label(img[0]) == query_label and i < k:
      true_guest += 1
    if label(img[0]) == query_label:
      total_instance += 1
    
  return true_guest/total_instance
    
def AP_at_k(k, query_name, arr):
  query_label = label(query_name)
  true_guest = 0
  AP = 0
  for i, img in enumerate(arr):
    if i == k:
      break
    if label(img[0]) == query_label:
      true_guest += 1
      AP += true_guest/(i+1)
  
  result = AP/true_guest if true_guest>0 else 0
  return result

# page    
st.title("Image retrieval")
st.header("Nguyễn Đức Tuệ - 21521644")

img = st.file_uploader("Ảnh", type = ["png", "jpg"])

if img is not None:
  st.image(img)
  if st.button("Retrieve"):
    euclid_dis = retrieve_img(img)
    st.write(f"precision: {precision_at_k(10, img.name, euclid_dis)}")
    st.write(f"recall: {recal_at_k(10, img.name, euclid_dis)}")
    st.write(f"AP: {AP_at_k(10, img.name, euclid_dis)}")
    for i in range(10):
      st.image(euclid_dis[i][0])

with st.sidebar:
  feature = st.selectbox("Feature", ["LBP"])

  model = st.selectbox("", ["Euclid distance"])