# -*- coding: utf-8 -*-
"""
Created on Sat Mar 9 02:37:45 2024

@author: Asus
"""
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import requests
model = tf.keras.models.load_model('D:/AI-ML/ConvolutedNets/Projects/CatDogClassifier/CatDogClassifier.h5')

def predict_class(image):
  processed_image = cv2.resize(image, (256, 256))  
  processed_image = processed_image.astype('float32') / 255

  predictions = model.predict(np.expand_dims(processed_image, axis=0))
  if predictions > 0.5:
    predicted_class = 'Dog' 
  else:
    predicted_class = 'Cat'
  return predicted_class

def main():
  st.title("Cat or Dog Classifier")
  st.write("Enter a hyperlink to an image or upload an image file:")

  uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
  image_url = st.text_input("Enter image URL (optional):", key="image_url")
  
  clear_button = st.button("Clear URL", key="clear_button")

  if "clear_requested" not in st.session_state:
    st.session_state["clear_requested"] = False

  if clear_button:
    st.session_state["clear_requested"] = True

  if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)  # Read uploaded image
  elif image_url:
    # Download image from URL using libraries like requests
    image_response = requests.get(image_url, stream=True)
    image = cv2.imdecode(np.asarray(bytearray(image_response.raw.read()), dtype="uint8"), cv2.IMREAD_COLOR)  # Read downloaded image

  if image is not None:
    predicted_class = "This function predicts cat or dog based on your model"

    st.image(image, channels="BGR")
    predicted_class = predict_class(image)
    st.write(f"Predicted Class: {predicted_class}")
  else:
    st.write("Please upload an image or enter a valid image URL.")


if __name__ == '__main__':
  main()
