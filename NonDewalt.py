# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 12:10:18 2024

@author: Debrup Basu
"""

import streamlit as st
from PIL import Image
from PIL import UnidentifiedImageError
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load the JSON file containing product information
with open('dewalt.json') as f:
    data = json.load(f)

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

st.title('Power Tool Image Recognition App')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the uploaded image
    img = image.resize((128, 128))
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = preprocess_input(img_array)
    st.markdown(f"Sorry I, could not recognise the image as it is not a Stanley Black & Decker Powertool.")
