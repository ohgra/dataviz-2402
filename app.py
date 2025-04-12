import streamlit as st
import numpy as np
import json
import shap
import time
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# ------------------------------------------------------------------
# Set page configuration to use wide layout and no sidebar
st.set_page_config(page_title="Explainable AI with Image", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------
# 1) Header
st.title("Explainable AI with Image")

# ------------------------------------------------------------------
# 2) About the Dataset Section
st.header("📊 About the Dataset")

st.write("""
We're working with two powerful datasets to bring deep learning to life:

- 🧠 **CIFAR-10** for neural network visualizations and feature maps — featuring 60,000 colorful 32×32 images across 10 object classes.
- 🔍 **ImageNet-50** for SHAP (Shapley Additive Explanations) — giving us high-resolution insights into model interpretability.

These datasets help us explore both **what the model learns** and **why it makes predictions**.
""")

# ------------------------------------------------------------------
# 3) Two Videos Side by Side for Neural Network Visualization
st.header("🔥 Neural Network in Action!")

st.write("""
Ever wondered what a neural network *actually sees*?  
We've taken vibrant **32×32×3 CIFAR-10 images**, flattened and fed them through powerful **3-layer and 4-layer neural networks**, trained on **50,000 examples**.

Now, watch the magic unfold ✨ — as we animate the predictions on **10,000 test images**.  
Experience deep learning, visually.
""")

col_video1, col_video2 = st.columns(2)
with col_video1:
    st.subheader("🔍 Exploring the 3-Layer Neural Network")
    video_html_3 = """
    <video width="100%" autoplay controls>
      <source src="https://raw.githubusercontent.com/ohgra/dataviz-2402/refs/heads/main/src/nnv/1.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    """
    st.markdown(video_html_3, unsafe_allow_html=True)
with col_video2:
    st.subheader("🔍 Exploring the 4-Layer Neural Network")
    video_html_4 = """
    <video width="100%" autoplay controls>
      <source src="https://raw.githubusercontent.com/ohgra/dataviz-2402/refs/heads/main/src/nnv/2.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    """
    st.markdown(video_html_4, unsafe_allow_html=True)

# ------------------------------------------------------------------
# 4) Section with 10 Radio Buttons and 7 Images Display
st.header("Visualizations per Class")
# List of class names
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer", 
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Create the radio button with vertical alignment and class names
selected_class = st.radio("Select Class", options=class_names, index=0)

# Display the selected class
st.write(f"You selected: {selected_class}")


# Display image for "class average value representation"
st.subheader("📊 Class-Wise Average Activation Representation")

st.write("""
For each class in the dataset, we've averaged the **activation values** from the network to capture how the model internally **represents each class**.

This provides insight into the **class-specific patterns** learned across the layers of our fully connected neural network.
""")

# Adjust the image path as needed. Here we assume an image file naming convention.
avg_image_path = f"src/class/avg/{selected_class}.png"
st.image(avg_image_path, caption=f"Class Average for Class {selected_class}", use_container_width=True)

# Display six images arranged as three rows, two images per row
st.subheader("🔍 Feature Kernels (using ResNet50) per Class")

st.write("""
In this section, we explore the **feature kernels** extracted from the **last 6 layers** of a **ResNet50** model. These kernels represent the learned features for each class, showing how the model detects important visual patterns specific to each category.

By examining these feature kernels, we gain deeper insights into the model's **understanding** of different object classes and how it distinguishes between them.
""")

col_left, col_right = st.columns(2)

with col_left:
    img_path_left = f"src/class/kernel/{selected_class}- (1).png"
    st.image(img_path_left, caption=f"Kernel 1", use_container_width=True)
with col_right:
    img_path_right = f"src/class/kernel/{selected_class}- (2).png"
    st.image(img_path_right, caption=f"Kernel 2", use_container_width=True)

col_left, col_right = st.columns(2)
with col_left:
    img_path_left = f"src/class/kernel/{selected_class}- (3).png"
    st.image(img_path_left, caption=f"Kernel 3", use_container_width=True)
with col_right:
    img_path_right = f"src/class/kernel/{selected_class}- (4).png"
    st.image(img_path_right, caption=f"Kernel 4", use_container_width=True)

col_left, col_right = st.columns(2)
with col_left:
    img_path_left = f"src/class/kernel/{selected_class}- (5).png"
    st.image(img_path_left, caption=f"Kernel 5", use_container_width=True)
with col_right:
    img_path_right = f"src/class/kernel/{selected_class}- (6).png"
    st.image(img_path_right, caption=f"Kernel 6", use_container_width=True)

# ------------------------------------------------------------------
# 5) Section for SHAP Values on an Image Based on Slider Input
st.header("🧠 SHAP Values for Image")

st.markdown("""
The SHAP values provide an explanation of how the **ResNet50 model** makes predictions on an image.  
The output highlights the most important regions of the image that influenced the model's decision, showing which pixels contribute positively or negatively to the predicted class.  
This visualization helps us understand the model's reasoning behind its predictions.
""")

slider_value = st.slider("Select an image index (0-48)", min_value=0, max_value=48, value=12, step=1)

progressbar=st.progress(0)
text =  st.empty()

for percent in range(101):
    progressbar.progress(percent)
    text.text(f"Calculating... {percent}%")
text.empty()
shap_image_path = f"src/shap/{slider_value}.png"
st.image(shap_image_path, caption=f"Shap Value for Image {slider_value}", use_container_width=True)
# model = ResNet50(weights="imagenet")

#     # File paths for the data and class names
# data_file = "src/data/imagenet50_224x224.npy"
# class_names_file = "src/data/imagenet_class_index.json"

#     # Load image data
# X = np.load(data_file, allow_pickle=True)
# X = np.clip(X, 0, 255).astype(np.uint8)
# y = None
#     # Load class names from JSON file
# with open(class_names_file, 'r') as f:
#     class_names = [v[1] for v in json.load(f).values()]
    
# def f(x):
#     tmp = x.copy()
#     preprocess_input(tmp)
#     return model(tmp)

# masker = shap.maskers.Image("inpaint_telea", X[0].shape)
# explainer = shap.Explainer(f, masker, output_names=class_names)

# # Use the slider value as the index for the image to explain.
# imgs = X[slider_value:slider_value+1]
# shap_values = explainer(imgs, max_evals=200, batch_size=50, outputs=shap.Explanation.argsort.flip[:4])
# plt.figure()
# shap.image_plot(shap_values, show=False)
# fig = plt.gcf()
# st.pyplot(fig)
