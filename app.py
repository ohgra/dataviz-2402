import streamlit as st
import numpy as np
import json
import shap
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
st.header("About the Dataset")
st.write("""
This dataset comprises images from ImageNet resized to 224x224 and stored in a NumPy array.
Each image represents a class and has been preprocessed for the ResNet50 model.
(Replace this text with your detailed description of the dataset.)
""")

# ------------------------------------------------------------------
# 3) Two Videos Side by Side for Neural Network Visualization
st.header("Neural Network Visualization")
col_video1, col_video2 = st.columns(2)
with col_video1:
    st.subheader("3 Layer")
    video_html_3 = """
    <video width="100%" autoplay controls>
      <source src="https://raw.githubusercontent.com/ohgra/dataviz-2402/refs/heads/main/src/nnv/1.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    """
    st.markdown(video_html_3, unsafe_allow_html=True)
with col_video2:
    st.subheader("4 Layer")
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
selected_class = st.radio("Select Class (1-10)", options=list(range(0, 10)))

# Display image for "class average value representation"
st.subheader("Class Average Value Representation")
# Adjust the image path as needed. Here we assume an image file naming convention.
avg_image_path = f"src/class/avg/{selected_class}.png"
st.image(avg_image_path, caption=f"Class Average for Class {selected_class}", use_column_width=True)

# Display six images arranged as three rows, two images per row
st.subheader("Feature Kernel (using ResNet50) per Class")
col_left, col_right = st.columns(2)

with col_left:
    img_path_left = f"src/class/kernel/{selected_class}- (1).png"
    st.image(img_path_left, caption=f"Kernel 1", use_column_width=True)
with col_right:
    img_path_right = f"src/class/kernel/{selected_class}- (2).png"
    st.image(img_path_right, caption=f"Kernel 2", use_column_width=True)

col_left, col_right = st.columns(2)
with col_left:
    img_path_left = f"src/class/kernel/{selected_class}- (3).png"
    st.image(img_path_left, caption=f"Kernel 3", use_column_width=True)
with col_right:
    img_path_right = f"src/class/kernel/{selected_class}- (4).png"
    st.image(img_path_right, caption=f"Kernel 4", use_column_width=True)

col_left, col_right = st.columns(2)
with col_left:
    img_path_left = f"src/class/kernel/{selected_class}- (5).png"
    st.image(img_path_left, caption=f"Kernel 5", use_column_width=True)
with col_right:
    img_path_right = f"src/class/kernel/{selected_class}- (6).png"
    st.image(img_path_right, caption=f"Kernel 6", use_column_width=True)

# ------------------------------------------------------------------
# 5) Section for SHAP Values on an Image Based on Slider Input
st.header("Shaply Values for Images")
slider_value = st.slider("Select an image index (0-48)", min_value=0, max_value=48, value=12, step=1)

model = ResNet50(weights="imagenet")

    # File paths for the data and class names
data_file = "src/data/imagenet50_224x224.npy"
class_names_file = "src/data/imagenet_class_index.json"

    # Load image data
X = np.load(data_file, allow_pickle=True)
X = np.clip(X, 0, 255).astype(np.uint8)
y = None
    # Load class names from JSON file
with open(class_names_file, 'r') as f:
    class_names = [v[1] for v in json.load(f).values()]
    
def f(x):
    tmp = x.copy()
    preprocess_input(tmp)
    return model(tmp)

masker = shap.maskers.Image("inpaint_telea", X[0].shape)
explainer = shap.Explainer(f, masker, output_names=class_names)

# Use the slider value as the index for the image to explain.
imgs = X[slider_value:slider_value+1]
shap_values = explainer(imgs, max_evals=200, batch_size=50, outputs=shap.Explanation.argsort.flip[:4])
plt.figure()
shap.image_plot(shap_values, show=False)
fig = plt.gcf()
st.pyplot(fig)
