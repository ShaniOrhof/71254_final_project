import streamlit as st
import cv2
import numpy as np
import skimage.io as io
from PIL import Image

# Streamlit app
st.title("Image Segmentation with K-Means Clustering")

# Function to segment using k-means
def segment_image_kmeans(img, k=3, attempts=10):
    # Convert MxNx3 image into Kx3 where K=MxN
    pixel_values = img.reshape((-1, 3))  # -1 reshape means, in this case, MxN

    # Convert the uint8 values to float as required by OpenCV's k-means
    pixel_values = np.float32(pixel_values)

    # Define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Flatten the labels array
    labels = labels.flatten()

    # Map each pixel to the centroid's color
    segmented_image = centers[labels.flatten()]

    # Reshape to the original image dimensions
    segmented_image = segmented_image.reshape(img.shape)

    return segmented_image

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    img = Image.open(uploaded_file)
    img = np.array(img)

    # Display the original image
    st.subheader("Original Image")
    st.image(img, channels="RGB")

    # K-means segmentation parameters
    st.subheader("Segmentation Settings")
    k = st.slider("Number of Clusters (k)", min_value=2, max_value=10, value=3, step=1)
    attempts = st.slider("K-Means Attempts", min_value=1, max_value=20, value=10, step=1)

    # Segment the image
    segmented_image = segment_image_kmeans(img, k=k, attempts=attempts)

    # Display the segmented image
    st.subheader("Segmented Image")
    st.image(segmented_image, channels="RGB")
