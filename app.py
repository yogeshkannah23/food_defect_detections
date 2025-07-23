import streamlit as st
import requests
from PIL import Image
import io
import os
from dotenv import load_dotenv

st.set_page_config(page_title="SS Suite - Food/Beverage Inspection", layout="centered")
st.title("SS Suite - Food/Beverage Inspection")

# Load environment variables
load_dotenv()
backend_url = os.getenv("BACKEND_URL") 

uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    if file_type.startswith("image"):
        st.image(uploaded_file, caption="Original Image", use_column_width=True)
        with st.spinner("Processing image..."):
            response = requests.post(
                f"{backend_url}/process/image",
                files={"file": uploaded_file.getvalue()}
            )
        if response.status_code == 200:
            st.image(Image.open(io.BytesIO(response.content)), caption="Detected Image", use_column_width=True)
        else:
            st.error("Failed to process image.")

    elif file_type.startswith("video"):
        st.video(uploaded_file, format="video/mp4")

        # Show a button to trigger processing
        if st.button("🔍 Process Video"):
            with st.spinner("Processing video..."):
                response = requests.post(
                    f"{backend_url}/process/video",
                    files={"file": uploaded_file.getvalue()}
                )
            if response.status_code == 200:
                st.success("Processed video is ready!")
                st.download_button(
                    label="📥 Download Processed Video",
                    data=response.content,
                    file_name="processed_output.mp4",
                    mime="video/mp4"
                )
            else:
                st.error("Failed to process video.")
