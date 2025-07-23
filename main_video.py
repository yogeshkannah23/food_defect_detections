import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import random
from PIL import Image


# Define class names
class_names = ['Apple', 'Rotten']

# Assign a unique color for each class
colors = {
    class_id: tuple(random.choices(range(256), k=3))
    for class_id in range(len(class_names))
}

# Load YOLO model
model = YOLO('model/best.onnx')

st.title("SS Suite - Food and Beverage Defect Identification System")

uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    if file_type.startswith("image"):
        # Handle image input
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_column_width=True)
        frame = np.array(image)

        results = model(image)
        result = results[0]

        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = class_names[class_id]
            confidence = box.conf.item()
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            color = colors[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

        st.image(frame, caption="Detected Image", use_column_width=True)

    elif file_type.startswith("video"):
        temp_input = tempfile.NamedTemporaryFile(delete=False)
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_path = temp_output.name

        cap = cv2.VideoCapture(temp_input_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        with st.spinner("Processing video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    results = model(frame)
                    result = results[0]

                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        class_name = class_names[class_id]
                        confidence = box.conf.item()
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        color = colors[class_id]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{class_name} {confidence:.2f}",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, color, 2)
                except Exception as e:
                    st.warning(f"Skipping frame due to error: {e}")
                    continue

                out.write(frame)

        cap.release()
        out.release()
        temp_output.close()
        st.success("Processed video:")

        with open(output_path, "rb") as file:
            st.download_button(
                label="📥 Download Processed Video",
                data=file,
                file_name="processed_output.mp4",
                mime="video/mp4"
            )
