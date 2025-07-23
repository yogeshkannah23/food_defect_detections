# routers/process.py
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from utils.model_utils import model, class_names, colors

from PIL import Image
import cv2
import numpy as np
import io
import tempfile

router = APIRouter()

@router.post("/image")
async def process_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = model(image)
    result = results[0]

    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = class_names[class_id]
        confidence = box.conf.item()
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        color = colors[class_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    _, buffer = cv2.imencode(".jpg", frame)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")


@router.post("/video")
async def process_video(file: UploadFile = File(...)):
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(await file.read())
    temp_input.close()

    cap = cv2.VideoCapture(temp_input.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

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
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        except Exception:
            continue
        out.write(frame)

    cap.release()
    out.release()

    with open(temp_output.name, "rb") as f:
        return StreamingResponse(io.BytesIO(f.read()), media_type="video/mp4")
