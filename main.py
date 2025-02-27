from fastapi import FastAPI, UploadFile, File, Form
from datetime import datetime
import os
from db import db_service
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define upload directory from .env
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI()


@app.post("/check-image")
async def check_image(captured_at: str = Form(...), file: UploadFile = File(...)):

    # Generate a unique filename (original name + timestamp)
    file_ext = os.path.splitext(file.filename)[1]  # Get file extension
    unique_filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}{file_ext}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

    # Save the file locally
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Insert record into the database
    frame_id = db_service.insert_video_frame(datetime.fromisoformat(captured_at), file_path)
    return {
        "image_path": file_path,
        "frame_id": frame_id,
        "captured_at": captured_at,
        "success": True
    }
@app.get("/last-five-frames")
async def get_last_five_frames():
    frames = db_service.get_last_five_frames()
    return {"frames": frames}


