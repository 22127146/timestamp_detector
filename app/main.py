from fastapi import FastAPI, File, UploadFile
from typing import List
from app.core import process_input_file
import shutil, os, json, tempfile

app = FastAPI()

@app.post("/analyze/")
async def analyze_media(
    files: List[UploadFile] = File(...),
    metadata_file: UploadFile = File(...)
):
    # Save metadata file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        shutil.copyfileobj(metadata_file.file, tmp)
        metadata_path = tmp.name

    # Read metadata from the temporary file
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Remove the temporary metadata file
    os.remove(metadata_path)

    results = []

    for file in files:
        # Save each uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        output = process_input_file(
            filepath=temp_path,
            metadata=metadata
        )
        results.append(output)

        os.remove(temp_path)

    return {"results": results}