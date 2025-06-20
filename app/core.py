from app.utils import extract_keyframes, detect_timestamp 
from pathlib import Path

def process_input_file(filepath, metadata):
    path = Path(filepath)
    keyframes = extract_keyframes(path)  # Extract frames from the video (if needed), or simply return the original image

    for frame in keyframes:
        result = detect_timestamp(
            image_path=frame,
            metadata=metadata
        )
        if result and result.get("timestamp"):
            return result

    return {
        "timestamp": None,
        "source": None,
        "confidence": 0.0,
        "keyframe_file": None
    }