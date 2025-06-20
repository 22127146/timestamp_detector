import os
import cv2
import time
import torch
import requests
import tempfile
import torchvision.transforms as T
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from difflib import SequenceMatcher
from serpapi import GoogleSearch
from open_clip import create_model_and_transforms

# Load model
model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='openai')
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

# Load environment variables
load_dotenv()
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

def upload_to_imgbb(image_path):
    with open(image_path, "rb") as f:
        res = requests.post(
            "https://api.imgbb.com/1/upload",
            params={"key": IMGBB_API_KEY},
            files={"image": f}
        )
    return res.json()["data"]["url"]

def extract_keyframes(video_path, frame_interval=5, threshold=0.92):
    keyframe_paths = []

    cap = cv2.VideoCapture(str(video_path))
    frame_id = 0
    saved_id = 0
    prev_feat = None

    # Create a temporary directory for keyframes
    keyframe_dir = tempfile.mkdtemp(prefix="keyframes_")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            # Convert frame → tensor (CLIP)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = T.ToPILImage()(image)
            image_tensor = preprocess(image_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = model.encode_image(image_tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)

            # Save keyframe if it's significantly different from the previous one
            if prev_feat is None or (feat @ prev_feat.T).item() < threshold:
                save_path = os.path.join(keyframe_dir, f"keyframe_{saved_id:03}.jpg")
                cv2.imwrite(save_path, frame)
                keyframe_paths.append(save_path)
                saved_id += 1
                prev_feat = feat

        frame_id += 1

    cap.release()
    return keyframe_paths

def parse_date_from_string(s):
    formats = [
        "%b %d, %Y, %H:%M",    # Oct 17, 2023, 14:25
        "%B %d, %Y, %H:%M",    # October 17, 2023, 14:25
        "%b %d, %Y",           # Oct 17, 2023
        "%B %d, %Y",           # October 17, 2023
        "%Y-%m-%d %H:%M",      # 2023-10-17 14:25
        "%Y-%m-%d",            # 2023-10-17
        "%d/%m/%Y %H:%M",      # 17/10/2023 14:25
        "%d/%m/%Y",            # 17/10/2023
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s.strip(), fmt)
        except:
            continue
    return None

def simple_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def detect_timestamp(image_path, metadata):
    text_query = f"{metadata['location']} {metadata['title']} {metadata['description']}"

    def search_by_text():
        search = GoogleSearch({
            "engine": "google",
            "q": text_query,
            "api_key": SERPAPI_API_KEY,
            "num": 20,
            "tbs": "sbd:1"
        })
        results = search.get_dict()
        return results.get("organic_results", [])

    text_results = search_by_text()
    print(f"Retrieved {len(text_results)} results from text search")

    print(f"\nProcessing image: {os.path.basename(image_path)}")

    # Upload image
    with open(image_path, "rb") as f:
        upload_response = requests.post(
            "https://api.imgbb.com/1/upload",
            params={"key": IMGBB_API_KEY},
            files={"image": f}
        )
    image_url = upload_response.json()["data"]["url"]
    print(f"Uploaded to imgbb: {image_url}")

    # Reverse image search
    search = GoogleSearch({
        "engine": "google_reverse_image",
        "image_url": image_url,
        "api_key": SERPAPI_API_KEY
    })
    results = search.get_dict()

    image_results = []
    for key, value in results.items():
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            print(f"Added {len(value)} results from field '{key}'")
            image_results.extend(value)

    print(f"Total of {len(image_results)} image search results")

    # Merge and score
    merged = text_results + image_results
    scored = []
    for res in merged:
        title = res.get("title", "")
        link = res.get("link", "")
        snippet = res.get("snippet", "")
        date = parse_date_from_string(res.get("date", ""))
        text = f"{title} {snippet}"
        sim = simple_similarity(text, text_query)
        scored.append({
            "title": title,
            "link": link,
            "date": date,
            "similarity": sim,
            "from_image": res in image_results
        })

    scored = sorted(scored, key=lambda x: (-x["similarity"], x["date"] or datetime.max))

    for item in scored:
        if item["date"]:
            date_str = item["date"].strftime("%Y-%m-%d %H:%M") if item["date"].hour or item["date"].minute else item["date"].strftime("%Y-%m-%d")
            print(f"\nMatch found:")
            print(f"Link: {item['link']}")
            print(f"Title: {item['title']}")
            print(f"Similarity: {item['similarity']:.2f}")
            print(f"Published date: {date_str}")
            
            result = {
                "timestamp": date_str,
                "source": item["link"],
                "confidence": item["similarity"]
            }
            
            if item["from_image"]:
                result["keyframe_file"] = image_url
            
            return result

    print("No reliable timestamp found.")
    return {
        "timestamp": None,
        "source": None,
        "confidence": 0.0
    }





