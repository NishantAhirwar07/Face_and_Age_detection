"""
Streamlit configuration and caching utilities.
"""

import streamlit as st
from pathlib import Path
from urllib.request import urlretrieve

from detector import FaceAgeDetector

MODEL_FILES = {
    "deploy_face.prototxt": "https://raw.githubusercontent.com/sr6033/face-detection-opencv/master/opencv_face_detector_uint8.pbtxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/sr6033/face-detection-opencv/raw/master/res10_300x300_ssd_iter_140000.caffemodel",
    "deploy_age.prototxt": "https://github.com/peabody124/age-gender-caffe/raw/master/deploy_age.prototxt",
    "age_net.caffemodel": "https://github.com/peabody124/age-gender-caffe/raw/master/age_net.caffemodel",
}


def ensure_model_files() -> None:
    """Download any missing model files into the local models directory."""
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(exist_ok=True)

    for file_name, url in MODEL_FILES.items():
        target_path = model_dir / file_name
        if target_path.exists():
            continue
        with st.spinner(f"Downloading missing model file: {file_name}..."):
            urlretrieve(url, target_path)


@st.cache_resource(show_spinner=True)
def get_detector() -> FaceAgeDetector:
    """Load the face-age detector model, cached for the session."""
    try:
        ensure_model_files()
        detector = FaceAgeDetector()
        detector.load()
        return detector
    except FileNotFoundError as e:
        st.error(
            f"❌ Model files not found!\n\n{e}\n\n"
            "**Solution:**\n"
            "1. Download model files from the links shown above\n"
            "2. Place them in a `models/` folder in the repo root\n"
            "3. Restart the app"
        )
        raise
    except RuntimeError as e:
        st.error(
            f"❌ OpenCV error!\n\n{e}\n\n"
            "**Troubleshooting:**\n"
            "- Ensure `opencv-python-headless` is in `requirements.txt`\n"
            "- Models may be corrupted - re-download them\n"
            "- Check system dependencies in `packages.txt`"
        )
        raise
