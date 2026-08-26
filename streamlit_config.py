"""
Streamlit configuration and caching utilities.
"""

import streamlit as st
import os
import sys
from detector import FaceAgeDetector


@st.cache_resource(show_spinner=True)
def get_detector() -> FaceAgeDetector:
    """Load the face-age detector model, cached for the session."""
    try:
        detector = FaceAgeDetector()
        detector.load()
        return detector
    except FileNotFoundError as e:
        st.error(
            f"❌ Model files not found!\n\n{e}\n\n"
            "**Solution:**\n"
            "1. Download model files from:\n"
            "   - Face: https://github.com/sr6033/face-detection-opencv/blob/master/res10_300x300_ssd_iter_140000.caffemodel\n"
            "   - Age: https://github.com/dill/nndb/tree/master/models\n"
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
