"""
Streamlit configuration and caching utilities.
"""

import streamlit as st
from detector import FaceAgeDetector


@st.cache_resource(show_spinner=False)
def get_detector() -> FaceAgeDetector:
    """Load the face-age detector model, cached for the session."""
    detector = FaceAgeDetector()
    detector.load()
    return detector
