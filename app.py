"""
Streamlit web app for face and age detection.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from streamlit_config import get_detector


def draw_predictions(image: np.ndarray, predictions: list) -> np.ndarray:
    """Draw bounding boxes and age labels on image."""
    output = image.copy()
    for (x, y, w, h), age_label, confidence in predictions:
        color = (0, 255, 0)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        label = f"{age_label} ({confidence:.2f})"
        cv2.putText(
            output,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
    return output


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Face & Age Detection",
        page_icon="👤",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.title("🎯 Face & Age Detection")
    st.markdown(
        "Upload an image or use your camera to detect faces and predict ages."
    )

    # Load model (cached)
    try:
        detector = get_detector()
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    confidence_threshold = st.sidebar.slider(
        "Face Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    input_method = st.sidebar.radio("Input Method", ["📸 Upload Image", "📷 Webcam"])

    # Main content
    col1, col2 = st.columns(2)

    if input_method == "📸 Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png", "bmp"]
        )

        if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")  # ← Add .convert("RGB")
image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            with col1:
                st.subheader("Original Image")
                st.image(image_pil, use_container_width=True)

            # Run detection
            if st.button("🔍 Detect Faces & Ages", type="primary"):
                with st.spinner("Processing..."):
                    try:
                        predictions = detector.detect_and_predict(image_cv, confidence_threshold)  # ← Pass slider

                        if predictions:
                            output_image = draw_predictions(image_cv, predictions)
                            output_image_rgb = cv2.cvtColor(
                                output_image, cv2.COLOR_BGR2RGB
                            )

                            with col2:
                                st.subheader("Detection Results")
                                st.image(output_image_rgb, use_container_width=True)

                            st.success(f"✅ Detected {len(predictions)} face(s)")

                            with st.expander("📊 Detailed Results"):
                                for i, (bbox, age, conf) in enumerate(predictions):
                                    st.write(
                                        f"**Face {i+1}**: Age {age} (Confidence: {conf:.2%})"
                                    )
                        else:
                            st.warning("⚠️ No faces detected in the image.")

                    except Exception as e:
                        st.error(f"❌ Error during detection: {e}")

    else:  # Webcam
        st.info("📷 Webcam input requires running locally with `streamlit run app.py`")
        picture = st.camera_input("Take a picture")

        if picture:
         image_pil = Image.open(uploaded_file).convert("RGB")  # ← Add .convert("RGB")
image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            with col1:
                st.subheader("Captured Image")
                st.image(image_pil, use_container_width=True)

            if st.button("🔍 Detect Faces & Ages", type="primary"):
                with st.spinner("Processing..."):
                    try:
                        predictions = detector.detect_and_predict(image_cv, confidence_threshold)  # ← Pass slider

                        if predictions:
                            output_image = draw_predictions(image_cv, predictions)
                            output_image_rgb = cv2.cvtColor(
                                output_image, cv2.COLOR_BGR2RGB
                            )

                            with col2:
                                st.subheader("Detection Results")
                                st.image(output_image_rgb, use_container_width=True)

                            st.success(f"✅ Detected {len(predictions)} face(s)")

                            with st.expander("📊 Detailed Results"):
                                for i, (bbox, age, conf) in enumerate(predictions):
                                    st.write(
                                        f"**Face {i+1}**: Age {age} (Confidence: {conf:.2%})"
                                    )
                        else:
                            st.warning("⚠️ No faces detected in the image.")

                    except Exception as e:
                        st.error(f"❌ Error during detection: {e}")

    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using [Streamlit](https://streamlit.io) | "
        "[GitHub](https://github.com/NishantAhirwar07/Face_and_Age_detection)"
    )


if __name__ == "__main__":
    main()
