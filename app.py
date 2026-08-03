# snippet to replace/augment your load() method
import os
import cv2

def load(self):
    base_dir = os.path.dirname(__file__)  # folder containing detector.py
    prototxt = os.path.join(base_dir, "models", "deploy_age.prototxt")
    caffemodel = os.path.join(base_dir, "models", "age_net.caffemodel")

    # Helpful debug/logging: raise a friendly error if files are missing
    if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
        raise FileNotFoundError(
            "Caffe model files not found.\n"
            f"Expected: {prototxt}\n"
            f"Expected: {caffemodel}\n"
            "Please place the .prototxt and .caffemodel under the package 'models/' directory "
            "or set the correct path in the code."
        )

    try:
        self._age_net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    except cv2.error as e:
        # Re-raise with actionable guidance
        raise RuntimeError(
            "OpenCV failed to load the Caffe model. Possible causes:\n"
            "- OpenCV wheel without DNN support was installed (use opencv-python-headless).\n"
            "- Model files are corrupted or incompatible.\n"
            f"Original error: {e}"
        ) from e
