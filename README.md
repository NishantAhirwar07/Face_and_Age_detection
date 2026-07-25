# Face Detection & Age Prediction

A Python/OpenCV notebook that detects faces in an image using a pretrained
TensorFlow SSD face detector, then predicts the approximate age of each
detected face using a pretrained Caffe age-classification model.

## How it works

1. **Face detection** — `opencv_face_detector_uint8.pb` (TensorFlow) locates
   all faces in the input image and draws bounding boxes around them.
2. **Age prediction** — for each detected face, a small padded crop is fed
   into `age_net.caffemodel` (Caffe), which classifies the face into one of
   8 age brackets.
3. The result is displayed with bounding boxes and predicted age labels
   overlaid on the original image.

## Project structure

```
.
├── Face_detection.ipynb     # Main notebook
├── requirements.txt         # Python dependencies
├── download_models.sh       # Downloads the pretrained model files
└── README.md
```

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd face-age-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the pretrained models

The model weight files are not committed to this repo (they're large
binaries). Download them with:

```bash
bash download_models.sh
```

This fetches the following files into the project root:

| File | Framework | Purpose |
|---|---|---|
| `opencv_face_detector.pbtxt` | TensorFlow | Face detector graph config |
| `opencv_face_detector_uint8.pb` | TensorFlow | Face detector weights |
| `age_deploy.prototxt` | Caffe | Age classifier architecture |
| `age_net.caffemodel` | Caffe | Age classifier weights |

### 4. Add an input image

Place an image (e.g. `kid1.jpg`) in the project root, or update the
`image_path` variable in the notebook to point to your own image.

### 5. Run

Open `Face_detection.ipynb` in Jupyter, JupyterLab, VS Code, or Google
Colab, and run all cells.

> **Note:** If running locally (not in Colab), replace the
> `from google.colab.patches import cv2_imshow` import and `cv2_imshow(frame)`
> call with standard OpenCV display code:
> ```python
> cv2.imshow("Result", frame)
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```

### 6. Known issue — missing constants

The `predict_age()` function references `MODEL_MEAN_VALUES` and `age_list`,
which need to be defined before it's called. Add this near the top of the
notebook (after the model-loading cell):

```python
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
```

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy

See `requirements.txt` for exact versions.

## Credits

Pretrained models sourced from the
[LearnOpenCV AgeGender project](https://github.com/spmallick/learnopencv/tree/master/AgeGender).

## License

MIT (or update to match your preferred license).
