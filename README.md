# Face Detection & Age Prediction

A Python/OpenCV web app (powered by **Streamlit**) that detects faces in images and estimates their age using pretrained neural networks.

## 🚀 Try It Live

**[Open Chronoscope on Streamlit Cloud](https://chronoscope.streamlit.app)**

## How it works

1. **Face detection** — `opencv_face_detector_uint8.pb` (TensorFlow) locates
   all faces in the input image and draws bounding boxes around them.
2. **Age prediction** — for each detected face, a small padded crop is fed
   into `age_net.caffemodel` (Caffe), which classifies the face into one of
   8 age brackets.
3. The result is displayed with bounding boxes and predicted age labels
   overlaid on the original image in a beautiful, interactive web interface.

## Features

- 📷 **Upload images** or **capture with camera** (live webcam support)
- 🎯 **Adjustable sensitivity** — control face detection confidence threshold
- 📊 **Real-time results** — instant face detection and age estimation
- 📥 **Download results** — save annotated images and JSON scan data
- 📈 **Scan history** — track all detections from your session
- 🎨 **Modern UI** — dark theme with retro-futuristic Chronoscope design

## Project structure

```
.
├── app.py                   # Main Streamlit web app
├── detector.py              # Core detection logic (UI-agnostic)
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── packages.txt            # System dependencies for deployment
├── Face_detection.ipynb     # Original notebook (reference)
└── README.md
```

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/NishantAhirwar07/Face_and_Age_detection
cd Face_and_Age_detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Model Files

Model weights are not committed to this repo (they're large binaries, 70+ MB total).
The app **automatically downloads them on first run** from these sources:

| File | Size | Framework | Purpose |
|---|---|---|---|
| `opencv_face_detector.pbtxt` | ~1 MB | TensorFlow | Face detector graph config |
| `opencv_face_detector_uint8.pb` | ~5 MB | TensorFlow | Face detector weights |
| `age_deploy.prototxt` | ~1 KB | Caffe | Age classifier architecture |
| `age_net.caffemodel` | ~70 MB | Caffe | Age classifier weights |

After download, they're cached in `./models/` for fast subsequent runs.

## Deploy to Streamlit Cloud

### Prerequisites

- GitHub account
- Streamlit Cloud account (free at [streamlit.io](https://streamlit.io))

### Steps

1. **Push this repo to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit deployment files"
   git push origin main
   ```

2. **Go to [Streamlit Cloud](https://share.streamlit.io)** and sign in with GitHub

3. **Click "Create app"** and select:
   - **Repository**: Your fork of this repo
   - **Branch**: `main`
   - **Main file path**: `app.py`

4. **Click "Deploy"**

Streamlit Cloud will:
- Install dependencies from `requirements.txt`
- Install system packages from `packages.txt`
- Download model files on first run (takes ~1 min on first load)
- Cache them for subsequent runs

Your app will be live at `https://<your-username>-face-detection.streamlit.app`

## Notes

- **First run takes ~60 seconds** as models are downloaded and cached
- **Subsequent runs are instant** (models stay cached in `.streamlit/cache`)
- **Upload limit**: 100 MB (configurable in `.streamlit/config.toml`)
- **Models auto-download** — no manual setup required

## Performance Tips

- Upload images under 10 MB for best performance
- JPEG format is slightly faster than PNG
- Adjust sensitivity slider to reduce false positives
- Lower crop margin for tighter face detection

## Troubleshooting

### "Model download failed" error
- Check your internet connection
- Try refreshing the page (models will retry download)
- Clear browser cache if it persists

### Slow performance on first load
- First load downloads ~80 MB of models — this is normal
- Subsequent sessions use cached models (instant load)

### Out of memory
- Try uploading a smaller image
- Close other browser tabs to free memory

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions

## Credits

**Model sources:**
- Face detector: [LearnOpenCV AgeGender](https://github.com/spmallick/learnopencv/tree/master/AgeGender)
- Age classifier: [eveningglow/age-and-gender-classification](https://github.com/eveningglow/age-and-gender-classification)

**Built with:**
- [OpenCV](https://opencv.org) — Computer vision
- [Streamlit](https://streamlit.io) — Web framework
- [TensorFlow](https://tensorflow.org) — Face detection
- [Caffe](https://caffe.berkeleyvision.org) — Age classification

## License

MIT — Feel free to use this for personal or commercial projects.

