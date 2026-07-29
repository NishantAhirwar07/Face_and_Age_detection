# Streamlit Deployment Guide

## Quick Start — Deploy in 5 Minutes

### 1. Prerequisites
- GitHub account
- This repository pushed to GitHub
- Streamlit Cloud account (free, 1 click signup at [streamlit.io](https://streamlit.io))

### 2. Deploy Steps

1. **Go to [Streamlit Cloud](https://share.streamlit.io)**
2. Click **"Create app"**
3. Fill in:
   - **Repository**: `NishantAhirwar07/Face_and_Age_detection`
   - **Branch**: `main` (or your target branch)
   - **Main file path**: `app.py`
4. Click **"Deploy"**
5. Wait 2-3 minutes (models download on first run)
6. **Done!** 🎉 Your app is live

Your URL will be: `https://[your-username]-face-age-detection.streamlit.app`

---

## What Happens During Deployment

### 1. Build Phase (~30 sec)
- Streamlit Cloud provisions a container
- Installs system packages from `packages.txt`
- Installs Python packages from `requirements.txt`

### 2. Runtime Phase
- App starts with `streamlit run app.py`
- Models are downloaded on **first user interaction** (not at startup)
- Models are cached in `.streamlit/cache/` for fast reloads

### 3. First User Experience
- Upload image or capture with camera
- Models auto-download (~60 sec, ~80 MB)
- Results show instantly

### 4. Subsequent Users
- Instant load (cached models)
- No download delay

---

## Files Explained

### `.streamlit/config.toml`
Streamlit configuration for cloud deployment:
```toml
[server]
headless = true              # Run without GUI (required for cloud)
maxUploadSize = 100          # Max 100 MB uploads

[theme]
primaryColor = "#F2A65A"     # Chronoscope amber accent
backgroundColor = "#0D1117"  # Dark mode
```

### `packages.txt`
System-level dependencies for Linux container:
```
libsm6              # OpenCV dependency
libxext6            # OpenCV dependency
libxrender-dev      # OpenCV rendering
```

### `requirements.txt`
Python packages:
```
opencv-python-headless>=4.8.0    # Headless (no GUI)
streamlit>=1.28.0                # Web framework
numpy>=1.24.0                    # Numerical computing
pillow>=9.5.0                    # Image processing
```

### `app.py`
Main Streamlit web app with UI, file upload, camera capture.

### `detector.py`
Core detection API (reusable for CLI, REST, batch, etc.).

---

## Custom Domain (Optional)

1. Go to your app's **Settings** in Streamlit Cloud
2. Under "Custom domains", add your domain
3. Update DNS CNAME records (Streamlit provides exact values)
4. Wait 5-10 minutes for DNS propagation

---

## Troubleshooting

### ❌ "Model download failed"
- Check internet (models are ~80 MB)
- Click "Rerun" to retry
- Check app logs in Streamlit Cloud dashboard

### ❌ "Out of memory"
- Models need ~1 GB RAM
- Upload smaller images
- Default Streamlit Cloud includes 1 GB RAM (sufficient)

### ❌ "Upload size exceeded"
- Max is 100 MB (set in `config.toml`)
- Compress images to <50 MB
- Or increase limit in `config.toml` (if needed)

### ❌ App won't deploy
- Check `app.py` for syntax errors
- Verify all imports exist in `requirements.txt`
- View deployment logs in Streamlit Cloud dashboard

### ✅ How to view logs
1. Go to your app on Streamlit Cloud
2. Click **"Manage app"** (gear icon)
3. Select **"View logs"** to see errors

---

## Scaling & Monitoring

### Performance
- **Cold start**: ~3 sec (models cached)
- **Inference time**: ~2-5 sec per image (depends on image size)
- **Concurrent users**: Streamlit Cloud handles auto-scaling

### Limits (Free Tier)
- 3 apps per account
- 1 GB app memory
- Stops after 7 days of inactivity (restarts on access)
- No persistent storage

### Upgrade to Pro
- Custom domains
- Persistent storage
- Priority compute
- Higher resource limits
- See [streamlit.io/cloud](https://streamlit.io/cloud) for pricing

---

## Advanced: Self-Hosted Deployment

If you want to deploy elsewhere (AWS, GCP, Heroku, VPS):

### Using Docker
```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build & Run
```bash
docker build -t chronoscope .
docker run -p 8501:8501 chronoscope
```

### Environment Variables
```bash
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

---

## Local Development

### Run locally
```bash
streamlit run app.py
```

### Auto-reload on file changes
```bash
streamlit run app.py --logger.level=debug
```

### Live URL for sharing (local testing only)
```bash
streamlit run app.py --logger.level=debug
# Shares localhost with ngrok (limited time)
```

---

## FAQ

**Q: Why do models take so long to download?**
A: The age classifier is ~70 MB. This only happens once per deployment; subsequent runs use cached models.

**Q: Can I use `opencv-python` instead of `opencv-python-headless`?**
A: No. The headless version doesn't require GUI libraries (saves space, required for cloud).

**Q: How do I update the app?**
A: Push changes to GitHub. Streamlit Cloud auto-deploys on git push (check "Auto deploy" in settings).

**Q: Can I use GPU for inference?**
A: Yes, but it requires paid Streamlit Pro. Upgrade in dashboard.

**Q: Where are model files stored?**
A: In `.streamlit/cache/models/` on Streamlit Cloud (ephemeral, re-downloaded if needed).

**Q: Can I run this on Hugging Face Spaces?**
A: Yes, create a Streamlit app repo on HF Spaces and push this code.

---

## Next Steps

1. ✅ Ensure `app.py`, `detector.py`, `requirements.txt` are in repo root
2. ✅ Push to GitHub
3. ✅ Go to [Streamlit Cloud](https://share.streamlit.io) and create app
4. ✅ Share your URL!

---

**Need help?**
- [Streamlit Docs](https://docs.streamlit.io)
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-cloud)
- [GitHub Issues](https://github.com/NishantAhirwar07/Face_and_Age_detection/issues)
