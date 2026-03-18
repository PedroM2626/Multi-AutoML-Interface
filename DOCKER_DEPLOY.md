# Docker deployment guide

This repository already includes `Dockerfile` at the project root which builds a Python 3.11 image and installs `requirements.txt`.

Quick local build (tags image `multi-automl:latest`):

```powershell
# build locally
docker build -t multi-automl:latest .

# run locally
docker run --rm -p 8501:8501 multi-automl:latest
```

Publish to Docker Hub (replace `myuser`):

```powershell
docker tag multi-automl:latest myuser/multi-automl:latest
docker push myuser/multi-automl:latest
```

Deploy to Google Cloud Run (using pushed image):

```bash
gcloud run deploy multi-automl \
  --image=myuser/multi-automl:latest \
  --platform=managed \
  --region=us-central1 \
  --allow-unauthenticated \
  --port=8501
```

Deploy to Render using Docker (manual):

1. Create a new Web Service in Render.
2. Choose "Docker" and point Render to the repository (or a Docker image URL).
3. Use `8501` as the internal port and the default health check.

Automated CI (recommended): push images from GitHub Actions to Docker Hub or GHCR,
then configure your cloud provider (Render / Cloud Run) to deploy that image.

Notes and tips
- Streamlit Community Cloud does not support custom Docker images; use Render/Cloud Run/other hosts for full Docker control.
- If your image build fails due to heavy compiled packages, prefer precompiled wheels and increase Docker build resources or offload heavy workloads to a separate service.
- If you'd like, I can scaffold a GitHub Actions workflow that builds and pushes images to Docker Hub or GHCR (you'll need to add repository secrets).
