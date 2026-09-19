# 🌿 Plant Disease Detector

An end-to-end deep learning system that identifies plant diseases from leaf images, served as a REST API and deployed on the cloud with a full CI/CD pipeline.

🔗 **Live API:** https://plant-disease-detector-api-ni6s.onrender.com/docs

> Note: This is deployed on a free-tier server, so the first request may take 30–50 seconds while it wakes up.

## Overview

This project trains a convolutional neural network to classify plant leaf images into 15 disease/health categories across crops like tomato, apple, corn, and grape. The model is served through a FastAPI backend, containerized with Docker, automatically validated on every push via GitHub Actions, and deployed live on Render.

## Tech Stack

- **Model:** TensorFlow / Keras, transfer learning with MobileNetV2
- **API:** FastAPI, Uvicorn
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Deployment:** Render
- **Dataset:** [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) (Kaggle)

## Results

- **Validation accuracy:** 98%
- Trained for 5 epochs using transfer learning (frozen MobileNetV2 base)
- 15 disease/health classes across multiple crop types

## How It Works

1. An image of a plant leaf is uploaded through the `/predict` endpoint
2. The image is preprocessed and resized to 224x224
3. The trained CNN predicts the most likely disease class
4. The API returns the predicted class and confidence score as JSON

## API Usage

**Endpoint:** `POST /predict`

\`\`\`bash
curl -X POST "https://plant-disease-detector-api-ni6s.onrender.com/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@leaf.jpg"
\`\`\`

**Example response:**
\`\`\`json
{
  "prediction": "Grape___Black_rot",
  "confidence": 0.896
}
\`\`\`

## Running Locally

\`\`\`bash
git clone https://github.com/jidil-safa/plant-disease-detector.git
cd plant-disease-detector

pip install -r requirements.txt

python -m uvicorn main:app --reload
\`\`\`

Then visit `http://127.0.0.1:8000/docs` to test it interactively.

### Running with Docker

\`\`\`bash
docker build -t plant-disease-api .
docker run -p 8000:8000 plant-disease-api
\`\`\`

## Project Structure

\`\`\`
plant-disease-detector/
├── main.py                  # FastAPI application
├── requirements.txt         # Python dependencies
├── Dockerfile                # Container definition
├── plant_disease_model.h5   # Trained CNN model
├── class_names.json         # Disease class labels
└── .github/workflows/       # CI/CD pipeline (GitHub Actions)
\`\`\`

## CI/CD

Every push to `main` automatically triggers a GitHub Actions workflow that builds the Docker image to verify the app still builds correctly before deployment.

## Future Improvements

- Expand to all 38 classes from the full dataset
- Add a simple frontend for image upload
- Add model versioning and automated retraining
- Convert model to TensorFlow Lite for faster inference

## License

MIT