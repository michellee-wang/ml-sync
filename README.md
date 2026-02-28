# ML Service - Music Genre Prediction & Fine-tuning

A Python-based machine learning service for music genre prediction and fine-tuning.

## Overview

This service provides:
- **Genre Prediction**: Classify music tracks into genres using pretrained models
- **Fine-tuning**: Adapt models to specific musical preferences
- **Audio Analysis**: Extract features from audio files for classification
- **API Integration**: RESTful API endpoints for model inference and training

## Architecture

```
services/ml-service/
├── src/
│   ├── api/          # FastAPI endpoints
│   ├── models/       # ML model definitions
│   ├── training/     # Training and fine-tuning scripts
│   └── inference/    # Inference pipeline
├── pretrained/       # Pretrained model weights
├── data/            # Training/validation data
├── scripts/         # Utility scripts
├── requirements.txt
└── Dockerfile
```

## Setup

### Local Development

1. **Create a virtual environment**:
```bash
cd services/ml-service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
Create a `.env` file in the service root:
```env
MODEL_PATH=./pretrained/genre_classifier
FINE_TUNE_DATA_PATH=./data/training
```

4. **Download pretrained models**:
```bash
python scripts/download_models.py
```

5. **Run the service**:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

1. **Build the image**:
```bash
docker build -t ml-service:latest .
```

2. **Run the container**:
```bash
docker run -p 8000:8000 \
  -v $(pwd)/pretrained:/app/pretrained \
  -v $(pwd)/data:/app/data \
  ml-service:latest
```

## Pretrained Models

### Loading Pretrained Models

The service uses pretrained transformer models for music classification:

1. **Base Model**: Loaded from `pretrained/` directory or Hugging Face Hub
2. **Model Types**:
   - Audio spectrogram transformers (AST)
   - BERT-based audio classifiers
   - Custom CNN architectures

Example loading code:
```python
from transformers import AutoModel, AutoFeatureExtractor

model = AutoModel.from_pretrained("./pretrained/genre_classifier")
feature_extractor = AutoFeatureExtractor.from_pretrained("./pretrained/genre_classifier")
```

### Supported Models

- **MIT/ast-finetuned-audioset**: Audio Spectrogram Transformer
- **facebook/musicgen**: Music generation and analysis
- **Custom models**: Fine-tuned on genre-specific datasets

## Fine-tuning

### Fine-tuning Process

1. **Prepare Training Data**:
- Audio features are extracted from audio files
- Data is split into train/validation sets

2. **Fine-tune Model**:
```bash
python src/training/fine_tune.py \
  --base-model ./pretrained/genre_classifier \
  --data-path ./data/training \
  --output-path ./models/fine_tuned \
  --epochs 10 \
  --batch-size 16
```

3. **Evaluation**:
- Validate on held-out test set
- Compare against base model performance
- Generate classification reports

### Fine-tuning Configuration

```python
{
  "learning_rate": 2e-5,
  "num_epochs": 10,
  "batch_size": 16,
  "warmup_steps": 500,
  "weight_decay": 0.01,
  "evaluation_strategy": "epoch"
}
```

## API Endpoints

### Health & Status

- `GET /health` - Service health check
- `GET /models` - List available models

### Inference

- `POST /predict` - Predict genre for audio file
  ```json
  {
    "audio_url": "https://...",
    "model_name": "genre_classifier"
  }
  ```

- `POST /analyze/features` - Extract audio features
  ```json
  {
    "audio_url": "https://..."
  }
  ```

### Training & Fine-tuning

- `POST /finetune/start` - Start fine-tuning job
  ```json
  {
    "base_model": "genre_classifier",
    "config": {
      "epochs": 10,
      "learning_rate": 2e-5
    }
  }
  ```

- `GET /finetune/status/{job_id}` - Check fine-tuning progress

- `POST /finetune/stop/{job_id}` - Stop fine-tuning job

### Model Management

- `POST /models/upload` - Upload custom model
- `DELETE /models/{model_name}` - Delete model
- `GET /models/{model_name}/metrics` - Get model performance metrics

## Data Flow

1. **Inference**:
   - Client sends audio file
   - Service extracts audio features
   - Model predicts genre probabilities
   - Returns top-k predictions with confidence scores

2. **Fine-tuning**:
   - Load training audio data and extract features
   - Fine-tune pretrained model on training data
   - Save fine-tuned model for predictions

## Performance Optimization

- **Batch Inference**: Process multiple tracks simultaneously
- **Model Caching**: Keep frequently-used models in memory
- **Feature Caching**: Store extracted features to avoid recomputation
- **GPU Acceleration**: Use CUDA for faster inference and training

## Monitoring

Prometheus metrics exposed at `/metrics`:
- Request latency
- Model inference time
- Training job status
- Error rates

## Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Integration tests
pytest tests/integration/
```

## Security Considerations

- API authentication required for fine-tuning endpoints
- Rate limiting on all endpoints
- Input validation for audio files

## License

MIT License
# ml-sync
