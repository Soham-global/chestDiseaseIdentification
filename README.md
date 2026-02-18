# 🫁 Chest Disease Classification using Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.2.2-blue.svg)
![DVC](https://img.shields.io/badge/DVC-Enabled-945DD6.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end deep learning solution for automated chest disease classification from CT scan images using transfer learning, MLflow experiment tracking, and DVC pipeline management.

[Features](#-features) • [Architecture](#-project-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Workflow](#-development-workflow)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Architecture](#-project-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Development Workflow](#-development-workflow)
- [Project Structure](#-project-structure)
- [API Endpoints](#-api-endpoints)
- [Model Pipeline](#-model-pipeline)
- [MLflow Integration](#-mlflow-integration)
- [DVC Pipeline](#-dvc-pipeline)
- [Docker Deployment](#-docker-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

This project implements an **end-to-end machine learning pipeline** for classifying chest diseases from CT scan images. It leverages **transfer learning** with pre-trained models, incorporates **MLflow** for experiment tracking and model versioning, and uses **DVC** for reproducible ML pipelines.

The system provides a **Flask-based web interface** for real-time predictions and includes comprehensive logging, modular architecture, and production-ready deployment configurations.

### 🎯 Key Objectives

- Automated chest disease detection from CT scan images
- Reproducible ML pipeline with version control
- Experiment tracking and model management
- Production-ready deployment with Docker
- RESTful API for seamless integration

---

## ✨ Features

- 🧠 **Transfer Learning**: Utilizes pre-trained ImageNet models for efficient training
- 📊 **MLflow Integration**: Complete experiment tracking, metrics logging, and model registry
- 🔄 **DVC Pipeline**: Version-controlled data and reproducible ML workflows
- 🌐 **Web Interface**: User-friendly Flask application for image upload and prediction
- 🔌 **REST API**: Easy integration with external applications
- 📦 **Modular Design**: Clean, maintainable code following software engineering best practices
- 🐳 **Docker Support**: Containerized deployment for consistency across environments
- 📝 **Comprehensive Logging**: Detailed logging for debugging and monitoring
- ⚙️ **Configuration Management**: YAML-based configuration for easy customization
- 🔁 **CI/CD Ready**: GitHub Actions workflow for automated testing and deployment

---

## 🏗️ Project Architecture

```
┌─────────────────┐
│  Data Ingestion │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Prepare Base    │
│ Model (VGG16)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │
│ with Transfer   │
│ Learning        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Evaluation│
│ with MLflow     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Flask Web App   │
│ for Prediction  │
└─────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | TensorFlow 2.12.0, Keras |
| **ML Ops** | MLflow 2.2.2, DVC, DagsHub |
| **Web Framework** | Flask, Flask-CORS |
| **Data Processing** | NumPy, Pandas, SciPy |
| **Visualization** | Matplotlib, Seaborn |
| **Configuration** | PyYAML, python-box |
| **Utilities** | gdown, tqdm, joblib |
| **Deployment** | Docker, GitHub Actions |

---

## 📥 Installation

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) Docker for containerized deployment

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/chestDiseaseIdentification.git
cd chestDiseaseIdentification
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Initialize DVC

```bash
dvc init
dvc remote add -d storage <your-remote-storage>
```

---

## 🚀 Usage

### Training the Model

#### Option 1: Using Python Script

```bash
python main.py
```

This will execute all pipeline stages:
1. Data Ingestion
2. Base Model Preparation
3. Model Training
4. Model Evaluation

#### Option 2: Using DVC Pipeline

```bash
dvc repro
```

### Running the Web Application

```bash
python app.py
```

The application will be available at `http://localhost:8080`

### Making Predictions via API

```python
import requests
import base64

# Read and encode image
with open("chest_scan.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Send prediction request
response = requests.post(
    "http://localhost:8080/predict",
    json={"image": image_data}
)

print(response.json())
```

---

## 🔄 Development Workflow

Follow this systematic workflow for development and experimentation:

### 1️⃣ Update Configuration Files

```yaml
# config/config.yaml - Define paths and directories
# params.yaml - Set hyperparameters
```

### 2️⃣ Update Entity Classes

Define data classes in `src/ChestDiseaseClassifier/entity/`

### 3️⃣ Update Configuration Manager

Modify `src/ChestDiseaseClassifier/config/configuration.py`

### 4️⃣ Implement Components

Create/update components in `src/ChestDiseaseClassifier/components/`

### 5️⃣ Create Pipeline Stages

Implement pipeline stages in `src/ChestDiseaseClassifier/pipeline/`

### 6️⃣ Update Main Pipeline

Integrate stages in `main.py`

### 7️⃣ Update DVC Pipeline

Configure `dvc.yaml` for reproducibility

### 8️⃣ Test and Validate

Run experiments and track with MLflow

---

## 📁 Project Structure

```
chestDiseaseIdentification/
│
├── .github/
│   └── workflows/
│       └── main.yaml              # CI/CD pipeline
│
├── artifacts/                     # Generated artifacts
│   ├── data_ingestion/
│   ├── prepare_base_model/
│   └── training/
│
├── config/
│   └── config.yaml                # Configuration settings
│
├── research/                      # Jupyter notebooks for experimentation
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_model_trainer.ipynb
│   └── 04_model_evaluation_with_mlflow.ipynb
│
├── src/ChestDiseaseClassifier/
│   ├── components/                # Core components
│   ├── config/                    # Configuration management
│   ├── constants/                 # Constants and paths
│   ├── entity/                    # Data classes
│   ├── pipeline/                  # Pipeline stages
│   └── utils/                     # Utility functions
│
├── static/                        # Static files for web app
│   └── styles.css
│
├── templates/                     # HTML templates
│   └── index.html
│
├── app.py                         # Flask application
├── main.py                        # Training pipeline
├── dvc.yaml                       # DVC pipeline configuration
├── params.yaml                    # Hyperparameters
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── Dockerfile                     # Docker configuration
└── README.md                      # Project documentation
```

---

## 🌐 API Endpoints

### `GET /`
Returns the web interface for image upload and prediction.

### `POST /train`
Triggers the complete training pipeline.

**Response:**
```json
"Training done successfully!"
```

### `POST /predict`
Accepts a base64-encoded image and returns prediction results.

**Request Body:**
```json
{
  "image": "<base64-encoded-image>"
}
```

**Response:**
```json
{
  "prediction": "Normal/Diseased",
  "confidence": 0.95
}
```

---

## 🔬 Model Pipeline

### Stage 1: Data Ingestion
- Downloads CT scan dataset from Google Drive
- Extracts and organizes data into training structure
- Validates data integrity

### Stage 2: Prepare Base Model
- Loads pre-trained VGG16 model (ImageNet weights)
- Removes top classification layers
- Adds custom classification head for binary classification
- Compiles model with specified optimizer and loss function

### Stage 3: Model Training
- Applies data augmentation (rotation, flip, zoom)
- Trains model on chest CT scan images
- Implements callbacks for early stopping and model checkpointing
- Saves trained model artifacts

### Stage 4: Model Evaluation
- Evaluates model on test dataset
- Logs metrics to MLflow (accuracy, loss, precision, recall)
- Generates confusion matrix and classification report
- Registers model in MLflow model registry

---

## 📊 MLflow Integration

### Tracking Experiments

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("chest-disease-classification")

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.tensorflow.log_model(model, "model")
```

### View MLflow UI

```bash
mlflow ui
```

Access at `http://localhost:5000`

---

## 🔄 DVC Pipeline

### Pipeline Stages

```yaml
stages:
  - data_ingestion
  - prepare_base_model
  - training
  - evaluation
```

### Run Specific Stage

```bash
dvc repro <stage_name>
```

### View Pipeline DAG

```bash
dvc dag
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t chest-disease-classifier .
```

### Run Container

```bash
docker run -p 8080:8080 chest-disease-classifier
```

### Docker Compose (Optional)

```bash
docker-compose up
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**

- GitHub: (https://github.com/Soham-global)
- LinkedIn: (https://www.linkedin.com/in/sohamkalsi/)

---

## 🙏 Acknowledgments

- Dataset: [Chest CT Scan Images Dataset]
- Pre-trained Model: VGG16 (ImageNet)
- MLflow for experiment tracking
- DVC for pipeline management

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ and Python

</div>
