# Rice Leaf Disease Identification

## Overview
Rice Leaf Disease Identification is a deep learning project aimed at detecting and classifying diseases in rice leaves using computer vision techniques. The project supports two different models:

1. **ResNet152V2**: A deep residual network with 152 layers, providing high accuracy
2. **MobileNetV2**: A lightweight model optimized for mobile and edge devices, offering faster inference

Both models utilize transfer learning and are trained on a dataset containing approximately 2600 images belonging to six different classes.

### Project Video
[Watch the video here](https://www.loom.com/share/3ac68f165e2f4d1bb7d67fe5c1256e99?sid=97793e34-a80f-4686-b068-bf4ce70e7d8e)


## Dataset
The dataset used for training the model is sourced from Kaggle and contains images of rice leaves affected by different diseases. You can access the dataset [here](https://www.kaggle.com/datasets/dedeikhsandwisaputra/rice-leafs-disease-dataset).

## Model Training

### ResNet152V2 Model
The ResNet152V2 model is trained using transfer learning techniques. This deep residual network provides high accuracy but requires more computational resources.

### MobileNetV2 Model
The MobileNetV2 model is a lightweight alternative that offers:
- Faster inference time
- Lower memory usage
- Better performance on mobile/edge devices
- Good accuracy with reduced computational requirements

### Training Process
Both models use transfer learning, allowing them to leverage pre-trained weights from ImageNet and fine-tune them for rice leaf disease classification. The training process includes:

1. **Data Augmentation**: Rotation, shifting, flipping, and zooming to increase dataset diversity
2. **Transfer Learning**: Using pre-trained weights as starting point
3. **Fine-tuning**: Unfreezing top layers for domain-specific learning
4. **Early Stopping**: Preventing overfitting
5. **Learning Rate Scheduling**: Adaptive learning rate reduction

#### ResNet152V2 Specific Features:
- **Attention Layer**: Custom attention mechanism to focus on important parts of the image
- **Advanced Augmentation**: More sophisticated data augmentation including brightness adjustment
- **Top-3 Accuracy**: Additional metric for better evaluation
- **Comprehensive Analysis**: Detailed performance analysis and per-class accuracy

## Classes
The model is trained to classify rice leaf images into the following disease classes:
- Healthy
- Brown spot
- Leaf blast
- Bacterial leaf blight
- Leaf scald
- Narrow brown spot

## Deployment
The trained model is deployed using Flask, a lightweight Python web framework. It accepts images of rice leaves as input and identifies the disease present in the leaf.

## Usage

### Using Pre-trained Models
1. Clone the repository.
2. Install the required dependencies: `pip install -r requirement.txt`
3. Run the Flask application: `python app.py`
4. Open http://127.0.0.1:5000/ in your browser
5. Upload an image of a rice leaf to the application
6. The application will predict the disease present in the leaf and display the result

### Training Your Own Models

#### Download Dataset
```bash
python download_dataset.py
```

#### Train MobileNetV2 Model
```bash
python train_mobilenet.py
```

This will:
- Create a MobileNetV2 model with transfer learning
- Train on the rice leaf disease dataset
- Save the trained model as `mobilenetv2_model.h5`
- Optionally perform fine-tuning for better accuracy
- Generate training plots and evaluation metrics

#### Train ResNet152V2 Model
```bash
python train_resnet.py
```

This will:
- Create a ResNet152V2 model with attention layer
- Train on the rice leaf disease dataset
- Save the trained model as `resnet152_model.h5`
- Include advanced data augmentation
- Generate comprehensive training plots and evaluation metrics
- Provide detailed performance analysis
- Optionally perform fine-tuning for better accuracy

#### Model Selection
You can choose which model to use by setting the `MODEL_TYPE` environment variable:
- `MODEL_TYPE=resnet` (default) - Uses ResNet152V2
- `MODEL_TYPE=mobilenet` - Uses MobileNetV2

Or modify the model selection in the web interface.

## Run locally on Windows (PowerShell)

If you run into long-path issues when installing TensorFlow inside a project on OneDrive, create a virtualenv at a short path and install requirements there.

1. Open PowerShell and run:

```powershell
venv\Scripts\activate


2. Start the app:

```powershell
python.exe app.py
```

3. Open http://127.0.0.1:5000/ in your browser.

Notes:
- The project expects model files in the project root:
  - `resnet152_model.h5` for ResNet152V2 model
  - `mobilenetv2_model.h5` for MobileNetV2 model
- If you don't have the models, the server will run but predictions will return an error indicating the model is not loaded.
- You can train your own MobileNetV2 model using the provided training script.
- If you want to run in-place virtualenv inside the project, use `.venv` but you may encounter Windows long-path issues when installing TensorFlow inside OneDrive.

## Image Quality Check

The application now includes automatic image quality assessment before processing:

### Quality Checks:
- **Resolution**: Minimum 224x224, Maximum 4096x4096
- **Blur Detection**: Uses Laplacian variance to detect blur (threshold: 20-50)
- **Brightness & Contrast**: Ensures proper lighting conditions (brightness: 20-240, contrast: 15+)
- **File Size**: Validates file size (0.1MB - 50MB)
- **Format**: Supports JPEG, PNG, BMP, TIFF

### Quality Check Behavior:
- **Flexible Mode**: Very lenient thresholds to accept most images
- **Disable Option**: Users can completely disable quality checks
- **Skip Option**: Users can bypass quality checks for experimental analysis
- **Warning Mode**: Borderline quality images show warnings but proceed
- **Real-world Friendly**: Thresholds adjusted for practical use

### Quality Check Options:
1. **Default**: Basic quality checks with flexible thresholds
2. **Disabled**: No quality checks - accepts all images
3. **Skip**: Bypass quality checks when errors occur

### Quality Scoring:
- **A+ (90-100)**: Excellent quality
- **A (80-89)**: Good quality
- **B (70-79)**: Acceptable quality
- **C (60-69)**: Poor quality
- **D (50-59)**: Very poor quality
- **F (0-49)**: Unacceptable quality

### User Experience:
- Images with quality issues are rejected with detailed feedback
- Warnings are shown for borderline quality images
- Quality score and grade are displayed with results
- Specific recommendations are provided for improvement

## Model Comparison

| Feature | ResNet152V2 | MobileNetV2 |
|---------|-------------|-------------|
| **Accuracy** | High | Good |
| **Speed** | Slower | Faster |
| **Model Size** | Large (~600MB) | Small (~14MB) |
| **Memory Usage** | High | Low |
| **Best For** | High accuracy needs | Mobile/Edge deployment |
| **Input Size** | 256x256 | 224x224 |

## Contributors
- [Umang Rathi](https://github.com/umangrathi110)


