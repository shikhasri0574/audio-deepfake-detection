# Audio Deepfake Detection Report

## 1. Introduction
Audio deepfake detection is a crucial task in safeguarding the authenticity of speech recordings. This project leverages deep learning to classify audio files as either *Real* or *Fake* using a Convolutional Neural Network (CNN).

## 2. Dataset
### 2.1 Data Collection
The dataset comprises real and fake audio samples. Fake samples were synthetically generated using AI-based voice cloning techniques. To ensure consistency, all audio files underwent preprocessing before model training.

### 2.2 Preprocessing
- **Resampling:** Audio files were resampled to 16 kHz.
- **Trimming/Padding:** All samples were adjusted to exactly 2 seconds in duration.
- **Feature Extraction:** Mel-Frequency Cepstral Coefficients (MFCC) were extracted to represent audio signals numerically.

## 3. My Approach
The approach to solving the **audio deepfake detection problem** was structured in multiple phases:

### **1️⃣ Understanding the Problem**
- The goal was to classify audio files as **real** or **fake** based on extracted features.
- Fake audio samples were generated using AI-based voice cloning techniques.

### **2️⃣ Dataset Preparation & Preprocessing**
- **Data Cleaning:** Ensured all audio files were of high quality and consistent format.
- **Standardization:** Resampled all files to **16 kHz** and trimmed/padded them to **2 seconds**.
- **Feature Extraction:** Used **Mel-Frequency Cepstral Coefficients (MFCCs)** for numerical representation.

### **3️⃣ Model Selection & Training**
- Decided to use a **Convolutional Neural Network (CNN)** due to its effectiveness in **audio spectrogram analysis**.
- Trained the model using:
  - **Cross-Entropy Loss** for classification
  - **Adam Optimizer** with a learning rate of **0.001**
  - **Batch size of 16** and **5 epochs**

### **4️⃣ Performance Evaluation**
- Used **Accuracy, Precision, Recall, F1-score, and Confusion Matrix** to analyze model performance.
- Identified and mitigated issues like **class imbalance and overfitting**.

### **5️⃣ Refinement & Future Improvements**
- Explored different architectures to improve generalization.
- Suggested adding **data augmentation** techniques like noise injection and pitch shifting.

## 4. Feature Extraction and Dataset Preparation
To convert raw audio into numerical representations suitable for model training, the following steps were performed:
- **Audio Length Fixing:** Each audio file was padded or truncated to a fixed length of 16,000 samples (1 second at 16 kHz).
- **MFCC Extraction:** 20 Mel-frequency cepstral coefficients (MFCCs) were extracted using a window size of 512 and a hop length of 160.
- **Time Step Normalization:** The MFCCs were either padded or truncated to ensure a fixed time-step length of 101.
- **Tensor Conversion:** The extracted features were converted into PyTorch tensors to be used for model training.

## 5. Model Architecture
The model is a **Convolutional Neural Network (CNN)** designed for audio classification:
- **Conv2D Layer 1:** 16 filters, kernel size (3x3), ReLU activation
- **MaxPooling Layer:** Reduces dimensionality
- **Conv2D Layer 2:** 32 filters, kernel size (3x3), ReLU activation
- **Fully Connected Layer:** 128 neurons, ReLU activation
- **Output Layer:** 2 neurons (Real/Fake) with Softmax activation

## 6. Training Process
### 6.1 Model Training
- **Loss Function:** Cross-Entropy Loss
- **Optimizer:** Adam (learning rate: 0.001)
- **Batch Size:** 16
- **Epochs:** 5
- **Hardware:** Trained on GPU (if available)

### 6.2 Performance Metrics
The model was evaluated using the following metrics:
- **Accuracy**
- **Precision, Recall, and F1-score**
- **Confusion Matrix**

## 7. Results
- **Training Accuracy:** 98.95%
- **Validation Accuracy:** 96.36%

### **Confusion Matrix**
![Confusion Matrix](Audio-Deepfake-Detection/images/output_confusion_metrics.png)

### **Classification Report**
![Classification Report](Audio-Deepfake-Detection/images/Screenshot%202025-03-31%20220920.png)

## 8. Challenges and Future Work
### 8.1 Challenges Faced:
- **Data Imbalance:** Ensured an equal distribution of Real and Fake samples.
- **Overfitting:** Used dropout layers and regularization to prevent overfitting.

### 8.2 Assumptions Made:
- All fake samples were synthetically generated and do not include real audio manipulations.
- MFCC features are sufficient for classification without additional spectral features.

### 8.3 Future Improvements:
- **Experiment with Transformer-based models** for improved feature extraction.
- **Apply data augmentation techniques** (e.g., noise injection, pitch shifting) to enhance model robustness.
- **Train on a larger dataset** for better generalization.

## 9. Model Deployment & Inference
The trained model can be loaded and used for inference using the following script:
```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("saved_models/audio_detection_model.pth", map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded successfully!")
