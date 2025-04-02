Part 1: Research & Selection
1. Identified Forgery Detection Approaches
Below are three models that show the most promise for detecting AI-generated human speech in real-time conversations:

1️⃣ ResNet-Based Detection Model
- **Key Technical Innovation:**  
  - Uses ResNet architecture for analyzing audio features.  
- **Reported Performance Metrics:**  
  - Competitive performance in ASVspoof challenge benchmarks.  
- **Why It’s Promising:**  
  - Suitable for real-time processing and analyzing live speech.  
- **Potential Limitations:**  
  - May struggle with dynamically changing conversational speech.  

2️⃣ DEEP-VOICE Dataset with XGBoost Classifier
- **Key Technical Innovation:**  
  - Employs XGBoost classifier trained on the DEEP-VOICE dataset.  
- **Reported Performance Metrics:**  
  - Achieved **99.3% accuracy** in 10-fold cross-validation.  
- **Why It’s Promising:**  
  - High accuracy suggests strong performance in AI voice detection.  
- **Potential Limitations:**  
  - Needs validation for real-time conversational settings.  

3️⃣ Pindrop’s Real-Time Deepfake Detection
- **Key Technical Innovation:**  
  - Designed for **real-time detection** of deepfake audio.  
- **Reported Performance Metrics:**  
  - Claims **over 90% accuracy** in distinguishing real vs. fake speech.  
- **Why It’s Promising:**  
  - Specifically built for analyzing live conversations.  
- **Potential Limitations:**  
  - Proprietary software may limit customization.  

These approaches are well-suited for real-time detection of AI-generated speech and align with our project requirements.
