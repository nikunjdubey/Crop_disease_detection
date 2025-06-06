## 🔍 Model Training & Analysis

This section provides a detailed breakdown of the training process, model architecture, and evaluation for the Plant Disease Detection system.

### 📦 Dataset
- **Source**: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Images**: 87,000+ images
- **Classes**: 38 (including healthy and diseased leaves)
- **Loading Method**: `image_dataset_from_directory` (with train/validation split)

## Model Live Demo
https://mainpy-yvip3qmpcr5atacyrzakxp.streamlit.app/

---

### 🧠 Model Architecture

The model is a deep **Convolutional Neural Network (CNN)** built using the Keras `Sequential` API.

**Model Layers:**
```text
Input: 128x128x3 RGB images
↓ Conv2D(32, 3x3) + ReLU
↓ Conv2D(32, 3x3) + ReLU
↓ MaxPooling2D(2x2)

↓ Conv2D(64, 3x3) + ReLU
↓ Conv2D(64, 3x3) + ReLU
↓ MaxPooling2D(2x2)

↓ Conv2D(128, 3x3) + ReLU
↓ Conv2D(128, 3x3) + ReLU
↓ MaxPooling2D(2x2)

↓ Conv2D(256, 3x3) + ReLU
↓ Conv2D(256, 3x3) + ReLU
↓ MaxPooling2D(2x2)

↓ Conv2D(512, 3x3) + ReLU
↓ Conv2D(512, 3x3) + ReLU
↓ MaxPooling2D(2x2)

↓ Dropout(0.25)
↓ Flatten
↓ Dense(1500) + ReLU
↓ Dropout(0.4)
↓ Dense(38) + Softmax (output layer)
⚙️ Compilation & Training Details
Loss Function: categorical_crossentropy

Optimizer: Adam with learning rate 0.0001

Metrics: accuracy

Epochs: 10

Batch Size: 32

Image Size: 128x128

📈 Training Performance
The model was trained for 10 epochs, and the training/validation metrics are:

✅ Training Accuracy: ~98.84%

✅ Validation Accuracy: ~98.43%

📉 Both training and validation loss decreased steadily.

The close match between training and validation metrics indicates that the model is well-generalized and not overfitting.

🧪 Evaluation Metrics
The model was evaluated on a validation/test set using the following metrics:

Classification Report: Includes precision, recall, and F1-score for each of the 38 classes.

Confusion Matrix: Plotted using Seaborn's heatmap.

Example Visualization:

python
Copy
Edit
plt.figure(figsize=(40,40))
sns.heatmap(cm, annot=True, annot_kws={'size':10})
🧪 Testing Dataset
Loaded using image_dataset_from_directory (no shuffling)

Predictions generated using model.predict()

Class labels recovered with argmax

Compared with true labels using:

classification_report

confusion_matrix

📁 Output Files
trained_model.keras: Final trained model

training_hist.json: Accuracy/loss history over epochs

✅ Model Highlights
Deep CNN with multiple convolution layers for feature extraction

Achieved high accuracy with minimal overfitting

Used Dropout for regularization

Comprehensive evaluation with visualizations

Excellent generalization capability

