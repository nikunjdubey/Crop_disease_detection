# 🌿 Plant Disease Detection using CNN

This repository contains a deep learning-based image classification model to detect plant diseases using Convolutional Neural Networks (CNN).
The model is trained on the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset), which consists of more than 87,000 images of healthy and diseased plant leaves.

## 📁 Dataset

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Classes**: 38 (including healthy and diseased leaves)
- **Images**: 87,000+ images across different plant types and diseases
- **Categories**: Apple, Tomato, Potato, Corn, Grape, and more

---

## 🧠 Model Summary

The CNN-based model is built using TensorFlow/Keras and trained to classify input leaf images into one of the 38 categories. 
It uses techniques such as data augmentation, dropout, and early stopping to improve performance and reduce overfitting.

## Model Live Demo
https://mainpy-yvip3qmpcr5atacyrzakxp.streamlit.app/


🔬 Train_plant_disease.ipynb Analysis Summary
📊 Final Training Cell Output:
python
Copy
Edit
model.fit(train_set, validation_data=val_set, epochs=10)
Output:

Shows training progress for 10 epochs.

Metrics Observed:

Training and validation accuracy increases steadily.

Final Epoch:

Training Accuracy: ~98.84%

Validation Accuracy: ~98.43%

Loss decreases across epochs, showing successful convergence.

✅ Conclusion:

Your model is well-trained and exhibits high accuracy.

There's no sign of overfitting; validation accuracy tracks closely with training accuracy.
