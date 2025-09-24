# Handwritten Digit Classifier

This repository contains projects for building **handwritten digit classifiers** using two different datasets and modeling approaches. The goal of these projects is to recognize digits (0–9) from images and evaluate the model performance using metrics like Accuracy and F1 score.

---

## Project 1: Kaggle Digit Recognizer Dataset (Logistic Regression)

- **Platform:** Jupyter Notebook.  
- **Dataset:** [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)  
- **Model:** Logistic Regression (Saga solver).  
- **Metrics:**  
  - Accuracy: 0.91  
  - F1 Score: Weighted average.  
- **Key Steps:**  
  1. Loaded and explored the dataset.  
  2. Preprocessed data by normalizing pixel values.  
  3. Splited data into training and test sets.  
  4. Trained Logistic Regression model.  
  5. Evaluated model with accuracy, F1 score, and confusion matrix.  
  6. Visualized misclassified digits for error analysis.  

---

## Project 2: MNIST Dataset (Artificial Neural Network)

- **Platform:** Google Colab.  
- **Dataset:** MNIST (handwritten digits, 28x28 grayscale images).  
- **Model:** Artificial Neural Network (ANN).  
- **Metrics:**  
  - Accuracy: 0.98  
  - F1 Score: Weighted average.  
- **Key Steps:**  
  1. Loaded and normalized MNIST dataset.  
  2. Built an ANN model with input, hidden, and output layers.  
  3. Compiled the model with appropriate loss function and optimizer.  
  4. Trained the model on the training data.  
  5. Evaluated performance on the test set.  
  6. Analyzed misclassifications and model performance metrics.  

---

## Insights & Takeaways

- Logistic Regression provides a **good baseline** for multiclass digit recognition.  
- ANN significantly improves accuracy by learning complex patterns in image data.  
- Visualization of misclassified digits helps identify **challenging digits** for the model.  
- Shows importance of preprocessing, evaluation metrics, and error analysis in classification tasks.  

---

## Tools & Libraries

- Python 3  
- Jupyter Notebook & Google Colab  
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow/keras`  


----



