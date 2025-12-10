# 🧠 Adult Income Prediction using K-Nearest Neighbors (KNN)

This project predicts whether a person earns **more than \$50K/year** using the **UCI Adult Census dataset**.  
The model is implemented using **K-Nearest Neighbors (KNN)** and includes full data preprocessing, training, evaluation, and prediction steps.

---

## 📌 Project Overview

The objective of this project is to classify a person's income level (`<=50K` or `>50K`) based on demographic and employment features.  
This dataset is widely used for benchmarking various machine learning algorithms.

---

## 🧹 Dataset Information

Dataset: **UCI Adult Census Income**

- Total rows: ~48,000  
- Features: 14  
- Target variable: `income` (`<=50K`, `>50K`)

Dataset includes **both categorical and numerical** features, requiring preprocessing such as:

- Label encoding for categorical variables  
- Train-test split  
- Scaling numerical columns  

---

## ⚙️ Model Used — KNN Classifier

KNN is a distance-based model that classifies new data points based on their **nearest neighbors**.

### ✔ Why KNN?
- Simple and effective  
- Works well for small to medium-sized datasets  
- No training time (lazy learning)

### ⚠ Challenges:
- Sensitive to feature scaling  
- Computation increases with dataset size  
- Performance depends on choosing the right value of *K*

---
## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
pip install numpy pandas scikit-learn

shell
Copy code

### 2️⃣ Run the Script
python Adult_income_KNN.py

yaml
Copy code

---

## 📊 Model Evaluation

The script outputs:

- Accuracy score  
- Classification report  
- Predictions on test data  

You will see metrics like:

Accuracy: 0.83
Precision, Recall, F1-score for each class

yaml
Copy code

---

## 🔍 Key Steps in Code
✔ Load & clean dataset  
✔ Encode categorical variables  
✔ Split into train and test sets  
✔ Standardize numerical columns  
✔ Train KNN classifier  
✔ Evaluate performance  
✔ Predict income categories  

---

## 🛠 Technologies Used
- Python  
- NumPy  
- Pandas  
- Scikit-learn  

---


## Author
**Kuldeep Patel**  
Machine Learning & AI Engineer

If you find this helpful, feel free to ⭐ the repository.
