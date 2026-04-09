# 📊 Supervised Machine Learning Explorer

## 🚀 Project Overview

The goal of this project is to build an interactive Streamlit application that allows users to explore supervised machine learning in a practical and intuitive way. 

With this app, users can upload their own dataset or select from built-in datasets, choose a target variable, and decide whether they want to perform a classification or regression task. The app then allows users to train different machine learning models, adjust hyperparameters, and observe how these changes impact model performance.

This project is designed to make machine learning more understandable by combining data exploration, model training, evaluation, and visualization into a single interface. It provides a hands-on experience where users can experiment and immediately see results.

---

## Deployed App
---

## Required Libraries and Versions

streamlit
pandas
numpy
matplotlib  
scikit-learn  


---

## App Features

### Models Used

Classification:
- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbors (KNN) Classifier  

Regression:
- Linear Regression  
- Decision Tree Regressor  
- K-Nearest Neighbors (KNN) Regressor  

### Hyperparameter Tuning

- Logistic Regression: C, max_iter  
- Decision Trees: max_depth, min_samples_split  
- KNN: n_neighbors, weights  

### Model Evaluation

Classification:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  
- ROC Curve  

Regression:
- R²  
- MAE  
- MSE  
- RMSE  
- Predicted vs Actual plot  

---

## References

https://docs.streamlit.io  
https://scikit-learn.org  
https://pandas.pydata.org  
https://matplotlib.org  

---

## Notes

- Use categorical targets for classification  
- Use numeric targets for regression  
- Adjust tree depth for better visualization performance  

