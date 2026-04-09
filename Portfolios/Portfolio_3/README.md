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

---
## Visuals from the Application
---

<table>
  <tr>
    <!-- IMAGE -->
    <td width="40%" align="center">
      <img src="https://github.com/mcalzada12/Calzada-data-science-portfolio/blob/main/Portfolios/Portfolio_3/images/app_screenshot.png?raw=true" width="90%" style="border-radius:10px;">
    </td>

    <!-- TEXT -->
    <td width="60%" valign="top">

### 🔍 [Machine Learning Models](https://github.com/mcalzada12/Calzada-data-science-portfolio/blob/main/Portfolios/Portfolio_3/MLSStreamlitApp.py)

This Streamlit application explores different datasets using supervised machine learning models. Users can choose between classification and regression depending on the target variable and experiment with different models and hyperparameters.

The app provides multiple tabs that guide the user through data exploration, training results, evaluation, and model comparison. It also allows users to upload their own datasets for a fully interactive experience.

**Tools & Technologies:**  
Python · Pandas · Streamlit · Matplotlib · Seaborn

    </td>
  </tr>

  <tr>
    <!-- IMAGE -->
    <td width="40%" align="center">
      <img src="https://res.cloudinary.com/aenetworks/image/upload/c_fill,ar_2,w_3840,h_1920,g_auto/dpr_auto/f_auto/q_auto:eco/v1/gettyimages-466313493-2?_a=BAVAZGB00" width="90%" style="border-radius:10px;">
    </td>

    <!-- TEXT -->
    <td width="60%" valign="top">

### 🏅 [Olympic Medalists Data Analysis](https://github.com/mcalzada12/Calzada-data-science-portfolio/tree/main/Portfolios/TidyData-Project)

In this Jupyter Notebook, I analyzed the 2008 Olympic dataset by applying tidy data principles to clean and structure the data. The project focuses on transforming raw data into a usable format and generating insights through visualization.

The analysis compares medal distribution across countries, sports, and gender, highlighting patterns and trends in the dataset.

**Tools & Technologies:**  
Python · Pandas · Jupyter Notebook · scikit-learn · Tidy Data Principles

    </td>
  </tr>
</table>




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

