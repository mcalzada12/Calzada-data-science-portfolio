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
    <td width="27%">
      <img src="<img width="2182" height="966" alt="image" src="https://github.com/user-attachments/assets/376a1a14-c8ad-4b01-9f94-57f16f13fc15" />
" alt="Training Results for Rgeression Models " width="70%">
    </td>
    <td width="55%" valign="top">

[Machine Learning Models ](https://github.com/mcalzada12/Calzada-data-science-portfolio/blob/main/Portfolios/Portfolio_3/MLSStreamlitApp.py)

This Streamlit project explores diffrente data sets through machine learning models, which coould be clssification or regression options, that can vary dependding on the target variable that wants to be predicted. The App aoffers several tabs which each one offers diffrente insight into what the model has achivied. The app also supports for users to upload their own files for machine learning models to be trained on. 

**Tools & Technologies:** Python, Pandas, Streamlit, Matplotlib, Seaborn

<tr>
  <td width="40%">
    <img src="https://res.cloudinary.com/aenetworks/image/upload/c_fill,ar_2,w_3840,h_1920,g_auto/dpr_auto/f_auto/q_auto:eco/v1/gettyimages-466313493-2?_a=BAVAZGB00" alt="Olympic Medalists" width="80%">
  </td>
  <td width="50%" valign="top">

[Olympic Medalists Data Analysis](https://github.com/mcalzada12/Calzada-data-science-portfolio/tree/main/Portfolios/TidyData-Project)

In this Jupyter Notebook, I worked with the 2008 Olympic dataset to clean, structure, and analyze medalist data.  
I applied tidy data principles to prepare the dataset, and created visualizations to compare medal distribution across countries, sports, and gender.  
The project focuses on the idea of tidy data preparation, and the process required to arrange dat aina way which is easier to manipulate and later on display. 


**Tools & Technologies:** Python, Pandas, Jupyter, sklearn, Applied Tidy Data Principles

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

