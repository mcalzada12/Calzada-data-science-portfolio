# 🥇 Olympic 2008 Medalists Data Set 

## 📊 About the Project 


This project focuse in trsnforming a raw data set of 2008 olympic medalists into a tidy format and performing exploratory analysis (EDA). the Main objective is to apply tidy data principles to restructure the data set and generate meaningful insight through visualizations and aggregation. 
---

But what exactly is data cleaning and the concept of "Tidy Data"? 
Tidy data helps the data have a clear sturcture to work with: 
- each variable has its own column
- each observation has its own row
- each type of observational unit has its own table

Once the data follows this format, it becomes much easier to summarize, compare, and visualize patterns in a meaningful way.

---

## What this notebook does

- reshaped the dataset from wide to long format using `melt()`
- removed missing values
- split combined column information into separate variables using `str.split()`
- cleaned string values using `str.replace()`
- renamed columns for clarity
- explored the cleaned data through visualizations
- created aggregations and pivot-style summaries to compare medal counts

The main purpose was not only to clean the data, but also to show why tidy data makes analysis much more organized and useful.

---

## Dataset Description
This dataset contains information on medalists from the 2008 Olympic Games. It was provided as part of the assignment and included athlete names along with medal outcomes stored across multiple columns. It has a total of 1875 athletes and medals to analyize from. 

### Pre-processing steps
The main pre-processing steps were:
1. Melting the dataset into long format
2. Dropping rows with missing medal values
3. Splitting the combined `gender_sport` column into:
   - `gender`
   - `sport`
4. Cleaning sport names with string replacement
5. Renaming columns to make the dataset easier to read

After these steps, the dataset was in tidy format and ready for exploratory analysis. 

---

## ⚙️ How to Run the Notebook
### Dependecies this notebook uses: 
- pandas
- matplotlib
- jupyter
- Python 


### 1. Clone or download the repository
```
git clone <https://github.com/mcalzada12/Calzada-data-science-portfolio/blob/main/Portfolios/TidyData-Project/Tidy_Data-project.ipynb>
cd <https://github.com/mcalzada12/Calzada-data-science-portfolio/tree/main/Portfolios/TidyData-Project>
```


### 2. Set up a python environmnet 
```
conda create -n olympics_env python=3.10 
conda activate olympics_env 
```
### 3. Install Libraries 
```
conda install pandas matplotlib jupyter python 

```

### 4. Launch Jupyter Notebook
`Jupyter notebook` 
This will open a browser window 

### 5. Open and Run the Notebook
- Open the file: olympics_08_tidy_analysis.ipynb
- Click "Run All" or run cells one by one from top to bottom





