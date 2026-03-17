import pandas as pd # Library for data manipulation
import seaborn as sns # Library for statistical plotting
import matplotlib.pyplot as plt # For creating custom plots
import streamlit as st # Framework for building interactive web apps
# ================================================================================
#Missing Data & Data Quality Checks
#
# This lecture covers:
# - Data Validation: Checking data types, missing values, and ensuring consistency.
# - Missing Data Handling: Options to drop or impute missing data.
# - Visualization: Using heatmaps and histograms to explore data distribution.
# ================================================================================
st.title("Missing Data & Data Quality Checks")
st.markdown("""
This lecture covers:
- **Data Validation:** Checking data types, missing values, and basic consistency.
- **Missing Data Handling:** Options to drop or impute missing data.
- **Visualization:** Using heatmaps and histograms to understand data distribution.
""")
# ------------------------------------------------------------------------------
# Load the Dataset
# ------------------------------------------------------------------------------
# Read the Titanic dataset from a CSV file.
df = pd.read_csv("data/titanic-1.csv")
# ------------------------------------------------------------------------------
# Display Summary Statistics
# ------------------------------------------------------------------------------
# Show key statistical measures like mean, standard deviation, etc.
st.write("**Summary Statistics**")
st.dataframe(df.describe())
# ------------------------------------------------------------------------------
# Check for Missing Values
# ------------------------------------------------------------------------------
# Display the count of missing values for each column.
st.write("**Number of Missing Values by Column**")
st.dataframe(df.isnull().sum())
# ------------------------------------------------------------------------------
# Visualize Missing Data
# ------------------------------------------------------------------------------
# Create a heatmap to visually indicate where missing values occur.
st.write("Heatmap of missing values")
# ================================================================================
# Interactive Missing Data Handling
# Create a matplotlib figure and axis for the heatmap 
fig,ax = plt.subplots()
# plots a heatmap of missing values using 
sns.heatmap(df.isnull(),cmap="viridis",cbar=False)

st.pyplot(fig)
# Users can select a numeric column and choose a method to address missing values.
# Options include:
# - Keeping the data unchanged
# - Dropping rows with missing values
# - Dropping columns if more than 50% of the values are missing
# - Imputing missing values with mean, median, or zero
# ================================================================================
column= st.selectbox("Choose a column to fill",
             df.select_dtypes(include=['number']).columns)

#provide options for how to handle missing data
method= st.radio("Choose a methods", 
         ["Original DF", "Drop Rows", 
         "Drop Columns (> 50% missing)",
         "Input Mean", "Input Median" , "Input zero"] )


# Work on a copy of the DataFrame so the original data remains unchanged.

df_clean= df.copy()

# Apply the selected method to handle missing data.
if method == "Drop Rows":
    pass
elif method == "Drop Rows":
#remove all rows that have contain any missing values 
    df_clean= df_clean.dropna()
elif method == "Drop Columns (> 50% missing)":
    # drop columns with 50% or more missing values(
    df_clean.drop(column == df_clean.columns[df_clean.columns.insull().mean()>0.50])
elif method == "Input Method":
    # fill missing values with the mean of the column
    df_clean[column]= df_clean[column].fillna(df_clean[column].mean())
elif method == 'Input Median': 
    # fill missing values with the median of the column
    df_clean[column]= df_clean[column].fillna(df_clean[column].median())
elif method == "Input zero":
    df_clean[column]= df_clean[column].fillna(0)



# ------------------------------------------------------------------------------
# Compare Data Distributions: Original vs. Cleaned



# Display side-by-side histograms and statistical summaries for the selected column.

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Data Distribution")
    # Plot a histogram ( with KDE) for the selected column from the original DataFrame.)
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True)
    plt.title(f'Original Distribution of {column}')
    st.pyplot(fig)
    st.subheader(f"{column}'s Original States")
    # Display Statistical summary for the slected colmn 
    st.write(df[column].describe())

with col2:
    st.subheader("Cleaned Data Distribution")
    # Plot a histogram ( with KDE) for the selected column from the cleaned DataFrame.
    fig, ax = plt.subplots()
    sns.histplot(df_clean[column], kde=True)
    plt.title(f'Cleaned Distribution of {column}')
    st.pyplot(fig)
    st.subheader(f"{column}'s Cleaned States")
    # Display Statistical summary for the slected colmn 
    st.write(df_clean[column].describe())





