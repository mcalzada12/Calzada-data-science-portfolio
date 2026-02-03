import streamlit as st 
import pandas as pd 
import altair as alt

st.title ("Welcome To My student Performeance App")
st.write ("This is a data review of student performance, getting to understand what works and what doesn't, "
"in students in order to look at key predictors for us students to get the best out of our time and energy. ")

### loading our csv file 
st.subheader("Exploring our data set")

# load the csv file

df= pd.read_csv("DATA/StudentPerformance.csv")
st.write("Here's our data")
st.dataframe(df)


# Display basic statistics
st.subheader("Performance Bar Chart")
st.subheader("Performance influenced by Extracurricular Activities")

# Prepare summary table
extra_perf = (
    df.groupby("Extracurricular Activities")["Performance Index"]
      .mean()
      .reset_index()
)


# Prepare summary table
sleep_pscores = (
    df.groupby("Sleep Hours")["Previous Scores"]
      .mean()
      .reset_index()
)


# Show table
st.dataframe(extra_perf)
st.dataframe(sleep_pscores)


#Scatter Plot 
st.subheader("Basic Statistics")
st.scatter_chart(df,x='Hours Studied',y='Performance Index')


st.subheader("Sleep Hours vs Performance")
st.scatter_chart(df,x='Sleep Hours',y='Performance Index')

st.subheader("Questions Practiced vs Performance")
st.scatter_chart(df,x='Sample Question Papers Practiced',y='Performance Index')

st.subheader("Hours Studied vs Previous Score")
st.scatter_chart(df,x='Hours Studied',y='Previous Scores')



