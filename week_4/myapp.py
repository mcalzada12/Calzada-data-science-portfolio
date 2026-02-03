import streamlit as st 
st .title("Hello , Streamlit! ")
st.markdown("# Hello, streamlit!")
st.write ("This is my first streamlit app. ")

if st.button ("Click me!"):
    st.write("You Clicked the button")
else:
    st.write("Click the button, and see what happens...")


### loading our csv file 
import pandas as pd 
st.subheader("Exploring our data set")

# load the csv file

df= pd.read_csv("data/sample_data-1.csv")
st.write("Here's our data")
st.dataframe(df)

City= st.selectbox("Select a city", df["City"].unique(), index= None)
filtered_df= df[df["City"] ==City]
st.write(f"People in {City}")
st.dataframe(filtered_df)

#add bar chart 
st.bar_chart(df["Salary"])







