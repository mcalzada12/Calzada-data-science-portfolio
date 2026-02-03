import streamlit as st 
st.title ("Welcome To my Whoop Data Review")
st.write ("Whoop has been a revolutionary pice of technology that was created in 2012 and has become one of the " \
"most used health tracking devices for high endurance atheletes")

### loading our csv file 
import pandas as pd 
st.subheader("Exploring our data set")

# load the csv file

df= pd.read_csv("Data/whoop_fitness_dataset_100k.csv")
st.write("Here's our data")
st.dataframe(df)





