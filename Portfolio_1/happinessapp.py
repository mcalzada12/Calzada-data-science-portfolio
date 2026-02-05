import streamlit as st 
import pandas as pd 
import altair as alt

st.title ("Welcome to the World Happiness Report 2019 Data Review App")
st.write ("This is a data review of the diffrente countrie's happines indicators and characteristics which position it /" \
"in diffrente rankings regarding several factors that might influence the general well being of its citizens.")

### loading our csv file 
st.header("Exploring our data set")

st.subheader("Select the Country that you think has the highest happiness rank! 😄")

if "clicked_btn" not in st.session_state:
    st.session_state.clicked_btn = None

with st.container(horizontal=True):
        if st.button('France'): 
            st.markdown('<p style="color:red;">Wrong Answer - Try Again!</p>', unsafe_allow_html=True)
        
        if st.button('Finland',key="fi"):
            st.markdown('<p style="color:green;">Correct Answer - Well Done! Finland is ranked 1st in the World Happiness Report 2019</p>', unsafe_allow_html=True)        
        if st.button('Denmark',key="dk"):
            st.markdown('<p style="color:red;">Wrong Answer - Try Again!</p>', unsafe_allow_html=True)
        
        if st.button('United States',key="us"):
            st.markdown('<p style="color:red;">Wrong Answer - Try Again!</p>', unsafe_allow_html=True)


st.subheader("Select the Country that you think has the lowest happiness rank 😞")
with st.container(horizontal=True):
    if st.button('Afghanistan'): 
        st.markdown('<p style="color:green;">Correct Answer - Well Done! Afghanistan is ranked 156th in the World Happiness Report 2019</p>', unsafe_allow_html=True) 
    
    if st.button('Rwanda'):
        st.markdown('<p style="color:red;">Wrong Answer - Try Again!</p>', unsafe_allow_html=True) 
    
    if st.button('Botswana'):
        st.markdown('<p style="color:red;">Wrong Answer - Try Again!</p>', unsafe_allow_html=True)

    if st.button('Yemen'):
        st.markdown('<p style="color:red;">Wrong Answer - Try Again!</p>', unsafe_allow_html=True)


# load the csv file

df= pd.read_csv("DATA/2019.csv")
st.write("Here's our data")

st.write("Data Summary")
st.write(df.describe())
















