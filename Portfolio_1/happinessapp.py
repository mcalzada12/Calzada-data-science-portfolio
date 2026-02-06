import streamlit as st 
import pandas as pd 
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go



st.set_page_config(layout="wide")

st.title ("  Welcome to the World Happiness Report 2019 Data Review App")
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
st.subheader("Here's our data")



# Load your data (replace with your actual data source)
data = pd.DataFrame(df)

with st.container(horizontal=True):
    country_filter = st.multiselect('Select Country (multiple) ', options=list(data['Country or region'].unique()))
    graph_y = st.multiselect('Select Category to observe (one) ', options=list(data.columns))

filtered_data = data[data['Country or region'].isin(country_filter)]

st.write(filtered_data)

fig = px.bar(filtered_data, x='Country or region', y=graph_y, color='Country or region')
st.plotly_chart(fig)   

st.title("Global Happiness: Are Countries Clustered or Spread Out?")

# Load data

# assume column name is 'Happiness Score'
# create bins for each continent 

continent_map = {
    # Europe
    "Finland": "Europe", "Denmark": "Europe", "Norway": "Europe", "Iceland": "Europe",
    "Netherlands": "Europe", "Switzerland": "Europe", "Sweden": "Europe",
    "Austria": "Europe", "Luxembourg": "Europe", "United Kingdom": "Europe",
    "Ireland": "Europe", "Germany": "Europe", "Belgium": "Europe", "France": "Europe",
    "Czech Republic": "Europe", "Malta": "Europe", "Spain": "Europe", "Slovakia": "Europe",
    "Poland": "Europe", "Italy": "Europe", "Ukraine": "Europe",
    "Lithuania": "Europe", "Slovenia": "Europe", "Russia": "Europe",
    "Latvia": "Europe", "Estonia": "Europe", "Portugal": "Europe",
    "Kosovo": "Europe", "Moldova": "Europe", "Hungary": "Europe",
    "Cyprus": "Europe", "Greece": "Europe", "Turkey": "Europe",
    "Bosnia and Herzegovina": "Europe", "North Macedonia": "Europe",
    "Serbia": "Europe", "Romania": "Europe", "Montenegro": "Europe",
    "Albania": "Europe", "Belarus": "Europe", "Bulgaria": "Europe",
    "Armenia": "Europe", "Georgia": "Europe",

    # North America
    "Canada": "North America", "United States": "North America",
    "Mexico": "North America", "Costa Rica": "North America",
    "Panama": "North America", "Trinidad & Tobago": "North America",
    "Guatemala": "North America", "Honduras": "North America",
    "El Salvador": "North America", "Nicaragua": "North America",
    "Dominican Republic": "North America", "Haiti": "North America",

    # South America
    "Chile": "South America", "Brazil": "South America",
    "Argentina": "South America", "Uruguay": "South America",
    "Colombia": "South America", "Ecuador": "South America",
    "Bolivia": "South America", "Paraguay": "South America",
    "Peru": "South America", "Venezuela": "South America",

    # Asia
    "Israel": "Asia", "United Arab Emirates": "Asia", "Saudi Arabia": "Asia",
    "Qatar": "Asia", "Kuwait": "Asia", "Bahrain": "Asia", "Oman": "Asia",
    "Uzbekistan": "Asia", "Kazakhstan": "Asia", "Turkmenistan": "Asia",
    "Kyrgyzstan": "Asia", "Azerbaijan": "Asia", "Lebanon": "Asia",
    "China": "Asia", "Japan": "Asia", "South Korea": "Asia",
    "Singapore": "Asia", "Thailand": "Asia", "Malaysia": "Asia",
    "Indonesia": "Asia", "Vietnam": "Asia", "Laos": "Asia",
    "Cambodia": "Asia", "Myanmar": "Asia", "Philippines": "Asia",
    "Mongolia": "Asia", "Iran": "Asia", "Iraq": "Asia",
    "Afghanistan": "Asia", "Pakistan": "Asia", "Bangladesh": "Asia",
    "Nepal": "Asia", "India": "Asia", "Bhutan": "Asia",
    "Sri Lanka": "Asia", "Jordan": "Asia", "Palestinian Territories": "Asia",
    "Yemen": "Asia",
    
    # Africa
    "South Africa": "Africa", "Namibia": "Africa", "Botswana": "Africa",
    "Zimbabwe": "Africa", "Zambia": "Africa", "Mozambique": "Africa",
    "Angola": "Africa", "Malawi": "Africa", "Tanzania": "Africa",
    "Kenya": "Africa", "Uganda": "Africa", "Rwanda": "Africa",
    "Burundi": "Africa", "Somalia": "Africa", "Ethiopia": "Africa",
    "South Sudan": "Africa", "Chad": "Africa", "Niger": "Africa",
    "Mali": "Africa", "Burkina Faso": "Africa", "Ghana": "Africa",
    "Ivory Coast": "Africa", "Liberia": "Africa", "Sierra Leone": "Africa",
    "Guinea": "Africa", "Benin": "Africa", "Togo": "Africa",
    "Senegal": "Africa", "Gabon": "Africa", "Congo (Brazzaville)": "Africa",
    "Congo (Kinshasa)": "Africa", "Algeria": "Africa", "Morocco": "Africa",
    "Tunisia": "Africa", "Egypt": "Africa", "Cameroon": "Africa",
    "Gambia": "Africa", "Mauritania": "Africa", "Madagascar": "Africa",
    "Lesotho": "Africa", "Comoros": "Africa", "Central African Republic": "Africa",

    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania"
}

# add a column to show the continen of each country 

df["Continent"] = df["Country or region"].map(continent_map)

# histogram to show the GDP per capita  across continents
fig_1 = px.bar(df, x='Continent', y='GDP per capita', template='seaborn')

# 1) Group by continent and compute average happiness score
avg_score_continent = (
    df.groupby('Continent', as_index=False)['Score']
      .mean()
)

# sort by avg score (highest first)
avg_score_continent = avg_score_continent.sort_values('Score', ascending=False)

custom_colors = {
    'Europe': '#264653',
    'Asia': '#2A9D8F',
    'North America': '#E9C46A',
    'South America': '#F4A261',
    'Africa': '#E76F51',
    'Oceania': '#8AB17D'
}

# 2) Bar chart of average happiness score per continent
fig_1 = px.bar(
    avg_score_continent,
    x='Continent',
    y='Score',
    template='seaborn',
    title='Global Happiness: Average Score by Continent',
    labels={'Score': 'Average Happiness Score'},
    color='Continent',
    color_discrete_map=custom_colors
)


st.plotly_chart(fig_1, use_container_width=True)






































