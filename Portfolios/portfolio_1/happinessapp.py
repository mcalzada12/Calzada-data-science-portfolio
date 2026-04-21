import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(
    page_title="World Happiness Report 2019",
    layout="wide",
    page_icon="🌍",
)

st.title("🌍 Welcome to the World Happiness Report 2019 Data Review App")
st.write(
    "This is a data review of the different countries' happiness indicators and "
    "characteristics which position them in different rankings regarding several "
    "factors that might influence the general well-being of their citizens."
)

# ----------------------------- LOAD DATA -----------------------------
df = pd.read_csv("DATA/2019.csv")

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
    "Australia": "Oceania", "New Zealand": "Oceania",
}

df["Continent"] = df["Country or region"].map(continent_map)

custom_colors = {
    "Europe": "#264653",
    "Asia": "#2A9D8F",
    "North America": "#E9C46A",
    "South America": "#F4A261",
    "Africa": "#E76F51",
    "Oceania": "#8AB17D",
}

# Factor columns (everything that's a numeric happiness driver)
factor_cols = [
    "GDP per capita",
    "Social support",
    "Healthy life expectancy",
    "Freedom to make life choices",
    "Generosity",
    "Perceptions of corruption",
]

# ----------------------------- QUIZ SECTION -----------------------------
st.header("🎯 Quick Quiz: Test Your Happiness Knowledge")

quiz_col1, quiz_col2 = st.columns(2)

with quiz_col1:
    st.subheader("Which country has the highest happiness rank? 😄")
    highest_choice = st.radio(
        "Pick one:",
        options=["France", "Finland", "Denmark", "United States"],
        index=None,
        key="highest_quiz",
    )
    if highest_choice == "Finland":
        st.success("✅ Correct! Finland is ranked 1st in the World Happiness Report 2019.")
    elif highest_choice is not None:
        st.error("❌ Wrong answer — try again!")

with quiz_col2:
    st.subheader("Which country has the lowest happiness rank? 😞")
    lowest_choice = st.radio(
        "Pick one:",
        options=["Afghanistan", "Rwanda", "Botswana", "Yemen"],
        index=None,
        key="lowest_quiz",
    )
    if lowest_choice == "Afghanistan":
        st.success("✅ Correct! Afghanistan is ranked 156th in the World Happiness Report 2019.")
    elif lowest_choice is not None:
        st.error("❌ Wrong answer — try again!")

st.divider()

# ----------------------------- DATA EXPLORER -----------------------------
st.header("🔎 Explore the Dataset")
st.caption("Filter by country and pick one or more metrics to compare.")

filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    country_filter = st.multiselect(
        "Select country (multiple allowed):",
        options=sorted(df["Country or region"].unique()),
        default=["Finland", "United States", "France", "Afghanistan"],
    )
with filter_col2:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    default_metric = ["Score"] if "Score" in numeric_cols else [numeric_cols[0]]
    graph_y = st.multiselect(
        "Select metric(s) to visualize:",
        options=numeric_cols,
        default=default_metric,
    )

filtered_data = df[df["Country or region"].isin(country_filter)]

if filtered_data.empty:
    st.info("Pick at least one country to see the data.")
else:
    st.dataframe(filtered_data, use_container_width=True)

    if graph_y:
        fig_filter = px.bar(
            filtered_data,
            x="Country or region",
            y=graph_y,
            color="Country or region",
            barmode="group",
            template="seaborn",
            title="Selected countries — selected metrics",
        )
        st.plotly_chart(fig_filter, use_container_width=True)
    else:
        st.info("Pick at least one metric to draw the chart.")

st.divider()

# ----------------------------- NEW ANALYSIS SECTION -----------------------------
st.header("📊 Deeper Look at the Happiness Drivers")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔥 Correlation Heatmap", "📦 Box Plots", "🔬 Scatter + Trendline", "🕸️ Country Radar"]
)

# --- TAB 1: Correlation heatmap ---
with tab1:
    st.subheader("How do the happiness factors relate to each other?")
    st.caption(
        "Values close to 1 mean two factors move together; values close to -1 mean "
        "they move in opposite directions; values near 0 mean little linear relationship."
    )

    corr_cols = ["Score"] + factor_cols
    corr_matrix = df[corr_cols].corr().round(2)

    fig_heat = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation matrix of Score and its six drivers",
    )
    fig_heat.update_layout(height=550)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Rank the drivers by their correlation with Score
    driver_corr = (
        corr_matrix["Score"]
        .drop("Score")
        .sort_values(ascending=False)
        .reset_index()
    )
    driver_corr.columns = ["Factor", "Correlation with Score"]
    st.write("**Drivers ranked by correlation with the happiness score:**")
    st.dataframe(driver_corr, use_container_width=True, hide_index=True)

# --- TAB 2: Box plots ---
with tab2:
    st.subheader("How are happiness factors distributed across continents?")
    st.caption("Each box shows the median, the middle 50% of countries, and any outliers.")

    box_metric = st.selectbox(
        "Pick a metric to compare across continents:",
        options=["Score"] + factor_cols,
        index=0,
    )

    fig_box = px.box(
        df.dropna(subset=["Continent"]),
        x="Continent",
        y=box_metric,
        color="Continent",
        color_discrete_map=custom_colors,
        points="all",
        hover_data=["Country or region"],
        template="seaborn",
        title=f"Distribution of {box_metric} across continents",
    )
    fig_box.update_layout(height=550, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # Quick summary table
    summary = (
        df.dropna(subset=["Continent"])
        .groupby("Continent")[box_metric]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
        .sort_values("mean", ascending=False)
    )
    st.write(f"**Summary of {box_metric} by continent:**")
    st.dataframe(summary, use_container_width=True)

# --- TAB 3: Scatter with trendline ---
with tab3:
    st.subheader("Does this factor actually predict happiness?")
    st.caption("Each dot is a country. The line shows the overall trend across all 156 countries.")

    scatter_col1, scatter_col2 = st.columns(2)
    with scatter_col1:
        x_factor = st.selectbox(
            "X-axis (the factor):",
            options=factor_cols,
            index=0,
        )
    with scatter_col2:
        color_by = st.selectbox(
            "Color the dots by:",
            options=["Continent", "Score"],
            index=0,
        )

    scatter_df = df.dropna(subset=["Continent"])

    if color_by == "Continent":
        fig_scatter = px.scatter(
            scatter_df,
            x=x_factor,
            y="Score",
            color="Continent",
            color_discrete_map=custom_colors,
            hover_name="Country or region",
            trendline="ols",
            trendline_scope="overall",
            trendline_color_override="black",
            template="seaborn",
            title=f"Happiness Score vs. {x_factor}",
        )
    else:
        fig_scatter = px.scatter(
            scatter_df,
            x=x_factor,
            y="Score",
            color="Score",
            color_continuous_scale="Viridis",
            hover_name="Country or region",
            trendline="ols",
            trendline_color_override="red",
            template="seaborn",
            title=f"Happiness Score vs. {x_factor}",
        )

    fig_scatter.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig_scatter.update_layout(height=600)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Report the correlation value for the chosen x
    r_value = df[[x_factor, "Score"]].corr().iloc[0, 1].round(3)
    st.metric(
        label=f"Pearson correlation between {x_factor} and Score",
        value=r_value,
        help="Closer to 1 = strong positive link; closer to 0 = weak link.",
    )

# --- TAB 4: Country radar chart ---
with tab4:
    st.subheader("Compare country profiles side by side")
    st.caption(
        "Each axis is one of the six happiness drivers. A bigger shape means a country "
        "scores higher on more factors overall."
    )

    default_radar = [c for c in ["Finland", "United States", "India", "Afghanistan"] if c in df["Country or region"].values]
    radar_countries = st.multiselect(
        "Pick up to 5 countries to compare:",
        options=sorted(df["Country or region"].unique()),
        default=default_radar,
        max_selections=5,
    )

    if radar_countries:
        # Normalize each factor to 0-1 so all axes share a scale
        radar_df = df[df["Country or region"].isin(radar_countries)].copy()
        normalized = df[factor_cols].copy()
        for col in factor_cols:
            col_min, col_max = df[col].min(), df[col].max()
            if col_max != col_min:
                normalized[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                normalized[col] = 0
        normalized["Country or region"] = df["Country or region"]
        radar_normalized = normalized[normalized["Country or region"].isin(radar_countries)]

        fig_radar = go.Figure()
        palette = ["#264653", "#E76F51", "#2A9D8F", "#E9C46A", "#8AB17D"]
        for i, country in enumerate(radar_countries):
            row = radar_normalized[radar_normalized["Country or region"] == country]
            if row.empty:
                continue
            values = row[factor_cols].values.flatten().tolist()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=factor_cols + [factor_cols[0]],
                    fill="toself",
                    name=country,
                    line_color=palette[i % len(palette)],
                    opacity=0.6,
                )
            )

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=600,
            title="Country profiles across the six happiness drivers (values normalized 0–1)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Show raw values for the selected countries
        st.write("**Raw values for the selected countries:**")
        st.dataframe(
            radar_df[["Country or region", "Overall rank", "Score"] + factor_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Pick at least one country to draw the radar chart.")