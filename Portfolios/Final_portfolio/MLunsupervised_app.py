"""
🔮 Unsupervised ML Explorer
A comprehensive Streamlit app for unsupervised machine learning exploration.

Features:
- Three algorithms: K-Means, Hierarchical clustering, PCA
- Built-in sample datasets + custom upload
- Rich visualizations: elbow plots, silhouette analysis, dendrograms, 3D plots
- Automated preprocessing pipeline
- Cluster profiling and interpretation
- Downloadable results
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import f_oneway
from plotly.subplots import make_subplots
import io
import time
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from st_flexible_callout_elements import flexible_callout
from scipy.stats import f_oneway



from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    make_blobs,
    make_moons,
    make_circles,
)

 
# PAGE CONFIG & STYLING

st.set_page_config(
    page_title="Unsupervised ML Explorer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
    
)

# Custom CSS 
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #667eea;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# DATA LOADING UTILITIES
@st.cache_data
def load_sample_dataset(name: str):
    """Load one of the built-in sample datasets."""
    if name == "Iris (flowers)":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["true_label"] = [data.target_names[i] for i in data.target]
        return df, "true_label"
    elif name == "Wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["true_label"] = [data.target_names[i] for i in data.target]
        return df, "true_label"
    elif name == "Breast Cancer":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["true_label"] = [data.target_names[i] for i in data.target]
        return df, "true_label"
    elif name == "Synthetic Blobs":
        X, y = make_blobs(n_samples=300, centers=4, n_features=5, random_state=42, cluster_std=1.2)
        df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(5)])
        df["true_label"] = [f"cluster_{i}" for i in y]
        return df, "true_label"
    elif name == "Two Moons":
        X, y = make_moons(n_samples=300, noise=0.08, random_state=42)
        df = pd.DataFrame(X, columns=["x", "y"])
        df["true_label"] = [f"moon_{i}" for i in y]
        return df, "true_label"
    elif name == "Concentric Circles":
        X, y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
        df = pd.DataFrame(X, columns=["x", "y"])
        df["true_label"] = [f"ring_{i}" for i in y]
        return df, "true_label"
    elif name == "Mall Customers (synthetic)":
        # Synthetic mall customers dataset
        np.random.seed(42)
        n = 200
        ages = np.concatenate([
            np.random.normal(25, 5, n // 4),
            np.random.normal(45, 8, n // 4),
            np.random.normal(35, 6, n // 4),
            np.random.normal(60, 7, n // 4),
        ])
        income = np.concatenate([
            np.random.normal(40, 10, n // 4),
            np.random.normal(80, 15, n // 4),
            np.random.normal(120, 20, n // 4),
            np.random.normal(50, 12, n // 4),
        ])
        spending = np.concatenate([
            np.random.normal(75, 10, n // 4),
            np.random.normal(40, 8, n // 4),
            np.random.normal(85, 12, n // 4),
            np.random.normal(20, 5, n // 4),
        ])
        df = pd.DataFrame({
            "age": np.clip(ages, 18, 80).round(1),
            "annual_income_k": np.clip(income, 10, 200).round(1),
            "spending_score": np.clip(spending, 1, 100).round(1),
        })
        return df, None
    return None, None


def get_dataset_description(name: str) -> str:
    descriptions = {
        "Iris (flowers)": "🌸 Classic dataset of 150 iris flowers with 4 measurements across 3 species. Perfect for clustering basics.",
        "Wine": "🍷 178 wines from 3 cultivars with 13 chemical properties. Good test of high-dimensional clustering.",
        "Breast Cancer": "🔬 569 tumor samples with 30 features. Binary clustering challenge with clinical relevance.",
        "Synthetic Blobs": "🟣 300 points in 5D forming 4 clear gaussian blobs. Easy mode — for verifying algorithms work.",
        "Two Moons": "🌙 Two interleaved half-circles. Linear methods fail; tests non-convex clustering.",
        "Concentric Circles": "⭕ Two nested rings. Notoriously hard for K-Means — useful for showing where centroid-based methods fail.",
        "Mall Customers (synthetic)": "🛍️ 200 customers with age, income, and spending score. Classic segmentation use case.",
    }
    return descriptions.get(name, "")


# PREPROCESSING
def preprocess_data(df: pd.DataFrame, features: list, scaler_type: str, handle_na: str):
    """Apply preprocessing pipeline and return cleaned + scaled data."""
    X = df[features].copy()

    # Handle missing values
    if handle_na == "Drop rows":
        mask = X.notna().all(axis=1)
        X = X.loc[mask]
        df_clean = df.loc[mask].reset_index(drop=True)
        X = X.reset_index(drop=True)
    elif handle_na == "Fill with mean":
        X = X.fillna(X.mean(numeric_only=True))
        df_clean = df.copy()
    elif handle_na == "Fill with median":
        X = X.fillna(X.median(numeric_only=True))
        df_clean = df.copy()
    else:
        df_clean = df.copy()

    # Scale
    if scaler_type == "StandardScaler (z-score)":
        scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler (0–1)":
        scaler = MinMaxScaler()
    elif scaler_type == "RobustScaler (median/IQR)":
        scaler = RobustScaler()
    else:
        scaler = None

    if scaler is not None:
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features, index=X.index)
    else:
        X_scaled = X.copy()

    return df_clean, X, X_scaled


# CLUSTERING METRICS HELPER
def compute_metrics(X, labels):
    """Compute silhouette, CH, and DB scores. Returns None for any that can't be computed."""
    metrics = {}
    unique_labels = set(labels) - {-1}  # -1 reserved for noise labels (defensive)
    n_clusters = len(unique_labels)

    if n_clusters >= 2:
        # Filter out noise points for metrics
        if -1 in labels:
            mask = labels != -1
            if mask.sum() > n_clusters:
                X_clean = X[mask]
                labels_clean = labels[mask]
            else:
                return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None, "n_clusters": n_clusters}
        else:
            X_clean = X
            labels_clean = labels
        try:
            metrics["silhouette"] = silhouette_score(X_clean, labels_clean)
        except Exception:
            metrics["silhouette"] = None
        try:
            metrics["calinski_harabasz"] = calinski_harabasz_score(X_clean, labels_clean)
        except Exception:
            metrics["calinski_harabasz"] = None
        try:
            metrics["davies_bouldin"] = davies_bouldin_score(X_clean, labels_clean)
        except Exception:
            metrics["davies_bouldin"] = None
    else:
        metrics = {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    metrics["n_clusters"] = n_clusters
    return metrics


# SIDEBAR — DATA INPUT
st.markdown('<div class="main-header">🔮 Unsupervised ML Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Discover hidden structure in your data with clustering and dimensionality reduction.</div>',
    unsafe_allow_html=True,
)
st.expander(" 📲 About This App! ").markdown(
    """
    This app is designed to help you explore unsupervised machine learning techniques on your datasets. 
    You can start with built-in sample datasets or upload your own CSV file. 
    The app provides tools for preprocessing, clustering, dimensionality reduction, and evaluation, all in an interactive interface.
    
    **Getting Started:**
    1. Choose a data source (sample dataset or upload your own).
    2. Select features and preprocessing options.
    3. Pick an algorithm and set hyperparameters.
    4. Explore the results through various visualizations and metrics.
    5. Download your clustered data for further analysis.
    
    Unsupervised learning can be a powerful way to find patterns and groupings in data without predefined labels. 
    Use this app to experiment and gain insights from your data!

    
    <font color="blue"> * bonus! we have added a new tab in the app that provides feature analysis diagnostics to help you decide which 
    features to keep or drop before training your model. This includes variance analysis, correlation heatmaps, cluster discriminative power, 
    and PCA loadings. These tools will guide you in selecting the most informative features for better clustering results. </font>

    
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("📁 1. Data Source")
    data_source = st.radio(
        "Choose data source:",
        ["Sample dataset", "Upload CSV"],
        help="Start with a sample dataset to explore, or upload your own data.",
    )

    df = None
    label_col = None

    if data_source == "Sample dataset":
        sample_name = st.selectbox(
            "Pick a dataset:",
            [
                "Iris (flowers)",
                "Wine",
                "Breast Cancer",
                "Mall Customers (synthetic)",
                "Synthetic Blobs",
                "Two Moons",
                "Concentric Circles",
            ],
        )
        df, label_col = load_sample_dataset(sample_name)
        st.info(get_dataset_description(sample_name))
    else:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
        else:
            st.warning("👆 Upload a CSV file or switch to a sample dataset.")

# MAIN AREA — IF NO DATA YET
if df is None:
    st.markdown(
        """
        <div class="info-box">
        <h3>👋 Welcome!</h3>
        <p>This app helps you explore unsupervised machine learning techniques on your data.</p>
        <p><strong>To get started:</strong></p>
        <ol>
            <li>Pick a sample dataset or upload your own CSV in the sidebar</li>
            <li>Choose features and preprocessing options</li>
            <li>Experiment with clustering and dimensionality reduction</li>
            <li>Interpret the results and download labeled data</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 Clustering")
        st.write("- **K-Means**: Fast, spherical clusters")
        st.write("- **Hierarchical**: Tree-based, with dendrograms")
    with col2:
        st.markdown("### 📐 Dimensionality Reduction")
        st.write("- **PCA**: Linear, fast, interpretable")
        st.write("- Optional: cluster on PCA output")
    with col3:
        st.markdown("### 📊 Evaluation")
        st.write("- **Elbow method** for optimal k")
        st.write("- **Silhouette analysis**")
        st.write("- **Calinski-Harabasz / Davies-Bouldin**")
        st.write("- **Cluster profiling**")
    st.stop()

# SIDEBAR — FEATURE SELECTION & PREPROCESSING
with st.sidebar:
    st.header("⚙️ 2. Features & Preprocessing")

    # Auto-detect numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Don't include the auto-detected label column in defaults
    default_features = [c for c in numeric_cols if c != label_col]

    if not default_features:
        st.error("❌ No numeric columns found in this dataset. Please upload data with numeric features.")
        st.stop()

    selected_features = st.multiselect(
        "Features to use:",
        options=numeric_cols,
        default=default_features,
        help="Only numeric features can be used for clustering. Pick at least 2.",
    )

    if len(selected_features) < 2:
        st.warning("Please select at least 2 features.")
        st.stop()

    # Optional: known label column (for color overlay only — not used for training)
    object_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    label_options = ["(none)"] + object_cols + numeric_cols
    default_idx = label_options.index(label_col) if label_col in label_options else 0
    label_col_choice = st.selectbox(
        "Known label column (optional, for comparison):",
        options=label_options,
        index=default_idx,
        help="If your data has known labels, pick them here to compare against discovered clusters. Not used for training.",
    )
    label_col = None if label_col_choice == "(none)" else label_col_choice

    scaler_type = st.selectbox(
        "Scaling method:",
        ["StandardScaler (z-score)", "MinMaxScaler (0–1)", "RobustScaler (median/IQR)", "None"],
        help="Most clustering algorithms are distance-based and need scaled features.",
    )

    handle_na = st.selectbox(
        "Handle missing values:",
        ["Drop rows", "Fill with mean", "Fill with median", "Leave as-is"],
    )

# Apply preprocessing
df_clean, X_raw, X_scaled = preprocess_data(df, selected_features, scaler_type, handle_na)

# SIDEBAR — ALGORITHM SELECTION
with st.sidebar:
    st.header("🧪 3. Algorithm")
    algorithm = st.selectbox(
        "Pick an algorithm:",
        [
            "K-Means Clustering",
            "Hierarchical Clustering",
            "PCA (Dimensionality Reduction)",
        ],
    )

    st.markdown("---")
    st.markdown("**Hyperparameters**")

    # Algorithm-specific hyperparameters
    params = {}
    if algorithm == "K-Means Clustering":
        st.caption(
            "💡 K-Means partitions points into k spherical clusters by minimizing distance to cluster centers."
        )
        params["n_clusters"] = st.slider(
            "Number of clusters (k)", 2, 15, 3,
            help="How many groups to split your data into. Use the elbow method or silhouette analysis to pick this. Too few → loses structure; too many → overfits to noise.",
        )
        params["init"] = st.selectbox(
            "Initialization", ["k-means++", "random"],
            help="How starting cluster centers are chosen. 'k-means++' spreads them apart smartly (recommended). 'random' picks them randomly — faster but often produces worse results.",
        )
        if params["init"] == "k-means++":
            st.caption(
                "📌 **k-means++**: spreads starting centers far apart on purpose. "
                "More reliable, converges faster, less likely to get stuck in a bad solution. *Recommended default.*"
            )
        else:
            st.caption(
                "📌 **random**: picks starting centers completely at random. "
                "Faster per run but can land in a bad local minimum — increase 'Number of initializations' to compensate."
            )
        params["n_init"] = st.slider(
            "Number of initializations", 1, 20, 10,
            help="K-Means is sensitive to starting positions. This runs the algorithm N times with different starts and keeps the best one. Higher = more reliable but slower.",
        )
        params["max_iter"] = st.slider(
            "Max iterations", 50, 1000, 300, step=50,
            help="Maximum refinement steps per run. The algorithm usually converges well before this limit. Increase only if you see convergence warnings.",
        )
        params["random_state"] = st.number_input(
            "Random seed", 0, 1000, 42,
            help="Fixes the randomness so results are reproducible. Change it to test how sensitive your clusters are to the random starting point.",
        )
    elif algorithm == "Hierarchical Clustering":
        st.caption(
            "💡 Hierarchical clustering builds a tree by repeatedly merging the closest pairs of points/clusters. Use the Dendrogram tab to visualize the merge structure."
        )
        params["n_clusters"] = st.slider(
            "Number of clusters", 2, 15, 3,
            help="Where to 'cut' the dendrogram tree. Use the Dendrogram tab to pick a sensible cut height visually — look for the largest vertical gap.",
        )
        params["linkage"] = st.selectbox(
            "Linkage method", ["ward", "complete", "average", "single"],
            help=(
                "How to measure distance between clusters when merging:\n\n"
                "• ward — minimizes variance within clusters (best default, balanced clusters)\n\n"
                "• complete — distance between farthest points (compact, equal-diameter clusters)\n\n"
                "• average — mean distance between all pairs (balanced)\n\n"
                "• single — distance between closest points (can produce stretched 'chaining' clusters)"
            ),
        )
        linkage_descriptions = {
            "ward": "📌 **Ward**: merges the pair of clusters that increases overall variance the least. "
                    "Tends to produce balanced, compact clusters. *Best general-purpose default.*",
            "complete": "📌 **Complete**: distance between two clusters = distance between their *farthest* members. "
                        "Produces tight, equal-sized clusters. Sensitive to outliers.",
            "average": "📌 **Average**: distance between two clusters = mean distance across all pairs of members. "
                       "A balanced compromise between single and complete linkage.",
            "single": "📌 **Single**: distance between two clusters = distance between their *closest* members. "
                      "Can find non-elliptical shapes but often produces stringy 'chaining' artifacts.",
        }
        st.caption(linkage_descriptions[params["linkage"]])
        if params["linkage"] == "ward":
            params["metric"] = "euclidean"
            st.caption("ℹ️ Ward linkage requires Euclidean distance.")
        else:
            params["metric"] = st.selectbox(
                "Distance metric", ["euclidean", "manhattan", "cosine"],
                help=(
                    "How to measure distance between two points:\n\n"
                    "• euclidean — straight-line distance (most common)\n\n"
                    "• manhattan — city-block distance, sum of absolute differences (more robust to outliers)\n\n"
                    "• cosine — angle between vectors, ignores magnitude (good for text or sparse data)"
                ),
            )
            metric_descriptions = {
                "euclidean": "📌 **Euclidean**: straight-line ('as the crow flies') distance. "
                             "Most natural for continuous numeric data. *Default choice.*",
                "manhattan": "📌 **Manhattan**: sum of absolute differences along each axis (like walking city blocks). "
                             "More robust to outliers than Euclidean.",
                "cosine": "📌 **Cosine**: measures the *angle* between vectors, ignoring magnitude. "
                          "Useful when you care about direction/pattern rather than scale (e.g., text data, ratings).",
            }
            st.caption(metric_descriptions[params["metric"]])
    elif algorithm == "PCA (Dimensionality Reduction)":
        st.caption(
            "💡 PCA finds linear combinations of features (called 'principal components') that capture the most variance. Useful for compressing data, removing correlated features, and visualizing high-dimensional data in 2D/3D."
        )
        max_comp = min(len(selected_features), len(X_scaled))
        params["n_components"] = st.slider(
            "Number of components", 2, max_comp, min(3, max_comp),
            help="How many principal components to keep. Each captures successively less variance. Look at the cumulative variance plot after training to pick a value that explains ~80–95% of total variance.",
        )
        params["follow_with_clustering"] = st.checkbox(
            "Cluster on PCA output (K-Means)", value=False,
            help="After reducing dimensions with PCA, run K-Means on the result. Often improves clustering quality and speed when you have many correlated or noisy features.",
        )
        if params["follow_with_clustering"]:
            params["n_clusters_after_pca"] = st.slider(
                "k for K-Means on PCA", 2, 15, 3,
                help="Number of clusters for the K-Means step that runs on the PCA output.",
            )


# MAIN AREA — TABS
tab_overview, tab_explore,tab_features, tab_model, tab_evaluate, tab_interpret, tab_export = st.tabs(
    [
        "📊 Overview",
        "🔍 Data Exploration",
        "📊 Feature Analysis",
        "🧪 Train Model",
        "📈 Evaluate",
        "🎨 Interpret",
        "💾 Export",
    ]
)


# TAB 1 — OVERVIEW
with tab_overview:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df_clean.shape[0]:,}")
    c2.metric("Columns", df_clean.shape[1])
    c3.metric("Numeric features used", len(selected_features))
    c4.metric("Missing values", int(df[selected_features].isna().sum().sum()))

    st.markdown("#### First 10 rows")
    st.dataframe(df_clean.head(10), use_container_width=True)

    with st.expander("📋 Summary statistics"):
        st.dataframe(df_clean[selected_features].describe(), use_container_width=True)

    with st.expander("📋 Column types & missing values"):
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(t) for t in df.dtypes],
            "Missing": df.isna().sum().values,
            "Unique": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(info_df, use_container_width=True)


# TAB 2 — DATA EXPLORATION

with tab_explore:
    st.subheader("Explore feature distributions and relationships")

    expl_cols = st.columns(2)
    with expl_cols[0]:
        st.markdown("#### 📉 Feature distributions")
        feat = st.selectbox("Pick a feature:", selected_features, key="dist_feat")
        fig = px.histogram(df_clean, x=feat, marginal="box", nbins=30, color_discrete_sequence=["#667eea"])
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with expl_cols[1]:
        st.markdown("#### 🔥 Correlation heatmap")
        corr = df_clean[selected_features].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🔗 Pairwise scatter (sampled to 500 points if larger)")
    df_sample = df_clean.sample(min(500, len(df_clean)), random_state=42) if len(df_clean) > 500 else df_clean

    pair_x = st.selectbox("X axis:", selected_features, index=0, key="pair_x")
    pair_y = st.selectbox(
        "Y axis:",
        selected_features,
        index=1 if len(selected_features) > 1 else 0,
        key="pair_y",
    )
    color_arg = label_col if label_col and label_col in df_sample.columns else None
    fig = px.scatter(
        df_sample,
        x=pair_x,
        y=pair_y,
        color=color_arg,
        opacity=0.7,
        height=500,
        title=f"{pair_y} vs. {pair_x}" + (f" (colored by {label_col})" if label_col else ""),
    )
    st.plotly_chart(fig, use_container_width=True)

# TAB 3 - Feature analysis for better decision maing in model building 
with tab_features:
    st.subheader("Which features matter most?")
    st.markdown(
        "Unsupervised models don't have a 'target' to predict, so feature importance is measured differently than in supervised learning. "
        "These four diagnostics help you decide which features to **keep**, **drop**, or **investigate further**."
    )
 
    feat_tabs = st.tabs([
        "📏 Variance",
        "🔗 Redundancy",
        "🎯 Cluster discriminative power",
        "📐 PCA loadings",
    ])
 
    # ---------------- 1. VARIANCE ----------------
    with feat_tabs[0]:
        st.markdown("#### Feature variance (after scaling)")
        st.caption(
            "Features with very low variance contain little information — they're nearly constant across all samples and "
            "won't help distinguish clusters. Consider dropping any feature whose variance is much smaller than the others."
        )
        # Variance on the unscaled data is what matters before scaling normalizes everything
        var_df = pd.DataFrame({
            "feature": selected_features,
            "raw_variance": X_raw.var().values,
            "raw_std": X_raw.std().values,
            "coef_of_variation": (X_raw.std() / X_raw.mean().abs().replace(0, np.nan)).values,
        }).sort_values("raw_variance", ascending=False)
 
        fig = px.bar(
            var_df, x="feature", y="raw_variance",
            color="raw_variance",
            color_continuous_scale="viridis",
            height=400,
            title="Raw variance per feature (before scaling)",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
 
        st.dataframe(
            var_df.style.format({
                "raw_variance": "{:.3f}",
                "raw_std": "{:.3f}",
                "coef_of_variation": "{:.3f}",
            }, na_rep="N/A").background_gradient(subset=["raw_variance"], cmap="viridis"),
            use_container_width=True,
        )
 
        # Auto-flag low-variance features
        low_var_threshold = var_df["raw_variance"].max() * 0.01
        low_var_feats = var_df[var_df["raw_variance"] < low_var_threshold]["feature"].tolist()
        if low_var_feats:
            st.warning(
                f"⚠️ These features have very low variance and likely add no signal: **{', '.join(low_var_feats)}**. "
                f"Consider removing them."
            )
        else:
            st.success("✅ All selected features have meaningful variance.")
 
    # ---------------- 2. REDUNDANCY ----------------
    with feat_tabs[1]:
        st.markdown("#### Feature redundancy (correlation analysis)")
        st.caption(
            "Highly correlated features carry the same signal twice and bias distance-based clustering toward whatever they measure. "
            "If two features have |correlation| > 0.9, you can usually drop one without losing information."
        )
        corr = X_raw.corr().abs()
        # Find pairs of highly correlated features (excluding self-correlations)
        upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr_pairs = (
            upper_tri.stack()
            .reset_index()
            .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "abs_correlation"})
            .sort_values("abs_correlation", ascending=False)
        )
 
        fig = px.imshow(
            X_raw.corr(),
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Pearson correlation between features",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
 
        st.markdown("##### Most correlated pairs")
        if len(high_corr_pairs) > 0:
            top_pairs = high_corr_pairs.head(10)
            st.dataframe(
                top_pairs.style.format({"abs_correlation": "{:.3f}"})
                .background_gradient(subset=["abs_correlation"], cmap="Reds"),
                use_container_width=True,
            )
 
            redundant = high_corr_pairs[high_corr_pairs["abs_correlation"] > 0.9]
            if len(redundant) > 0:
                pairs_str = "; ".join([
                    f"`{r.feature_1}` ↔ `{r.feature_2}` ({r.abs_correlation:.2f})"
                    for r in redundant.head(5).itertuples()
                ])
                st.warning(
                    f"⚠️ Highly correlated pairs (|r| > 0.9): {pairs_str}. "
                    f"Each pair likely contains one redundant feature."
                )
            elif high_corr_pairs.iloc[0]["abs_correlation"] > 0.7:
                st.info("ℹ️ Some moderate correlations exist but no severe redundancy. Probably fine.")
            else:
                st.success("✅ No problematic feature redundancy detected.")
        else:
            st.info("Need at least 2 features to compute correlations.")
 
    # ---------------- 3. CLUSTER DISCRIMINATIVE POWER ----------------
    with feat_tabs[2]:
        st.markdown("#### Cluster discriminative power")
        st.caption(
            "**The closest thing to feature importance for clustering.** "
            "Trains a quick K-Means and ranks features by how strongly their values differ across the discovered clusters. "
            "Features at the top are doing most of the work in separating clusters."
        )
        st.markdown(
            "**How it works:** Run K-Means with the chosen k, then for each feature compute the **F-statistic** "
            "(ratio of between-cluster variance to within-cluster variance). Higher F = the feature differs sharply across clusters = more important."
        )
 
        disc_k = st.slider("Number of clusters to test with", 2, 10, 3, key="disc_k")
        if st.button("Compute feature importance", key="disc_btn", type="primary"):
            with st.spinner("Running K-Means and computing F-statistics..."):
                km = KMeans(n_clusters=disc_k, n_init=10, random_state=42)
                disc_labels = km.fit_predict(X_scaled)
 
                rows = []
                for feat in selected_features:
                    groups = [X_raw.loc[disc_labels == c, feat].values for c in range(disc_k)]
                    # Filter out empty groups (shouldn't happen with k-means but defensive)
                    groups = [g for g in groups if len(g) > 0]
                    try:
                        f_stat, p_val = f_oneway(*groups)
                    except Exception:
                        f_stat, p_val = np.nan, np.nan
                    # Also report how spread out the cluster means are (relative effect size)
                    cluster_means = [g.mean() for g in groups]
                    overall_mean = X_raw[feat].mean()
                    overall_std = X_raw[feat].std()
                    # Eta-squared (effect size): proportion of variance explained by clustering
                    if overall_std > 0:
                        between_var = np.mean([(m - overall_mean) ** 2 for m in cluster_means])
                        eta_sq = between_var / X_raw[feat].var() if X_raw[feat].var() > 0 else 0
                    else:
                        eta_sq = 0
                    rows.append({
                        "feature": feat,
                        "F_statistic": f_stat,
                        "p_value": p_val,
                        "eta_squared": eta_sq,
                    })
 
                disc_df = pd.DataFrame(rows).sort_values("F_statistic", ascending=False)
 
            fig = px.bar(
                disc_df, x="F_statistic", y="feature",
                orientation="h",
                color="F_statistic",
                color_continuous_scale="Viridis",
                height=max(300, 50 * len(selected_features)),
                title=f"Feature importance — F-statistic across {disc_k} K-Means clusters",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
 
            # Format and color the table
            disc_display = disc_df.copy()
            disc_display["significant"] = disc_display["p_value"].apply(
                lambda p: "✅" if p < 0.001 else ("✓" if p < 0.05 else "✗")
            )
            st.dataframe(
                disc_display.style.format({
                    "F_statistic": "{:.2f}",
                    "p_value": "{:.2e}",
                    "eta_squared": "{:.3f}",
                }).background_gradient(subset=["F_statistic"], cmap="viridis"),
                use_container_width=True,
            )
            st.caption(
                "**eta_squared** is the proportion of each feature's variance explained by cluster assignment "
                "(0 = no separation, 1 = perfect separation). **significant**: ✅ p<0.001, ✓ p<0.05, ✗ not significant."
            )
 
            # Recommendations
            top_3 = disc_df.head(3)["feature"].tolist()
            bottom = disc_df[disc_df["F_statistic"] < 1.0]["feature"].tolist()
 
            st.markdown("##### 💡 Recommendations")
            st.markdown(
                f"- **Top discriminating features:** `{'`, `'.join(top_3)}` — these are doing most of the clustering work."
            )
            if bottom:
                st.markdown(
                    f"- **Low-importance features (F < 1):** `{'`, `'.join(bottom)}` — "
                    f"these barely vary across clusters. Consider removing them and re-running to see if clusters get cleaner."
                )
            else:
                st.markdown("- **All features contribute meaningfully** to cluster separation.")
 
    # ---------------- 4. PCA LOADINGS ----------------
    with feat_tabs[3]:
        st.markdown("#### Feature contribution to principal components")
        st.caption(
            "PCA finds the directions of maximum variance. The **loadings** tell you how much each original feature "
            "contributes to each principal component — a way of asking 'which features drive the data's overall structure?'"
        )
        n_pcs = st.slider("Number of components to inspect", 2, min(5, len(selected_features)), min(3, len(selected_features)), key="pca_load_n")
        if st.button("Compute PCA loadings", key="pca_load_btn", type="primary"):
            pca_load = PCA(n_components=n_pcs)
            pca_load.fit(X_scaled)
            # components_ has shape (n_components, n_features) — each row is a PC
            loadings = pd.DataFrame(
                pca_load.components_.T,
                index=selected_features,
                columns=[f"PC{i+1}" for i in range(n_pcs)],
            )
 
            # Heatmap of loadings
            fig = px.imshow(
                loadings,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title=f"PCA loadings — top {n_pcs} components",
                labels=dict(x="Component", y="Feature", color="Loading"),
            )
            fig.update_layout(height=max(300, 40 * len(selected_features)))
            st.plotly_chart(fig, use_container_width=True)
 
            # Summed absolute loadings — overall feature importance to top PCs (weighted by variance explained)
            evr = pca_load.explained_variance_ratio_
            weighted_importance = (np.abs(loadings) * evr).sum(axis=1).sort_values(ascending=False)
            imp_df = pd.DataFrame({
                "feature": weighted_importance.index,
                "weighted_importance": weighted_importance.values,
            })
 
            fig = px.bar(
                imp_df, x="weighted_importance", y="feature",
                orientation="h",
                color="weighted_importance",
                color_continuous_scale="Viridis",
                height=max(300, 50 * len(selected_features)),
                title="Weighted importance: |loading| × variance explained, summed across PCs",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
 
            st.caption(
                f"Top {n_pcs} components explain **{evr.sum()*100:.1f}%** of total variance. "
                f"Per component: " + ", ".join([f"PC{i+1} = {v*100:.1f}%" for i, v in enumerate(evr)])
            )
 
            # Top contributors per PC
            st.markdown("##### Top contributing feature per component")
            for i in range(n_pcs):
                pc_col = f"PC{i+1}"
                top_feat = loadings[pc_col].abs().idxmax()
                top_val = loadings.loc[top_feat, pc_col]
                direction = "positively" if top_val > 0 else "negatively"
                st.markdown(
                    f"- **{pc_col}** ({evr[i]*100:.1f}% of variance): "
                    f"most influenced by `{top_feat}` (loading = {top_val:.2f}, contributes {direction})"
                )



# TAB 4 — TRAIN MODEL

with tab_model:
    st.subheader(f"Run: {algorithm}")

    run_btn = st.button("🚀 Train model", type="primary", use_container_width=True)

    # Use session state to persist results
    if run_btn:
        with st.spinner(f"Training {algorithm}..."):
            t0 = time.time()
            model = None
            labels = None
            embedding = None
            extra_info = {}

            try:
                if algorithm == "K-Means Clustering":
                    model = KMeans(
                        n_clusters=params["n_clusters"],
                        init=params["init"],
                        n_init=params["n_init"],
                        max_iter=params["max_iter"],
                        random_state=params["random_state"],
                    )
                    labels = model.fit_predict(X_scaled)
                    extra_info["inertia"] = model.inertia_
                    extra_info["centers"] = model.cluster_centers_

                elif algorithm == "Hierarchical Clustering":
                    model = AgglomerativeClustering(
                        n_clusters=params["n_clusters"],
                        linkage=params["linkage"],
                        metric=params["metric"],
                    )
                    labels = model.fit_predict(X_scaled)

                elif algorithm == "PCA (Dimensionality Reduction)":
                    model = PCA(n_components=params["n_components"])
                    embedding = model.fit_transform(X_scaled)
                    extra_info["explained_variance_ratio"] = model.explained_variance_ratio_
                    extra_info["components"] = model.components_
                    if params.get("follow_with_clustering"):
                        km = KMeans(
                            n_clusters=params["n_clusters_after_pca"],
                            n_init=10,
                            random_state=42,
                        )
                        labels = km.fit_predict(embedding)
                        extra_info["kmeans_inertia"] = km.inertia_

                elapsed = time.time() - t0

                # Store everything in session state
                st.session_state["model"] = model
                st.session_state["labels"] = labels
                st.session_state["embedding"] = embedding
                st.session_state["extra_info"] = extra_info
                st.session_state["X_scaled"] = X_scaled
                st.session_state["X_raw"] = X_raw
                st.session_state["df_clean"] = df_clean
                st.session_state["selected_features"] = selected_features
                st.session_state["algorithm"] = algorithm
                st.session_state["elapsed"] = elapsed
                st.session_state["params"] = params
                st.session_state["label_col"] = label_col

                st.success(f"✅ Training complete in {elapsed:.2f}s")

            except Exception as e:
                st.error(f"❌ Error training model: {e}")
                st.exception(e)

    # Display results if available
    if "labels" in st.session_state or "embedding" in st.session_state:
        labels = st.session_state.get("labels")
        embedding = st.session_state.get("embedding")
        extra_info = st.session_state.get("extra_info", {})
        algo = st.session_state.get("algorithm")

        st.markdown("---")
        st.markdown("### 📊 Results")
        st.expander("Result Explanations").markdown(
             """ **Silhuette Score:** 
             *Measures how similar an object is to its own cluster compared to other clusters. Range: [-1, 1].*
               1. Higher is better. Values near +1 indicate that the sample is far from the neighboring clusters
               2. Values near 0 indicate that the sample is on or very close to the decision boundary between two neighboring clusters. 
               3. Negative values indicate that those samples might have been assigned to the wrong cluster.

            **Calinski-Harabasz Index:** 
            *Ratio of between-cluster variance to within-cluster variance. Higher is better.*
            1. Higher values indicate better-defined clusters.
            2. Sensitive to the number of clusters — can be used to help choose k by

            **Davies-Bouldin Index:**
            *Average similarity between each cluster and its most similar one, where similarity is the ratio of within-cluster distances to between-cluster distances. Lower is better.*
            1. Lower values indicate better clustering (well-separated, compact clusters).
            2. Values > 1 often indicate overlapping clusters, while values close to 0 indicate very distinct clusters.


             """
        )

        # Quick metrics
        if labels is not None:
            metrics = compute_metrics(st.session_state["X_scaled"].values, labels)
            mcols = st.columns(4)
            mcols[0].metric("Clusters found", metrics["n_clusters"])
            if metrics.get("silhouette") is not None:
                mcols[1].metric("Silhouette", f"{metrics['silhouette']:.3f}", help="Higher is better. Range: [-1, 1]")
            else:
                mcols[1].metric("Silhouette", "N/A")
            if metrics.get("calinski_harabasz") is not None:
                mcols[2].metric("Calinski-Harabasz", f"{metrics['calinski_harabasz']:.1f}", help="Higher is better")
            else:
                mcols[2].metric("Calinski-Harabasz", "N/A")
            if metrics.get("davies_bouldin") is not None:
                mcols[3].metric("Davies-Bouldin", f"{metrics['davies_bouldin']:.3f}", help="Lower is better")
            else:
                mcols[3].metric("Davies-Bouldin", "N/A")

        # Visualization — primary 2D view
        st.markdown("#### 🖼️ Cluster visualization")

        df_viz = st.session_state["df_clean"].copy()
        if labels is not None:
            df_viz["cluster"] = [f"Cluster {l}" if l != -1 else "Noise" for l in labels]

        # Pick what to plot
        if embedding is not None and embedding.shape[1] >= 2:
            # Plot using the embedding
            df_viz["dim1"] = embedding[:, 0]
            df_viz["dim2"] = embedding[:, 1]
            color = "cluster" if labels is not None else None
            fig = px.scatter(
                df_viz,
                x="dim1",
                y="dim2",
                color=color,
                opacity=0.75,
                height=550,
                title=f"{algo} — 2D embedding",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
            st.plotly_chart(fig, use_container_width=True)

            if embedding.shape[1] >= 3:
                df_viz["dim3"] = embedding[:, 2]
                fig3d = px.scatter_3d(
                    df_viz,
                    x="dim1", y="dim2", z="dim3",
                    color=color,
                    opacity=0.75,
                    height=600,
                    title=f"{algo} — 3D embedding",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig3d.update_traces(marker=dict(size=4))
                st.plotly_chart(fig3d, use_container_width=True)
        elif labels is not None:
            # Project original data to 2D via PCA for visualization
            n_dims = st.session_state["X_scaled"].shape[1]
            if n_dims >= 2:
                pca_viz = PCA(n_components=2)
                proj = pca_viz.fit_transform(st.session_state["X_scaled"])
                df_viz["pc1"] = proj[:, 0]
                df_viz["pc2"] = proj[:, 1]
                fig = px.scatter(
                    df_viz,
                    x="pc1",
                    y="pc2",
                    color="cluster",
                    opacity=0.75,
                    height=550,
                    title=f"{algo} clusters projected to 2D via PCA",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    labels={
                        "pc1": f"PC1 ({pca_viz.explained_variance_ratio_[0]*100:.1f}%)",
                        "pc2": f"PC2 ({pca_viz.explained_variance_ratio_[1]*100:.1f}%)",
                    },
                )
                fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
                st.plotly_chart(fig, use_container_width=True)

                # Optional: choose two original features
                with st.expander("👀 View clusters in original feature space"):
                    fc1, fc2 = st.columns(2)
                    feat_x = fc1.selectbox("X axis", st.session_state["selected_features"], key="cx")
                    feat_y = fc2.selectbox(
                        "Y axis",
                        st.session_state["selected_features"],
                        index=1 if len(st.session_state["selected_features"]) > 1 else 0,
                        key="cy",
                    )
                    fig2 = px.scatter(
                        df_viz,
                        x=feat_x,
                        y=feat_y,
                        color="cluster",
                        opacity=0.75,
                        height=500,
                        color_discrete_sequence=px.colors.qualitative.Bold,
                    )
                    st.plotly_chart(fig2, use_container_width=True)

        # Algorithm-specific extras
        if algo == "PCA (Dimensionality Reduction)" and "explained_variance_ratio" in extra_info:
            st.markdown("#### 📐 Explained variance")
            evr = extra_info["explained_variance_ratio"]
            cumvar = np.cumsum(evr)
            evr_df = pd.DataFrame({
                "Component": [f"PC{i+1}" for i in range(len(evr))],
                "Variance explained": evr,
                "Cumulative": cumvar,
            })
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=evr_df["Component"], y=evr_df["Variance explained"], name="Per component", marker_color="#667eea"),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=evr_df["Component"],
                    y=evr_df["Cumulative"],
                    name="Cumulative",
                    mode="lines+markers",
                    marker=dict(color="#f5576c", size=10),
                    line=dict(width=3),
                ),
                secondary_y=True,
            )
            fig.update_yaxes(title_text="Per-component variance", secondary_y=False)
            fig.update_yaxes(title_text="Cumulative variance", secondary_y=True, range=[0, 1.05])
            fig.update_layout(height=400, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(evr_df, use_container_width=True)

    else:
        st.info("👆 Click **Train model** to run the selected algorithm with your chosen hyperparameters.")



# TAB 5 — EVALUATE

with tab_evaluate:
    st.subheader("Diagnostic plots & model selection tools")
    st.write("These plots help you choose good hyperparameters — they run independently of the main model.")

    diag_tabs = st.tabs(["📉 Elbow method", "🌳 Dendrogram", "🎯 Method comparison"])

    # --- Elbow method
    with diag_tabs[0]:
        st.markdown("#### Elbow method for K-Means")
        st.caption("The 'elbow' in the inertia curve suggests a good choice of k.")
        max_k = st.slider("Max k to test", 3, 20, 10, key="elbow_max_k")
        if st.button("Run elbow analysis", key="elbow_btn"):
            inertias = []
            silhouettes = []
            ks = list(range(2, max_k + 1))
            progress = st.progress(0)
            for i, k in enumerate(ks):
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                lbl = km.fit_predict(X_scaled)
                inertias.append(km.inertia_)
                try:
                    silhouettes.append(silhouette_score(X_scaled, lbl))
                except Exception:
                    silhouettes.append(np.nan)
                progress.progress((i + 1) / len(ks))
            progress.empty()

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Inertia (lower is better)", "Silhouette score (higher is better)"),
            )
            fig.add_trace(
                go.Scatter(x=ks, y=inertias, mode="lines+markers", name="Inertia",
                           line=dict(color="#667eea", width=3), marker=dict(size=10)),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=ks, y=silhouettes, mode="lines+markers", name="Silhouette",
                           line=dict(color="#f5576c", width=3), marker=dict(size=10)),
                row=1, col=2,
            )
            fig.update_xaxes(title_text="k", row=1, col=1)
            fig.update_xaxes(title_text="k", row=1, col=2)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            best_k = ks[int(np.nanargmax(silhouettes))]
            st.success(f"💡 Suggested k by silhouette: **{best_k}** (silhouette = {silhouettes[ks.index(best_k)]:.3f})")


    # --- Dendrogram
    with diag_tabs[1]:
        st.markdown("#### Hierarchical clustering dendrogram")
        st.caption("The y-axis shows merge distances. Cut horizontally to choose number of clusters.")
        with st.expander("📖 How to read a dendrogram", expanded=False):
            st.markdown(
                """
                A dendrogram is a tree showing **how points were merged** into clusters, step by step.
 
                **The axes:**
                - **X-axis:** individual samples (or clusters of samples — numbers in parentheses are cluster sizes after truncation).
                - **Y-axis:** the *distance* at which two clusters were merged. Higher = more dissimilar.
 
                **How to read it:**
                - Each horizontal line connects two clusters that were merged. The height tells you how far apart they were.
                - Short merges near the bottom = points that are very similar.
                - Tall merges near the top = clusters that are quite different from each other.
 
                **How to choose the number of clusters (k):**
                1. Imagine a horizontal line cutting across the tree.
                2. Count how many vertical lines it crosses — that's your k.
                3. **The best place to cut is where you'd cross the largest vertical gap** (a long stretch with no merges happening). This gap means the clusters above the line are very different from each other, so cutting there gives you well-separated groups.
 
                **Color meaning:** scipy automatically colors clusters that are below an internal threshold differently. Same-colored branches belong to the same "natural" cluster at scipy's chosen cut height.
                """
            )
        dend_link = st.selectbox("Linkage", ["ward", "complete", "average", "single"], key="dend_link")
        truncate_at = st.slider("Show last N merges", 5, 50, 30, key="dend_trunc")
        suggest_k = st.slider("Show suggested cut for k =", 2, 10, 3, key="dend_k")

        if st.button("Plot dendrogram", key="dend_btn"):
            n_for_dendro = min(500, len(X_scaled))
            if len(X_scaled) > 500:
                st.info(f"📌 Sampled {n_for_dendro} points (out of {len(X_scaled)}) for dendrogram.")
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X_scaled), size=n_for_dendro, replace=False)
                X_dend = X_scaled.values[idx]
            else:
                X_dend = X_scaled.values

            with st.spinner("Computing linkage..."):
                Z = linkage(X_dend, method=dend_link)
            
            distances = Z[:, 2]
            if suggest_k <= len(distances):
                cut_height = (distances[-suggest_k] + distances[-suggest_k + 1]) / 2 if suggest_k > 1 else distances[-1] * 1.1
            else:
                cut_height = None

            fig, ax = plt.subplots(figsize=(12, 6))
            dendrogram(
                Z,
                truncate_mode="lastp",
                p=truncate_at,
                show_leaf_counts=True,
                leaf_rotation=90,
                leaf_font_size=9,
                color_threshold=0.7 * max(Z[:, 2]),
                ax=ax,
            )

            # Draw the suggested cut line
            if cut_height is not None:
                ax.axhline(
                    y=cut_height, color="red", linestyle="--", linewidth=2,
                    label=f"Cut for k={suggest_k} (height ≈ {cut_height:.2f})",
                )
                ax.legend(loc="upper right")

            ax.set_title(f"Dendrogram (linkage = {dend_link})")
            ax.set_xlabel("Sample index or (cluster size)")
            ax.set_ylabel("Distance")
            st.pyplot(fig)
            plt.close(fig)

    # --- Method comparison
    with diag_tabs[2]:
        st.markdown("#### Compare clustering methods")
        st.caption("Side-by-side scoreboard of clustering variants on the current data.")
        cmp_k = st.slider("Number of clusters (k)", 2, 10, 3, key="cmp_k")
        if st.button("Run comparison", key="cmp_btn"):
            results = []
            algos = {
                "K-Means": KMeans(n_clusters=cmp_k, n_init=10, random_state=42),
                "Hierarchical (ward)": AgglomerativeClustering(n_clusters=cmp_k, linkage="ward"),
                "Hierarchical (complete)": AgglomerativeClustering(n_clusters=cmp_k, linkage="complete"),
                "Hierarchical (average)": AgglomerativeClustering(n_clusters=cmp_k, linkage="average"),
                "Hierarchical (single)": AgglomerativeClustering(n_clusters=cmp_k, linkage="single"),
            }
            for name, m in algos.items():
                t0 = time.time()
                try:
                    lbl = m.fit_predict(X_scaled)
                    metrics = compute_metrics(X_scaled.values, lbl)
                    results.append({
                        "Method": name,
                        "Clusters": metrics["n_clusters"],
                        "Silhouette": metrics.get("silhouette"),
                        "Calinski-H": metrics.get("calinski_harabasz"),
                        "Davies-B": metrics.get("davies_bouldin"),
                        "Time (s)": round(time.time() - t0, 3),
                    })
                except Exception as e:
                    results.append({
                        "Method": name,
                        "Clusters": "Error",
                        "Silhouette": None,
                        "Calinski-H": None,
                        "Davies-B": None,
                        "Time (s)": str(e)[:30],
                    })
            res_df = pd.DataFrame(results)
            st.dataframe(
                res_df.style.format({
                    "Silhouette": "{:.3f}",
                    "Calinski-H": "{:.1f}",
                    "Davies-B": "{:.3f}",
                }, na_rep="N/A").background_gradient(subset=["Silhouette"], cmap="RdYlGn"),
                use_container_width=True,
            )



# TAB 6 — INTERPRET

with tab_interpret:
    st.subheader("Make sense of the clusters")

    if "labels" not in st.session_state or st.session_state.get("labels") is None:
        st.info("👈 First train a clustering model in the **Train Model** tab.")
    else:
        labels = st.session_state["labels"]
        df_int = st.session_state["df_clean"].copy()
        df_int["cluster"] = labels
        feat_list = st.session_state["selected_features"]

        # Cluster sizes
        st.markdown("#### 📦 Cluster sizes")
        size_df = df_int["cluster"].value_counts().sort_index().reset_index()
        size_df.columns = ["cluster", "count"]
        size_df["cluster"] = size_df["cluster"].apply(lambda x: f"Cluster {x}" if x != -1 else "Noise")
        size_df["pct"] = (size_df["count"] / size_df["count"].sum() * 100).round(1)

        fig = px.bar(
            size_df, x="cluster", y="count", text="count",
            color="cluster",
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=350,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Cluster profiles — mean per feature
        st.markdown("#### 🎨 Cluster profiles (mean per feature)")
        profile = df_int.groupby("cluster")[feat_list].mean().round(3)
        st.dataframe(
            profile.style.background_gradient(axis=0, cmap="RdYlBu_r"),
            use_container_width=True,
        )

        # Heatmap of standardized profiles
        st.markdown("#### 🌡️ Standardized profile heatmap")
        st.caption("Each cell = z-score of feature mean for that cluster. Lets you compare across features.")
        profile_z = (profile - profile.mean()) / profile.std(ddof=0).replace(0, 1)
        fig = px.imshow(
            profile_z,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-2, zmax=2,
            labels=dict(x="Feature", y="Cluster", color="z-score"),
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Boxplots per feature, split by cluster
        st.markdown("#### 📦 Feature distributions by cluster")
        feat_for_box = st.selectbox("Feature to inspect:", feat_list, key="box_feat")
        df_box = df_int[df_int["cluster"] != -1] if -1 in df_int["cluster"].values else df_int
        df_box = df_box.copy()
        df_box["cluster_str"] = df_box["cluster"].apply(lambda x: f"Cluster {x}")
        fig = px.box(
            df_box, x="cluster_str", y=feat_for_box,
            color="cluster_str", points="outliers",
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=400,
        )
        fig.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        # Compare to known labels if available
        known_label = st.session_state.get("label_col")
        if known_label and known_label in df_int.columns:
            st.markdown(f"#### ⚖️ Compare clusters to known labels ({known_label})")
            ct = pd.crosstab(df_int["cluster"], df_int[known_label])
            fig = px.imshow(
                ct,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                labels=dict(x=known_label, y="Cluster", color="Count"),
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Row-normalized
            ct_norm = ct.div(ct.sum(axis=1), axis=0)
            with st.expander("View row-normalized (purity per cluster)"):
                st.dataframe(
                    ct_norm.style.format("{:.2%}").background_gradient(cmap="Blues"),
                    use_container_width=True,
                )

        # Auto-generated cluster narrative
        st.markdown("#### 📝 Auto-generated cluster summary")
        narrative_lines = []
        for c in sorted(df_int["cluster"].unique()):
            sub = df_int[df_int["cluster"] == c]
            size = len(sub)
            pct = size / len(df_int) * 100
            # Find top 3 distinguishing features (z-score furthest from 0)
            if c in profile_z.index:
                top_feats = profile_z.loc[c].abs().sort_values(ascending=False).head(3).index.tolist()
                desc_parts = []
                for f in top_feats:
                    z = profile_z.loc[c, f]
                    direction = "high" if z > 0 else "low"
                    desc_parts.append(f"{direction} `{f}` ({profile.loc[c, f]:.2f})")
                narrative_lines.append(
                    f"**Cluster {c}** ({size} points, {pct:.1f}%): characterized by {', '.join(desc_parts)}."
                )
        for line in narrative_lines:
            st.markdown(f"- {line}")



# TAB 7 — EXPORT
with tab_export:
    st.subheader("Download your results")

    if "labels" in st.session_state and st.session_state.get("labels") is not None:
        out_df = st.session_state["df_clean"].copy()
        out_df["cluster"] = st.session_state["labels"]
        if st.session_state.get("embedding") is not None:
            emb = st.session_state["embedding"]
            for i in range(emb.shape[1]):
                out_df[f"embedding_dim{i+1}"] = emb[:, i]

        st.dataframe(out_df.head(20), use_container_width=True)
        csv = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download labeled data (CSV)",
            csv,
            file_name="clustered_data.csv",
            mime="text/csv",
            type="primary",
        )

        # Run summary
        st.markdown("#### Run summary")
        summary = {
            "Algorithm": st.session_state.get("algorithm"),
            "Hyperparameters": st.session_state.get("params", {}),
            "Features used": st.session_state.get("selected_features"),
            "Rows": len(out_df),
            "Clusters found": int(len(set(st.session_state["labels"]) - {-1})),
            "Training time (s)": round(st.session_state.get("elapsed", 0), 3),
        }
        st.json(summary)

    elif st.session_state.get("embedding") is not None:
        out_df = st.session_state["df_clean"].copy()
        emb = st.session_state["embedding"]
        for i in range(emb.shape[1]):
            out_df[f"embedding_dim{i+1}"] = emb[:, i]
        st.dataframe(out_df.head(20), use_container_width=True)
        csv = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download embedded data (CSV)",
            csv,
            file_name="embedded_data.csv",
            mime="text/csv",
            type="primary",
        )
    else:
        st.info("Train a model first — then come back here to download results.")



# FOOTER

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; padding: 1rem;">
    Built with Streamlit · scikit-learn · Plotly · matplotlib<br>
    🔮 <em>Unsupervised ML Explorer</em>
    </div>
    """,
    unsafe_allow_html=True,
)