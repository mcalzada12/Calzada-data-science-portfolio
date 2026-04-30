
# 🔮 Unsupervised ML Explorer
 
> *Discover hidden structure in your data — no labels required.*
 
An interactive Streamlit app that turns unsupervised machine learning into an exploration playground. Upload your own data or pick from seven sample datasets, tune hyperparameters in real time, and watch clusters emerge — with built-in guidance every step of the way.
 
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
 
---
 
## ✨ What makes it different
 
Most ML demos throw a single algorithm at you and call it done. This app is different — it's built to **teach you what's happening** while you experiment:
 
- 🎓 **Every hyperparameter has a plain-language description** that updates as you change selections
- 🔍 **Built-in feature importance analysis** for unsupervised learning (rare!)
- 📖 **Auto-generated interpretations** of dendrograms and cluster profiles
- 🎯 **Smart recommendations** — the app suggests optimal `k`, flags low-variance features, and warns about redundancy
- 🌗 **Side-by-side method comparison** so you can see which algorithm fits your data best
---
 
## 🛠️ Features
 
### Three classic algorithms
| Algorithm | When to use it |
|---|---|
| 🎯 **K-Means** | Fast clustering when you expect roughly spherical, balanced groups |
| 🌳 **Hierarchical** | When you want a tree of merges and the flexibility to choose `k` afterward |
| 📐 **PCA** | Compress data, kill correlations, or visualize high-dimensional structure |
 
### Seven sample datasets
**Real-world classics** — Iris 🌸, Wine 🍷, Breast Cancer 🔬
**Practical use cases** — Mall Customers 🛍️ (segmentation)
**Algorithm stress-tests** — Synthetic Blobs 🟣, Two Moons 🌙, Concentric Circles ⭕
 
…or upload your own CSV.
 
### Diagnostic suite
- 📉 **Elbow method and Silhuette Test** with both inertia and silhouette curves
- 🌳 **Interactive dendrogram** with adjustable cut lines and auto-interpretation
- 🎯 **Method comparison scoreboard** — run all variants side-by-side
- Three quality metrics: **Silhouette**, **Calinski-Harabasz**, **Davies-Bouldin**
### Feature analysis tab *(the secret weapon)*
- 📏 **Variance** — auto-flags near-constant features
- 🔗 **Redundancy** — detects feature pairs with |correlation| > 0.9
- 🎯 **Cluster discriminative power** — F-statistic feature importance for clustering
- 📐 **PCA loadings** — see which features drive each principal component
### Cluster interpretation
- Cluster size breakdown with proportions
- Per-cluster feature profiles + standardized z-score heatmap
- Box plots split by cluster
- Cross-tabulation against known labels (when available)
- **Auto-generated natural-language summaries** of each cluster
### Polished UX
- Custom gradient styling and color-coded tabs
- Interactive Plotly charts with zoom, hover, and 3D rotation
- Session-state persistence — train once, explore everywhere
- One-click CSV download with cluster labels and embeddings
---
 
## 🚀 Quick start
 
### Run locally
```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py
```
 
Then open `http://localhost:8501` in your browser.
 
### Deploy to Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and pick `app.py`
4. Done — your app is live
---
 
## 📁 Project structure
 
```
.
├── app.py              # The full Streamlit application
├── requirements.txt    # Python dependencies
├── sample_data.csv     # Sample CSV for testing the upload flow
└── README.md           # You're here
```
 
---
 
## 🎯 Suggested walkthrough
 
Try this 5-minute tour to see the app in action:
 
1. **Sidebar → Sample dataset → "Mall Customers"** — start with a familiar segmentation problem
2. **Overview tab** — get a feel for the dataset shape and basic stats
3. **Feature Analysis tab → Cluster discriminative power** — see which features will drive clustering before you even run it
4. **Sidebar → K-Means**, leave default `k=3`
5. **Train Model tab → 🚀 Train model** — watch clusters appear in 2D and 3D
6. **Evaluate tab → Elbow method** — verify `k=3` was a reasonable choice
7. **Interpret tab** — read the auto-generated cluster narrative
8. **Export tab** — download a labeled CSV
**Bonus challenge:** Try the **Two Moons** dataset with K-Means at `k=2`. K-Means will fail — and you'll see exactly why centroid-based methods struggle with non-convex shapes.
 
---
 
## 🧠 Tech stack
 
- **[Streamlit](https://streamlit.io)** — UI framework
- **[scikit-learn](https://scikit-learn.org)** — ML algorithms and metrics
- **[Plotly](https://plotly.com/python/)** — interactive charts (2D + 3D)
- **[Matplotlib](https://matplotlib.org)** — silhouette and dendrogram plots
- **[SciPy](https://scipy.org)** — hierarchical clustering and statistical tests
- **[pandas](https://pandas.pydata.org)** + **[NumPy](https://numpy.org)** — data plumbing
---
 
## 📚 What you'll learn
 
This app is designed to demystify unsupervised learning. By the time you've explored it, you'll understand:
 
- Why **scaling** matters before clustering (and which scaler to pick)
- The difference between `k-means++` and `random` initialization (and why one is the default)
- How **linkage methods** (ward, complete, average, single) shape hierarchical clusters
- How to read a **dendrogram** and pick a cut height
- Why **PCA before clustering** can dramatically improve results
- How to measure **feature importance** when there's no target label
---
 
## 🤝 Contributing
 
Found a bug? Have an idea for a new diagnostic? PRs welcome — open an issue first to discuss bigger changes.
 
---
 
## 📄 License
 
MIT — do whatever you want with it.
 
---
 
<div align="center">
**Built with ❤️ and ☕ for everyone who's ever stared at a dataset and wondered _"what's hiding in here?"_**
 
🔮 *Unsupervised ML Explorer*
 
</div>

