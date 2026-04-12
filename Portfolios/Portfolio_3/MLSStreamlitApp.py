import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import (
    load_breast_cancer,
    load_wine,
    fetch_california_housing,
    load_diabetes

)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

st.set_page_config(page_title="Supervised ML Explorer", layout="wide")

st.title(" 📊 Supervised Machine Learning Explorer")
st.write(
    "Upload a dataset or use a sample dataset, choose classification or regression, "
    "train models, tune hyperparameters, and evaluate performance."
)

# how to use this app guide 
button_expander = st.expander("How to Use This App")
with button_expander:
    st.write(
        """
        1. **Choose Dataset**: Select a sample dataset or upload your own CSV file.
        2. **Select Target Variable**: Choose the column you want to predict.
        3. **Select Task Type**: Choose between classification (categorical target) or regression (numeric target).
        4. **Train-Test Split**: Adjust the test size and random state for reproducibility.
        5. **Choose Model**: Select a machine learning model to train.
        6. **Hyperparameters**: Tune the hyperparameters for the selected model.
        7. **Train Model**: Click the button to train the model and view results in the tabs.
        
        Use the tabs to explore data overview, training results, evaluation metrics, and compare different models.
        """
    )
     

# sample data for users

def load_sample_dataset(name):
    if name == "Diabetes":
        data = load_diabetes(as_frame=True)
        df = data.frame
    elif name == "Breast Cancer":
        data = load_breast_cancer(as_frame=True)
        df = data.frame
    elif name == "Wine":
        data = load_wine(as_frame=True)
        df = data.frame
    elif name == "California Housing":
        data = fetch_california_housing(as_frame=True)
        df = data.frame
    
    return df

# target column to guarantee it's not included in the features and to handle categorical targets for classification and numeric targets for regression

def preprocess_features(df, target_column):
    X = df.drop(columns=[target_column]).copy()
    X = pd.get_dummies(X, drop_first=True)
    return X


def preprocess_target_for_classification(y):
    if y.dtype == "object" or str(y.dtype) == "category":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
    else:
        y = y.values
    return y


def preprocess_target_for_regression(y):
    return pd.to_numeric(y, errors="coerce")


# comparison of models with default settings to give users a benchmark for how different algorithms perform on their dataset with minimal tuning

def compare_classification_models(X_train, X_test, y_train, y_test, random_state):
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=300))
        ]),
        "Decision Tree Classifier": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(random_state=random_state))
        ]),
        "KNN Classifier": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        "Random Forest Classifier": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=random_state))
        ])  
    }

    results = []
# for classification models we use accuracy, precision, recall, and F1 score to give a well-rounded view of performance, 
# especially for imbalanced datasets.

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, preds, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, preds, average="weighted", zero_division=0)
        })

    return pd.DataFrame(results)

# for regression models where our predictable target variable is numeric or continuous

def compare_regression_models(X_train, X_test, y_train, y_test, random_state):
    models = {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Decision Tree Regressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeRegressor(random_state=random_state))
        ]),
        "KNN Regressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor())
        ]),
        "Random Forest Regressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(random_state=random_state))
        ])  
    }

    results = []
# results for regression models 

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)

        results.append({
            "Model": name,
            "R²": r2_score(y_test, preds),
            "MAE": mean_absolute_error(y_test, preds),
            "MSE": mse,
            "RMSE": rmse
        })

    return pd.DataFrame(results)

# SIDE BAR FOR USER INPUTS 

st.sidebar.header("1. Choose Dataset")

dataset_option = st.sidebar.radio(
    "Select data source:",
    ["Use Sample Dataset", "Upload CSV File"]
)

df = None

if dataset_option == "Use Sample Dataset":
    sample_name = st.sidebar.selectbox(
        "Choose a sample dataset:",
        ["Diabetes", "Breast Cancer", "Wine", "California Housing"]
    )
    df = load_sample_dataset(sample_name)

elif dataset_option == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)


if df is not None:
    st.sidebar.header("2. Select Target Variable")
    target_column = st.sidebar.selectbox("Choose the target column:", df.columns)

    st.sidebar.header("3. Select Task Type")
    task_type = st.sidebar.radio(
        "Choose supervised learning task:",
        ["Classification", "Regression"]
    )

    X = preprocess_features(df, target_column)
    y_original = df[target_column]

    if task_type == "Classification":
        y = preprocess_target_for_classification(y_original)
    else:
        y = preprocess_target_for_regression(y_original)
        valid_idx = y.notna()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

    st.sidebar.header("4. Train-Test Split")
    test_size = st.sidebar.slider(
        "Select test size:",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05
    )

    random_state = st.sidebar.number_input(
        "Random state:",
        min_value=0,
        max_value=1000,
        value=42,
        step=1
    )

    st.sidebar.header("5. Choose Model")

    if task_type == "Classification":
        model_name = st.sidebar.selectbox(
            "Select a classification model:",
            ["Logistic Regression", "Decision Tree Classifier", "KNN Classifier", 'Random Forest Classifier']
        )
    else:
        model_name = st.sidebar.selectbox(
            "Select a regression model:",
            ["Linear Regression", "Decision Tree Regressor", "KNN Regressor", 'Random Forest Regressor']
        )

    st.sidebar.header("6. Hyperparameters")

    if model_name == "Logistic Regression":
        C = st.sidebar.slider("Regularization strength (C)", 0.01, 10.0, 1.0)
        max_iter = st.sidebar.slider("Max iterations", 100, 1000, 300, step=50)

    elif model_name == "Decision Tree Classifier":
        max_depth = st.sidebar.slider("Max depth", 1, 20, 5)
        min_samples_split = st.sidebar.slider("Min samples split", 2, 20, 2)
        criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])

    elif model_name == "KNN Classifier":
        n_neighbors = st.sidebar.slider("Number of neighbors (k)", 1, 20, 5)
        weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
    elif model_name == 'Random Forest Classifier':
        n_estimators = st.sidebar.slider("Number of trees (n_estimators)", 10, 200, 100, step=10)
        max_depth = st.sidebar.slider("Max depth", 1, 20, 5)
        min_samples_split = st.sidebar.slider("Min samples split", 2, 20, 2)    

    elif model_name == "Decision Tree Regressor":
        max_depth = st.sidebar.slider("Max depth", 1, 20, 5)
        min_samples_split = st.sidebar.slider("Min samples split", 2, 20, 2)

    elif model_name == "KNN Regressor":
        n_neighbors = st.sidebar.slider("Number of neighbors (k)", 1, 20, 5)
        weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
    elif model_name == 'Random Forest Regressor':
        n_estimators = st.sidebar.slider("Number of trees (n_estimators)", 10, 200, 100, step=10)
        max_depth = st.sidebar.slider("Max depth", 1, 20, 5)
        min_samples_split = st.sidebar.slider("Min samples split", 2, 20, 2)

    show_tree = False
    plot_depth = 3

    if model_name in ["Decision Tree Classifier", "Decision Tree Regressor", 'Random Forest Classifier', 'Random Forest Regressor']:
        st.sidebar.header("7. Tree Visualization")
        show_tree = st.sidebar.checkbox("Show Decision Tree", value=False)
        plot_depth = st.sidebar.slider("Tree plot depth", 1, 10, 3)

    train_model = st.sidebar.button("Train Model")

# MAIN APP CONTENT WITH TABS FOR ORGANIZED DISPLAY OF DATA, RESULTS, EVALUATION, AND COMPARISON


    tab1, tab2, tab3, tab4 = st.tabs(
        ["Data Overview", "Training Results", "Evaluation and Visuals", "Comparison"]
    )

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isna().sum().sum()))

        with st.expander("See Column Types"):
            st.write(df.dtypes)

        with st.expander("See Missing Values by Column"):
            st.write(df.isna().sum())

        st.subheader("Processed Features")
        st.write(f"Feature matrix shape: {X.shape}")
        st.write(f"Task type selected: {task_type}")

# MODELS AVAIALBEL FOR SLECTIOM AND TRAINING CODE

    if train_model:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state
            )

            if model_name == "Logistic Regression":
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(C=C, max_iter=max_iter))
                ])

            elif model_name == "Decision Tree Classifier":
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        criterion=criterion,
                        random_state=random_state
                    ))
                ])

            elif model_name == "KNN Classifier":
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier(
                        n_neighbors=n_neighbors,
                        weights=weights
                    ))
                ])
            elif model_name == 'Random Forest Classifier':
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=random_state
                    ))
                ])
            

            elif model_name == "Linear Regression":
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression())
                ])

            elif model_name == "Decision Tree Regressor":
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", DecisionTreeRegressor(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=random_state
                    ))
                ])

            elif model_name == "KNN Regressor":
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsRegressor(
                        n_neighbors=n_neighbors,
                        weights=weights
                    ))
                ])
            elif model_name == 'Random Forest Regressor':
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=random_state
                    ))
                ])

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
# TABS BASED ON MODEL TYPE AND SELECTION 

            if task_type == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                results_df = pd.DataFrame({
                    "Actual": y_test,
                    "Predicted": y_pred
                }).reset_index(drop=True)

                fig_cm, ax_cm = plt.subplots()
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(ax=ax_cm)

                comparison_df = compare_classification_models(
                    X_train, X_test, y_train, y_test, random_state
                )
# TAB 2 FOR CLASSIFICATION MODELS performance metrics and prediction 
                with tab2:
                    st.subheader("Model Performance")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{accuracy:.3f}")
                    m2.metric("Precision", f"{precision:.3f}")
                    m3.metric("Recall", f"{recall:.3f}")
                    m4.metric("F1 Score", f"{f1:.3f}")

                    st.subheader("Prediction Preview")
                    st.dataframe(results_df.head(20))
# tab 3 for classification models (decision tree and randome forrest)

                with tab3:
                    st.subheader("Confusion Matrix")
                    st.pyplot(fig_cm)

                    if len(np.unique(y)) == 2 and hasattr(model.named_steps["model"], "predict_proba"):
                        y_probs = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_probs)
                        auc_score = roc_auc_score(y_test, y_probs)

                        fig_roc, ax_roc = plt.subplots()
                        ax_roc.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
                        ax_roc.plot([0, 1], [0, 1], linestyle="--")
                        ax_roc.set_xlabel("False Positive Rate")
                        ax_roc.set_ylabel("True Positive Rate")
                        ax_roc.set_title("ROC Curve")
                        ax_roc.legend()

                        st.subheader("ROC Curve")
                        st.pyplot(fig_roc)
                        st.metric("AUC Score", f"{auc_score:.3f}")
                    else:
                        st.info("ROC curve is only shown for binary classification.")

                    st.subheader("Interpretation")
                    if accuracy > 0.9:
                        st.write("This classification model performed very well on the test set.")
                    elif accuracy > 0.75:
                        st.write("This classification model performed reasonably well, with room for improvement.")
                    else:
                        st.write("This classification model had weaker performance, so different settings or a different model may work better.")

                    if model_name == "Decision Tree Classifier":
                        tree_model = model.named_steps["model"]
                        importance_df = pd.DataFrame({
                            "Feature": X.columns,
                            "Importance": tree_model.feature_importances_
                        }).sort_values(by="Importance", ascending=False)

                        fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
                        ax_imp.barh(
                            importance_df["Feature"].head(10)[::-1],
                            importance_df["Importance"].head(10)[::-1]
                        )
                        ax_imp.set_title("Top 10 Feature Importances")

                        st.subheader("Feature Importances")
                        st.dataframe(importance_df.head(10))
                        st.pyplot(fig_imp)
                    

                        if show_tree:
                            st.subheader("Decision Tree Visualization")
                            fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
                            plot_tree(
                                tree_model,
                                feature_names=X.columns,
                                class_names=[str(c) for c in np.unique(y_train)],
                                filled=True,
                                rounded=True,
                                fontsize=8,
                                max_depth=plot_depth,
                                ax=ax_tree
                            )
                            st.pyplot(fig_tree)
                # random forrect classifier feature importanc and visualization

                    if model_name == "Random Forest Classifier":
                        forest_model = model.named_steps["model"]
                        importance_df = pd.DataFrame({
                            "Feature": X.columns,
                            "Importance": forest_model.feature_importances_
                        }).sort_values(by="Importance", ascending=False)

                        fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
                        ax_imp.barh(
                            importance_df["Feature"].head(10)[::-1],
                            importance_df["Importance"].head(10)[::-1]
                        )
                        ax_imp.set_title("Top 10 Feature Importances")

                        st.subheader("Feature Importances")
                        st.dataframe(importance_df.head(10))
                        st.pyplot(fig_imp)
                        
                        if show_tree:
                            st.subheader("Random Forest Visualization")
                            st.write("Visualizing individual trees from a random forest can be complex, but here is one of the trees in the forest:")
                            fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
                            plot_tree(
                                forest_model.estimators_[0],
                                feature_names=X.columns,
                                class_names=[str(c) for c in np.unique(y_train)],
                                filled=True,
                                rounded=True,
                                fontsize=8,
                                max_depth=plot_depth,
                                ax=ax_tree
                            )
                            st.pyplot(fig_tree)
                
# TAB 4 model comaprison, performance metrics and roc and auc curve compariosn between models (classification)
                with tab4:
                    st.subheader("Model Comparison")
                    st.dataframe(comparison_df)
                    st.subheader("ROC Curve Comparison Across Models")
                    def plot_model_roc_comparison(X_train, X_test, y_train, y_test, random_state):
                        models = {
                            "Logistic Regression": Pipeline([
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                                ("model", LogisticRegression(max_iter=300))
                            ]),
                            "Decision Tree Classifier": Pipeline([
                                ("imputer", SimpleImputer(strategy="median")),
                                ("model", DecisionTreeClassifier(random_state=random_state))
                            ]),
                            "KNN Classifier": Pipeline([
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                                ("model", KNeighborsClassifier())
                            ]),
                            "Random Forest Classifier": Pipeline([
                                ("imputer", SimpleImputer(strategy="median")),
                                ("model", RandomForestClassifier(random_state=random_state))
                            ])

                        }

                        fig, ax = plt.subplots(figsize=(8, 6))

                        for name, model in models.items():
                            model.fit(X_train, y_train)

                            if hasattr(model.named_steps["model"], "predict_proba"):
                                y_probs = model.predict_proba(X_test)[:, 1]
                                fpr, tpr, _ = roc_curve(y_test, y_probs)
                                auc_score = roc_auc_score(y_test, y_probs)
                                ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

                        ax.plot([0, 1], [0, 1], linestyle="--")
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.set_title("ROC Curve Comparison")
                        ax.legend()

                        return fig
                    roc_fig = plot_model_roc_comparison(X_train, X_test, y_train, y_test, random_state)
                    st.pyplot(roc_fig)
                    

            else:# this else is to hanlde regression models (not clasfication = regression)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)

                results_df = pd.DataFrame({
                    "Actual": y_test,
                    "Predicted": y_pred
                }).reset_index(drop=True)

                comparison_df = compare_regression_models(
                    X_train, X_test, y_train, y_test, random_state
                )

                with tab2:
                    st.subheader("Model Performance")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("R²", f"{r2:.3f}")
                    m2.metric("MAE", f"{mae:.3f}")
                    m3.metric("MSE", f"{mse:.3f}")
                    m4.metric("RMSE", f"{rmse:.3f}")

                    st.subheader("Prediction Preview")
                    st.dataframe(results_df.head(20))

                with tab3:
                    st.subheader("Predicted vs Actual")

                    fig_reg, ax_reg = plt.subplots()
                    ax_reg.scatter(y_test, y_pred)
                    ax_reg.set_xlabel("Actual Values")
                    ax_reg.set_ylabel("Predicted Values")
                    ax_reg.set_title("Predicted vs Actual")

                    min_val = min(np.min(y_test), np.min(y_pred))
                    max_val = max(np.max(y_test), np.max(y_pred))
                    ax_reg.plot([min_val, max_val], [min_val, max_val], linestyle="--")

                    st.pyplot(fig_reg)

                    st.subheader("Interpretation")
                    if r2 > 0.8:
                        st.write("This regression model explains a large portion of the variation in the target variable.")
                    elif r2 > 0.5:
                        st.write("This regression model has moderate predictive power, but there is still room for improvement.")
                    else:
                        st.write("This regression model is not explaining much of the variation, so a different model or better features may be needed.")

                    if model_name == "Decision Tree Regressor":
                        tree_model = model.named_steps["model"]
                        importance_df = pd.DataFrame({
                            "Feature": X.columns,
                            "Importance": tree_model.feature_importances_
                        }).sort_values(by="Importance", ascending=False)

                        fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
                        ax_imp.barh(
                            importance_df["Feature"].head(10)[::-1],
                            importance_df["Importance"].head(10)[::-1]
                        )
                        ax_imp.set_title("Top 10 Feature Importances")

                        st.subheader("Feature Importances")
                        st.dataframe(importance_df.head(10))
                        st.pyplot(fig_imp)

                        if show_tree:
                            st.subheader("Decision Tree Visualization")
                            fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
                            plot_tree(
                                tree_model,
                                feature_names=X.columns,
                                filled=True,
                                rounded=True,
                                fontsize=8,
                                max_depth=plot_depth,
                                ax=ax_tree
                            )
                            st.pyplot(fig_tree)
                    if model_name == "Random Forest Regressor":
                        forest_model = model.named_steps["model"]
                        importance_df = pd.DataFrame({
                            "Feature": X.columns,
                            "Importance": forest_model.feature_importances_
                        }).sort_values(by="Importance", ascending=False)

                        fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
                        ax_imp.barh(
                            importance_df["Feature"].head(10)[::-1],
                            importance_df["Importance"].head(10)[::-1]
                        )
                        ax_imp.set_title("Top 10 Feature Importances")

                        st.subheader("Feature Importances")
                        st.dataframe(importance_df.head(10))
                        st.pyplot(fig_imp)
                        
                        if show_tree:
                            st.subheader("Random Forest Visualization")
                            st.write("Visualizing individual trees from a random forest can be complex, but here is one of the trees in the forest:")
                            fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
                            plot_tree(
                                forest_model.estimators_[0],
                                feature_names=X.columns,
                                filled=True,
                                rounded=True,
                                fontsize=8,
                                max_depth=plot_depth,
                                ax=ax_tree
                            )
                            st.pyplot(fig_tree)

                        
        

                with tab4:
                    st.subheader("Model Comparison")
                    st.dataframe(comparison_df)
                

        except Exception as e:
            st.error(f"An error occurred during model training: {e}")

else:
    st.info("Please upload a dataset or choose a sample dataset from the sidebar.")



