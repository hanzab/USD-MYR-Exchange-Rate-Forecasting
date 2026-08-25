# ============================================================
# BSD 3523 MACHINE LEARNING PROJECT
# REGRESSION MODEL COMPARISON
# ============================================================

import os
import time
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Machine Learning Model Comparison",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_FILE = "ml_data.csv"

TARGET_COLUMN = "USD"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"Cannot find '{DATA_FILE}'. "
            f"Make sure the CSV file is in the same "
            f"folder as ml_app.py."
        )

    df = pd.read_csv(DATA_FILE)

    return df


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df):

    data = df.copy()

    # --------------------------------------------------------
    # Check target
    # --------------------------------------------------------

    if TARGET_COLUMN not in data.columns:

        raise ValueError(
            f"Column '{TARGET_COLUMN}' does not exist.\n\n"
            f"Available columns:\n{list(data.columns)}"
        )

    # --------------------------------------------------------
    # Remove completely empty columns
    # --------------------------------------------------------

    data = data.dropna(
        axis=1,
        how="all"
    )

    # --------------------------------------------------------
    # Convert target to numeric
    # --------------------------------------------------------

    data[TARGET_COLUMN] = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce"
    )

    # --------------------------------------------------------
    # DATE PROCESSING
    # --------------------------------------------------------

    date_columns = []

    for column in data.columns:

        if column.lower() == "date":

            data[column] = pd.to_datetime(
                data[column],
                errors="coerce"
            )

            data["year"] = (
                data[column].dt.year
            )

            data["month"] = (
                data[column].dt.month
            )

            data["quarter"] = (
                data[column].dt.quarter
            )

            date_columns.append(
                column
            )

    # --------------------------------------------------------
    # Remove original Date column
    # --------------------------------------------------------

    if date_columns:

        data = data.drop(
            columns=date_columns
        )

    # --------------------------------------------------------
    # CONVERT CATEGORICAL COLUMNS
    # --------------------------------------------------------

    categorical_columns = (
        data.select_dtypes(
            include=[
                "object",
                "category"
            ]
        ).columns.tolist()
    )

    for column in categorical_columns:

        if column != TARGET_COLUMN:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Keep numeric columns
    # --------------------------------------------------------

    numeric_columns = (
        data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
    )

    if TARGET_COLUMN not in numeric_columns:

        raise ValueError(
            f"'{TARGET_COLUMN}' is not numeric "
            f"and could not be converted."
        )

    data = data[numeric_columns]

    # --------------------------------------------------------
    # Replace infinity
    # --------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Remove rows with missing target
    # --------------------------------------------------------

    data = data.dropna(
        subset=[TARGET_COLUMN]
    )

    # --------------------------------------------------------
    # Fill missing predictor values
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in data.columns
        if column != TARGET_COLUMN
    ]

    for column in feature_columns:

        median_value = (
            data[column].median()
        )

        if pd.isna(median_value):

            median_value = 0

        data[column] = (
            data[column]
            .fillna(median_value)
        )

    # --------------------------------------------------------
    # Remove constant columns
    # --------------------------------------------------------

    constant_columns = []

    for column in feature_columns:

        if data[column].nunique() <= 1:

            constant_columns.append(
                column
            )

    if constant_columns:

        data = data.drop(
            columns=constant_columns
        )

    return data


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
):

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = {

        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE
            ),

        "Decision Tree":
            DecisionTreeRegressor(
                random_state=RANDOM_STATE
            ),

        "SVR":
            SVR(
                kernel="rbf",
                C=100,
                gamma="scale"
            ),

        "XGBoost":
            XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                eval_metric="rmse"
            )
    }

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    results = []

    trained_models = {}

    predictions = {}

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    progress = st.progress(0)

    status = st.empty()

    total_models = len(models)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for index, (
        name,
        model
    ) in enumerate(
        models.items()
    ):

        status.write(
            f"Training **{name}**..."
        )

        start_time = time.time()

        # ----------------------------------------------------
        # FIT
        # ----------------------------------------------------

        model.fit(
            X_train_scaled,
            y_train
        )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        preds = model.predict(
            X_test_scaled
        )

        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        training_time = (
            time.time()
            - start_time
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                preds
            )
        )

        mae = mean_absolute_error(
            y_test,
            preds
        )

        r2 = r2_score(
            y_test,
            preds
        )

        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        results.append(
            {
                "Model": name,
                "RMSE": rmse,
                "MAE": mae,
                "R2 Score": r2,
                "Training Time (s)": training_time
            }
        )

        trained_models[
            name
        ] = model

        predictions[
            name
        ] = preds

        progress.progress(
            int(
                (
                    (index + 1)
                    / total_models
                )
                * 100
            )
        )

    status.success(
        "All models trained successfully."
    )

    return (
        pd.DataFrame(results),
        trained_models,
        predictions
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # TITLE
    # ========================================================

    st.title(
        "📊 Machine Learning Regression Analysis"
    )

    st.write(
        "Regression model comparison for predicting USD."
    )

    st.divider()

    # ========================================================
    # LOAD DATA
    # ========================================================

    try:

        df = load_data()

    except Exception as error:

        st.error(
            "Dataset loading failed."
        )

        st.exception(error)

        st.stop()

    # ========================================================
    # SIDEBAR
    # ========================================================

    st.sidebar.title(
        "⚙️ Model Settings"
    )

    st.sidebar.success(
        "Dataset loaded successfully"
    )

    st.sidebar.write(
        f"File: `{DATA_FILE}`"
    )

    st.sidebar.write(
        f"Rows: **{len(df):,}**"
    )

    st.sidebar.write(
        f"Columns: **{len(df.columns):,}**"
    )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    try:

        model_df = prepare_data(
            df
        )

    except Exception as error:

        st.error(
            "Data preparation failed."
        )

        st.exception(error)

        st.stop()

    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    st.header(
        "1. Dataset Overview"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Original Rows",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Original Columns",
            f"{len(df.columns):,}"
        )

    with col3:

        st.metric(
            "Clean Rows",
            f"{len(model_df):,}"
        )

    with col4:

        st.metric(
            "Features",
            f"{len(model_df.columns) - 1:,}"
        )

    # ========================================================
    # VIEW DATA
    # ========================================================

    with st.expander(
        "👁️ View Dataset"
    ):

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    with st.expander(
        "🔍 View Cleaned Model Dataset"
    ):

        st.dataframe(
            model_df,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # PREPARE X AND Y
    # ========================================================

    X = model_df.drop(
        columns=[
            TARGET_COLUMN
        ]
    )

    y = model_df[
        TARGET_COLUMN
    ]

    # ========================================================
    # CHECK FEATURES
    # ========================================================

    if X.shape[1] == 0:

        st.error(
            "No predictor features remain "
            "after preprocessing."
        )

        st.stop()

    # ========================================================
    # FEATURE LIST
    # ========================================================

    with st.expander(
        "📋 Model Features"
    ):

        feature_df = pd.DataFrame(
            {
                "Feature": X.columns
            }
        )

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    st.header(
        "2. Train-Test Split"
    )

    test_size = st.sidebar.slider(
        "Test Size",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=RANDOM_STATE
        )
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Training Rows",
            f"{len(X_train):,}"
        )

    with col2:

        st.metric(
            "Testing Rows",
            f"{len(X_test):,}"
        )

    with col3:

        st.metric(
            "Training %",
            f"{(1 - test_size) * 100:.0f}%"
        )

    with col4:

        st.metric(
            "Testing %",
            f"{test_size * 100:.0f}%"
        )

    # ========================================================
    # SCALING
    # ========================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # ========================================================
    # MODEL TRAINING
    # ========================================================

    st.header(
        "3. Model Training"
    )

    st.info(
        "The following six regression models "
        "will be trained:"
    )

    st.write(
        """
        1. Linear Regression  
        2. Random Forest  
        3. Gradient Boosting  
        4. Decision Tree  
        5. Support Vector Regression (SVR)  
        6. XGBoost
        """
    )

    train_button = st.button(
        "🚀 Train All Models",
        type="primary",
        use_container_width=True
    )

    if not train_button:

        st.warning(
            "Click **Train All Models** to start "
            "the machine learning process."
        )

        return

    # ========================================================
    # TRAIN
    # ========================================================

    try:

        (
            results,
            trained_models,
            predictions
        ) = train_models(
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test
        )

    except Exception as error:

        st.error(
            "Model training failed."
        )

        st.exception(error)

        st.stop()

    # ========================================================
    # SORT RESULTS
    # ========================================================

    results = (
        results
        .sort_values(
            by="RMSE",
            ascending=True
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.header(
        "4. Model Performance Comparison"
    )

    st.dataframe(
        results.round(4),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_model_name = (
        results.iloc[0]["Model"]
    )

    best_rmse = (
        results.iloc[0]["RMSE"]
    )

    best_mae = (
        results.iloc[0]["MAE"]
    )

    best_r2 = (
        results.iloc[0]["R2 Score"]
    )

    st.success(
        f"🏆 Best Performing Model: "
        f"**{best_model_name}**"
    )

    # ========================================================
    # BEST MODEL METRICS
    # ========================================================

    metric1, metric2, metric3 = (
        st.columns(3)
    )

    with metric1:

        st.metric(
            "RMSE",
            f"{best_rmse:,.4f}"
        )

    with metric2:

        st.metric(
            "MAE",
            f"{best_mae:,.4f}"
        )

    with metric3:

        st.metric(
            "R² Score",
            f"{best_r2:,.4f}"
        )

    # ========================================================
    # RMSE CHART
    # ========================================================

    st.subheader(
        "RMSE Comparison"
    )

    fig_rmse = px.bar(
        results,
        x="Model",
        y="RMSE",
        title=(
            "Model RMSE Comparison "
            "— Lower is Better"
        ),
        text_auto=".4f"
    )

    fig_rmse.update_layout(
        xaxis_tickangle=-25
    )

    st.plotly_chart(
        fig_rmse,
        use_container_width=True
    )

    # ========================================================
    # MAE CHART
    # ========================================================

    st.subheader(
        "MAE Comparison"
    )

    fig_mae = px.bar(
        results,
        x="Model",
        y="MAE",
        title=(
            "Model MAE Comparison "
            "— Lower is Better"
        ),
        text_auto=".4f"
    )

    fig_mae.update_layout(
        xaxis_tickangle=-25
    )

    st.plotly_chart(
        fig_mae,
        use_container_width=True
    )

    # ========================================================
    # R2 CHART
    # ========================================================

    st.subheader(
        "R² Score Comparison"
    )

    fig_r2 = px.bar(
        results,
        x="Model",
        y="R2 Score",
        title=(
            "Model R² Comparison "
            "— Higher is Better"
        ),
        text_auto=".4f"
    )

    fig_r2.update_layout(
        xaxis_tickangle=-25
    )

    st.plotly_chart(
        fig_r2,
        use_container_width=True
    )

    # ========================================================
    # TRAINING TIME
    # ========================================================

    st.subheader(
        "Training Time Comparison"
    )

    fig_time = px.bar(
        results,
        x="Model",
        y="Training Time (s)",
        title="Training Time by Model",
        text_auto=".4f"
    )

    fig_time.update_layout(
        xaxis_tickangle=-25
    )

    st.plotly_chart(
        fig_time,
        use_container_width=True
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.header(
        "5. Feature Importance"
    )

    tree_models = [
        "Random Forest",
        "Gradient Boosting",
        "Decision Tree",
        "XGBoost"
    ]

    if best_model_name in tree_models:

        best_model = (
            trained_models[
                best_model_name
            ]
        )

        importance = pd.Series(
            best_model.feature_importances_,
            index=X.columns
        )

        importance = (
            importance
            .sort_values(
                ascending=False
            )
        )

        importance_table = (
            importance
            .reset_index()
        )

        importance_table.columns = [
            "Feature",
            "Importance"
        ]

        st.dataframe(
            importance_table.round(4),
            use_container_width=True,
            hide_index=True
        )

        fig_importance = px.bar(
            importance_table
            .sort_values(
                "Importance",
                ascending=True
            ),
            x="Importance",
            y="Feature",
            orientation="h",
            title=(
                f"Feature Importance — "
                f"{best_model_name}"
            ),
            text_auto=".4f"
        )

        st.plotly_chart(
            fig_importance,
            use_container_width=True
        )

    else:

        st.info(
            f"Feature importance is not directly "
            f"available for {best_model_name}."
        )

    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    st.header(
        "6. Actual vs Predicted"
    )

    selected_model_name = (
        st.selectbox(
            "Select Model",
            list(
                trained_models.keys()
            )
        )
    )

    selected_prediction = (
        predictions[
            selected_model_name
        ]
    )

    comparison_df = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted":
                selected_prediction
        }
    )

    # --------------------------------------------------------
    # METRICS FOR SELECTED MODEL
    # --------------------------------------------------------

    selected_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            selected_prediction
        )
    )

    selected_mae = (
        mean_absolute_error(
            y_test,
            selected_prediction
        )
    )

    selected_r2 = (
        r2_score(
            y_test,
            selected_prediction
        )
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    with c1:

        st.metric(
            "Selected Model RMSE",
            f"{selected_rmse:,.4f}"
        )

    with c2:

        st.metric(
            "Selected Model MAE",
            f"{selected_mae:,.4f}"
        )

    with c3:

        st.metric(
            "Selected Model R²",
            f"{selected_r2:,.4f}"
        )

    # ========================================================
    # ACTUAL VS PREDICTED CHART
    # ========================================================

    fig_actual = px.scatter(
        comparison_df,
        x="Actual",
        y="Predicted",
        title=(
            f"{selected_model_name}: "
            "Actual vs Predicted"
        )
    )

    min_value = min(
        comparison_df["Actual"].min(),
        comparison_df["Predicted"].min()
    )

    max_value = max(
        comparison_df["Actual"].max(),
        comparison_df["Predicted"].max()
    )

    fig_actual.add_shape(
        type="line",
        x0=min_value,
        y0=min_value,
        x1=max_value,
        y1=max_value,
        line=dict(
            dash="dash"
        )
    )

    fig_actual.update_layout(
        xaxis_title="Actual USD",
        yaxis_title="Predicted USD"
    )

    st.plotly_chart(
        fig_actual,
        use_container_width=True
    )

    # ========================================================
    # PREDICTION TABLE
    # ========================================================

    st.subheader(
        "Prediction Results"
    )

    comparison_df["Error"] = (
        comparison_df["Actual"]
        -
        comparison_df["Predicted"]
    )

    comparison_df[
        "Absolute Error"
    ] = (
        comparison_df["Error"]
        .abs()
    )

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # MODEL RANKING
    # ========================================================

    st.header(
        "7. Final Model Ranking"
    )

    ranking = results.copy()

    ranking.insert(
        0,
        "Rank",
        range(
            1,
            len(ranking) + 1
        )
    )

    st.dataframe(
        ranking.round(4),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.header(
        "8. Export Results"
    )

    results_csv = results.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Model Results",
        data=results_csv,
        file_name=(
            "model_performance_results.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )

    # ========================================================
    # FOOTER
    # ========================================================

    st.divider()

    st.caption(
        "BSD 3523 Machine Learning Project | "
        "Regression Model Comparison"
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()