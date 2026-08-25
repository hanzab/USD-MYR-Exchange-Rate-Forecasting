# ============================================================
# MODEL TRAINING
# ============================================================

from xgboost import XGBRegressor

# Prepare data
X = df.drop(
    columns=[
        "USD",
        "Date",
        "year",
        "month"
    ],
    errors="ignore"
)

y = df["USD"]

# Remove rows containing missing values
model_data = pd.concat(
    [X, y],
    axis=1
).dropna()

X = model_data.drop(
    columns=["USD"]
)

y = model_data["USD"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "SVR":
        SVR(
            kernel="rbf"
        ),

    "XGBoost":
        XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            eval_metric="rmse"
        )
}


# ============================================================
# TRAIN MODELS
# ============================================================

results = []

trained_models = {}

predictions = {}


with st.spinner("Training models..."):

    for name, model in models.items():

        start_time = time.time()

        # Linear/SVR/XGBoost can use numeric data.
        # We use scaled data consistently here.
        model.fit(
            X_train_scaled,
            y_train
        )

        preds = model.predict(
            X_test_scaled
        )

        training_time = (
            time.time() - start_time
        )

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

        results.append(
            {
                "Model": name,
                "RMSE": rmse,
                "MAE": mae,
                "R2 Score": r2,
                "Training Time (s)": training_time
            }
        )

        trained_models[name] = model

        predictions[name] = preds


# ============================================================
# RESULTS
# ============================================================

res_df = pd.DataFrame(
    results
).sort_values(
    by="RMSE",
    ascending=True
).reset_index(
    drop=True
)


st.subheader(
    "Model Performance Comparison"
)

st.dataframe(
    res_df.round(4),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# BEST MODEL
# ============================================================

best_model_name = (
    res_df.iloc[0]["Model"]
)

st.success(
    f"🏆 Best Performing Model: "
    f"**{best_model_name}**"
)


# ============================================================
# RMSE
# ============================================================

fig_rmse = px.bar(
    res_df,
    x="Model",
    y="RMSE",
    title="Model RMSE Comparison — Lower is Better"
)

st.plotly_chart(
    fig_rmse,
    use_container_width=True
)


# ============================================================
# R2
# ============================================================

fig_r2 = px.bar(
    res_df,
    x="Model",
    y="R2 Score",
    title="Model R² Comparison — Higher is Better"
)

st.plotly_chart(
    fig_r2,
    use_container_width=True
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

if best_model_name in [
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "Decision Tree"
]:

    st.subheader(
        f"Feature Importance — {best_model_name}"
    )

    best_model = trained_models[
        best_model_name
    ]

    importance = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    ).sort_values(
        ascending=True
    )

    fig_feature = px.bar(
        importance,
        orientation="h",
        title="Feature Importance"
    )

    st.plotly_chart(
        fig_feature,
        use_container_width=True
    )


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

st.subheader(
    "Actual vs Predicted"
)

selected_model_name = st.selectbox(
    "Select Model",
    list(models.keys())
)

selected_prediction = predictions[
    selected_model_name
]

comparison_df = pd.DataFrame(
    {
        "Actual": y_test.values,
        "Predicted":
            selected_prediction
    }
)

fig_actual = px.scatter(
    comparison_df,
    x="Actual",
    y="Predicted",
    title=(
        f"{selected_model_name}: "
        "Actual vs Predicted"
    )
)

st.plotly_chart(
    fig_actual,
    use_container_width=True
)