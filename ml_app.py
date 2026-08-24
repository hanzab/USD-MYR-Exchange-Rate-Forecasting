import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Malaysia USD Exchange Rate Predictor",
    page_icon="🇲🇾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('ml_data.csv')
    # Create a Date column for plotting
    df['Date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1))
    return df

df = load_data()

# Sidebar Navigation
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Flag_of_Malaysia.svg/800px-Flag_of_Malaysia.svg.png", 
            width=150)
    st.title("Navigation")
    page = st.radio("Go to", ["🏠 Overview", "📊 Data Exploration", "🤖 Model Training", "📈 Forecast"])
    
    st.markdown("---")
    st.info("""
    **Course:** BSD3523 Machine Learning
    **University:** Universiti Malaysia Pahang al-Sultan Abdullah (UMPSA)
    **Team:** 
    - Muhammad Danial Bin Issham
    - Ain Mardhiah Binti Abdul Hamid
    - Haizatul Syifa Binti Mansor
    - Hamizan Nasri Bin Zulkairi
    - Siti Nurul Insyirah Binti Mohd Fauzi
    """)

# --- PAGE 1: OVERVIEW ---
if page == "🏠 Overview":
    st.title("Malaysia USD Exchange Rate Predictor")
    
    st.markdown("""
    ## Project Overview
    
    This project aims to predict the USD/MYR exchange rate using various economic indicators,
    including GDP, GNI, migration trends, and leading economic indicators.
    
    ### **Data Sources:**
    1. **MEI (Main Economic Indicators)** - Leading, Coincident, Lagging indices
    2. **GDP/GNI Annual Data** - Gross Domestic Product and Gross National Income
    3. **Net Migration Statistics** - Migration trends in Malaysia
    4. **Monthly Exchange Rates** - USD/MYR exchange rates
    5. **CPI Inflation Data** - Consumer Price Index inflation rates
    """)
    
    # Key Metrics
    current_usd = df['USD'].iloc[-1]
    avg_usd = df['USD'].mean()
    
    # Calculate GDP Growth (Year-over-Year comparison of latest months)
    latest_gdp = df['gdp'].iloc[-1]
    prev_year_gdp = df['gdp'].iloc[-13] if len(df) > 12 else df['gdp'].iloc[0]
    gdp_growth = ((latest_gdp - prev_year_gdp) / prev_year_gdp) * 100
    
    avg_migration = df['Net migration'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current USD/MYR", f"{current_usd:.4f}")
    col2.metric("Average USD/MYR", f"{avg_usd:.4f}")
    col3.metric("GDP Growth (YoY)", f"{gdp_growth:.2f}%")
    col4.metric("Avg Net Migration", f"{avg_migration:,.0f}")

    st.divider()

    # Dataset Information
    st.subheader("📋 Dataset Information")
    c1, c2, c3 = st.columns(3)
    c1.write(f"**Rows:** {df.shape[0]}")
    c2.write(f"**Columns:** {df.shape[1]}")
    c3.write(f"**Time Period:** {df['year'].min()} - {df['year'].max()}")
    
    if st.checkbox("Show Raw Data"):
        st.dataframe(df.head(10))

    # Economic Trends Visualization
    st.subheader("📈 Economic Trends")
    trend_col = st.selectbox("Select Indicator for Trend Analysis", 
                             ['USD', 'gdp', 'gni', 'inflation', 'Net migration', 'leading', 'coincident'])
    
    fig_trend = px.line(df, x='Date', y=trend_col, title=f'{trend_col} Trend Over Time',
                        line_shape='spline', render_mode='svg')
    fig_trend.update_layout(hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)

# --- PAGE 2: DATA EXPLORATION ---
elif page == "📊 Data Exploration":
    st.title("🔍 Exploratory Data Analysis (EDA)")
    
    # Filter for numeric columns only to prevent errors in KDE/Boxplots
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove 'year' and 'month' from analysis as they are categorical in nature
    analysis_cols = [c for c in numeric_cols if c not in ['year', 'month', 'Date']]

    tab1, tab2, tab3, tab4 = st.tabs(["Skewness (KDE)", "Outliers (Boxplot)", "Correlation Heatmap", "USD Monthly Analysis"])

    with tab1:
        st.subheader("Distribution & Skewness")
        var_kde = st.selectbox("Select variable for KDE", analysis_cols, key="kde_sel")
        
        try:
            # Remove NaN values for better plotting
            kde_data = df[[var_kde]].dropna()
            
            # Set seaborn style
            sns.set_style("darkgrid")
            
            # Create KDE plot with histogram
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(kde_data[var_kde], kde=True, color='#3366CC', ax=ax, stat='density')
            skewness = kde_data[var_kde].skew()
            ax.set_title(f"{var_kde} | Skewness: {round(skewness, 2)}", fontsize=14, fontweight='bold')
            ax.set_xlabel(f"{var_kde} Value", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error creating KDE plot: {e}")

    with tab2:
        st.subheader("Outlier Detection")
        var_box = st.selectbox("Select variable for Boxplot", analysis_cols, key="box_sel")
        
        try:
            box_data = df[[var_box]].dropna()
            
            fig_box = px.box(
                box_data, y=var_box, 
                title=f"Boxplot of {var_box}", 
                color_discrete_sequence=['#EF553B']
            )
            st.plotly_chart(fig_box, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error creating boxplot: {e}")

    with tab3:
        st.subheader("Feature Correlation")
        try:
            # Ensure only numeric data is used for correlation
            corr_data = df[analysis_cols].dropna()
            corr = corr_data.corr()
            
            fig_heat = px.imshow(
                corr, 
                text_auto=".2f", 
                aspect="auto", 
                color_continuous_scale='RdBu_r', 
                title="Correlation Heatmap (Numeric Indicators)"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error creating correlation heatmap: {e}")

    with tab4:
        st.subheader("USD/MYR Monthly Seasonality")
        try:
            # Ensure month is treated as a category for the X-axis
            df_monthly = df.copy()
            df_monthly['month_name'] = pd.Categorical(
                df_monthly['month'].apply(lambda x: pd.to_datetime(f'2020-{int(x):02d}-01').strftime('%B')),
                categories=["January", "February", "March", "April", "May", "June", 
                           "July", "August", "September", "October", "November", "December"],
                ordered=True
            )
            
            fig_month = px.box(
                df_monthly, 
                x='month_name', 
                y='USD', 
                title="USD/MYR Variation by Month",
                color='month_name'
            )
            st.plotly_chart(fig_month, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error creating monthly analysis: {e}")

# --- PAGE 3: MODEL TRAINING ---
elif page == "🤖 Model Training":
    st.title("🤖 Model Training & Comparison")

    # Prepare Data
    X = df.drop(columns=['USD', 'Date', 'year', 'month'])
    y = df['USD']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models Dictionary
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "SVR": SVR(kernel='rbf'),
        "XGBoost": xgb.XGBRegressor(objective ='reg:squarederror', random_state=42)
    }

    results = []
    trained_models = {}

    with st.spinner('Training models...'):
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2 Score": r2})
            trained_models[name] = model

    # Performance Comparison
    res_df = pd.DataFrame(results).sort_values(by="RMSE")
    st.subheader("Model Performance Comparison")
    st.dataframe(res_df.style.highlight_min(subset=['RMSE', 'MAE'], color='lightgreen')
                         .highlight_max(subset=['R2 Score'], color='lightgreen'))

    # Plot Comparison
    fig_comp = px.bar(res_df, x='Model', y='RMSE', color='RMSE', title="Model RMSE Comparison (Lower is Better)")
    st.plotly_chart(fig_comp, use_container_width=True)

    # Best Model Selection
    best_model_name = res_df.iloc[0]['Model']
    st.success(f"Best Performing Model: **{best_model_name}**")

    # Feature Importance (for tree-based models)
    if best_model_name in ["Random Forest", "Gradient Boosting", "XGBoost", "Decision Tree"]:
        st.subheader(f"Feature Importance ({best_model_name})")
        feat_importances = pd.Series(trained_models[best_model_name].feature_importances_, index=X.columns)
        fig_feat = px.bar(feat_importances.sort_values(), orientation='h', title="Feature Importance")
        st.plotly_chart(fig_feat, use_container_width=True)

    # Actual vs Predicted
    st.subheader("Actual vs Predicted Comparison")
    selected_model_name = st.selectbox("Select Model to Visualize Results", list(models.keys()))
    sel_model = trained_models[selected_model_name]
    y_pred = sel_model.predict(X_test_scaled)
    
    res_plot_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index()
    fig_res = px.scatter(res_plot_df, x='Actual', y='Predicted', trendline="ols",
                         title=f"{selected_model_name}: Actual vs Predicted")
    st.plotly_chart(fig_res, use_container_width=True)

# --- PAGE 4: FORECASTING ---
elif page == "📈 Forecast":
    st.title("🔮USD/MYR Forecast")
    
    # 1. Prepare Data and Model
    # We include year and month in the features so the model can capture time trends/seasonality
    X = df.drop(columns=['USD', 'Date'])
    y = df['USD']
    
    # We'll use Random Forest as it's generally robust for this type of tabular forecasting
    best_model = RandomForestRegressor(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_model.fit(X_scaled, y)

    st.info("""
    This tool forecasts the USD/MYR exchange rate for the next 12 months. 
    The model uses historical seasonal patterns (months) and projected economic growth to estimate future rates.
    """)
    
    # 2. Sidebar - Forecast Scenario Settings
    st.sidebar.subheader("Forecast Scenarios")
    gdp_growth_annual = st.sidebar.slider("Expected Annual GDP Growth (%)", -5.0, 10.0, 3.0)
    gni_growth_annual = st.sidebar.slider("Expected Annual GNI Growth (%)", -5.0, 10.0, 3.0)
    coincident_adj = st.sidebar.slider("Coincident Indicator Adjustment", -10.0, 10.0, 0.0)
    lagging_adj = st.sidebar.slider("Lagging Indicator Adjustment", -10.0, 10.0, 0.0)
    leading_adj = st.sidebar.slider("Leading Indicator Adjustment", -10.0, 10.0, 0.0)
    
    # 3. Generate 12-Month Future Data
    last_row = df.iloc[-1]
    last_date = df['Date'].max()
    
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 13)]
    future_rows = []

    for i, f_date in enumerate(future_dates, 1):
        new_row = last_row.copy()
        
        # Update temporal features
        new_row['year'] = f_date.year
        new_row['month'] = f_date.month
        
        # Project GDP (compounded monthly growth based on annual input)
        monthly_gdp_growth_rate = (1 + gdp_growth_annual/100)**(1/12)
        new_row['gdp'] = last_row['gdp'] * (monthly_gdp_growth_rate ** i)
        
        # Project GNI (compounded monthly growth based on annual input)
        monthly_gni_growth_rate = (1 + gni_growth_annual/100)**(1/12)
        new_row['gni'] = last_row['gni'] * (monthly_gni_growth_rate ** i)
        
        # Adjust Economic Indicators
        new_row['coincident'] = last_row['coincident'] + (coincident_adj * i / 12)
        new_row['lagging'] = last_row['lagging'] + (lagging_adj * i / 12)
        new_row['leading'] = last_row['leading'] + (leading_adj * i / 12)
        
        future_rows.append(new_row.drop(['USD', 'Date']))

    future_df = pd.DataFrame(future_rows)
    
    # 4. Run Predictions
    future_scaled = scaler.transform(future_df)
    predictions = best_model.predict(future_scaled)
    
    # Prepare result dataframe
    forecast_results = pd.DataFrame({
        'Date': future_dates,
        'USD_Forecast': predictions
    })

    # 5. Visualizations
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Forecast Table")
        # Formatting for display
        display_df = forecast_results.copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y - %b')
        st.dataframe(display_df.style.format({'USD_Forecast': '{:.4f}'}))

    with col2:
        st.subheader("Visual Trend")
        fig_forecast = go.Figure()
        
        # Plot last 24 months of history for context
        history_context = df.tail(24)
        fig_forecast.add_trace(go.Scatter(
            x=history_context['Date'], y=history_context['USD'],
            name='Historical (Last 24m)', mode='lines+markers'
        ))
        
        # Plot 12 month forecast
        fig_forecast.add_trace(go.Scatter(
            x=forecast_results['Date'], y=forecast_results['USD_Forecast'],
            name='12-Month Forecast', mode='lines+markers',
            line=dict(color='red', dash='dash')
        ))
        
        fig_forecast.update_layout(
            xaxis_title="Date",
            yaxis_title="USD/MYR",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

    # 6. Summary Metrics
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Highest Projected", f"{predictions.max():.4f}")
    m2.metric("Lowest Projected", f"{predictions.min():.4f}")
    m3.metric("Average Forecast", f"{predictions.mean():.4f}")