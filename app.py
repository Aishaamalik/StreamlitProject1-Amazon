import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import io
import base64
from datetime import datetime
import re
from wordcloud import WordCloud
import time

# Amazon Color Scheme
AMAZON_COLORS = {
    'primary': '#FF9900',  # Amazon Orange
    'secondary': '#232F3E',  # Amazon Dark Blue
    'accent': '#146EB4',  # Amazon Blue
    'light_gray': '#F3F3F3',
    'dark_gray': '#37475A',
    'success': '#067D62',
    'warning': '#FF6B35',
    'danger': '#B12704'
}

# Custom CSS for Amazon styling
def load_custom_css():
    st.markdown(f"""
    <style>
    /* Main theme colors */
    .main {{
        background-color: white;
    }}
    
    /* Sidebar styling */
    .css-1d391kg {{
        background-color: {AMAZON_COLORS['light_gray']};
    }}
    
    /* Header styling */
    .amazon-header {{
        background: linear-gradient(90deg, {AMAZON_COLORS['secondary']}, {AMAZON_COLORS['dark_gray']});
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }}
    
    /* Metric cards */
    .metric-card {{
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid {AMAZON_COLORS['primary']};
        margin: 0.5rem 0;
    }}
    
    /* Warning boxes */
    .warning-box {{
        background-color: #FFF3CD;
        border: 1px solid #FFEAA7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }}
    
    /* Success boxes */
    .success-box {{
        background-color: #D4EDDA;
        border: 1px solid #C3E6CB;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: {AMAZON_COLORS['light_gray']};
        border-radius: 5px 5px 0 0;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {AMAZON_COLORS['primary']};
        color: white;
    }}
    
    /* Button styling */
    .stButton > button {{
        background-color: {AMAZON_COLORS['primary']};
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }}
    
    .stButton > button:hover {{
        background-color: {AMAZON_COLORS['warning']};
        transform: translateY(-2px);
    }}
    
    /* Download button styling */
    .download-button {{
        background-color: {AMAZON_COLORS['success']};
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem;
        transition: all 0.3s;
    }}
    
    .download-button:hover {{
        background-color: {AMAZON_COLORS['accent']};
        transform: translateY(-2px);
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display:none;}}
    </style>
    """, unsafe_allow_html=True)

# Data loading and caching
@st.cache_data
def load_data():
    """Load and preprocess the Amazon dataset"""
    try:
        df = pd.read_csv('amazon.csv')
        
        # Basic data cleaning
        df = df.copy()
        
        # Clean price columns
        if 'discounted_price' in df.columns:
            df['discounted_price_clean'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').str.strip()
            df['discounted_price_numeric'] = pd.to_numeric(df['discounted_price_clean'], errors='coerce')
        
        if 'actual_price' in df.columns:
            df['actual_price_clean'] = df['actual_price'].astype(str).str.replace('₹', '').str.replace(',', '').str.strip()
            df['actual_price_numeric'] = pd.to_numeric(df['actual_price_clean'], errors='coerce')
        
        # Clean discount percentage
        if 'discount_percentage' in df.columns:
            df['discount_percentage_clean'] = df['discount_percentage'].astype(str).str.replace('%', '').str.strip()
            df['discount_percentage_numeric'] = pd.to_numeric(df['discount_percentage_clean'], errors='coerce')
        
        # Clean rating
        if 'rating' in df.columns:
            df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Clean rating count
        if 'rating_count' in df.columns:
            df['rating_count_clean'] = df['rating_count'].astype(str).str.replace(',', '').str.strip()
            df['rating_count_numeric'] = pd.to_numeric(df['rating_count_clean'], errors='coerce')
        
        # Create price range categories
        if 'discounted_price_numeric' in df.columns:
            df['price_range'] = pd.cut(df['discounted_price_numeric'], 
                                     bins=[0, 500, 1000, 2000, 5000, float('inf')], 
                                     labels=['₹0-500', '₹500-1K', '₹1K-2K', '₹2K-5K', '₹5K+'])
        
        # Create rating categories
        if 'rating_numeric' in df.columns:
            df['rating_category'] = pd.cut(df['rating_numeric'], 
                                         bins=[0, 2, 3, 4, 5], 
                                         labels=['Poor (0-2)', 'Fair (2-3)', 'Good (3-4)', 'Excellent (4-5)'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def preprocess_for_modeling(df, target_column, feature_columns):
    """Preprocess data for machine learning models"""
    try:
        # Create a copy for modeling
        model_df = df[feature_columns + [target_column]].copy()
        
        # Remove rows with missing target values
        model_df = model_df.dropna(subset=[target_column])
        
        # Handle categorical variables
        le_dict = {}
        for col in model_df.columns:
            if model_df[col].dtype == 'object' and col != target_column:
                le = LabelEncoder()
                model_df[col] = le.fit_transform(model_df[col].astype(str))
                le_dict[col] = le
        
        # Handle missing values in features
        for col in feature_columns:
            if model_df[col].dtype in ['int64', 'float64']:
                model_df[col] = model_df[col].fillna(model_df[col].median())
            else:
                model_df[col] = model_df[col].fillna('Unknown')
        
        return model_df, le_dict
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None, None

def show_warning(message):
    """Display warning message with Amazon styling"""
    st.markdown(f"""
    <div class="warning-box">
        <strong>⚠️ Warning:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def show_success(message):
    """Display success message with Amazon styling"""
    st.markdown(f"""
    <div class="success-box">
        <strong>✅ Success:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, delta=None):
    """Create a metric card with Amazon styling"""
    delta_html = ""
    if delta:
        delta_color = AMAZON_COLORS['success'] if delta > 0 else AMAZON_COLORS['danger']
        delta_html = f'<p style="color: {delta_color}; margin: 0;">Δ {delta}</p>'
    
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="margin: 0; color: {AMAZON_COLORS['secondary']};">{title}</h4>
        <h2 style="margin: 0.5rem 0; color: {AMAZON_COLORS['primary']};">{value}</h2>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def apply_filters(df):
    """Apply sidebar filters to the dataframe"""
    st.sidebar.markdown("## 🔍 Data Filters")
    
    filtered_df = df.copy()
    
    # Category filter
    if 'category' in df.columns:
        categories = df['category'].dropna().unique()
        if len(categories) > 1:
            selected_categories = st.sidebar.multiselect(
                "Select Categories",
                options=categories,
                default=categories[:5] if len(categories) > 5 else categories
            )
            if selected_categories:
                filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    # Price range filter
    if 'discounted_price_numeric' in df.columns:
        min_price = float(df['discounted_price_numeric'].min())
        max_price = float(df['discounted_price_numeric'].max())
        if not np.isnan(min_price) and not np.isnan(max_price):
            price_range = st.sidebar.slider(
                "Price Range (₹)",
                min_value=min_price,
                max_value=max_price,
                value=(min_price, max_price),
                step=100.0
            )
            filtered_df = filtered_df[
                (filtered_df['discounted_price_numeric'] >= price_range[0]) &
                (filtered_df['discounted_price_numeric'] <= price_range[1])
            ]
    
    # Rating filter
    if 'rating_numeric' in df.columns:
        min_rating = st.sidebar.slider(
            "Minimum Rating",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.1
        )
        filtered_df = filtered_df[filtered_df['rating_numeric'] >= min_rating]
    
    # Sample size for performance
    sample_size = st.sidebar.slider(
        "Sample Size (for performance)",
        min_value=100,
        max_value=min(len(filtered_df), 10000),
        value=min(len(filtered_df), 2000),
        step=100
    )
    
    if len(filtered_df) > sample_size:
        filtered_df = filtered_df.sample(n=sample_size, random_state=42)
    
    # Check for sufficient data
    if len(filtered_df) < 10:
        show_warning("Selected filters result in very few data points. Consider adjusting filters.")
    
    return filtered_df

def data_overview_tab(df):
    """Data Overview tab implementation"""
    st.markdown("""
    <div class="amazon-header">
        <h2>📊 Data Overview</h2>
        <p>Comprehensive overview of your Amazon dataset</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("Total Records", f"{len(df):,}")
    
    with col2:
        create_metric_card("Total Columns", len(df.columns))
    
    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        create_metric_card("Missing Data", f"{missing_pct:.1f}%")
    
    with col4:
        if 'discounted_price_numeric' in df.columns:
            avg_price = df['discounted_price_numeric'].mean()
            create_metric_card("Avg Price", f"₹{avg_price:.0f}")
        else:
            create_metric_card("Data Types", len(df.dtypes.unique()))
    
    # Dataset Information
    st.markdown("### 📋 Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Column Information")
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null %': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(info_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Statistical Summary")
        if df.select_dtypes(include=[np.number]).shape[1] > 0:
            st.dataframe(df.describe(), use_container_width=True)
        else:
            st.info("No numerical columns found for statistical summary")
    
    # Data Quality Issues
    st.markdown("### 🔍 Data Quality Assessment")
    
    quality_issues = []
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        quality_issues.append(f"Found {duplicates} duplicate rows")
    
    # Check for missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        quality_issues.append(f"Columns with missing values: {', '.join(missing_cols[:5])}")
    
    # Check for data type issues
    for col in df.columns:
        if 'price' in col.lower() and df[col].dtype == 'object':
            quality_issues.append(f"Price column '{col}' contains non-numeric data")
    
    if quality_issues:
        for issue in quality_issues:
            show_warning(issue)
    else:
        show_success("No major data quality issues detected!")
    
    # Sample data
    st.markdown("### 🔬 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

def data_visualization_tab(df):
    """Data Visualization tab implementation"""
    st.markdown("""
    <div class="amazon-header">
        <h2>📈 Data Visualization</h2>
        <p>Interactive visualizations of your Amazon dataset</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for sufficient data
    if len(df) < 5:
        show_warning("Insufficient data for meaningful visualizations. Please adjust your filters.")
        return
    
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Distribution Analysis", "Correlation Analysis", "Category Analysis", "Price Analysis", "Rating Analysis"]
    )
    
    if viz_type == "Distribution Analysis":
        st.markdown("#### 📊 Distribution Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns available for distribution analysis")
            return
        
        col = st.selectbox("Select column for distribution", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(df, x=col, nbins=30, 
                             title=f"Distribution of {col}",
                             color_discrete_sequence=[AMAZON_COLORS['primary']])
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color=AMAZON_COLORS['secondary'])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(df, y=col, 
                        title=f"Box Plot of {col}",
                        color_discrete_sequence=[AMAZON_COLORS['accent']])
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color=AMAZON_COLORS['secondary'])
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Analysis":
        st.markdown("#### 🔗 Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for correlation analysis")
            return
        
        corr_matrix = df[numeric_cols].corr()
        
        # Correlation heatmap
        fig = px.imshow(corr_matrix, 
                       text_auto=True,
                       aspect="auto",
                       color_continuous_scale="RdBu_r",
                       title="Correlation Heatmap")
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color=AMAZON_COLORS['secondary'])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Strong correlations
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if strong_corrs:
            st.markdown("#### Strong Correlations (|r| > 0.5)")
            for col1, col2, corr in strong_corrs:
                st.write(f"**{col1}** ↔ **{col2}**: {corr:.3f}")
    
    elif viz_type == "Category Analysis":
        st.markdown("#### 📂 Category Analysis")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if not categorical_cols:
            st.warning("No categorical columns available for analysis")
            return
        
        cat_col = st.selectbox("Select categorical column", categorical_cols)
        
        if df[cat_col].nunique() > 20:
            show_warning(f"Column '{cat_col}' has {df[cat_col].nunique()} unique values. Showing top 20.")
            top_categories = df[cat_col].value_counts().head(20)
        else:
            top_categories = df[cat_col].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig = px.bar(x=top_categories.index, y=top_categories.values,
                        title=f"Distribution of {cat_col}",
                        color_discrete_sequence=[AMAZON_COLORS['primary']])
            fig.update_xaxes(tickangle=45)
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color=AMAZON_COLORS['secondary'])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie chart
            fig = px.pie(values=top_categories.values, names=top_categories.index,
                        title=f"Proportion of {cat_col}",
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color=AMAZON_COLORS['secondary'])
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Price Analysis":
        st.markdown("#### 💰 Price Analysis")
        
        if 'discounted_price_numeric' not in df.columns:
            st.warning("No price data available for analysis")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution by category
            if 'category' in df.columns and df['category'].nunique() <= 10:
                fig = px.box(df, x='category', y='discounted_price_numeric',
                           title="Price Distribution by Category",
                           color_discrete_sequence=[AMAZON_COLORS['accent']])
                fig.update_xaxes(tickangle=45)
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color=AMAZON_COLORS['secondary'])
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Price range distribution
                if 'price_range' in df.columns:
                    price_dist = df['price_range'].value_counts().sort_index()
                    fig = px.bar(x=price_dist.index, y=price_dist.values,
                               title="Distribution by Price Range",
                               color_discrete_sequence=[AMAZON_COLORS['primary']])
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Discount analysis
            if 'discount_percentage_numeric' in df.columns:
                fig = px.scatter(df, x='actual_price_numeric', y='discount_percentage_numeric',
                               title="Discount vs Original Price",
                               color_discrete_sequence=[AMAZON_COLORS['warning']])
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color=AMAZON_COLORS['secondary'])
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Rating Analysis":
        st.markdown("#### ⭐ Rating Analysis")
        
        if 'rating_numeric' not in df.columns:
            st.warning("No rating data available for analysis")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            fig = px.histogram(df, x='rating_numeric', nbins=20,
                             title="Rating Distribution",
                             color_discrete_sequence=[AMAZON_COLORS['success']])
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color=AMAZON_COLORS['secondary'])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rating vs Price relationship
            if 'discounted_price_numeric' in df.columns:
                fig = px.scatter(df, x='rating_numeric', y='discounted_price_numeric',
                               title="Rating vs Price",
                               color_discrete_sequence=[AMAZON_COLORS['accent']])
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color=AMAZON_COLORS['secondary'])
                )
                st.plotly_chart(fig, use_container_width=True)

def eda_tab(df):
    """Exploratory Data Analysis tab implementation"""
    st.markdown("""
    <div class="amazon-header">
        <h2>🔍 Exploratory Data Analysis</h2>
        <p>Deep dive into your Amazon dataset</p>
    </div>
    """, unsafe_allow_html=True)
    
    # EDA sections
    eda_section = st.selectbox(
        "Select EDA Section",
        ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", "Text Analysis", "Outlier Detection"]
    )
    
    if eda_section == "Univariate Analysis":
        st.markdown("#### 📊 Univariate Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Numerical Variables Summary")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                for col in numeric_cols[:5]:  # Show first 5 numeric columns
                    st.markdown(f"**{col}**")
                    col_stats = df[col].describe()
                    stats_df = pd.DataFrame({
                        'Statistic': col_stats.index,
                        'Value': col_stats.values
                    })
                    st.dataframe(stats_df, use_container_width=True)
            else:
                st.info("No numerical columns found")
        
        with col2:
            st.markdown("##### Categorical Variables Summary")
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                for col in categorical_cols[:3]:  # Show first 3 categorical columns
                    st.markdown(f"**{col}**")
                    unique_count = df[col].nunique()
                    top_values = df[col].value_counts().head(3)
                    st.write(f"Unique values: {unique_count}")
                    st.write("Top 3 values:")
                    for val, count in top_values.items():
                        st.write(f"- {val}: {count}")
            else:
                st.info("No categorical columns found")
    
    elif eda_section == "Bivariate Analysis":
        st.markdown("#### 🔗 Bivariate Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        analysis_type = st.radio(
            "Select Analysis Type",
            ["Numeric vs Numeric", "Numeric vs Categorical", "Categorical vs Categorical"]
        )
        
        if analysis_type == "Numeric vs Numeric" and len(numeric_cols) >= 2:
            col1 = st.selectbox("Select first numeric variable", numeric_cols)
            col2 = st.selectbox("Select second numeric variable", [c for c in numeric_cols if c != col1])
            
            # Scatter plot
            fig = px.scatter(df, x=col1, y=col2,
                           title=f"Relationship between {col1} and {col2}",
                           color_discrete_sequence=[AMAZON_COLORS['primary']])
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color=AMAZON_COLORS['secondary'])
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation coefficient
            if not df[col1].isna().all() and not df[col2].isna().all():
                corr_coef = df[col1].corr(df[col2])
                st.metric("Correlation Coefficient", f"{corr_coef:.3f}")
        
        elif analysis_type == "Numeric vs Categorical" and len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            num_col = st.selectbox("Select numeric variable", numeric_cols)
            cat_col = st.selectbox("Select categorical variable", categorical_cols)
            
            if df[cat_col].nunique() <= 10:
                # Box plot
                fig = px.box(df, x=cat_col, y=num_col,
                           title=f"{num_col} by {cat_col}",
                           color_discrete_sequence=[AMAZON_COLORS['accent']])
                fig.update_xaxes(tickangle=45)
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color=AMAZON_COLORS['secondary'])
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical summary
                summary_stats = df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std']).round(3)
                st.dataframe(summary_stats, use_container_width=True)
            else:
                show_warning(f"Categorical variable '{cat_col}' has too many unique values ({df[cat_col].nunique()}) for effective visualization.")
    
    elif eda_section == "Text Analysis":
        st.markdown("#### 📝 Text Analysis")
        
        text_cols = [col for col in df.columns if df[col].dtype == 'object']
        if not text_cols:
            st.warning("No text columns found for analysis")
            return
        
        text_col = st.selectbox("Select text column for analysis", text_cols)
        
        # Text statistics
        df_text = df[text_col].dropna().astype(str)
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_metric_card("Total Records", len(df_text))
            create_metric_card("Unique Values", df_text.nunique())
            
            # Average text length
            avg_length = df_text.str.len().mean()
            create_metric_card("Avg Text Length", f"{avg_length:.1f}")
        
        with col2:
            # Word cloud (if text data is available)
            if len(df_text) > 0:
                try:
                    # Combine all text and create word cloud
                    all_text = ' '.join(df_text.head(1000))  # Limit for performance
                    
                    if len(all_text.strip()) > 0:
                        wordcloud = WordCloud(
                            width=400, 
                            height=300,
                            background_color='white',
                            colormap='viridis'
                        ).generate(all_text)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                except Exception as e:
                    st.info("Could not generate word cloud for this text data")
        
        # Most common values
        st.markdown("##### Most Common Values")
        top_values = df_text.value_counts().head(10)
        if len(top_values) > 0:
            fig = px.bar(x=top_values.values, y=top_values.index, orientation='h',
                        title=f"Top 10 Most Common {text_col}",
                        color_discrete_sequence=[AMAZON_COLORS['primary']])
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color=AMAZON_COLORS['secondary'])
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif eda_section == "Outlier Detection":
        st.markdown("#### 🎯 Outlier Detection")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns available for outlier detection")
            return
        
        col = st.selectbox("Select column for outlier detection", numeric_cols)
        
        # Calculate outliers using IQR method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_metric_card("Total Outliers", outlier_count)
        
        with col2:
            create_metric_card("Outlier %", f"{outlier_percentage:.2f}%")
        
        with col3:
            create_metric_card("IQR", f"{IQR:.2f}")
        
        # Outlier visualization
        fig = px.box(df, y=col, title=f"Outlier Detection for {col}")
        fig.add_hline(y=upper_bound, line_dash="dash", line_color="red", 
                     annotation_text="Upper Bound")
        fig.add_hline(y=lower_bound, line_dash="dash", line_color="red", 
                     annotation_text="Lower Bound")
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color=AMAZON_COLORS['secondary'])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if outlier_count > 0:
            st.markdown("##### Outlier Details")
            st.dataframe(outliers[[col]].describe(), use_container_width=True)

def statistical_analysis_tab(df):
    """Statistical Analysis tab implementation"""
    st.markdown("""
    <div class="amazon-header">
        <h2>📈 Statistical Analysis</h2>
        <p>Advanced statistical tests and insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    analysis_type = st.selectbox(
        "Select Statistical Test",
        ["Descriptive Statistics", "Normality Tests", "Correlation Tests", "Chi-Square Test", "T-Test/ANOVA"]
    )
    
    if analysis_type == "Descriptive Statistics":
        st.markdown("#### 📊 Descriptive Statistics")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns available for descriptive statistics")
            return
        
        selected_cols = st.multiselect("Select columns for analysis", numeric_cols, default=numeric_cols[:3])
        
        if selected_cols:
            desc_stats = df[selected_cols].describe()
            
            # Add additional statistics
            additional_stats = pd.DataFrame({
                col: {
                    'variance': df[col].var(),
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis(),
                    'range': df[col].max() - df[col].min()
                } for col in selected_cols
            }).T
            
            # Combine statistics
            full_stats = pd.concat([desc_stats.T, additional_stats], axis=1)
            st.dataframe(full_stats, use_container_width=True)
            
            # Interpretation
            st.markdown("##### Statistical Interpretation")
            for col in selected_cols:
                skew = df[col].skew()
                kurt = df[col].kurtosis()
                
                skew_interp = "symmetric" if abs(skew) < 0.5 else ("right-skewed" if skew > 0 else "left-skewed")
                kurt_interp = "mesokurtic" if abs(kurt) < 1 else ("leptokurtic" if kurt > 0 else "platykurtic")
                
                st.write(f"**{col}**: Distribution is {skew_interp} and {kurt_interp}")
    
    elif analysis_type == "Normality Tests":
        st.markdown("#### 📈 Normality Tests")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns available for normality testing")
            return
        
        col = st.selectbox("Select column for normality test", numeric_cols)
        
        # Remove missing values
        data = df[col].dropna()
        
        if len(data) < 3:
            show_warning("Insufficient data for normality testing")
            return
        
        # Shapiro-Wilk test (for small samples)
        if len(data) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(data)
            st.write(f"**Shapiro-Wilk Test**")
            st.write(f"Statistic: {shapiro_stat:.4f}")
            st.write(f"p-value: {shapiro_p:.4f}")
            st.write(f"Result: {'Normal distribution' if shapiro_p > 0.05 else 'Not normal distribution'} (α = 0.05)")
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        st.write(f"**Kolmogorov-Smirnov Test**")
        st.write(f"Statistic: {ks_stat:.4f}")
        st.write(f"p-value: {ks_p:.4f}")
        st.write(f"Result: {'Normal distribution' if ks_p > 0.05 else 'Not normal distribution'} (α = 0.05)")
        
        # Q-Q plot
        fig, ax = plt.subplots(figsize=(8, 6))
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot for {col}")
        st.pyplot(fig)
    
    elif analysis_type == "Correlation Tests":
        st.markdown("#### 🔗 Correlation Tests")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for correlation testing")
            return
        
        col1 = st.selectbox("Select first variable", numeric_cols)
        col2 = st.selectbox("Select second variable", [c for c in numeric_cols if c != col1])
        
        # Remove missing values
        data_clean = df[[col1, col2]].dropna()
        
        if len(data_clean) < 3:
            show_warning("Insufficient data for correlation testing")
            return
        
        # Pearson correlation
        pearson_corr, pearson_p = pearsonr(data_clean[col1], data_clean[col2])
        st.write(f"**Pearson Correlation**")
        st.write(f"Correlation coefficient: {pearson_corr:.4f}")
        st.write(f"p-value: {pearson_p:.4f}")
        st.write(f"Result: {'Significant correlation' if pearson_p < 0.05 else 'No significant correlation'} (α = 0.05)")
        
        # Spearman correlation
        spearman_corr, spearman_p = spearmanr(data_clean[col1], data_clean[col2])
        st.write(f"**Spearman Correlation**")
        st.write(f"Correlation coefficient: {spearman_corr:.4f}")
        st.write(f"p-value: {spearman_p:.4f}")
        st.write(f"Result: {'Significant correlation' if spearman_p < 0.05 else 'No significant correlation'} (α = 0.05)")
        
        # Scatter plot with trend line
        fig = px.scatter(data_clean, x=col1, y=col2, 
                        title=f"Correlation between {col1} and {col2}",
                        trendline="ols",
                        color_discrete_sequence=[AMAZON_COLORS['primary']])
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color=AMAZON_COLORS['secondary'])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Chi-Square Test":
        st.markdown("#### ✖️ Chi-Square Test of Independence")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if len(categorical_cols) < 2:
            st.warning("Need at least 2 categorical columns for chi-square testing")
            return
        
        col1 = st.selectbox("Select first categorical variable", categorical_cols)
        col2 = st.selectbox("Select second categorical variable", [c for c in categorical_cols if c != col1])
        
        # Create contingency table
        contingency_table = pd.crosstab(df[col1], df[col2])
        
        if contingency_table.size == 0 or contingency_table.min().min() < 5:
            show_warning("Contingency table has cells with count < 5. Chi-square test may not be reliable.")
        
        # Perform chi-square test
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        st.write(f"**Chi-Square Test Results**")
        st.write(f"Chi-square statistic: {chi2_stat:.4f}")
        st.write(f"p-value: {p_value:.4f}")
        st.write(f"Degrees of freedom: {dof}")
        st.write(f"Result: {'Variables are dependent' if p_value < 0.05 else 'Variables are independent'} (α = 0.05)")
        
        # Show contingency table
        st.markdown("##### Contingency Table")
        st.dataframe(contingency_table, use_container_width=True)
        
        # Heatmap of contingency table
        fig = px.imshow(contingency_table.values,
                       x=contingency_table.columns,
                       y=contingency_table.index,
                       text_auto=True,
                       title=f"Contingency Table: {col1} vs {col2}",
                       color_continuous_scale="Blues")
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color=AMAZON_COLORS['secondary'])
        )
        st.plotly_chart(fig, use_container_width=True)

def model_application_tab(df):
    """Model Application tab implementation"""
    st.markdown("""
    <div class="amazon-header">
        <h2>🤖 Model Application</h2>
        <p>Apply machine learning models to your Amazon dataset</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model type selection
    model_type = st.selectbox(
        "Select Model Type",
        ["Classification", "Regression"]
    )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    all_cols = numeric_cols + categorical_cols
    
    if len(all_cols) < 2:
        show_warning("Insufficient columns for modeling. Need at least 2 columns (1 target + 1 feature).")
        return
    
    # Target variable selection
    if model_type == "Classification":
        suitable_targets = []
        for col in all_cols:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 20:  # Reasonable for classification
                suitable_targets.append(col)
        
        if not suitable_targets:
            show_warning("No suitable target variables found for classification. Target should have 2-20 unique values.")
            return
        
        target_col = st.selectbox("Select target variable for classification", suitable_targets)
    else:
        if not numeric_cols:
            show_warning("No numeric columns available for regression modeling.")
            return
        target_col = st.selectbox("Select target variable for regression", numeric_cols)
    
    # Feature selection
    available_features = [col for col in all_cols if col != target_col]
    if not available_features:
        show_warning("No features available after selecting target variable.")
        return
    
    feature_cols = st.multiselect(
        "Select feature variables",
        available_features,
        default=available_features[:5] if len(available_features) >= 5 else available_features
    )
    
    if not feature_cols:
        show_warning("Please select at least one feature variable.")
        return
    
    # Preprocess data
    model_df, le_dict = preprocess_for_modeling(df, target_col, feature_cols)
    
    if model_df is None or len(model_df) < 10:
        show_warning("Insufficient data for modeling after preprocessing.")
        return
    
    # Check target variable for classification
    if model_type == "Classification":
        unique_targets = model_df[target_col].nunique()
        if unique_targets < 2:
            show_warning("Target variable must have at least 2 classes for classification.")
            return
        elif unique_targets > 20:
            show_warning("Target variable has too many classes. Consider grouping or using regression.")
            return
    
    # Train-test split
    test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
    
    X = model_df[feature_cols]
    y = model_df[target_col]
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if model_type == "Classification" else None
        )
    except ValueError as e:
        show_warning(f"Error in train-test split: {str(e)}. Try adjusting the target variable or increasing data size.")
        return
    
    # Model training
    if st.button("Train Models", type="primary"):
        with st.spinner("Training models..."):
            models = {}
            results = {}
            
            if model_type == "Classification":
                # Classification models
                models = {
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                    "SVM": SVC(random_state=42, probability=True)
                }
                
                # Train and evaluate models
                for name, model in models.items():
                    try:
                        # Scale features for SVM and Logistic Regression
                        if name in ["SVM", "Logistic Regression"]:
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        
                        results[name] = {
                            'model': model,
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'predictions': y_pred
                        }
                        
                        if name in ["SVM", "Logistic Regression"]:
                            results[name]['scaler'] = scaler
                        
                    except Exception as e:
                        show_warning(f"Error training {name}: {str(e)}")
                        continue
            
            else:  # Regression
                models = {
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                    "Linear Regression": LinearRegression(),
                    "SVR": SVR()
                }
                
                # Train and evaluate models
                for name, model in models.items():
                    try:
                        # Scale features for SVR and Linear Regression
                        if name in ["SVR", "Linear Regression"]:
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)
                        
                        results[name] = {
                            'model': model,
                            'mse': mse,
                            'rmse': rmse,
                            'r2_score': r2,
                            'predictions': y_pred
                        }
                        
                        if name in ["SVR", "Linear Regression"]:
                            results[name]['scaler'] = scaler
                        
                    except Exception as e:
                        show_warning(f"Error training {name}: {str(e)}")
                        continue
            
            # Display results
            if results:
                st.markdown("### 📊 Model Performance")
                
                if model_type == "Classification":
                    # Classification results
                    results_df = pd.DataFrame({
                        'Model': list(results.keys()),
                        'Accuracy': [results[name]['accuracy'] for name in results.keys()],
                        'Precision': [results[name]['precision'] for name in results.keys()],
                        'Recall': [results[name]['recall'] for name in results.keys()],
                        'F1-Score': [results[name]['f1_score'] for name in results.keys()]
                    })
                    
                    # Find best model
                    best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
                    
                else:
                    # Regression results
                    results_df = pd.DataFrame({
                        'Model': list(results.keys()),
                        'RMSE': [results[name]['rmse'] for name in results.keys()],
                        'R² Score': [results[name]['r2_score'] for name in results.keys()],
                        'MSE': [results[name]['mse'] for name in results.keys()]
                    })
                    
                    # Find best model (highest R²)
                    best_model_name = results_df.loc[results_df['R² Score'].idxmax(), 'Model']
                
                st.dataframe(results_df, use_container_width=True)
                
                show_success(f"Best performing model: **{best_model_name}**")
                
                # Store results in session state for other tabs
                st.session_state.model_results = results
                st.session_state.best_model_name = best_model_name
                st.session_state.model_type = model_type
                st.session_state.feature_cols = feature_cols
                st.session_state.target_col = target_col
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Performance comparison
                    if model_type == "Classification":
                        fig = px.bar(results_df, x='Model', y='Accuracy',
                                   title="Model Accuracy Comparison",
                                   color_discrete_sequence=[AMAZON_COLORS['primary']])
                    else:
                        fig = px.bar(results_df, x='Model', y='R² Score',
                                   title="Model R² Score Comparison",
                                   color_discrete_sequence=[AMAZON_COLORS['primary']])
                    
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color=AMAZON_COLORS['secondary'])
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Actual vs Predicted for best model
                    best_predictions = results[best_model_name]['predictions']
                    
                    if model_type == "Classification":
                        # Confusion matrix
                        cm = confusion_matrix(y_test, best_predictions)
                        fig = px.imshow(cm, text_auto=True,
                                      title=f"Confusion Matrix - {best_model_name}",
                                      color_continuous_scale="Blues")
                    else:
                        # Scatter plot for regression
                        fig = px.scatter(x=y_test, y=best_predictions,
                                       title=f"Actual vs Predicted - {best_model_name}",
                                       labels={'x': 'Actual', 'y': 'Predicted'},
                                       color_discrete_sequence=[AMAZON_COLORS['accent']])
                        # Add diagonal line
                        min_val = min(y_test.min(), best_predictions.min())
                        max_val = max(y_test.max(), best_predictions.max())
                        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                               mode='lines', name='Perfect Prediction',
                                               line=dict(dash='dash', color='red')))
                    
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color=AMAZON_COLORS['secondary'])
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                show_warning("No models were successfully trained. Please check your data and try again.")

def model_interpretation_tab(df):
    """Model Interpretation tab implementation"""
    st.markdown("""
    <div class="amazon-header">
        <h2>🔍 Model Interpretation</h2>
        <p>Understand how your models make predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if models have been trained
    if 'model_results' not in st.session_state:
        show_warning("Please train models first in the Model Application tab.")
        return
    
    results = st.session_state.model_results
    best_model_name = st.session_state.best_model_name
    model_type = st.session_state.model_type
    feature_cols = st.session_state.feature_cols
    
    # Model selection for interpretation
    selected_model = st.selectbox("Select model for interpretation", list(results.keys()))
    
    model_obj = results[selected_model]['model']
    
    # Feature Importance
    st.markdown("### 📊 Feature Importance")
    
    try:
        if hasattr(model_obj, 'feature_importances_'):
            # Tree-based models
            importances = model_obj.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title=f"Feature Importance - {selected_model}",
                        color_discrete_sequence=[AMAZON_COLORS['primary']])
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color=AMAZON_COLORS['secondary'])
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top features
            st.markdown("#### Top 5 Most Important Features")
            for i, (_, row) in enumerate(importance_df.head().iterrows()):
                st.write(f"{i+1}. **{row['Feature']}**: {row['Importance']:.4f}")
        
        elif hasattr(model_obj, 'coef_'):
            # Linear models
            if model_type == "Classification" and len(model_obj.coef_.shape) > 1:
                # Multi-class classification
                coef_mean = np.abs(model_obj.coef_).mean(axis=0)
            else:
                coef_mean = np.abs(model_obj.coef_.flatten())
            
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': coef_mean
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title=f"Coefficient Magnitude - {selected_model}",
                        color_discrete_sequence=[AMAZON_COLORS['accent']])
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color=AMAZON_COLORS['secondary'])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Feature importance not available for this model type.")
    
    except Exception as e:
        show_warning(f"Error calculating feature importance: {str(e)}")
    
    # SHAP Analysis (for supported models)
    st.markdown("### 🎯 SHAP Analysis")
    
    try:
        # Prepare data for SHAP
        X_sample = st.session_state.X_test.head(100)  # Use smaller sample for performance
        
        if selected_model == "Random Forest":
            # Tree explainer for Random Forest
            explainer = shap.TreeExplainer(model_obj)
            shap_values = explainer.shap_values(X_sample)
            
            if model_type == "Classification" and isinstance(shap_values, list):
                # Multi-class classification - use first class
                shap_values = shap_values[0]
            
            # SHAP summary plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, 
                            plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance - {selected_model}")
            st.pyplot(fig)
            
            # SHAP waterfall plot for first prediction
            if len(X_sample) > 0:
                st.markdown("#### SHAP Explanation for Sample Prediction")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if hasattr(explainer, 'expected_value'):
                    if isinstance(explainer.expected_value, np.ndarray):
                        expected_value = explainer.expected_value[0]
                    else:
                        expected_value = explainer.expected_value
                else:
                    expected_value = 0
                
                # Create explanation object manually
                shap_exp = shap.Explanation(
                    values=shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                    base_values=expected_value,
                    data=X_sample.iloc[0].values,
                    feature_names=feature_cols
                )
                
                shap.waterfall_plot(shap_exp, show=False)
                st.pyplot(fig)
        
        else:
            st.info("SHAP analysis is currently available only for Random Forest models.")
    
    except Exception as e:
        st.info("SHAP analysis not available for this model configuration.")
    
    # Model Performance Details
    st.markdown("### 📈 Model Performance Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Performance Metrics")
        if model_type == "Classification":
            metrics = {
                'Accuracy': results[selected_model]['accuracy'],
                'Precision': results[selected_model]['precision'],
                'Recall': results[selected_model]['recall'],
                'F1-Score': results[selected_model]['f1_score']
            }
        else:
            metrics = {
                'RMSE': results[selected_model]['rmse'],
                'R² Score': results[selected_model]['r2_score'],
                'MSE': results[selected_model]['mse']
            }
        
        for metric, value in metrics.items():
            create_metric_card(metric, f"{value:.4f}")
    
    with col2:
        st.markdown("#### Model Recommendations")
        
        if model_type == "Classification":
            accuracy = results[selected_model]['accuracy']
            if accuracy > 0.9:
                st.success("🎉 Excellent model performance!")
            elif accuracy > 0.8:
                st.info("👍 Good model performance.")
            elif accuracy > 0.7:
                st.warning("⚠️ Moderate performance. Consider feature engineering.")
            else:
                st.error("❌ Poor performance. Review data quality and features.")
        else:
            r2 = results[selected_model]['r2_score']
            if r2 > 0.9:
                st.success("🎉 Excellent model fit!")
            elif r2 > 0.7:
                st.info("👍 Good model fit.")
            elif r2 > 0.5:
                st.warning("⚠️ Moderate fit. Consider feature engineering.")
            else:
                st.error("❌ Poor fit. Review data quality and features.")
        
        # General recommendations
        st.markdown("##### General Recommendations:")
        st.write("• Consider feature scaling for linear models")
        st.write("• Try feature selection techniques")
        st.write("• Experiment with hyperparameter tuning")
        st.write("• Collect more data if possible")

def export_tab(df):
    """Export tab implementation"""
    st.markdown("""
    <div class="amazon-header">
        <h2>📁 Export</h2>
        <p>Download your analysis results and models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Export options
    export_type = st.selectbox(
        "Select Export Type",
        ["Cleaned Dataset", "Analysis Code", "Trained Models", "Full Report"]
    )
    
    if export_type == "Cleaned Dataset":
        st.markdown("### 📊 Export Cleaned Dataset")
        
        # Data cleaning options
        st.markdown("#### Cleaning Options")
        
        remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
        remove_missing = st.checkbox("Remove rows with missing target values", value=True)
        standardize_prices = st.checkbox("Standardize price formats", value=True)
        
        # Apply cleaning
        cleaned_df = df.copy()
        
        if remove_duplicates:
            before_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            after_count = len(cleaned_df)
            st.write(f"Removed {before_count - after_count} duplicate rows")
        
        if standardize_prices:
            # Keep numeric price columns
            price_cols = [col for col in cleaned_df.columns if 'price' in col.lower() and '_numeric' in col]
            if price_cols:
                st.write(f"Standardized {len(price_cols)} price columns")
        
        # Show cleaned data info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_metric_card("Original Rows", len(df))
        
        with col2:
            create_metric_card("Cleaned Rows", len(cleaned_df))
        
        with col3:
            create_metric_card("Columns", len(cleaned_df.columns))
        
        # Export options
        st.markdown("#### Export Format")
        export_format = st.selectbox("Select format", ["CSV", "Excel", "JSON"])
        
        if st.button("Generate Download Link"):
            if export_format == "CSV":
                csv_data = cleaned_df.to_csv(index=False)
                b64 = base64.b64encode(csv_data.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="amazon_cleaned_data.csv" class="download-button">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            elif export_format == "Excel":
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    cleaned_df.to_excel(writer, sheet_name='Cleaned_Data', index=False)
                excel_data = output.getvalue()
                b64 = base64.b64encode(excel_data).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="amazon_cleaned_data.xlsx" class="download-button">Download Excel</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            elif export_format == "JSON":
                json_data = cleaned_df.to_json(orient='records', indent=2)
                b64 = base64.b64encode(json_data.encode()).decode()
                href = f'<a href="data:file/json;base64,{b64}" download="amazon_cleaned_data.json" class="download-button">Download JSON</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    elif export_type == "Analysis Code":
        st.markdown("### 💻 Export Analysis Code")
        
        # Generate analysis code
        analysis_code = f'''
# Amazon Dataset Analysis Code
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
df = pd.read_csv('amazon.csv')

# Data preprocessing
def preprocess_data(df):
    """Clean and preprocess the Amazon dataset"""
    df = df.copy()
    
    # Clean price columns
    if 'discounted_price' in df.columns:
        df['discounted_price_clean'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').str.strip()
        df['discounted_price_numeric'] = pd.to_numeric(df['discounted_price_clean'], errors='coerce')
    
    if 'actual_price' in df.columns:
        df['actual_price_clean'] = df['actual_price'].astype(str).str.replace('₹', '').str.replace(',', '').str.strip()
        df['actual_price_numeric'] = pd.to_numeric(df['actual_price_clean'], errors='coerce')
    
    # Clean rating
    if 'rating' in df.columns:
        df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')
    
    return df

# Preprocess the data
df_processed = preprocess_data(df)

# Basic analysis
print("Dataset shape:", df_processed.shape)
print("\\nDataset info:")
print(df_processed.info())

print("\\nNumerical summary:")
print(df_processed.describe())

# Data visualization examples
def create_visualizations(df):
    """Create basic visualizations"""
    
    # Price distribution
    if 'discounted_price_numeric' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df['discounted_price_numeric'].dropna(), bins=30, alpha=0.7)
        plt.title('Price Distribution')
        plt.xlabel('Price (₹)')
        plt.ylabel('Frequency')
        plt.show()
    
    # Rating distribution
    if 'rating_numeric' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df['rating_numeric'].dropna(), bins=20, alpha=0.7)
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.show()

# Run visualizations
create_visualizations(df_processed)

# Machine learning example
def train_basic_model(df, target_col, feature_cols):
    """Train a basic machine learning model"""
    
    # Prepare data
    model_df = df[feature_cols + [target_col]].dropna()
    
    # Encode categorical variables
    le_dict = {{}}
    for col in feature_cols:
        if model_df[col].dtype == 'object':
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))
            le_dict[col] = le
    
    X = model_df[feature_cols]
    y = model_df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    if y.nunique() <= 20:  # Classification
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Classification Accuracy: {{accuracy:.4f}}")
    else:  # Regression
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"Regression R² Score: {{r2:.4f}}")
    
    return model, le_dict

# Example usage:
# model, encoders = train_basic_model(df_processed, 'target_column', ['feature1', 'feature2'])

print("Analysis complete!")
'''
        
        st.code(analysis_code, language='python')
        
        if st.button("Download Analysis Code"):
            b64 = base64.b64encode(analysis_code.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="amazon_analysis_code.py" class="download-button">Download Python Code</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    elif export_type == "Trained Models":
        st.markdown("### 🤖 Export Trained Models")
        
        if 'model_results' not in st.session_state:
            show_warning("No trained models available. Please train models first in the Model Application tab.")
            return
        
        results = st.session_state.model_results
        best_model_name = st.session_state.best_model_name
        
        st.markdown(f"#### Available Models ({len(results)} trained)")
        
        for model_name in results.keys():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{model_name}**")
                if model_name == best_model_name:
                    st.write("🏆 Best Model")
            
            with col2:
                if st.session_state.model_type == "Classification":
                    accuracy = results[model_name]['accuracy']
                    st.write(f"Accuracy: {accuracy:.3f}")
                else:
                    r2 = results[model_name]['r2_score']
                    st.write(f"R²: {r2:.3f}")
            
            with col3:
                if st.button(f"Save {model_name}", key=f"save_{model_name}"):
                    # Save model using joblib
                    model_obj = results[model_name]['model']
                    model_data = joblib.dump(model_obj, f"{model_name.lower().replace(' ', '_')}_model.pkl")
                    
                    show_success(f"Model {model_name} saved successfully!")
        
        # Export all models
        if st.button("Export All Models", type="primary"):
            models_info = {
                'model_type': st.session_state.model_type,
                'best_model': best_model_name,
                'feature_columns': st.session_state.feature_cols,
                'target_column': st.session_state.target_col,
                'models': {}
            }
            
            for model_name, model_data in results.items():
                models_info['models'][model_name] = {
                    'performance': {k: v for k, v in model_data.items() if k != 'model' and not isinstance(v, np.ndarray)}
                }
            
            # Create JSON with model information
            models_json = json.dumps(models_info, indent=2, default=str)
            b64 = base64.b64encode(models_json.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="models_info.json" class="download-button">Download Models Info</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    elif export_type == "Full Report":
        st.markdown("### 📄 Export Full Analysis Report")
        
        # Generate comprehensive report
        report = f"""
# Amazon Dataset Analysis Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Overview
- **Total Records**: {len(df):,}
- **Total Columns**: {len(df.columns)}
- **Missing Data**: {(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%

## Column Information
{df.dtypes.to_frame('Data Type').to_string()}

## Statistical Summary
{df.describe().to_string()}

## Data Quality Assessment
"""
        
        # Add data quality issues
        quality_issues = []
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            quality_issues.append(f"- Found {duplicates} duplicate rows")
        
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            quality_issues.append(f"- Columns with missing values: {', '.join(missing_cols[:5])}")
        
        if quality_issues:
            report += "\n".join(quality_issues)
        else:
            report += "- No major data quality issues detected"
        
        # Add model results if available
        if 'model_results' in st.session_state:
            report += f"""

## Machine Learning Results
- **Model Type**: {st.session_state.model_type}
- **Best Model**: {st.session_state.best_model_name}
- **Features Used**: {', '.join(st.session_state.feature_cols)}
- **Target Variable**: {st.session_state.target_col}

### Model Performance:
"""
            results = st.session_state.model_results
            for model_name, model_data in results.items():
                report += f"\n**{model_name}**:\n"
                if st.session_state.model_type == "Classification":
                    report += f"- Accuracy: {model_data['accuracy']:.4f}\n"
                    report += f"- Precision: {model_data['precision']:.4f}\n"
                    report += f"- Recall: {model_data['recall']:.4f}\n"
                    report += f"- F1-Score: {model_data['f1_score']:.4f}\n"
                else:
                    report += f"- RMSE: {model_data['rmse']:.4f}\n"
                    report += f"- R² Score: {model_data['r2_score']:.4f}\n"
                    report += f"- MSE: {model_data['mse']:.4f}\n"
        
        report += """

## Recommendations
1. Consider feature engineering to improve model performance
2. Collect more data if possible to enhance model accuracy
3. Implement cross-validation for more robust model evaluation
4. Monitor model performance on new data regularly
5. Consider ensemble methods for better predictions

---
Report generated by Amazon Analytics Dashboard
"""
        
        st.text_area("Report Preview", report, height=400)
        
        if st.button("Download Full Report"):
            b64 = base64.b64encode(report.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="amazon_analysis_report.txt" class="download-button">Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Amazon Analytics Dashboard",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Header
    st.markdown("""
    <div class="amazon-header">
        <h1>🛒 Amazon Analytics Dashboard</h1>
        <p>Comprehensive analysis and machine learning platform for Amazon data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading Amazon dataset..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load dataset. Please check that 'amazon.csv' exists in the current directory.")
        return
    
    # Apply filters
    filtered_df = apply_filters(df)
    
    # Show filter results
    if len(filtered_df) != len(df):
        st.sidebar.success(f"Showing {len(filtered_df):,} of {len(df):,} records")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Data Overview", 
        "📈 Data Visualization", 
        "🔍 EDA", 
        "📉 Statistical Analysis",
        "🤖 Model Application",
        "🔍 Model Interpretation",
        "📁 Export"
    ])
    
    with tab1:
        data_overview_tab(filtered_df)
    
    with tab2:
        data_visualization_tab(filtered_df)
    
    with tab3:
        eda_tab(filtered_df)
    
    with tab4:
        statistical_analysis_tab(filtered_df)
    
    with tab5:
        model_application_tab(filtered_df)
    
    with tab6:
        model_interpretation_tab(filtered_df)
    
    with tab7:
        export_tab(filtered_df)

if __name__ == "__main__":
    main()