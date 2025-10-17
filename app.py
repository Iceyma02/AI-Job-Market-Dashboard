import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="AI Job Market Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🤖 AI Job Market Analysis Dashboard")
st.markdown("Comprehensive analysis of AI job market trends, skills demand, and salary insights")

# Load data function with error handling
@st.cache_data
def load_data():
    try:
        # Try to load your datasets
        df1 = pd.read_csv('data/raw/ai_job_dataset.csv')
        return df1
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return sample data structure if files not found
        return pd.DataFrame({
            'job_title': ['Sample AI Engineer', 'Data Scientist', 'ML Engineer'],
            'salary_usd': [100000, 120000, 110000],
            'experience_level': ['Mid', 'Senior', 'Mid'],
            'industry': ['Tech', 'Tech', 'Finance'],
            'remote_ratio': [100, 50, 0]
        })

# Load data
df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Experience level filter
experience_levels = ['All'] + sorted(df['experience_level'].unique().tolist())
selected_experience = st.sidebar.selectbox(
    "Experience Level",
    experience_levels
)

# Industry filter
industries = ['All'] + sorted(df['industry'].unique().tolist())
selected_industry = st.sidebar.selectbox(
    "Industry",
    industries
)

# Apply filters
filtered_df = df.copy()
if selected_experience != 'All':
    filtered_df = filtered_df[filtered_df['experience_level'] == selected_experience]
if selected_industry != 'All':
    filtered_df = filtered_df[filtered_df['industry'] == selected_industry]

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Jobs Analyzed",
        f"{len(filtered_df):,}",
        f"{len(filtered_df) - len(df)}" if len(filtered_df) != len(df) else ""
    )

with col2:
    avg_salary = filtered_df['salary_usd'].mean()
    st.metric(
        "Average Salary",
        f"${avg_salary:,.0f}",
        f"${avg_salary - df['salary_usd'].mean():,.0f}" if len(filtered_df) != len(df) else ""
    )

with col3:
    remote_jobs = len(filtered_df[filtered_df['remote_ratio'] == 100])
    st.metric(
        "Remote Jobs",
        f"{remote_jobs}",
        f"{(remote_jobs/len(filtered_df)*100):.1f}%"
    )

with col4:
    hybrid_jobs = len(filtered_df[(filtered_df['remote_ratio'] > 0) & (filtered_df['remote_ratio'] < 100)])
    st.metric(
        "Hybrid Jobs",
        f"{hybrid_jobs}",
        f"{(hybrid_jobs/len(filtered_df)*100):.1f}%"
    )

# Charts and visualizations
tab1, tab2, tab3, tab4 = st.tabs(["📊 Salary Analysis", "🔧 Skills", "🌍 Locations", "📈 Trends"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Salary Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(filtered_df['salary_usd'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Salary (USD)')
        ax.set_ylabel('Number of Jobs')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Salary by Experience Level")
        salary_by_exp = filtered_df.groupby('experience_level')['salary_usd'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        salary_by_exp.plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_ylabel('Average Salary (USD)')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

with tab2:
    st.subheader("Top Industries Hiring AI Talent")
    top_industries = filtered_df['industry'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    top_industries.plot(kind='bar', ax=ax, color='lightgreen')
    ax.set_ylabel('Number of Jobs')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

with tab3:
    st.subheader("Remote Work Distribution")
    remote_counts = [
        len(filtered_df[filtered_df['remote_ratio'] == 0]),
        len(filtered_df[(filtered_df['remote_ratio'] > 0) & (filtered_df['remote_ratio'] < 100)]),
        len(filtered_df[filtered_df['remote_ratio'] == 100])
    ]
    labels = ['On-site', 'Hybrid', 'Fully Remote']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(remote_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    ax.axis('equal')
    st.pyplot(fig)

with tab4:
    st.subheader("Company Size Distribution")
    if 'company_size' in filtered_df.columns:
        company_sizes = filtered_df['company_size'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(company_sizes.values, labels=company_sizes.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.info("Company size data not available in the filtered dataset")

# Data table
st.subheader("📋 Job Data Preview")
st.dataframe(
    filtered_df.head(100)[['job_title', 'salary_usd', 'experience_level', 'industry', 'remote_ratio']],
    use_container_width=True
)

# Footer
st.markdown("---")
st.markdown("**AI Job Market Analysis** • Built with Streamlit")
