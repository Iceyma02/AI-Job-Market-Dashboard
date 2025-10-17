# app.py - Enhanced AI Job Market Dashboard with GitHub Data Loading
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import re
from collections import Counter
import requests
import io

# Page configuration
st.set_page_config(
    page_title="AI Job Market Dashboard", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Theme Management ----------
def get_theme_css(is_dark):
    if is_dark:
        return """
        <style>
            .main { background-color: #0E1117; color: #FAFAFA; }
            .main-header { font-size: 2.6rem; color: #66B2FF; text-align: center; margin-bottom: 1.2rem; font-weight:700; }
            .section-header { font-size: 1.4rem; color: #66B2FF; border-bottom: 3px solid #66B2FF; padding-bottom: 0.35rem; margin-top: 1.2rem; margin-bottom: 0.8rem; }
            .metric-card { background: linear-gradient(135deg, #1E3A8A 0%, #3730A3 100%); padding: 1rem; border-radius: 12px; color: white; text-align:center; margin:0.4rem; box-shadow:0 4px 10px rgba(0,0,0,0.3); }
            .insight-box { background: #1F2937; padding: 0.9rem 1rem; border-radius: 10px; border-left: 6px solid #66B2FF; margin:0.6rem 0; box-shadow:0 2px 6px rgba(0,0,0,0.2); color: #E5E7EB; }
            .insight-box h4 { color:#66B2FF; margin:0 0 0.2rem 0; font-size:1rem; }
            .sidebar-filter { background-color:#374151; padding:0.8rem; border-radius:8px; margin:0.4rem 0; color: #E5E7EB; }
            .small-muted { color:#9CA3AF; font-size:0.9rem; }
            .stRadio > div { background-color: #374151; padding: 10px; border-radius: 8px; }
            .stButton button { background-color: #2563EB; color: white; border: none; border-radius: 6px; padding: 0.5rem 1rem; }
            .stButton button:hover { background-color: #1D4ED8; }
        </style>
        """
    else:
        return """
        <style>
            .main-header { font-size: 2.6rem; color: #1f77b4; text-align: center; margin-bottom: 1.2rem; font-weight:700; }
            .section-header { font-size: 1.4rem; color: #2E86AB; border-bottom: 3px solid #2E86AB; padding-bottom: 0.35rem; margin-top: 1.2rem; margin-bottom: 0.8rem; }
            .metric-card { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); padding: 1rem; border-radius: 12px; color: white; text-align:center; margin:0.4rem; box-shadow:0 4px 10px rgba(0,0,0,0.12); }
            .insight-box { background: #f0f8ff; padding: 0.9rem 1rem; border-radius: 10px; border-left: 6px solid #2E86AB; margin:0.6rem 0; box-shadow:0 2px 6px rgba(0,0,0,0.06);}
            .insight-box h4 { color:#2E86AB; margin:0 0 0.2rem 0; font-size:1rem; }
            .sidebar-filter { background-color:#f8f9fa; padding:0.8rem; border-radius:8px; margin:0.4rem 0; }
            .small-muted { color:#666; font-size:0.9rem; }
        </style>
        """

# Initialize session state for theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ---------- Data Loading from GitHub ----------
@st.cache_data
def load_data_from_github():
    """Load CSV files directly from GitHub raw URLs"""
    
    # Replace with your actual GitHub raw file URLs
    github_files = {
        "ai_job_dataset": "https://raw.githubusercontent.com/Iceyma02/AI-Job-Market-Dashboard/main/data/raw/ai_job_dataset.csv",
        "linkedin_jobs_analysis": "https://raw.githubusercontent.com/Iceyma02/AI-Job-Market-Dashboard/main/data/raw/linkedin_jobs_analysis.csv", 
        "ai_job_market": "https://raw.githubusercontent.com/Iceyma02/AI-Job-Market-Dashboard/main/data/raw/ai_job_market.csv"
    }
    
    dfs = {}
    
    for name, url in github_files.items():
        try:
            # Try to load from GitHub
            response = requests.get(url)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
                dfs[name] = df
                st.sidebar.success(f"✅ Loaded {name} from GitHub")
            else:
                st.sidebar.warning(f"⚠️ Could not load {name} from GitHub")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {name}: {e}")
    
    # If GitHub loading fails, try local files as fallback
    if not dfs:
        st.sidebar.info("🔄 Trying local files...")
        local_files = {
            "ai_job_dataset": "data/raw/ai_job_dataset.csv",
            "linkedin_jobs_analysis": "data/raw/linkedin_jobs_analysis.csv",
            "ai_job_market": "data/raw/ai_job_market.csv"
        }
        
        for name, path in local_files.items():
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    dfs[name] = df
                    st.sidebar.success(f"✅ Loaded {name} locally")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {name} locally: {e}")
    
    return dfs

@st.cache_data
def load_sample_data():
    """Create sample data if no files are found"""
    st.sidebar.warning("📊 Using sample data - replace with your CSV files")
    return {
        "ai_job_dataset": pd.DataFrame({
            'job_title': ['AI Engineer', 'Data Scientist', 'ML Researcher', 'AI Product Manager'],
            'salary_usd': [120000, 110000, 130000, 125000],
            'experience_level': ['Senior', 'Mid', 'Senior', 'Mid'],
            'company_location': ['USA', 'Canada', 'UK', 'Germany'],
            'required_skills': ['Python, TensorFlow', 'Python, SQL, ML', 'PyTorch, Research', 'Product, AI'],
            'industry': ['Tech', 'Tech', 'Research', 'Tech'],
            'remote_ratio': [100, 50, 0, 100]
        })
    }

# ---------- Helper Functions ----------
def clean_salary(df):
    s = pd.Series([np.nan] * len(df), index=df.index, dtype=float)

    if 'salary_usd' in df.columns:
        try:
            s = pd.to_numeric(df['salary_usd'], errors='coerce')
        except:
            pass

    if s.isna().all() and 'salary_range_usd' in df.columns:
        def parse_range(val):
            if pd.isna(val): return np.nan
            val = str(val)
            val = val.replace('USD','').replace('$','').replace(',', '').lower()
            val = re.sub(r'(\d+(?:\.\d+)?)k', lambda m: str(float(m.group(1)) * 1000), val)
            nums = re.findall(r'(\d+(?:\.\d+)?)', val)
            nums = [float(x) for x in nums]
            if len(nums) == 0: return np.nan
            return float(np.mean(nums))
        s = df['salary_range_usd'].apply(parse_range)

    if s.isna().all() and 'salary' in df.columns:
        def parse_any_salary(val):
            if pd.isna(val): return np.nan
            val = str(val)
            val = val.replace(',', '').replace('$','').lower()
            val = re.sub(r'(\d+(?:\.\d+)?)k', lambda m: str(float(m.group(1)) * 1000), val)
            nums = re.findall(r'(\d+(?:\.\d+)?)', val)
            if not nums: return np.nan
            nums = [float(x) for x in nums]
            return float(np.mean(nums))
        s = df['salary'].apply(parse_any_salary)

    return s

def unify_location(df):
    for col in ['company_location', 'location', 'employee_residence', 'company', 'company_name']:
        if col in df.columns:
            locs = df[col].astype(str).fillna("Unknown")
            return locs
    return pd.Series(["Unknown"] * len(df), index=df.index)

def gather_skills_series(df):
    possible = ['required_skills', 'skills_required', 'skills', 'tools_preferred']
    skills_col = None
    for c in possible:
        if c in df.columns:
            skills_col = df[c]
            break
    if skills_col is None:
        return pd.Series([np.nan]*len(df), index=df.index)
    
    cleaned = skills_col.fillna('').astype(str).apply(lambda x: re.split(r'[\,\;\|\n\/]', x))
    cleaned = cleaned.apply(lambda lst: [s.strip().lower() for s in lst if s and s.strip()])
    return cleaned

def make_skill_counter(sk_series):
    all_sk = []
    for lst in sk_series:
        if isinstance(lst, (list, tuple)):
            all_sk.extend(lst)
    return Counter(all_sk)

def get_plotly_theme(is_dark):
    if is_dark:
        return {
            'template': 'plotly_dark',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': 'white'}
        }
    else:
        return {
            'template': 'plotly_white'
        }

# ---------- Data Loading Section ----------
st.sidebar.title("📊 Data Source")
st.sidebar.markdown("Loading data from GitHub repository...")

# Load data
dfs = load_data_from_github()

# If no data loaded, use sample data
if not dfs:
    dfs = load_sample_data()

# Show loaded datasets
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Loaded Datasets")
for name, df in dfs.items():
    st.sidebar.markdown(f"**{name}**: {df.shape[0]:,} rows, {df.shape[1]} cols")

# ---------- Theme Toggle ----------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Theme Settings")
if st.sidebar.button("🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode"):
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.rerun()

# Apply theme CSS
st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ---------- Dataset Selection ----------
dataset_names = list(dfs.keys())
chosen = st.sidebar.selectbox("Choose dataset to analyze", dataset_names, index=0)
df = dfs[chosen].copy()

# Build normalized columns
df['salary_clean'] = clean_salary(df)
df['location_unified'] = unify_location(df)
skills_series = gather_skills_series(df)
df['skills_list'] = skills_series.apply(lambda x: x if isinstance(x, list) else [])

# Add AI role detection
def guess_ai_role(row):
    title = str(row.get('job_title') or row.get('title') or "").lower()
    skills = " ".join(row.get('skills_list', [])).lower()
    keywords = ['ai', 'machine learning', 'ml', 'deep learning', 'nlp', 'computer vision', 'data scientist', 'data science']
    text = title + " " + skills
    return any(k in text for k in keywords)

if 'is_ai_role' not in df.columns:
    df['is_ai_role'] = df.apply(guess_ai_role, axis=1)

# ---------- Enhanced Sidebar Filters ----------
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Data Filters")

# Location filter
locs = df['location_unified'].fillna("Unknown").unique()
selected_locs = st.sidebar.multiselect(
    "📍 Location(s)", 
    options=sorted(locs), 
    default=list(locs)[:min(6, len(locs))]
)

# AI role filter
role_choice = st.sidebar.radio("🤖 Role filter", ["All Roles", "AI Roles Only", "Non-AI Roles Only"])

# Salary slider
salary_available = df['salary_clean'].dropna()
if not salary_available.empty:
    smin = int(max(0, np.nanpercentile(salary_available, 1)))
    smax = int(np.nanpercentile(salary_available, 99))
    salary_sel = st.sidebar.slider(
        "💰 Salary (USD) range", 
        min_value=smin, 
        max_value=smax, 
        value=(smin, smax),
        help="Adjust to filter jobs by salary range"
    )
else:
    salary_sel = None

# Experience level filter (if available)
if 'experience_level' in df.columns:
    exp_levels = df['experience_level'].dropna().unique()
    selected_exp = st.sidebar.multiselect(
        "🎯 Experience Level",
        options=sorted(exp_levels),
        default=list(exp_levels)
    )
else:
    selected_exp = None

# Apply filters
filtered = df.copy()
if selected_locs:
    filtered = filtered[filtered['location_unified'].isin(selected_locs)]
if role_choice == "AI Roles Only":
    filtered = filtered[filtered['is_ai_role'] == True]
elif role_choice == "Non-AI Roles Only":
    filtered = filtered[filtered['is_ai_role'] == False]
if salary_sel and 'salary_clean' in filtered.columns:
    filtered = filtered[
        (filtered['salary_clean'].notna()) & 
        (filtered['salary_clean'] >= salary_sel[0]) & 
        (filtered['salary_clean'] <= salary_sel[1])
    ]
if selected_exp and 'experience_level' in filtered.columns:
    filtered = filtered[filtered['experience_level'].isin(selected_exp)]

# Show active filters summary
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Active Filters:**")
st.sidebar.markdown(f"- 📍 Locations: {len(selected_locs)} selected")
st.sidebar.markdown(f"- 🤖 Roles: {role_choice}")
if salary_sel:
    st.sidebar.markdown(f"- 💰 Salary: ${salary_sel[0]:,} - ${salary_sel[1]:,}")

# ---------- Main Dashboard Layout ----------
st.markdown('<div class="main-header">🧠 AI Job Market Intelligence Dashboard</div>', unsafe_allow_html=True)

sections = [
    "🏠 Dashboard Overview",
    "📊 Market Analysis", 
    "💰 Salary Insights",
    "🛠️ Skills Intelligence", 
    "🌍 Geographic Trends",
    "📈 AI Impact Report"
]

selected_section = st.radio("Choose section", sections, horizontal=True, label_visibility="collapsed")

# Get plotly theme config
plotly_theme = get_plotly_theme(st.session_state.dark_mode)

# ---- Insight Box Helper ----
def show_insight(title, content, details=None):
    details_html = f"<p style='margin:0;font-size:0.9rem;color:#9CA3AF'><em>{details}</em></p>" if details else ""
    st.markdown(f"""
    <div class="insight-box">
        <h4>{title}</h4>
        <p style="margin:0.1rem 0 0.4rem 0;">{content}</p>
        {details_html}
    </div>
    """, unsafe_allow_html=True)

# ---------- DASHBOARD OVERVIEW ----------
if selected_section == "🏠 Dashboard Overview":
    st.markdown('<div class="section-header">📈 Executive Summary</div>', unsafe_allow_html=True)
    
    total_jobs = len(filtered)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"<div class='metric-card'><h3>📊 Total Jobs</h3><h2>{total_jobs:,}</h2></div>", unsafe_allow_html=True)
    
    with col2:
        ai_jobs = int(filtered['is_ai_role'].sum())
        perc = (ai_jobs / total_jobs * 100) if total_jobs > 0 else 0
        st.markdown(f"<div class='metric-card'><h3>🤖 AI Roles</h3><h2>{ai_jobs:,}</h2><p class='small-muted'>{perc:.1f}% of total</p></div>", unsafe_allow_html=True)
    
    with col3:
        if 'salary_clean' in filtered.columns and filtered['salary_clean'].dropna().size > 0:
            avg_salary = int(filtered['salary_clean'].dropna().mean())
            st.markdown(f"<div class='metric-card'><h3>💰 Avg Salary</h3><h2>${avg_salary:,}</h2></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='metric-card'><h3>💰 Avg Salary</h3><h2>N/A</h2></div>", unsafe_allow_html=True)
    
    with col4:
        companies_col = next((col for col in ['company_name', 'company'] if col in filtered.columns), None)
        if companies_col:
            unique_companies = filtered[companies_col].nunique()
            st.markdown(f"<div class='metric-card'><h3>🏢 Companies</h3><h2>{unique_companies:,}</h2></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='metric-card'><h3>🏢 Companies</h3><h2>N/A</h2></div>", unsafe_allow_html=True)

    # Quick Insights
    st.markdown('<div class="section-header">💡 Quick Insights</div>', unsafe_allow_html=True)
    
    try:
        # Insight 1: Total Market
        show_insight("📊 Total Market", f"**{total_jobs:,}** job listings analyzed", "Filtered based on your selections")
        
        # Insight 2: AI Focus
        ai_count = int(filtered['is_ai_role'].sum())
        ai_pct = (ai_count / total_jobs * 100) if total_jobs > 0 else 0
        show_insight("🤖 AI Focus", f"**{ai_pct:.1f}%** are AI-related roles", f"{ai_count:,} positions in AI/ML")
        
        # Insight 3: Salary Overview
        if 'salary_clean' in filtered.columns and filtered['salary_clean'].dropna().size > 0:
            avg_sal = int(filtered['salary_clean'].dropna().mean())
            sal_count = filtered['salary_clean'].notna().sum()
            show_insight("💰 Salary Overview", f"**${avg_sal:,}** average salary", f"{sal_count:,} jobs with salary data")
        else:
            show_insight("💰 Salary Data", "Salary information not available", "Check dataset columns")
        
        # Insight 4: Skills Coverage
        sk_count = filtered['skills_list'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
        if sk_count > 0:
            show_insight("🛠️ Skills Data", f"**{sk_count:,}** skill mentions", "Detailed analysis in Skills tab")
        else:
            show_insight("🛠️ Skills Data", "Limited skills data", "Try different dataset")
            
    except Exception as e:
        st.error(f"Error generating insights: {e}")

    # Data Preview
    st.markdown('<div class="section-header">📋 Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(filtered.head(10), use_container_width=True)

# ---------- MARKET ANALYSIS ----------
elif selected_section == "📊 Market Analysis":
    st.markdown('<div class="section-header">📊 Market Analysis</div>', unsafe_allow_html=True)
    
    # Top Job Titles
    if 'job_title' in filtered.columns or 'title' in filtered.columns:
        jt_col = 'job_title' if 'job_title' in filtered.columns else 'title'
        top_jobs = filtered[jt_col].value_counts().head(12)
        if not top_jobs.empty:
            fig = px.bar(
                x=top_jobs.values, y=top_jobs.index, 
                orientation='h', 
                title="📋 Top Job Titles",
                labels={'x':'Count','y':'Job Title'}
            )
            fig.update_layout(**plotly_theme, height=450)
            st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # AI vs Non-AI Distribution
        if 'is_ai_role' in filtered.columns:
            counts = filtered['is_ai_role'].value_counts()
            labels = ['AI Roles', 'Non-AI Roles']
            values = [int(counts.get(True, 0)), int(counts.get(False, 0))]
            if sum(values) > 0:
                fig = px.pie(
                    values=values, names=labels, 
                    title="🤖 AI vs Non-AI Roles",
                    color=labels,
                    color_discrete_map={'AI Roles': '#00CC96', 'Non-AI Roles': '#EF553B'}
                )
                fig.update_layout(**plotly_theme)
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Industry Distribution
        if 'industry' in filtered.columns:
            top_ind = filtered['industry'].value_counts().head(8)
            if not top_ind.empty:
                fig = px.pie(
                    values=top_ind.values, names=top_ind.index,
                    title="🏭 Top Industries"
                )
                fig.update_layout(**plotly_theme)
                st.plotly_chart(fig, use_container_width=True)

    # Skills/Tools Distribution
    if 'skills_list' in filtered.columns:
        counter = make_skill_counter(filtered['skills_list'])
        if counter:
            top_tools = counter.most_common(15)
            fig = px.bar(
                x=[c for _, c in top_tools], 
                y=[t for t, _ in top_tools], 
                orientation='h',
                title="🛠️ Top Skills/Tools Mentioned"
            )
            fig.update_layout(**plotly_theme, height=400)
            st.plotly_chart(fig, use_container_width=True)

# ---------- SALARY INSIGHTS ----------
elif selected_section == "💰 Salary Insights":
    st.markdown('<div class="section-header">💰 Salary Insights</div>', unsafe_allow_html=True)
    
    if 'salary_clean' not in filtered.columns or filtered['salary_clean'].dropna().empty:
        st.info("No salary data available in the current view.")
    else:
        sal_data = filtered['salary_clean'].dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Salary Distribution
            fig = px.histogram(
                sal_data, x=sal_data, 
                nbins=40,
                title="📊 Salary Distribution",
                labels={'x': 'Salary (USD)', 'y': 'Count'}
            )
            fig.update_layout(**plotly_theme)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Salary by AI Role
            if 'is_ai_role' in filtered.columns:
                box_df = filtered[['salary_clean','is_ai_role']].dropna()
                if not box_df.empty:
                    fig = px.box(
                        box_df, x='is_ai_role', y='salary_clean',
                        title="🤖 Salary: AI vs Non-AI",
                        labels={'is_ai_role': 'AI Role', 'salary_clean': 'Salary (USD)'},
                        color='is_ai_role'
                    )
                    fig.update_layout(**plotly_theme)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Salary Statistics
        st.markdown('<div class="section-header">📈 Salary Statistics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average", f"${int(sal_data.mean()):,}")
        col2.metric("Median", f"${int(sal_data.median()):,}")
        col3.metric("Range", f"${int(sal_data.min()):,}-${int(sal_data.max()):,}")
        col4.metric("Sample Size", f"{len(sal_data):,}")

# ---------- SKILLS INTELLIGENCE ----------
elif selected_section == "🛠️ Skills Intelligence":
    st.markdown('<div class="section-header">🛠️ Skills Intelligence</div>', unsafe_allow_html=True)
    
    if 'skills_list' in filtered.columns:
        skill_counter = make_skill_counter(filtered['skills_list'])
        if skill_counter:
            top_skills = skill_counter.most_common(20)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Skills Bar Chart
                fig = px.bar(
                    x=[v for _, v in top_skills], 
                    y=[k for k, _ in top_skills], 
                    orientation='h',
                    title="📊 Top Skills by Frequency",
                    labels={'x': 'Frequency', 'y': 'Skill'}
                )
                fig.update_layout(**plotly_theme, height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Skills Summary
                st.markdown("### 🎯 Top Skills Summary")
                total_mentions = sum(skill_counter.values())
                for i, (skill, count) in enumerate(top_skills[:10], 1):
                    percentage = (count / total_mentions) * 100
                    st.metric(
                        label=f"{i}. {skill.title()}",
                        value=f"{count:,}",
                        delta=f"{percentage:.1f}%"
                    )

# ---------- GEOGRAPHIC TRENDS ----------
elif selected_section == "🌍 Geographic Trends":
    st.markdown('<div class="section-header">🌍 Geographic Trends</div>', unsafe_allow_html=True)
    
    if 'location_unified' in filtered.columns:
        loc_counts = filtered['location_unified'].value_counts().head(15)
        
        if not loc_counts.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Location Bar Chart
                fig = px.bar(
                    x=loc_counts.values, 
                    y=loc_counts.index,
                    orientation='h',
                    title="📍 Top Locations by Job Count",
                    labels={'x': 'Number of Jobs', 'y': 'Location'}
                )
                fig.update_layout(**plotly_theme, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Location Pie Chart
                top_pie = loc_counts.head(8)
                fig = px.pie(
                    values=top_pie.values, 
                    names=top_pie.index,
                    title="🌎 Location Distribution"
                )
                fig.update_layout(**plotly_theme)
                st.plotly_chart(fig, use_container_width=True)

# ---------- AI IMPACT REPORT ----------
elif selected_section == "📈 AI Impact Report":
    st.markdown('<div class="section-header">📈 AI Impact Report</div>', unsafe_allow_html=True)
    
    total = len(filtered)
    ai_count = int(filtered['is_ai_role'].sum()) if 'is_ai_role' in filtered.columns else 0
    ai_pct = (ai_count / total * 100) if total > 0 else 0
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("AI Market Share", f"{ai_pct:.1f}%")
    
    with col2:
        if 'salary_clean' in filtered.columns:
            ai_salary = filtered[filtered['is_ai_role'] == True]['salary_clean'].mean()
            st.metric("Avg AI Salary", f"${int(ai_salary):,}" if not pd.isna(ai_salary) else "N/A")
    
    with col3:
        if 'salary_clean' in filtered.columns:
            non_ai_salary = filtered[filtered['is_ai_role'] == False]['salary_clean'].mean()
            premium = ((ai_salary - non_ai_salary) / non_ai_salary * 100) if non_ai_salary > 0 else 0
            st.metric("Salary Premium", f"{premium:+.1f}%")
    
    with col4:
        st.metric("AI Job Count", f"{ai_count:,}")

    # Market Insights
    st.markdown('<div class="section-header">🚀 Market Insights</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_insight(
            "🎯 AI Market Growth", 
            f"AI roles constitute **{ai_pct:.1f}%** of the current job market with strong growth potential.",
            "Demand for AI skills continues to outpace supply."
        )
        
        show_insight(
            "💰 Salary Advantage", 
            "AI professionals command significant salary premiums due to specialized skill requirements.",
            "Upskill in AI/ML to capture higher compensation."
        )
    
    with col2:
        show_insight(
            "🌍 Geographic Distribution", 
            "AI jobs are concentrated in tech hubs but remote opportunities are growing rapidly.",
            "Location flexibility expands career options."
        )
        
        show_insight(
            "🛠️ Skills Evolution", 
            "Continuous learning in AI frameworks and cloud technologies is essential for career advancement.",
            "Stay updated with emerging tools and methodologies."
        )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#666;'>"
    "🧠 <strong>AI Job Market Intelligence Dashboard</strong> • Built with Streamlit • "
    f"{'🌙 Dark Mode' if st.session_state.dark_mode else '☀️ Light Mode'}"
    "</div>", 
    unsafe_allow_html=True
)
