🧠 AI Job Market Dashboard
📌 Overview

The AI Job Market Dashboard is a data-driven project that explores how Artificial Intelligence (AI) is reshaping global employment trends. It analyzes job growth, automation risks, and skill demand across multiple industries such as Technology, Finance, Healthcare, Education, and Manufacturing.

This project provides insights into how AI is creating, transforming, and disrupting jobs worldwide — using data visualization and Natural Language Processing (NLP) to reveal patterns and trends from real-world datasets.

🎯 Objectives

Analyze the impact of AI on global and industry-specific job trends

Identify skills in demand in the AI era

Visualize job creation vs automation displacement rates

Perform text analysis on job listings using NLP to uncover keyword trends

Build an interactive dashboard for exploration and storytelling

🧰 Tech Stack
Category	Tools / Libraries
Programming	Python
Data Handling	Pandas, NumPy
Visualization	Matplotlib, Seaborn, Plotly
NLP	NLTK, spaCy, WordCloud
Dashboard	Streamlit
Data Sources	Kaggle, LinkedIn Jobs API, OECD, World Economic Forum Reports
📂 Project Structure
AI-Job-Market-Dashboard/
│
├── data/
│   ├── raw/                     # Original datasets
│   ├── processed/               # Cleaned and transformed data
│
├── notebooks/
│   ├── data_cleaning.ipynb      # Data preprocessing and cleaning
│   ├── eda_visuals.ipynb        # Exploratory data analysis and visualizations
│   ├── nlp_analysis.ipynb       # NLP skill and job description analysis
│
├── dashboard/
│   ├── app.py                   # Streamlit dashboard script
│
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
└── LICENSE                      # License file (optional)

📊 Dashboard Features

✅ Interactive job trend visualizations by sector & region
✅ Keyword and skill frequency WordClouds
✅ AI automation risk percentage by industry
✅ Salary and demand correlation plots
✅ Search filters for job role, region, and skill

📦 Installation & Setup

1️⃣ Clone this repo:

git clone https://github.com/<your-username>/AI-Job-Market-Dashboard.git
cd AI-Job-Market-Dashboard


2️⃣ Install dependencies:

py -m pip install -r requirements.txt


3️⃣ Run the dashboard locally:

streamlit run dashboard/app.py


4️⃣ Optional (for NLP setup):

python -m spacy download en_core_web_sm

🧠 Data Sources (Examples)

You can use or merge data from:

Kaggle – Global AI Job Trends Dataset

LinkedIn Jobs API or scraped listings

OECD Employment Outlook

World Economic Forum – Future of Jobs Report

🧩 Future Enhancements

Add real-time job listing scraping from APIs

Integrate machine learning models to predict future AI job trends

Include AI skill gap analysis by region

Add sentiment analysis on AI-related job posts

📸 Dashboard Preview

(Add your screenshots here once your visuals are done)
Example:

![Dashboard Overview](images/dashboard_preview.png)

👤 Author

Icey Manjengwa
🎓 Data Analyst | AI Enthusiast | BCA Graduate with Distinction
📍 Passionate about data storytelling and emerging technologies
📧 [YourEmail@example.com
]
🌐 LinkedIn
 | GitHub

⭐ Acknowledgments

Special thanks to open data communities and AI researchers who make global workforce insights possible
