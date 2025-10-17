# 🧠 AI Job Market Intelligence Dashboard

![Dashboard Overview](images/Dashboard%20Overview.png)

## 📖 Overview

The **AI Job Market Intelligence Dashboard** is an interactive Streamlit application that provides comprehensive analysis of the AI job market. This dashboard processes and visualizes data from multiple job market datasets, offering insights into AI roles, salary trends, required skills, geographic distributions, and market impact.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-job-market-dashboard-ncflwmukusaxbgyu3faqfs.streamlit.app/))

## 📊 Features

### 🏠 **Dashboard Overview**
- Executive summary with key metrics
- Total jobs, AI roles percentage, average salary, and company count
- Quick insights with actionable intelligence
- Interactive data preview

![Dashboard Overview](images/Dashboard%20Overview.png)

### 📈 **Market Analysis**
- Top job titles distribution
- AI vs Non-AI roles comparison
- Industry hiring patterns
- Skills and tools demand analysis

![Market Analysis](images/Market%20Analysis.png)

### 💰 **Salary Insights**
- Salary distribution analysis
- AI vs Non-AI salary comparison
- Statistical breakdown (mean, median, range)
- Interactive salary filtering

![Salary Insights](images/Salary%20Insights.png)

### 🛠️ **Skills Intelligence**
- Top demanded skills visualization
- Skills frequency analysis
- Skills categorization (Programming, ML Frameworks, Cloud, Data Tools)
- Skills comparison between AI and non-AI roles

![Skills Intelligence](images/Skills%20Intelligence.png)

### 🌍 **Geographic Trends**
- Job distribution by location
- Top hiring locations analysis
- Salary variations by geography
- Regional market insights

![Geographic Trends](images/Geographic%20Trends.png)

### 📊 **AI Impact Report**
- AI market share analysis
- Salary premium calculations
- Market growth insights
- Skills gap identification

![AI Impact Report](images/AI%20Impact%20Report.png)

## 🛠️ Technology Stack

### **Backend & Data Processing**
- **Python 3.8+** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations

### **Visualization & UI**
- **Streamlit** - Interactive web application framework
- **Plotly** - Interactive charts and graphs
- **Plotly Express** - Simplified plotting interface

### **Data Sources & Processing**
- **Multiple CSV datasets** integration
- **Custom data cleaning pipelines**
- **Salary parsing algorithms** (handles various formats: USD, ranges, k notation)
- **Skills extraction and normalization**
- **Location unification** across datasets

### **Features**
- **Dark/Light mode** toggle
- **Responsive design** for all screen sizes
- **Interactive filters** (location, salary, role type)
- **Real-time data updates**
- **Multi-dataset support**

## 📁 Project Structure

```
AI-Job-Market-Dashboard/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── data/                          # Data directory
│   ├── ai_job_dataset.csv         # Primary dataset
│   ├── ai_job_market.csv          # Additional market data
│   └── linkedin_jobs_analysis.csv # LinkedIn job insights
│
└── images/                        # Screenshots and assets
    ├── Dashboard Overview.png
    ├── Market Analysis.png
    ├── Salary Insights.png
    ├── Skills Intelligence.png
    ├── Geographic Trends.png
    ├── AI Impact Report.png
    └── Left side bar.png
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Iceyma02/AI-Job-Market-Dashboard.git
   cd AI-Job-Market-Dashboard
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the dashboard**
   - Open your web browser
   - Navigate to `http://localhost:8501`

### Requirements
```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
scikit-learn>=1.3.0
```

## 📈 Key Findings

### 🎯 **Market Overview**
- **15,000+** job listings analyzed across multiple datasets
- **86.2%** of roles are AI-focused, indicating strong market demand
- Average salary of **$115,000+** for AI positions
- **20+** major companies actively hiring in AI space

### 💰 **Salary Insights**
- AI roles command **1.2% salary premium** over non-AI roles
- Salary range: **$32,515 - $390,085** across all positions
- Clear correlation between specialized AI skills and higher compensation

### 🛠️ **Skills Analysis**
- **Python** is the most demanded skill (1,400+ mentions)
- **TensorFlow** and **PyTorch** lead ML framework requirements
- **SQL** remains essential for data-related roles
- Cloud platforms (**AWS, Azure, GCP**) show growing importance

### 🌍 **Geographic Trends**
- **Switzerland** leads in AI job concentration
- Strong presence in **India, China, France, Canada, Germany**
- Remote opportunities showing significant growth
- Tech hubs continue to dominate hiring

### 📊 **Role Distribution**
- **AI Product Manager** and **Machine Learning Engineer** are top roles
- **NLP Engineers** and **Data Scientists** in high demand
- Emerging roles in **Robotics** and **Autonomous Systems**

## 🎨 Customization

### Adding New Datasets
1. Place CSV files in the `data/` directory
2. The app automatically detects and integrates new datasets
3. Supported columns: `job_title`, `salary`, `skills`, `location`, `company`

### Modifying Visualizations
- Edit chart configurations in respective sections
- Customize colors in the theme configuration
- Add new plot types using Plotly Express

### Theme Customization
- Toggle between dark/light modes using sidebar button
- Modify CSS in the `get_theme_css()` function
- Customize Plotly themes in `get_plotly_theme()`

## 🔍 Usage Guide

### Data Filters
- **Location Filter**: Select specific countries or regions
- **Role Type**: Filter by AI roles, non-AI roles, or all roles
- **Salary Range**: Adjust to focus on specific compensation levels
- **Experience Level**: Filter by seniority (if data available)

### Navigation
- Use the horizontal radio buttons to switch between sections
- Each section provides specialized insights and visualizations
- Interactive charts allow zooming, panning, and data point inspection

### Data Interpretation
- Hover over charts for detailed information
- Use filter combinations for targeted analysis
- Export insights using Streamlit's built-in functionality

## 🤝 Contributing

We welcome contributions! Please feel free to submit pull requests or open issues for:

- New visualization ideas
- Additional data sources
- Performance improvements
- Bug fixes
- Documentation enhancements

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Anesu Manjengwa**
- 📧 Email: [manjengwap10@gmail.com](mailto:manjengwap10@gmail.com)
- 💼 LinkedIn: [Anesu Manjengwa](https://www.linkedin.com/in/anesu-manjengwa-684766247)
- 🐙 GitHub: [Iceyma02](https://github.com/Iceyma02)
- 🔗 Portfolio: [AI Job Market Dashboard](https://github.com/Iceyma02/AI-Job-Market-Dashboard)

## 🙏 Acknowledgments

- Data sources and contributors
- Streamlit community for excellent documentation
- Plotly for powerful visualization capabilities
- Open-source community for continuous inspiration

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/Iceyma02/AI-Job-Market-Dashboard/issues) page
2. Create a new issue with detailed description
3. Contact via email for direct support

---

<div align="center">

**⭐ Don't forget to star this repository if you find it helpful!**

*Built with ❤️ using Streamlit and Python*

</div>
