# 🗽 NYC Housing Market Linear Regression 🏠

This repository contains a data-driven analysis of the NYC housing market using linear regression techniques. By leveraging statistical models, this project provides insights into the factors influencing housing prices in New York City.

## 📄 Project Overview

Housing market analysis in NYC is a complex task due to the diversity of neighborhoods, apartment types, economic factors, and social dynamics. This repository adopts machine learning techniques, specifically **Simple Linear Regression** and **Multiple Linear Regression**, to uncover trends and predictions in housing prices.

## 📂 Contents

The project repository includes the following files:

- **📓 Jupyter Notebooks**
  - [`Simple Linear Regression.ipynb`](./Simple%20Linear%20Regression.ipynb): Builds and evaluates a simple linear regression model for NYC housing price data.
  - [`Multiple Linear Regression.ipynb`](./Multiple%20Linear%20Regression.ipynb): Explores multivariate relationships using multiple linear regression methods.

- **📊 Datasets**
  - [`NY-House-Dataset.csv`](./NY-House-Dataset.csv): Full dataset containing housing features and pricing information.
  - [`NY-Housing-SimpleData.csv`](./NY-Housing-SimpleData.csv): Filtered dataset for simpler use cases and initial experiments.

- **🔧 Scripts**
  - [`app.py`](./app.py): Python script to preprocess data and deploy regression models.

- **📋 Requirements**
  - [`requirements.txt`](./requirements.txt): Specifies required Python packages for the project.

## 🎯 Project Objectives

The primary goals of this project include:
- 🔍 Analyzing the relationship between housing features (e.g., square footage, location) and prices.
- 🤖 Building predictive models to estimate house prices based on input features.
- 💡 Providing actionable insights into key drivers of housing market fluctuations.

## 🧪 Methodology

1. **🔄 Data Preprocessing:**
   - Cleaning and organizing datasets.
   - Handling missing values and outliers.
   - Splitting datasets into training and testing phases.

2. **📊 Exploratory Data Analysis (EDA):**
   - Visualizing pricing trends across neighborhoods.
   - Understanding the distribution of features.

3. **📈 Model Building:**
   - Constructing and evaluating simple and multiple linear regression models.
   - Assessing accuracy through metrics like Mean Squared Error (MSE) and R-squared.

4. **🚀 Deployment:**
   - Using Python scripts to make the regression models accessible for further experimentation.

## ⚙️ Requirements

Install the following dependencies to run the notebooks and scripts:
- Python 3.8+
- Jupyter Notebook
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

Install these packages using:
```bash
pip install -r requirements.txt
```

## 🖱️ How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/SagarRudagi/NYC-Housing-Market-Linear-Regression.git
   ```
2. Navigate the Jupyter Notebooks for analysis and visualizations.
3. Use `app.py` for additional data processing and model experiments.

## 🔮 Future Work

- Include more features like crime rates, school ratings, and commute times.
- Experiment with advanced regression techniques or machine learning models.
- Deploy a web-based tool for interactive price exploration.

---

📌 **Author:** Sagar Rudagi  
Feel free to 🌟 this repository, contribute, open issues, or provide suggestions for improvements!