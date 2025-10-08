# Telco Customer Churn Analysis & Prediction

**Project Description:**  
Customer churn is a critical issue for telecom companies because acquiring new customers is often more expensive than retaining existing ones. This project aims to analyze customer behavior, identify factors driving churn, and build predictive models to anticipate which customers are most likely to leave. By combining statistical inference, machine learning, and interpretability techniques, the project provides actionable insights for targeted retention strategies.

---

## Project Objectives

1. **Exploratory Data Analysis (EDA)**  
   - Analyze customer demographics, finances, services, and contract information.
   - Understand relationships between features and churn.
   - Visualize correlations and feature distributions.

2. **Statistical Inference**  
   - Conduct Chi-Square tests for categorical features to detect significant relationships with churn.
   - Fit Logit, Probit, Cloglog, Penalized Logit, and Bayesian Approximation models to evaluate predictive power.
   - Examine marginal effects to understand how each feature impacts churn probability.

3. **Machine Learning Modeling**  
   - Prepare dataset with label encoding and one-hot encoding.
   - Split data into Train (70%), Test (20%), and Outsample (10%) sets.
   - Train a Gradient Boosting Classifier using Randomized Search CV with 10-fold cross-validation to find optimal hyperparameters.
   - Evaluate model performance using Accuracy, AUC, LogLoss, and F1-score.
   - Visualize feature importance and ROC curves.

4. **Risk Segmentation & SHAP Analysis**  
   - Identify high-risk customers based on predicted probabilities (top 20% risk segment).
   - Profile high-risk customers with descriptive statistics and visualizations.
   - Use SHAP values to interpret model predictions and feature contributions.

---

## Installation

```bash
pip install pandas numpy scikit-learn pycaret shap matplotlib seaborn statsmodels
