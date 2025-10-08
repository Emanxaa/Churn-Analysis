# %% [markdown]
# # Telco Churn Prediction Using Gradient Boosting
# 
# **Objective**: Predict customer churn using Gradient Boosting and identify high-risk customers. 
# Steps include:
# 1. Data preprocessing (Label Encoding, One-Hot Encoding, type conversion)
# 2. Train-test-outsample split
# 3. Hyperparameter tuning using Randomized Search CV
# 4. Model evaluation using Accuracy, AUC, LogLoss, F1
# 5. ROC Curve visualization
# 6. Feature importance (both tree-based and SHAP)
# 7. High-risk segment identification and profiling

# %%
import pandas as pd

# %% [markdown]
# ## 1. Load Dataset
df=pd.read_csv("D:\\DATA SCIENCE\\CHURN PROJ\\TELCO_CLEANED.csv")
df.head()

# %% [markdown]
# ## 2. Define Encoding Strategies
# Label encoding for simple categorical features, One-Hot encoding for multi-category features

# %%
Encod_Label = ['gender', 'Partner', 'Dependents', 'PhoneService','Churn']
Num_label = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
Encod_OneHot = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

# %% [markdown]
# ## 3. Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].dtype

# %% [markdown]
# ## 4. Label Encoding
from sklearn.preprocessing import LabelEncoder

# %%
le = LabelEncoder()
for col in Encod_Label:
    df[col] = le.fit_transform(df[col])

# %% [markdown]
# ## 5. One-Hot Encoding
for col in Encod_OneHot:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=[col], axis=1, inplace=False)

# %%
print(df.dtypes)

# %% [markdown]
# ## 6. Identify Numeric and Categorical Features
numeric_features = df.select_dtypes(include=['int64', 'int32', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()

# %% [markdown]
# ## 7. Fix column names & fill missing
df.rename(columns={'Churn_x':'Churn'}, inplace=True)
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# %% [markdown]
# ## 8. PyCaret Setup (Optional)
from pycaret.classification import *

# Ensure target exists
assert 'Churn' in df.columns, "Column 'Churn' not found!"

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in df.columns if col not in numeric_features + ['Churn','customer ID']]

exp_clf = setup(
    data=df,
    target='Churn',
    session_id=123,
    numeric_features=numeric_features,
    categorical_features=categorical_features,
    fold_shuffle=True
)

# %% [markdown]
# ## 9. Compare Top Models by AUC
top3_auc = compare_models(n_select=3, sort='AUC')
print("Top 3 model berdasarkan AUC:")
for i, model in enumerate(top3_auc, 1):
    print(f"{i}. {model}") 

# %% [markdown]
# ## 10. Train-Test-Outsample Split
# Split data into train (70%), test (20%), outsample (10%) using stratification to preserve class ratio

# %%
import numpy as np
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train_full, X_outsample, y_train_full, y_outsample = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.222, random_state=42, stratify=y_train_full)
# 0.222 * 0.9 ≈ 20% total

print(f"Train: {len(X_train)}, Test: {len(X_test)}, Outsample: {len(X_outsample)}")

# %% [markdown]
# ## 11. Define Gradient Boosting Model & Hyperparameter Space
from sklearn.ensemble import GradientBoostingClassifier

# %%
gboost = GradientBoostingClassifier(random_state=42)

param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None]
}

# %% [markdown]
# ## 12. Randomized Search CV
from sklearn.model_selection import RandomizedSearchCV

# %%
random_search = RandomizedSearchCV(
    estimator=gboost,
    param_distributions=param_dist,
    n_iter=30,
    cv=10,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
print("\n=== Best Hyperparameters ===")
print(random_search.best_params_)

# %% [markdown]
# ## 13. Evaluate Model Performance
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, f1_score

# %%
def evaluate_model(model, X, y, label="Dataset"):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    ll = log_loss(y, y_prob)
    f1 = f1_score(y, y_pred)
    print(f"\n[{label}] Accuracy: {acc:.3f} | AUC: {auc:.3f} | LogLoss: {ll:.3f} | F1: {f1:.3f}")
    return acc, auc, ll, f1, y_prob

train_metrics = evaluate_model(best_model, X_train, y_train, "Train")
test_metrics = evaluate_model(best_model, X_test, y_test, "Test")
outsample_metrics = evaluate_model(best_model, X_outsample, y_outsample, "Outsample")

y_train_prob = train_metrics[4]
y_test_prob = test_metrics[4]
y_out_prob = outsample_metrics[4]

# %% [markdown]
# ## 14. Plot ROC Curve for Outsample
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# %%
fpr, tpr, _ = roc_curve(y_outsample, y_out_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_outsample, y_out_prob):.3f}')
plt.plot([0,1], [0,1], 'k--')
plt.title('ROC Curve - Outsample')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## 15. Feature Importance (Gradient Boosting)

# %%
import seaborn as sns
import pandas as pd

importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,6))
sns.barplot(x=importances[:15], y=importances.index[:15], palette='Blues_r')
plt.title("Top 15 Feature Importances (Gradient Boosting)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# %% [markdown]
# ## 16. Identify High-Risk Segment (Top 20% Probability)

# %%
def identify_risk_segment(probs, quantile=0.8):
    threshold = np.quantile(probs, quantile)
    return np.where(probs >= threshold, 'High Risk', 'Low Risk')

train_risk = identify_risk_segment(y_train_prob)
test_risk = identify_risk_segment(y_test_prob)
outsample_risk = identify_risk_segment(y_out_prob)

df_train = X_train.copy()
df_train['churn'] = y_train
df_train['risk_segment'] = train_risk

# %% [markdown]
# ## 17. Profiling High Risk Segment
high_risk_profile = df_train[df_train['risk_segment']=='High Risk'].describe()
print(high_risk_profile)

# %% [markdown]
# ## 18. Visualize Feature Distribution by Risk Segment
for col in X_train.columns[:5]:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='risk_segment', y=col, data=df_train)
    plt.title(f'{col} by Risk Segment')
    plt.show()

# %% [markdown]
# ## 19. SHAP Feature Importance
import shap

# %%
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_train)

# Summary bar plot
shap.summary_plot(shap_values, X_train, plot_type="bar")

# Dependence plot for top feature
important_feature = X_train.columns[np.argmax(np.abs(shap_values.values).mean(0))]
shap.dependence_plot(important_feature, shap_values.values, X_train)
