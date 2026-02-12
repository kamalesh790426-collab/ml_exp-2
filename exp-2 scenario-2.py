Python 3.11.5 (tags/v3.11.5:cce6ba9, Aug 24 2023, 14:38:34) [MSC v.1936 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
# Name: Kamalesh.M
# Roll No: 24BAD054
# =========================================
# Stock Price Movement Prediction using Logistic Regression
# =========================================
# ===============================
# 1. Import Required Libraries
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
   accuracy_score, precision_score, recall_score,
   f1_score, confusion_matrix, roc_curve, auc,
   classification_report
)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
# ===============================
# 2. Load Dataset
# ===============================
df = pd.read_csv("LICI - 10 minute data.csv")
print("Dataset Shape:", df.shape)
print(df.head())
# ===============================
# 3. Create Binary Target Variable
# ===============================
df['price_movement'] = np.where(df['close'] > df['open'], 1, 0)
# ===============================
# 4. Select Features & Target
# ===============================
features = ['open', 'high', 'low', 'volume']
target = 'price_movement'
X = df[features]
y = df[target]
# ===============================
# 5. Handle Missing Values
# ===============================
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
# ===============================
# 6. Feature Scaling
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# ===============================
# 7. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
   X_scaled, y, test_size=0.2, random_state=42
)
print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)
# ===============================
# 8. Train Logistic Regression
# ===============================
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
# ===============================
# 9. Predictions
# ===============================
y_pred = log_reg.predict(X_test)
# ===============================
# 10. Model Evaluation
# ===============================
print("\n--- Base Model Performance ---")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
           xticklabels=["Decrease", "Increase"],
           yticklabels=["Decrease", "Increase"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
# ===============================
# 11. ROC Curve
# ===============================
y_prob = log_reg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"Base Model AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
# ===============================
# 12. Feature Importance (Random Forest)
# ===============================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importance_df = pd.DataFrame({
   'Feature': features,
   'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(6,4))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance (Random Forest)")
plt.show()
# ===============================
# 13. Hyperparameter Tuning
# ===============================
... param_grid = {
...    'C': [0.01, 0.1, 1, 10],
...    'penalty': ['l2'],
...    'solver': ['lbfgs']
... }
... grid = GridSearchCV(
...    LogisticRegression(max_iter=1000),
...    param_grid,
...    cv=5,
...    scoring='accuracy'
... )
... grid.fit(X_train, y_train)
... print("\nBest Parameters:", grid.best_params_)
... # ===============================
... # 14. Evaluate Tuned Model
... # ===============================
... best_model = grid.best_estimator_
... y_best_pred = best_model.predict(X_test)
... print("\n--- Tuned Model Performance ---")
... print("Tuned Accuracy:",
...      accuracy_score(y_test, y_best_pred))
... print("Tuned Precision:",
...      precision_score(y_test, y_best_pred))
... print("Tuned Recall:",
...      recall_score(y_test, y_best_pred))
... print("Tuned F1 Score:",
...      f1_score(y_test, y_best_pred))
... # ROC for Tuned Model
... y_best_prob = best_model.predict_proba(X_test)[:, 1]
... fpr_best, tpr_best, _ = roc_curve(y_test, y_best_prob)
... roc_auc_best = auc(fpr_best, tpr_best)
... plt.figure(figsize=(6,4))
... plt.plot(fpr, tpr, label=f"Base AUC = {roc_auc:.2f}")
... plt.plot(fpr_best, tpr_best, label=f"Tuned AUC = {roc_auc_best:.2f}")
... plt.plot([0,1], [0,1], linestyle="--")
... plt.xlabel("False Positive Rate")
... plt.ylabel("True Positive Rate")
... plt.title("ROC Curve Comparison")
... plt.legend()
