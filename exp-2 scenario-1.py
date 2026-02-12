Python 3.11.5 (tags/v3.11.5:cce6ba9, Aug 24 2023, 14:38:34) [MSC v.1936 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
# Name: Kamalesh.M

# Roll No: 24BAD053

# =========================================

# Water Temperature Prediction using Regression Models

# =========================================

# ===============================

# 1. Import Required Libraries

# ===============================

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.feature_selection import SelectKBest, f_regression

# ===============================

# 2. Load Dataset

# ===============================

bottle = "bottle.csv"

cast = "cast.csv"

b_df = pd.read_csv(bottle, low_memory=False)

c_df = pd.read_csv(cast, low_memory=False)

print("Bottle Dataset Shape:", b_df.shape)

print("Cast Dataset Shape  :", c_df.shape)

# ===============================

# 3. Merge Required Columns

# ===============================

merged = pd.merge(

    b_df,

    c_df[['Cst_Cnt', 'Sta_ID', 'Lat_Dec', 'Lon_Dec']],

    on=['Cst_Cnt', 'Sta_ID'],

    how='left'

)

print("Merged Dataset Shape:", merged.shape)

# ===============================

# 4. Select Features & Target

# ===============================

features = ['Depthm', 'Salnty', 'O2ml_L', 'Lat_Dec', 'Lon_Dec']

target = 'T_degC'

merged = merged.dropna(subset=features + [target])

X = merged[features]

y = merged[target]

print("Final Dataset Shape:", X.shape)

# ===============================

# 5. Correlation Heatmap

# ===============================

plt.figure(figsize=(8,6))

sns.heatmap(merged[features + [target]].corr(),

            annot=True, cmap="coolwarm")

plt.title("Correlation Matrix")

plt.show()

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

print("Training Data:", X_train.shape)

print("Testing Data :", X_test.shape)

# ===============================

# 8. Model Evaluation Function

# ===============================

def evaluate_model(model, X_train, X_test, y_train, y_test, name):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    print(f"\n{name} Performance")

    print("MSE :", mse)

    print("RMSE:", rmse)

    print("R²  :", r2)

    return y_pred, r2

# ===============================

# 9. Linear Regression

# ===============================

lr = LinearRegression()

lr_pred, lr_r2 = evaluate_model(lr, X_train, X_test, y_train, y_test, "Linear Regression")

# ===============================

# 10. Ridge Regression

# ===============================

ridge = Ridge(alpha=1.0)

ridge_pred, ridge_r2 = evaluate_model(ridge, X_train, X_test, y_train, y_test, "Ridge Regression")

# ===============================

# 11. Lasso Regression

# ===============================

lasso = Lasso(alpha=0.01)

lasso_pred, lasso_r2 = evaluate_model(lasso, X_train, X_test, y_train, y_test, "Lasso Regression")

# ===============================

# 12. Actual vs Predicted Plot

# ===============================

plt.figure(figsize=(7,5))

plt.scatter(y_test, lr_pred, alpha=0.5)

plt.xlabel("Actual Temperature (°C)")

plt.ylabel("Predicted Temperature (°C)")

plt.title("Actual vs Predicted (Linear Regression)")

plt.show()

# ===============================

# 13. Residual Distribution

# ===============================

residuals = y_test - lr_pred

plt.figure(figsize=(7,5))

sns.histplot(residuals, kde=True)

plt.xlabel("Residual Error")

plt.title("Residual Error Distribution")

plt.show()

# ===============================

# 14. Feature Selection

# ===============================

selector = SelectKBest(score_func=f_regression, k=3)

X_selected = selector.fit_transform(X_scaled, y)

... selected_features = np.array(features)[selector.get_support()]
... 
... print("\nTop 3 Selected Features:", selected_features)
... 
... # ===============================
... 
... # 15. Feature Importance
... 
... # ===============================
... 
... coeff_df = pd.DataFrame({
... 
...     'Feature': features,
... 
...     'Linear Coefficient': lr.coef_
... 
... })
... 
... print("\nFeature Importance:")
... 
... print(coeff_df)
... 
... # ===============================
... 
... # 16. Model Comparison
... 
... # ===============================
... 
... comparison = pd.DataFrame({
... 
...     "Model": ["Linear", "Ridge", "Lasso"],
... 
...     "R2 Score": [lr_r2, ridge_r2, lasso_r2]
... 
... })
... 
... print("\nModel Comparison:")
... 
... print(comparison)
