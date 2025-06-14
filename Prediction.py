# Library Import
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Suppress Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load Dataset
for dirpath, _, files in os.walk('/kaggle/input'):
    for file in files:
        print(os.path.join(dirpath, file))

train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

# Missing Value Inspection
missing_values = train.isnull().sum().sort_values(ascending=False)
missing_values = missing_values[missing_values > 0]
print("Missing values:\n", missing_values)

# Data Type Overview
print("\nData types:\n", train.dtypes)

# SalePrice Distribution
plt.figure(figsize=(10, 6))
sns.histplot(train['SalePrice'], kde=True)
plt.title('SalePrice Distribution')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()

# Log Transformation
train['LogSalePrice'] = np.log1p(train['SalePrice'])

plt.figure(figsize=(10, 6))
sns.histplot(train['LogSalePrice'], kde=True)
plt.title('Log(SalePrice) Distribution')
plt.xlabel('LogSalePrice')
plt.ylabel('Frequency')
plt.show()

print("Skewness of LogSalePrice:", train['LogSalePrice'].skew())

# Remove Outliers
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train, x='GrLivArea', y='SalePrice')
plt.title('GrLivArea vs SalePrice (Before Removing Outliers)')
plt.show()

train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=train, x='GrLivArea', y='SalePrice')
plt.title('GrLivArea vs SalePrice (After Removing Outliers)')
plt.show()

# Drop Columns with >50% Missing
threshold = len(train) * 0.5
train = train.drop(columns=[col for col in train.columns if train[col].isnull().sum() > threshold])
test = test.drop(columns=[col for col in test.columns if test[col].isnull().sum() > threshold])

print("Remaining columns in train:\n", train.columns.tolist())
print("Remaining columns in test:\n", test.columns.tolist())

# Imputation for Specific Features
if 'LotFrontage' in train.columns and 'Neighborhood' in train.columns:
    train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    if 'LotFrontage' in test.columns:
        test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

if all(col in train.columns for col in ['GarageYrBlt', 'GarageType', 'YearBuilt']):
    train['GarageYrBlt'] = train.apply(lambda row: row['YearBuilt'] if pd.isna(row['GarageYrBlt']) and row['GarageType'] != 'None' else row['GarageYrBlt'], axis=1)
    test['GarageYrBlt'] = test.apply(lambda row: row['YearBuilt'] if pd.isna(row['GarageYrBlt']) and row['GarageType'] != 'None' else row['GarageYrBlt'], axis=1)
    train['GarageYrBlt'].fillna(0, inplace=True)
    test['GarageYrBlt'].fillna(0, inplace=True)

if 'MasVnrType' in train.columns:
    mode_masvnr = train['MasVnrType'].mode()[0]
    train['MasVnrType'].fillna(mode_masvnr, inplace=True)
    if 'MasVnrType' in test.columns:
        test['MasVnrType'].fillna(mode_masvnr, inplace=True)

if 'MasVnrArea' in train.columns and 'MasVnrType' in train.columns:
    train['MasVnrArea'] = train.apply(lambda row: 0 if row['MasVnrType'] == 'None' else row['MasVnrArea'], axis=1)
    test['MasVnrArea'] = test.apply(lambda row: 0 if row['MasVnrType'] == 'None' else row['MasVnrArea'], axis=1)
    train['MasVnrArea'].fillna(train['MasVnrArea'].median(), inplace=True)
    test['MasVnrArea'].fillna(test['MasVnrArea'].median(), inplace=True)

# General Missing Value Imputation
for col in train.columns:
    if col not in ['LotFrontage', 'GarageYrBlt', 'MasVnrType', 'MasVnrArea']:
        if train[col].dtype in ['int64', 'float64']:
            train[col].fillna(train[col].median(), inplace=True)
            if col in test.columns:
                test[col].fillna(test[col].median(), inplace=True)
        else:
            train[col].fillna('None', inplace=True)
            if col in test.columns:
                test[col].fillna('None', inplace=True)

print("Total missing values after processing:", train.isnull().sum().sum())

# Feature Engineering
train['TotalBath'] = train['FullBath'] + 0.5 * train['HalfBath'] + train['BsmtFullBath'] + 0.5 * train['BsmtHalfBath']
test['TotalBath'] = test['FullBath'] + 0.5 * test['HalfBath'] + test['BsmtFullBath'] + 0.5 * test['BsmtHalfBath']

train['HouseAge'] = train['YrSold'] - train['YearBuilt']
test['HouseAge'] = test['YrSold'] - test['YearBuilt']

mean_price = train.groupby('Neighborhood')['SalePrice'].mean()
train['Neighborhood_Encoded'] = train['Neighborhood'].map(mean_price)
test['Neighborhood_Encoded'] = test['Neighborhood'].map(mean_price).fillna(mean_price.mean())

train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

# Correlation Analysis
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = train[numerical_cols].corr()
print("Top correlated features with SalePrice:\n", corr_matrix['SalePrice'].sort_values(ascending=False).head(11))

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Skewness Correction
skewed_cols = train[numerical_cols].apply(lambda x: x.skew()).sort_values(ascending=False)
skewed_cols = skewed_cols[skewed_cols > 0.5].index
for col in skewed_cols:
    if col not in ['SalePrice', 'LogSalePrice']:
        train[col] = np.log1p(train[col])
        if col in test.columns:
            test[col] = np.log1p(test[col])

# Visualization
plt.figure(figsize=(12, 6))
sns.boxplot(x='Neighborhood', y='SalePrice', data=train)
plt.xticks(rotation=45)
plt.title('SalePrice by Neighborhood')
plt.xlabel('Neighborhood')
plt.ylabel('SalePrice')
plt.show()

# One-hot Encoding
train_encoded = pd.get_dummies(train.drop(columns=['SalePrice', 'LogSalePrice']),
                               columns=train.select_dtypes(include='object').columns)
test_encoded = pd.get_dummies(test, columns=test.select_dtypes(include='object').columns)
train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

# Model Preparation
X = train_encoded.drop(columns=['Id'])
y = train['LogSalePrice']
X_test = test_encoded.drop(columns=['Id'], errors='ignore')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training with XGBoost
xgb = XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=6, random_state=42)
xgb.fit(X_train, y_train)

# Validation & Evaluation
y_pred = xgb.predict(X_val)
rmse_log = np.sqrt(mean_squared_error(y_val, y_pred))
print("RMSE (Log SalePrice):", rmse_log)

y_pred_original = np.expm1(y_pred)
y_val_original = np.expm1(y_val)
rmse_original = np.sqrt(mean_squared_error(y_val_original, y_pred_original))
print("RMSE (Original Scale):", rmse_original)

plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Actual LogSalePrice')
plt.ylabel('Predicted LogSalePrice')
plt.title('Actual vs Predicted (LogSalePrice)')
plt.show()

# Cross-Validation
cv_scores = cross_val_score(xgb, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_cv = np.sqrt(-cv_scores.mean())

# Submission
final_preds = np.expm1(xgb.predict(X_test))
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': final_preds})
submission.to_csv('submission.csv', index=False)
print("Submission preview:\n", submission.head())
