# 🏡 House Price Prediction

This project is part of the Kaggle competition [Home Data for ML Course](https://www.kaggle.com/competitions/home-data-for-ml-course), aimed at building a machine learning model to predict house prices in Ames, Iowa, based on various property features.

---

## 📌 Project Overview

Accurately predicting house prices is a classic regression problem in data science. This project demonstrates the application of exploratory data analysis, feature engineering, and regression modeling to predict housing prices using the Ames Housing dataset.

---

## 📊 Dataset

The dataset contains 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. The target variable is:

- `SalePrice` – the property's sale price in USD.

📁 Key files:
- `train.csv` – Training data with features and target variable.
- `test.csv` – Test data for which predictions are to be made.

---

## ⚙️ Tools & Libraries

- Python 3.x
- Jupyter Notebook
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- XGBoost / LightGBM (optional for advanced modeling)

---

## 🔍 Methodology

1. **Data Cleaning** – Handling missing values, outliers, and inconsistent data.
2. **Exploratory Data Analysis (EDA)** – Understanding distributions, correlations, and variable importance.
3. **Feature Engineering** – Transforming skewed data, encoding categoricals, scaling numeric features.
4. **Modeling** – Using multiple regression algorithms:
    - Linear Regression
    - Ridge & Lasso Regression
    - Decision Trees / Random Forest
    - Gradient Boosting (XGBoost / LightGBM)
5. **Model Evaluation** – Using RMSE and cross-validation.

---

## 📈 Results

- Final model achieves competitive Root Mean Squared Error (RMSE) on the test dataset.
- Feature importance plot shows key predictors such as `OverallQual`, `GrLivArea`, and `GarageCars`.

---

## 🧠 Key Learnings

- The importance of feature preprocessing and outlier removal.
- Regularization techniques (Lasso/Ridge) help reduce overfitting.
- Gradient boosting methods outperform linear models in many cases.

---

## 🚀 How to Run

```bash
# Install required packages
pip install -r requirements.txt

# Launch notebook
jupyter notebook house-price-prediction.ipynb
````

---

## 📂 Project Structure

```
house-price-prediction/
│
├── house-price-prediction.ipynb   # Main notebook
├── train.csv                      # Training data
├── test.csv                       # Test data
├── requirements.txt               # Required Python packages
├── .gitignore                     # Files to exclude from git
└── README.md                      # Project documentation
```

---

## 🏁 Credits

* Competition hosted by [Kaggle](https://www.kaggle.com/competitions/home-data-for-ml-course)
* Dataset: Ames Housing Dataset by Dean De Cock

---

## 📬 Contact

Created by **\[Usriatur Rohma]**
📧 Email: \[usriaturrohmah13@gmail.com(mailto:usriaturrohmah13@gmail.com)]
🔗 LinkedIn: \[[Usriatur Rohma](https://www.linkedin.com/in/usriaturrohma/)]

---


