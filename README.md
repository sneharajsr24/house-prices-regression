# 🏡 Advanced House Price Prediction using XGBoost

This project presents an end-to-end machine learning pipeline for predicting house prices using the **Ames Housing Dataset**. It integrates advanced feature engineering, correlation-based feature selection, and hyperparameter tuning using **XGBoost**, all structured within a clean and modular `scikit-learn` pipeline.

---

## 📌 Key Features

- ✅ **Custom Feature Engineering**: Derived meaningful features like `HouseAge`, `TotalSF`, `RemodAge`, `OverallGrade`, and more.
- 🔍 **Feature Selection**: Retained highly correlated features using a custom correlation-based transformer.
- 🧠 **Model Tuning**: Optimized XGBoost hyperparameters via `RandomizedSearchCV` over 50 iterations.
- 📈 **Performance**: Achieved **R² = 0.882** and **RMSE ≈ 27,383** on the test set.
- 🧱 **Modular Design**: Built with `Pipeline`, `BaseEstimator`, and `TransformerMixin` for clarity and reuse.

---

## 📁 Dataset

This project uses the **Ames Housing Dataset** from Kaggle:  
🔗 [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

Please download the dataset and place `train.csv` and `test.csv` in the project root directory.

---

## 🛠️ Tools & Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- RandomizedSearchCV
- Custom Transformers
- Label Encoding

---
