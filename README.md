# 🏠 Rental Price Predictor

A machine learning project that predicts monthly rental prices 
across 6 major Indian cities using property features.

**Live Demo**: [link after you deploy online]

---

## Problem Statement
Predicting rental prices is difficult due to high variability 
across cities, property sizes, and furnishing conditions. 
This project builds an ML model to estimate monthly rent 
given key property features.

---

## Dataset
- **Source**: [House Rent Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset) (Kaggle)
- 4,746 rental listings across Mumbai, Delhi, Bangalore, 
  Chennai, Hyderabad, and Kolkata
- 12 features including BHK, Size, Furnishing Status, City, Floor info

---

## Key Decisions & Why

**1. Log-transformed Rent (target variable)**
Rent was heavily right-skewed (max ₹35,00,000 vs median ₹16,000).
Applied log transformation to normalize distribution before modeling.

**2. Parsed Floor column**
Floor was stored as text ("1 out of 3"). Split into two numeric 
columns: Floor_Number and Total_Floors. 4 unparseable rows 
filled with median.

**3. Dropped Area Locality**
2,235 unique neighborhood values — too high cardinality to encode 
meaningfully. Used City instead.

**4. One-hot encoded categoricals**
Converted City, Furnishing Status, Tenant Preferred, Area Type, 
and Point of Contact into binary columns for model compatibility.

---

## Models Compared

| Model | MAE | RMSE |
|---|---|---|
| Linear Regression (baseline) | ₹11,320 | ₹31,569 |
| Random Forest ✅ | ₹10,769 | ₹34,628 |

Random Forest selected as final model — better average accuracy 
(MAE) despite slightly higher RMSE on luxury outliers.

---

## Key Findings

- **Point of Contact** was the strongest predictor (importance: 0.38),
  acting as a proxy for property quality — agents list premium properties,
  owners list budget ones
- **Bathroom count** (0.20) outranked BHK (0.026) as a quality signal
- Both models struggle on luxury listings (>₹1,00,000) due to 
  underrepresentation in training data (<5% of rows)
- City_Mumbai confirmed as strong location signal (#4 overall)

---

## What I'd Improve With More Time
- Scrape live listings for fresher data
- Add locality-level features via OpenStreetMap
- Try XGBoost with hyperparameter tuning
- Handle luxury listings separately (two-model approach)

---

## Tech Stack
- Python, Pandas, NumPy, Scikit-learn
- Matplotlib for visualization
- Streamlit for deployment

---

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```