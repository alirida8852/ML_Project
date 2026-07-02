# 🏠 Rental Price Predictor

A machine learning project that predicts monthly rental prices 
across 6 major Indian cities using property features.

**Live Demo**: [coming soon]

---

## Problem Statement

Predicting rental prices is difficult due to high variability 
across cities, property sizes, and furnishing conditions. 
This project builds an end-to-end ML pipeline to estimate 
monthly rent given key property features — from raw data 
cleaning to a deployed web app.

---

## Dataset

- **Source**: [House Rent Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset) (Kaggle)
- 4,746 rental listings across Mumbai, Delhi, Bangalore, 
  Chennai, Hyderabad, and Kolkata
- 12 original features including BHK, Size, Furnishing Status, 
  City, Floor info, Tenant Preferred, Point of Contact

---

## Project Structure
ml-rent-prediction/
├── data/
│   └── House_Rent_Dataset.csv
├── notebooks/
│   └── 01_eda.ipynb
├── app.py
├── model.pkl
├── columns.pkl
├── requirements.txt
└── README.md

---

## Key Decisions & Why

**1. Log-transformed Rent (target variable)**
Rent was heavily right-skewed (max ₹35,00,000 vs median ₹16,000).
Applied log transformation using np.log1p() to normalize the 
distribution before modeling. Predictions are converted back 
to actual ₹ using np.expm1() for reporting.

**2. Parsed Floor column**
Floor was stored as messy text ("1 out of 3", "Ground out of 2").
Split into two numeric columns: Floor_Number and Total_Floors.
Ground floor treated as 0. 4 unparseable rows filled with median.

**3. Dropped Area Locality**
2,235 unique neighborhood values — too high cardinality to encode 
meaningfully for a dataset of 4,746 rows. Used City instead.

**4. One-hot encoded categoricals**
Converted City, Furnishing Status, Tenant Preferred, Area Type, 
and Point of Contact into binary columns. Used drop_first=True 
to avoid the dummy variable trap.

**5. Feature Engineering**
Created two derived features without using the target variable 
(avoiding data leakage):
- floor_ratio = Floor_Number / Total_Floors — captures how high 
  up proportionally the property is
- bhk_per_bathroom = BHK / Bathroom — captures layout quality

Initially considered price_per_sqft but rejected it as it would 
cause data leakage (uses Rent, the target variable).

---

## Models Compared

| Model | MAE | RMSE |
|---|---|---|
| Linear Regression (baseline) | ₹11,320 | ₹31,569 |
| Linear Regression + features | ₹11,107 | ₹30,517 |
| Random Forest ✅ | ₹10,769 | ₹34,628 |
| Random Forest + features | ₹10,922 | ₹36,196 |
| XGBoost + features | ₹11,183 | ₹37,346 |

**Random Forest selected as final model** — best MAE (₹10,769) 
across all variations. Engineered features improved Linear 
Regression but slightly hurt tree-based models, suggesting 
Random Forest already captured those relationships internally.

---

## Key Findings

- **Point of Contact** was the strongest predictor (importance: 0.38) 
  — agents list premium properties, owners list budget ones, 
  making this a proxy for property quality
- **Bathroom count** (0.20) outranked BHK (0.026) as a quality 
  signal — number of bathrooms is a stronger indicator of 
  property standard than bedroom count in Indian rentals
- **City_Mumbai** confirmed as strong location signal (#4 overall)
- Both models struggle on luxury listings (>₹1,00,000) due to 
  underrepresentation in training data (<5% of rows)
- Engineered features helped linear models but not tree-based 
  models — an important lesson in feature engineering impact

---

## Error Analysis

Predicted vs Actual plots show both models perform well on 
typical listings (₹5,000–₹50,000) but diverge on luxury 
listings above ₹1,00,000. This is expected — only ~5% of 
training data represents high-end properties, so the model 
has limited examples to learn from in that range.

---

## Real World Sanity Check

| Input | Predicted Rent |
|---|---|
| Kolkata, 1BHK, 400sqft, Unfurnished, Owner | ₹5,655 |
| Mumbai, 3BHK, 1500sqft, Furnished, Agent | ₹67,272 |

12x difference between budget and premium listings — 
consistent with real Indian rental market pricing.

---

## What I'd Improve With More Time

- Scrape live listings via API for fresher, larger dataset
- Add locality-level features via OpenStreetMap 
  (distance to metro, schools, hospitals)
- Handle luxury listings separately with a two-model approach
- Try XGBoost with proper hyperparameter tuning (GridSearchCV)
- Add confidence intervals based on prediction variance

---

## Tech Stack

- **Python** — Pandas, NumPy, Scikit-learn, XGBoost
- **Visualization** — Matplotlib
- **Deployment** — Streamlit

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

Rida Ali — [LinkedIn:https://www.linkedin.com/in/rida-ali-a25537276] 
[GitHub: https://github.com/alirida8852]