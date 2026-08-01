# What Drives the Price of a Car?

Practical Application II — UC Berkeley Post Graduate Program in AI and Machine Learning

## Notebook

[prompt_II.ipynb](prompt_II.ipynb)

---

## Data Problem

A used car dealership wants to know what makes a car worth more or less money. We build a regression model to predict `price` from car features and identify which features matter most — so the dealership knows what to look for when buying and pricing inventory.

**Evaluation metric:** RMSE (Root Mean Squared Error) — tells us how far off our price predictions are in dollars. We also track R² to measure how much of the price variation our model explains.

---

## Data Understanding

- Loaded 426,880 listings with 18 features
- Identified key quality issues: extreme price outliers ($0 to $3B), odometer readings up to 10M miles, and heavy missingness in `condition` (41%), `cylinders` (42%), `drive` (31%), and `size` (72%)
- Explored distributions of continuous variables (price, year, odometer) and categorical variables (condition, fuel, transmission, type)

---

## Data Preparation

1. **Remove outliers** — kept prices $500–$150k, odometer < 500k miles, year 1980–2024
2. **Handle missing values and drop sparse columns** — filled `condition` NaN with "unknown", extracted numeric cylinders, dropped `size` (72% missing)
3. **Drop irrelevant columns** — removed `id`, `VIN`, `region`, `state`, `model`, `manufacturer`, `paint_color`, `title_status`
4. **Encode categories** — ordinal encoding for `condition`; one-hot encoding for `fuel`, `transmission`, `drive`, `type`
5. **Split into train/test sets** — 80/20 split (297k train, 74k test)
6. **Scale numeric features** — StandardScaler fitted on training data only (year, odometer, cylinders)

---

## Modeling

1. **Ridge Regression** — L2 regularization; GridSearchCV over alpha `[1, 10, 100, 1000, 10000]` with 5-fold CV
2. **Lasso Regression** — L1 regularization; GridSearchCV over alpha `[0.01, 0.1, 1, 10, 100, 1000]` with 5-fold CV
3. **Random Forest** — ensemble of decision trees; GridSearchCV over n_estimators, max_depth, min_samples_leaf with 3-fold CV

---

## Evaluation

- Random Forest outperformed both linear models significantly
- Ridge and Lasso both plateaued at ~$9,200 RMSE — linear models hit a ceiling due to non-linear price relationships
- Random Forest predicted prices within ~$5,500 on average and explained over 85% of price variation
- Top price drivers: **year**, **odometer**, **condition**, **cylinders**, **vehicle type**, **drive type**

---

## Key Recommendations for the Dealership

1. **Stock newer vehicles** — year is the single strongest price driver; prioritize 2015 and newer
2. **Watch the odometer** — under 80,000 miles commands the best margins
3. **Condition matters** — excellent/like-new condition justifies a meaningful price premium
4. **Focus on trucks and SUVs** — consistently outprice sedans and hatchbacks
5. **4WD is a selling point** — buyers pay more for four-wheel drive

## Next Steps

- Add manufacturer back using target encoding to capture brand value
- Incorporate geographic data (state/region) for regional price differences
- Retrain the model periodically as market conditions change
