# Medical Insurance Cost Prediction

A regression project that predicts individual medical insurance charges based on demographic and lifestyle attributes, benchmarked across 5 algorithms and deployed as a desktop inference app.

## Problem Statement

Insurance charges vary significantly based on a person's age, BMI, smoking status, and other factors. This project builds and compares multiple regression models to predict `charges` for a given individual, and packages the best-performing model into a usable PyQt5 desktop application for real-time predictions.

## Dataset

- **Source:** [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) (1,338 records)
- **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- **Target:** `charges` (USD)

**Preprocessing:**
- `sex` and `smoker` label-encoded (female→0/male→1, no→0/yes→1)
- `region` one-hot encoded into `region_northeast`, `region_northwest`, `region_southeast`, `region_southwest`

## Approach

Five regression models were trained and evaluated to compare linear vs. tree-based (ensemble) approaches on this dataset:

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 79.59% | 5,940.03 | 4,069.04 |
| LassoCV | 79.42% | 5,965.25 | 4,076.61 |
| RidgeCV | 79.58% | 5,942.52 | 4,071.43 |
| Random Forest | 88.26% | 4,506.03 | 2,700.27 |
| **XGBoost** ✅ | **89.21%** | **4,320.15** | **2,498.54** |

**Final model: XGBoost Regressor** (`n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42`)

### Key Insight

Linear models (Linear Regression, LassoCV, RidgeCV) plateaued around **79-80% R²**, since insurance charges depend on non-linear interactions — for example, smoking status disproportionately increases cost at higher BMI. Tree-based ensemble models (Random Forest, XGBoost) capture these interactions automatically without manual feature engineering, pushing R² to **88-89%**. XGBoost outperformed Random Forest across every metric due to its sequential, error-correcting boosting mechanism, making it the final choice for deployment.

## Evaluation Metrics Explained

- **R² (Coefficient of Determination):** Percentage of variance in charges explained by the model.
- **MAE (Mean Absolute Error):** Average absolute rupee/dollar difference between predicted and actual charges — the most intuitive "typical error" metric.
- **RMSE (Root Mean Squared Error):** Similar to MAE but penalizes large errors more heavily, since differences are squared before averaging. RMSE > MAE indicates the presence of some large-error outliers in the dataset.
- **Adjusted R²:** Penalizes R² for the number of features used, preventing inflated scores from irrelevant predictors. Given the sample size relative to the 9 features used here, adjusted R² stays close to the standard R² score.

## Model Limitations

A subset of predictions — particularly non-smokers with normal BMI who still incur unusually high charges — show larger prediction errors. This is a known characteristic of this dataset: charges are also influenced by underlying health conditions that are not captured in the available features (age, sex, bmi, children, smoker, region). This limitation is inherent to the feature set, not the model choice, and explains why R² plateaus around 89% across tree-based models.

## Project Structure

```
├── insurance.csv                      # Dataset
├── model_training.ipynb / .py         # Training & evaluation of all 5 models
├── medical_insurance_xgb_model.pkl    # Final trained XGBoost model
├── insurance_predictor_app.py         # PyQt5 desktop app for real-time inference
└── README.md
```

## Desktop Application

A PyQt5-based GUI app (`insurance_predictor_app.py`) loads the trained XGBoost model and provides real-time cost predictions from user-entered inputs (age, sex, BMI, children, smoker status, region), demonstrating an end-to-end pipeline from model training to deployment.

**To run:**
```bash
pip install pyqt5 xgboost joblib pandas
python insurance_predictor_app.py
```

## Tech Stack

Python, pandas, scikit-learn, XGBoost, PyQt5, joblib

## Future Improvements

- Add adjusted R² and cross-validation (StratifiedKFold/GridSearchCV) for more robust model selection
- Feature engineering: smoker × bmi and smoker × age interaction terms (shown to improve linear model performance in related experiments)
- Deploy as a web app (Streamlit/Flask) in addition to the desktop version
"# insurance_cost_prediction" 
