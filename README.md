
# NYC Taxi Trip Duration Prediction

Predict the duration of NYC taxi trips using historical trip data. This project demonstrates a complete ML workflow data preprocessing, baseline and candidate models, final model selection, and evaluation.

---

## Project Overview

The goal is to predict taxi trip duration based on features such as

- Pickup and dropoff coordinates  
- Pickup datetime  
- Passenger count  
- Vendor ID and other trip metadata  

This is a regression problem using Python ML libraries and tree-based ensembles.

---

## Repository Structure

```
NYC-Taxi-Trip-Duration
│
├─ notebooks               # Jupyter  Colab notebooks
│   ├─ eda.ipynb            # Exploratory Data Analysis notebook
│   ├─ preprocessing.ipynb  # Preprocessing notebook
│   └─ modeling.ipynb       # Modeling notebook
│
├─ results               # Stores model outputs
│   ├─ baselines         # Baseline model metrics
│   ├─ candidates        # Candidate model metrics
│   └─ final             # Final model and test results
│
├─ src                   # Source code
│   ├─ modeling.py        # Modeling scripts
│   └─ preprocessing.py   # Preprocessing functions
│
├─ .gitignore
├─ README.md
└─ requirements.txt


```
## Dataset

The dataset is not included in this repository due to privacy. You can download the original NYC Taxi Trip Duration dataset from Kaggle [NYC Taxi Trip Duration](httpswww.kaggle.comcompetitionsnyc-taxi-trip-durationoverview)

#### Once downloaded, place the CSV files into a folder named `dataraw` in the repository, or adjust paths in the notebooks accordingly.
---

## Setup Instructions

1. Clone the repository

```bash
git clone httpsgithub.comEng-MoazNYC-Taxi-Trip-Duration.git
cd NYC-Taxi-Trip-Duration
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Mount Google Drive (for Colab usage)

```python
from google.colab import drive
drive.mount('contentdrive')
```

4. Run the modeling notebook

Open `notebookspreprocessing.ipynb` and `notebooksmodeling.ipynb` to preprocess data, train models, evaluate performance, and save results.

---

## Models

Baseline Models  

- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Random Forest (default parameters)  

Candidate Models (Tree-based Ensembles)  

- Gradient Boosting  
- XGBoost  
- LightGBM  
- CatBoost  

Final Model Catboost (chosen based on validation metrics MAE, RMSE, R²)

---

## Evaluation Metrics

- MAE – Mean Absolute Error  
- RMSE – Root Mean Squared Error  
- R² – Coefficient of Determination
- 
### Performance on Validation Set (Sample)

| Model                | MAE     | RMSE    | R²      |
|----------------------|---------|---------|---------|
| Linear Regression    | 0.29978 | 0.41291 | 0.68268 |
| Ridge                | 0.30007 | 0.41420 | 0.68069 |
| Lasso                | 0.42823 | 0.54735 | 0.44241 |
| RandomForest_Default | 0.02094 | 0.04404 | 0.99639 |
| XGBoost              | 0.02444 | 0.03811 | 0.99730 |
| LightGBM             | 0.02824 | 0.04418 | 0.99637 |
| CatBoost             | 0.02081 | 0.03042 | 0.99828 |
| Final Model (CatBoost)| 0.01297 | 0.01993 | 0.99918 |


---

## Saving Models and Results

- Baseline results → `resultsbaselinesbaseline_results.json`  
- Candidate model results → `resultscandidatescandidate_results.json`  
- Final model → `resultsfinalfinal_model.pkl`  
- Final evaluation on test set → `resultsfinaltesting_results.json`  

---

## Usage Example

Preprocess data using `srcpreprocessing.py`, train models, and make predictions

```python
import joblib
model = joblib.load('resultsfinalfinal_model.pkl')
preds = model.predict(X_test)
```

---

## Dependencies

- Python ≥ 3.8  
- pandas, numpy, matplotlib, seaborn  
- scikit-learn  
- xgboost, lightgbm, catboost  

---

## Notes

- CatBoost training artifacts (`catboost_info`) are excluded from version control via `.gitignore`.  
- Hyperparameter tuning can be added, but current validation scores are already very high.  
- The repository is structured for professional ML workflow, ensuring reproducibility and model tracking.


