# NYC Taxi Trip Duration Prediction

A machine learning project to predict NYC taxi trip durations using historical trip data. This project demonstrates a complete ML pipeline including data cleaning, exploratory data analysis, feature engineering (temporal and geospatial), and modeling with various algorithms such as Linear Regression, Random Forest, XGBoost, and LightGBM. The focus is on feature importance analysis, model interpretability, and RMSE optimization.

## Overview

This project aims to predict the duration of taxi trips in New York City based on various features such as pickup/dropoff locations, time of day, date, and other trip characteristics. The prediction model can help taxi companies optimize their operations, improve customer experience, and better estimate arrival times.

**Key Features:**
- Complete data preprocessing pipeline
- Comprehensive exploratory data analysis
- Advanced feature engineering including temporal and geospatial features
- Multiple machine learning algorithms comparison
- Model performance evaluation and interpretation
- Focus on achieving optimal RMSE scores

## Dataset

The project uses NYC taxi trip data which includes:
- **Pickup datetime**: Timestamp when the trip started
- **Pickup/Dropoff coordinates**: Geographical coordinates (longitude, latitude)
- **Passenger count**: Number of passengers in the trip
- **Trip duration**: Target variable (in seconds)

**Data Source**: The dataset is typically sourced from the NYC Taxi and Limousine Commission (TLC) trip record data.

## Exploratory Data Analysis (EDA)

The EDA phase includes:
- **Data quality assessment**: Missing values, outliers, and data distribution analysis
- **Temporal patterns**: Trip duration variations by hour, day, month, and season
- **Geospatial analysis**: Popular pickup/dropoff locations and route patterns
- **Feature correlations**: Relationships between different variables
- **Statistical summaries**: Descriptive statistics and data visualization

Key insights discovered through EDA will guide feature engineering and model selection decisions.

## Feature Engineering

Advanced feature engineering techniques applied:

**Temporal Features:**
- Hour of day, day of week, month, season
- Holiday indicators
- Rush hour flags
- Weekend/weekday classification

**Geospatial Features:**
- Haversine distance between pickup and dropoff points
- Direction of travel (bearing)
- Clustering of popular locations
- Proximity to landmarks and airports

**Derived Features:**
- Speed calculations
- Traffic density indicators
- Weather integration (if available)

## Modeling

Multiple machine learning algorithms are implemented and compared:

1. **Linear Regression**: Baseline model for interpretability
2. **Random Forest**: Ensemble method for capturing non-linear relationships
3. **XGBoost**: Gradient boosting for high performance
4. **LightGBM**: Efficient gradient boosting variant

**Model Evaluation:**
- Primary metric: Root Mean Square Error (RMSE)
- Cross-validation for robust performance estimation
- Feature importance analysis
- Model interpretability using SHAP values

## Results

Performance comparison of different models:
- Model accuracy metrics (RMSE, MAE, R²)
- Feature importance rankings
- Computational efficiency analysis
- Prediction examples and case studies

*Detailed results and visualizations will be updated as the project progresses.*

## Next Steps

Future improvements and extensions:
- [ ] Integration of real-time traffic data
- [ ] Weather data incorporation
- [ ] Deep learning approaches (neural networks)
- [ ] Ensemble methods combining multiple models
- [ ] Real-time prediction API development
- [ ] Model deployment considerations
- [ ] A/B testing framework for model validation

## Project Structure

```
NYC-Taxi-Trip-Duration/
├── data/                   # Data files (raw, processed, external)
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code modules
├── models/                 # Trained model files
├── reports/                # Generated reports and visualizations
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── .gitignore             # Git ignore rules
```

## Installation and Usage

1. Clone the repository:
```bash
git clone https://github.com/Eng-Moaz/NYC-Taxi-Trip-Duration.git
cd NYC-Taxi-Trip-Duration
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Follow the notebooks in order for complete analysis pipeline.
