import os
import src.utility_function as myFn
from dotenv import load_dotenv

IS_DEBUG = False

HAS_ESG = False
IS_LARGE_DATASET = True
HAS_BACKUP_REGRESSORS = False

TOP_N_FEATURES = 20

RF_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 10, 15],
    "min_samples_leaf": [5, 10, 20],
    "max_features": ["sqrt", "log2", 1.0]
}

XGB_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1]
}

# KNN_GRID = {
#     "n_neighbors": [3, 5, 7],
#     "weights": ["uniform", "distance"],
#     "metric": ["euclidean", "manhattan"]
# }

KNN_GRID = {
    "n_neighbors": [5],
    "weights": ["distance"]
}

LR_GRID = {
    "OLS": {
        "fit_intercept": [True, False] 
    },
    "Lasso": {
        # "alpha": [0.001, 0.005, 0.01, 0.1, 1.0] 
        "alpha": [0.005]
    },
    "Ridge": {
        # "alpha": [0.1, 1.0, 10.0, 100.0]
        "alpha": [1.0]
    }
}

load_dotenv()
try:
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    if not FRED_API_KEY:
        print("API Token not loaded.")
except Exception as e:
    print(f"Fail to load API key: {e}")
    FRED_API_KEY = ""

# Set date range
if IS_LARGE_DATASET:
    START_DATE = "1998-01-01"
    END_DATE = "2025-12-31"

    TRAIN_START_DATE = "2000-01-01"
    TRAIN_END_DATE = "2024-12-31"

    HOLDOUT_START_DATE = "2025-01-01"
    HOLDOUT_END_DATE = "2025-12-31"

else:
    START_DATE = "2018-01-01"
    END_DATE = "2025-12-31"

    TRAIN_START_DATE = "2020-01-01"
    TRAIN_END_DATE = "2025-06-30"

    HOLDOUT_START_DATE = "2025-07-01"
    HOLDOUT_END_DATE = "2025-12-31"

TRAINING_START_YEAR, TRAINING_END_YEAR, TRAINING_START_QUARTER, TRAINING_END_QUARTER = myFn.set_training_periods(TRAIN_START_DATE, TRAIN_END_DATE)
HOLDOUT_START_YEAR, HOLDOUT_END_YEAR, HOLDOUT_START_QUARTER, HOLDOUT_END_QUARTER = myFn.set_holdout_periods(HOLDOUT_START_DATE, HOLDOUT_END_DATE)

# Set template file name
start_year_label = str((TRAINING_START_YEAR % 100)).zfill(2)
end_year_label = str((HOLDOUT_END_YEAR % 100)).zfill(2)

if HAS_ESG:
    TEMPLATE_DIR_NAME = f"{start_year_label}_{end_year_label}_ESG"
else:
    TEMPLATE_DIR_NAME = f"{start_year_label}_{end_year_label}_NoESG"