# Financial & ESG Stock Prediction Project

This project fetches, processes, and merges multiple data sources—including daily stock transactions, macroeconomic indicators, technology indices, and ESG (Environmental, Social, and Governance) data—to train machine learning models. 

The pipeline automatically engineers time-series features (QoQ, YoY, moving averages, etc.) and evaluates multiple models (Random Forest, XGBoost, KNN, OLS, Lasso, Ridge) to predict quarterly stock returns and volatility.

---

## 📂 Project Structure

Before running the code, ensure your project directory looks like this:

```text
Final_Project/
├── .env                    # Stores API_KEYs (NEVER upload to GitHub)
├── requirements.txt        # Records project package versions (pandas, scikit-learn, fredapi, groq, etc.)
├── config.py               # Centralizes management of all global variables, date ranges, and toggles (HAS_ESG, etc.)
├── main.py                 # The main orchestrator (Project entry point)
│
├── src/                    # Core algorithms and logic modules
│   ├── __init__.py         # Makes 'src' an importable package (can be left empty)
│   ├── data_loader.py      # Responsible for fetching all external data (FRED API, Google Drive reads)
│   ├── model_cleaner.py    # Handles frequency conversion, feature engineering, and data cleaning/preprocessing
│   ├── model_runner.py     # Responsible for model training and prediction
│   └── function.py         # Stores utility/helper functions
│
├── data/                   # Dataset repository
│   ├── raw/                # Original downloaded files (Not uploaded to GitHub)
│   └── processed/          # Cleaned data ready for direct access/use
│
└── experiments/            # Dedicated directory for storing execution results
    ├── <start_year>_<end_year>_<has_esg>_<exe_date>_<exe_time>/
    │   ├── models/         # Stores Model1~4.csv or .pkl model files generated during this run
    │   ├── figures/        # Stores feature importance charts (.png) generated during this run
    │   ├── evaluation.csv  # Evaluation results table for all parameter combinations from this run
    │   ├── importance.csv  # Feature importance ranking table from this run
    │   └── config.json     # The "ID card" of this experiment (records parameters and variable configurations at the time)
    │
    └──  ...                # Each execution generates a new directory, ensuring results are never overwritten
```

---

## 🛠️ Prerequisites & Installation

### 1. Environment Setup

Ensure you have **Python 3.7+** installed. It is highly recommended to use a virtual environment to avoid dependency conflicts.

```bash
# Create a virtual environment named .venv
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate     # For MacOS / Linux:
.venv\Scripts\activate        # For Windows:
```

### 2. Install Dependencies

With the virtual environment activated, install all necessary Python libraries:

```bash
pip3 install -r requirements.txt
```

### 3. Configure Environment Variables (`.env`)

This project uses the FRED (Federal Reserve Economic Data) API to fetch macroeconomic indices.

1. Get a free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html).
2. Create a file named `.env` in the root directory (same folder as `main.py`).
3. Add your API key to the file:

```text
FRED_API_KEY = "<your_actual_api_key_here>"
```

---

## 🚀 How to Run

Execute the entire pipeline with a single command:

```bash
python3 main.py
```

### Pipeline Workflow & Outputs

1. **Data Loading:** Fetches datasets from Google Drive/Sheets and reads local transaction CSVs. *(Note: Downloading large files may take a moment).*
2. **Data Transformation:** Converts daily/monthly/annual data into a unified quarterly format.
3. **Feature Engineering:** Creates lagged variables, QoQ/YoY growth rates, and volatility metrics.
4. **Modeling & Evaluation:** Trains and evaluates Random Forest, XGBoost, KNN, and Linear Regression models.
5. **Artifact Generation:** * A new timestamped folder will be automatically created under `experiments/` for each run.
* Trait datasets, trained models (`.pkl`), evaluation metrics (`evaluation.csv`), and feature importance charts (`.png`) will be saved there. **Previous results will never be overwritten.**



---

## ⚙️ Configuration

You can customize the pipeline execution by modifying the global variables in `config.py`.

### Key Parameters

| Variable | Type | Description |
| --- | --- | --- |
| `IS_DEBUG` | Boolean | `True` for quick testing with subset data; `False` for full run. |
| `HAS_ESG` | Boolean | Enable/Disable ESG data integration. |
| `HAS_TECH_IDX` | Boolean | Enable/Disable technology indices features. |
| `IS_LARGE_DATASET` | Boolean | Toggle between large historical data (1998+) and small data (2018+). |
| `TOP_N_FEATURES` | Integer | Number of top features to keep after feature selection. |

### Hyperparameter Tuning Grids

You can also tweak the hyperparameter grids for models directly in `config.py`:

```python
RF_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 10, 15],
    "min_samples_leaf": [5, 10, 20],
    "max_features": ["sqrt", "log2"]
}

XGB_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1]
}
```

---

## 📝 Maintenance

### Update Requirements

After installing new packages or updating existing ones, make sure to update your `requirements.txt` file:

```bash
pip3 freeze > requirements.txt
```

---

## 🔗 References

* [Python Installation and Virtual Environment on macOS](https://medium.com/@jarebox1917/python-macos-安裝-python-詳細步驟-教學筆記-7f7c7e8cd5c7)
```