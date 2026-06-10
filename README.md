# Financial & ESG Stock Prediction Project

This project fetches, processes, and merges multiple data sources—including daily stock transactions, macroeconomic indicators, technology indices, and ESG (Environmental, Social, and Governance) data—to train machine learning models. 

The pipeline automatically engineers time-series features (QoQ, YoY, moving averages, etc.) and evaluates multiple models (Random Forest, XGBoost, KNN, OLS, Lasso, Ridge) to predict quarterly stock returns and volatility.

## 📂 Project Structure

Before running the code, ensure your project directory looks like this:

```text
your_project_folder/
│
├── project.py             # Main execution script
├── function.py            # Helper functions for data processing and modeling
├── requirements.txt       # List of Python dependencies
├── .env                   # Configuration file for API keys (Needs to be created)
│
├── file/                  # Directory for local dataset inputs
│   └── FinMind_transaction_2000_2026.csv  (or the 2019_2025 version)
│
├── models/                # Directory where final modeling datasets will be saved
└── results/               # Directory where model evaluation and feature importance logs will be saved

```

## 🛠️ Prerequisites & Installation

### 1. Python Environment

Ensure you have Python 3.7+ installed. It is highly recommended to use a virtual environment to avoid dependency conflicts.

### 2. Setup Virtual Environment

Create a virtual environment named `.venv` in your project folder:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

```bash
# For MacOS / Linux
source .venv/bin/activate

# For Windows
.venv\Scripts\activate
```

### 3. Install Required Packages

With the virtual environment activated, install all necessary Python libraries:

```bash
pip3 install -r requirements.txt
```

### 4. Setup FRED API Key (.env)

This project uses the FRED (Federal Reserve Economic Data) API to fetch macroeconomic indices.

1. Get a free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html).
2. Create a file named `.env` in the same directory as `project.py`.
3. Add your API key to the `.env` file like this:

```text
FRED_API_KEY="<your_actual_api_key_here>"
```

## 🚀 How to Run

Once your folders are set up and dependencies are installed, you can execute the entire pipeline with a single command:

```bash
python3 project.py
```

### What to expect during execution:

1. **Data Loading:** The script will download datasets from Google Drive/Sheets and read your local transaction CSV. *(Note: Downloading large files from Google Drive may take a moment).*
2. **Data Transformation:** It will convert daily/monthly/annual data into a unified quarterly format.
3. **Feature Engineering:** Creates lagged variables, QoQ/YoY growth rates, and volatility metrics.
4. **Modeling:** Trains and evaluates Random Forest, XGBoost, KNN, and Linear Regression models.
5. **Output:** Intermediate datasets used for training will be saved in the `./models/` directory (e.g., `Final_Model1.csv`).
* Detailed model evaluation metrics (MSE, MAE, R-Squared, Adj-R-Squared) and feature importance rankings will be saved in the `./results/` directory with a timestamp.



## ⚙️ Configuration

You can customize the pipeline execution by modifying the global variables at the top of `project.py`.

**1. Dataset Size, ESG Toggle, and Feature Count:**

```python
IS_LARGE_DATASET = True  # Set to False to use the smaller 2018-2025 dataset
HAS_ESG = True           # Toggle ESG data inclusion
TOP_N_FEATURES = 20      # Number of top features to retain for model evaluation
```

**2. Date Range Adjustment:**
Modify the date ranges for training and holdout periods within the conditional blocks:

```python
if IS_LARGE_DATASET:
    START_DATE = "1998-01-01"
    END_DATE = "2025-12-31"
    # Adjust TRAIN and HOLDOUT periods accordingly...
else:
    START_DATE = "2018-01-01"
    END_DATE = "2025-12-31"
    # Adjust TRAIN and HOLDOUT periods accordingly...
```

## Update Requirements
After installing new packages or updating existing ones, make sure to update your `requirements.txt` file:

```
pip3 freeze > requirements.txt
```

---
## References
- [Python Installation and Virtual Environment on macOS](https://medium.com/@jarebox1917/python-macos-安裝-python-詳細步驟-教學筆記-7f7c7e8cd5c7)