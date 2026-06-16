# `main.py` Logic & Project Structure
* load datas -> convert to quarterly -> merge and add time series features -> combine to final dataset -> model training & prediction -> evaluation & visualization


# Data Merging & Feature Engineering Logic

| $t - 1$ |   $t$   | $t + 1$ |
|:-------:|:-------:|:-------:|
|OP, ESG, Market | Transactions, Technology | $\hat{y}$ |


# Terms

| Term    | Definition |
|:-------:|:-----------|
|  Model  | Dataset that is ready for model training and prediction. |
|Algorithm| Model training and prediction logic (e.g., Random Forest, XGBoost, KNN, OLS, Lasso, Ridge). |
|  Sample  | Model + Parameters |