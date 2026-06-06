import os
import numpy as np
import pandas as pd
import function as myfn
from dotenv import load_dotenv
from fredapi import Fred
from datetime import datetime

HAS_ESG = False
IS_LARGE_DATASET = True
TOP_N_FEATURES = 20

# Global Variables & Configurations
if IS_LARGE_DATASET:
    START_DATE = "1998-01-01"
    END_DATE = "2025-12-31"

    TRAIN_START_DATE = "2000-01-01"
    TRAIN_END_DATE = "2024-12-31"

    HOLDOUT_START_DATE = "2025-01-01"
    HOLDOUT_END_DATE = "2025-12-31"

    path_transaction = "./file/FinMind_transaction_2000_2026.parquet"    
else:
    START_DATE = "2018-01-01"
    END_DATE = "2025-12-31"

    TRAIN_START_DATE = "2020-01-01"
    TRAIN_END_DATE = "2025-06-30"

    HOLDOUT_START_DATE = "2025-07-01"
    HOLDOUT_END_DATE = "2025-12-31"

    path_transaction = "./file/FinMind_transaction_2019_2025.parquet"

TWSE_BASE_URL = "https://openapi.twse.com.tw/v1"

myfn.set_top_n_features(n = TOP_N_FEATURES)
# Set up training and holdout periods for the project using the function module. This will allow the functions to reference these periods when processing data and training models.
myfn.set_training_periods(training_start_date = TRAIN_START_DATE, training_end_date = TRAIN_END_DATE)
myfn.set_holdout_periods(holdout_start_date = HOLDOUT_START_DATE, holdout_end_date = HOLDOUT_END_DATE)
myfn.set_esg_large_dataset_flags(esg_flag = HAS_ESG, large_dataset_flag = IS_LARGE_DATASET)

# Check Dictionary
print("##### Configuration Check #####")
for directory in ["./figures", "./results", "./models"]:
    os.makedirs(directory, exist_ok = True)
    print(f"Directory '{directory}' is ready.")

print("Configuration Check Complete.\n")

# FRED API
load_dotenv()
try:
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    if not FRED_API_KEY:
        print("API Token not loaded.")
except Exception as e:
    print(f"Fail to load API key: {e}")
    FRED_API_KEY = ""

print("##### Start Loading Data #####")

print("\nTarget Period:", START_DATE, "to", END_DATE)
print(f"Training Period: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
print(f"Holdout Period: {HOLDOUT_START_DATE} to {HOLDOUT_END_DATE}")

# ==========================================
# 4.1 Daily Transaction Data
# ==========================================

# df_daily_transaction = myfn.get_file_from_drive("https://drive.google.com/file/d/1YI_wwPSRQBzAOlLLaRxIGp2twCoYMBWV/view?usp=sharing")
df_daily_transaction = pd.read_parquet(path_transaction)
df_daily_transaction["Date"] = df_daily_transaction["Date"].apply(myfn.convert_date)
# df_daily_transaction = df_daily_transaction[df_daily_transaction["Date"].between(START_DATE, END_DATE)]
df_daily_transaction["StockID"] = df_daily_transaction["StockID"].astype(str).str.strip()
df_daily_transaction["ClosingPrice"] = pd.to_numeric(df_daily_transaction["ClosingPrice"], errors = "coerce")
df_daily_transaction["TradingVolume"] = pd.to_numeric(df_daily_transaction["TradingVolume"], errors = "coerce")
df_daily_transaction["TradingMoney"] = pd.to_numeric(df_daily_transaction["TradingMoney"], errors = "coerce")
df_daily_transaction["TradingTurnover"] = pd.to_numeric(df_daily_transaction["TradingTurnover"], errors = "coerce")
df_daily_transaction["Spread"] = pd.to_numeric(df_daily_transaction["Spread"], errors = "coerce")
df_daily_transaction = df_daily_transaction.dropna()
df_daily_transaction = df_daily_transaction.sort_values(by = ["StockID", "Date"])
myfn.describe_data(df_daily_transaction, "Daily Transaction Data")

# ==========================================
# 4.2 Company Category
# ==========================================

df_company_category = myfn.get_file_from_drive("https://docs.google.com/spreadsheets/d/1dJqfJVqryl2XhHoW9Fa01SXzxVkHpfnDQTaMF58Urh4/edit?gid=84991497#gid=84991497")
df_company_category = df_company_category.rename(columns = {"Code": "StockID"})
df_company_category["StockID"] = df_company_category["StockID"].astype(str).str.strip()
df_company_category = df_company_category[["StockID", "SubCategory"]].copy()
myfn.describe_data(df_company_category, "Company Category Data")

# ==========================================
# 4.3 Tech Index (FRED & Github)
# ==========================================

fred = Fred(api_key = FRED_API_KEY)
series_to_concat = {}
try:
    series_to_concat["VIX"] = fred.get_series("VIXCLS")
    series_to_concat["SOX"] = fred.get_series("NASDAQSOX")
    series_to_concat["DJIA"] = fred.get_series("DJIA")
    series_to_concat["IXIC"] = fred.get_series("NASDAQCOM")
    series_to_concat["SP500"] = fred.get_series("SP500")
except Exception as e:
    print(f"FRED API request error: {e}")

try:
    data_FnG = pd.read_csv("https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/refs/heads/main/fear-greed.csv")
    data_FnG["Date"] = pd.to_datetime(data_FnG["Date"], errors = "coerce")
except Exception as e:
    print(f"FnG request error: {e}")
    data_FnG = pd.DataFrame(columns = ["Fear Greed", "Rating"])

df_tech_idx = pd.concat(series_to_concat, axis = 1)
df_tech_idx = df_tech_idx.reset_index()
df_tech_idx = df_tech_idx.rename(columns = {"index": "Date"})

if not data_FnG.empty:
    data_FnG = data_FnG.rename(columns = {"Fear Greed": "FnG", "Rating": "FnGRating"})
    df_tech_idx = pd.merge(
        df_tech_idx,
        data_FnG[["Date", "FnG"]],
        on = "Date",
        how = "left"
    )
df_tech_idx["Date"] = df_tech_idx["Date"].apply(myfn.convert_date)
df_tech_idx = df_tech_idx.dropna()
# df_tech_idx = df_tech_idx[df_tech_idx["Date"].between(START_DATE, END_DATE)]
myfn.describe_data(df_tech_idx, "Technology Index Data")

# ==========================================
# 4.4 Macro (Rediscount, CPI, Unemp)
# ==========================================

df_rediscount_rate = myfn.get_file_from_drive("https://docs.google.com/spreadsheets/d/1qy3yNGYnj5vPCCCDd7XtzZ5wr2eDm-zw9ChmwgvuUAE/edit?gid=834914550#gid=834914550")
df_rediscount_rate["Duration"] = df_rediscount_rate["Duration"].apply(myfn.convert_date)
myfn.describe_data(df_rediscount_rate, "Rediscount Rate Data")

df_cpi = myfn.get_file_from_drive("https://docs.google.com/spreadsheets/d/1CHlZ0XWqH0e1TlRbBu7t43tzK4Wu_pHcjIWwKB_Hv7g/edit?usp=sharing")
df_cpi["Duration"] = df_cpi["Duration"].apply(myfn.convert_date)
df_cpi = df_cpi[["Duration", "OverallIndex"]].copy()
myfn.describe_data(df_cpi, "CPI Data")

df_unemployment_rate = myfn.get_file_from_drive("https://docs.google.com/spreadsheets/d/1OuZ-xTaNkT0nCx-eb0ABtThWDmaLYYRMTja9yJrqwYQ/edit?gid=606389627#gid=606389627")
df_unemployment_rate["Duration"] = df_unemployment_rate["Duration"].apply(myfn.convert_date)
myfn.describe_data(df_unemployment_rate, "Unemployment Rate Data")

# ====================================================================================
# 4.5 Financial Data (EPS, Revenue, ROE&ROA&GrossMargin)
# ====================================================================================

df_eps = myfn.get_file_from_drive("https://drive.google.com/file/d/1TnuzhXPyIpJJtKM3vxjkF9d9GqQWN4mj/view?usp=sharing")
df_eps["Duration"] = df_eps["Duration"].apply(myfn.convert_date)
df_eps = df_eps[["StockID", "Duration", "EPS"]].copy()
df_eps = df_eps.sort_values(by = ["StockID", "Duration"])
myfn.describe_data(df_eps, "EPS Data")

df_revenue = myfn.get_file_from_drive("https://drive.google.com/file/d/11IFSInZP7sJmxYzxi8mPb6Akt4FoY5g0/view?usp=sharing")
df_revenue["Duration"] = df_revenue["Duration"].apply(myfn.convert_date)
myfn.describe_data(df_revenue, "Revenue Data")

df_roe_roa_grossmargin = myfn.get_file_from_drive("https://docs.google.com/spreadsheets/d/1ngHwal_vAQ4IQeON8vT0D0IMkzz3cOseiDROuU4KqK4/edit?pli=1&gid=1778576688#gid=1778576688")
df_roe_roa_grossmargin["Duration"] = df_roe_roa_grossmargin["Duration"].apply(myfn.convert_date)
myfn.describe_data(df_roe_roa_grossmargin, "ROE, ROA, Gross Margin Data")

# ==========================================
# 4.6 ESG Data
# ==========================================

esg_file_list = {
    "green_gas_emission": "https://drive.google.com/file/d/1dLyrCvc7_-Ly8MVqj_UPiZF0CN_X_58z/view?usp=share_link",
    "climate_related_issues_management": "https://drive.google.com/file/d/1cq66js-9UHBieVHA37BvQTauhRHuo8Te/view?usp=share_link",
    "energy_management": "https://drive.google.com/file/d/1W8cHBtTPgldiUI2nd5lnjmlaL5G52G1z/view?usp=share_link",
    "water_resource_management": "https://drive.google.com/file/d/1qe1WUxERBphKk0Mqohtd-rETAN28Q1pO/view?usp=share_link",
    "waste_management": "https://drive.google.com/file/d/10dTzpsMqEbjK_uYNXzMIzxHKGku6Na8Y/view?usp=share_link",
    "human_resource_management": "https://drive.google.com/file/d/1kjrc2_aWSo61E4Ay5i1lxQNGGnd2iobo/view?usp=share_link",
    "occupational_health_and_safety": "https://drive.google.com/file/d/1_ssFF-JoW5YrDUK6jyxIDbO53zzhU_EZ/view?usp=share_link",
    "board_of_directors": "https://drive.google.com/file/d/19mwCXiPAw0OxZCyXx7GebJSOho7xjS3j/view?usp=share_link",
    "functional_committees": "https://drive.google.com/file/d/1qM4o4A58NXxBOrcInQjeu048kb0r0mxb/view?usp=sharing_link",
    "shareholding_and_control": "https://drive.google.com/file/d/1qyRDxd33_rgwkc5WRK7_hlZ-_AGyO4dx/view?usp=share_link",
    "investor_communication": "https://drive.google.com/file/d/1ZcXJXJZav9Bo4Q6vnbt15nilBdD7oT4K/view?usp=share_link"
}

df_esg = []

for index, (esg_topic, file_url) in enumerate(esg_file_list.items()):
    print(f"[{index + 1}/{len(esg_file_list)}] Processing: {esg_topic}")

    df_temp = myfn.get_file_from_drive(file_url)
    df_temp = df_temp.rename(columns = {"Code": "StockID", "Category": "ESGCategory"})
    df_esg.append(df_temp)

df_esg = pd.concat(df_esg, ignore_index = True)
df_esg = df_esg.rename(columns = {"Code": "StockID", "Category": "ESGCategory"})

# wide -> long
df_esg = df_esg.melt(
    id_vars = ["StockID", "ESGCategory"],
    var_name = "Duration",
    value_name = "IsDisclosed"
)

df_esg["Duration"] = df_esg["Duration"].astype(str)

df_esg = df_esg.drop_duplicates(subset = ["Duration", "StockID", "ESGCategory"])

# long -> wide
df_esg = df_esg.pivot(
    index = ["Duration", "StockID"],
    columns = "ESGCategory",
    values = "IsDisclosed"
).reset_index()
df_esg.columns.name = None
df_esg = df_esg.rename(columns = {"Duration": "Duration", "StockID": "StockID"})

myfn.describe_data(df_esg, "ESG Data (Long Format)")

# ==========================================
# 5. Convert to Quarterly
# ==========================================
print("##### Converting to Quarterly #####")

# Annually -> Quarterly
esg_category_cols = [col for col in df_esg.columns if col not in ["Duration", "StockID"]]
df_esg_quarterly = myfn.annual_to_quarterly(df_esg, target_cols = esg_category_cols, method = "duplicate")
myfn.describe_data(df_esg_quarterly, "ESG Data (Quarterly)")

# Quarterly -> Quarterly
df_eps_quarterly = myfn.quarter_to_quarterly(df_eps, target_cols = ["EPS"])
df_revenue_quarterly = myfn.quarter_to_quarterly(df_revenue, target_cols = ["Last3mCumulativeRevenue", "Last3mCumulativeRevenueGrowthRate"])
df_roe_roa_grossmargin_quarterly = myfn.quarter_to_quarterly(df_roe_roa_grossmargin, target_cols = ["ROE", "ROA", "GrossMargin"])

# Monthly -> Quarterly
df_cpi_quarterly = myfn.month_to_quarterly(df_cpi, target_cols = ["OverallIndex"])
df_unemployment_rate_quarterly = myfn.month_to_quarterly(df_unemployment_rate, target_cols = ["TotalUnempPct"])
df_rediscount_rate_quarterly = myfn.month_to_quarterly(df_rediscount_rate, target_cols = ["RediscountRate"])

# Daily -> Quarterly
df_daily_transaction_tmp_sum = myfn.daily_to_quarterly(df_daily_transaction, target_cols = ["TradingVolume", "TradingMoney", "TradingTurnover"], method = "sum")
df_daily_transaction_tmp_mean = myfn.daily_to_quarterly(df_daily_transaction, target_cols = ["ClosingPrice", "Spread"], method = "mean")

df_transaction_quarterly = pd.merge(
    df_daily_transaction_tmp_sum,
    df_daily_transaction_tmp_mean,
    on = ["Year", "Quarter", "StockID"],
    how = "inner"
)
df_tech_idx_quarterly = myfn.daily_to_quarterly(df_tech_idx, target_cols = ["VIX", "SOX", "DJIA", "IXIC", "SP500", "FnG"], method = "mean")

# ====================================================================================
# Split, Merge and Extract Time Series Features
# ====================================================================================

# DV: y_{t + 1}
df_dv = df_transaction_quarterly[["StockID", "Year", "Quarter", "ClosingPrice"]].copy()
df_dv["QuarterlyReturn"] = df_dv.groupby("StockID")["ClosingPrice"].pct_change()
df_dv["QuarterlyReturn_Lag1"] = df_dv.groupby("StockID")["QuarterlyReturn"].shift(1)
df_dv["QuarterlyReturn_Lag2"] = df_dv.groupby("StockID")["QuarterlyReturn"].shift(2)
df_dv["QuarterlyReturn_Lag3"] = df_dv.groupby("StockID")["QuarterlyReturn"].shift(3)

window_size = 4
df_dv["QuarterlyLogReturn"] = np.log(df_dv["ClosingPrice"] / df_dv.groupby("StockID")["ClosingPrice"].shift(1))
df_dv["QuarterlyVolatility"] = df_dv.groupby("StockID")["QuarterlyLogReturn"].rolling(window = window_size).std().reset_index(0, drop = True)
df_dv["QuarterlyVolatility_Lag1"] = df_dv.groupby("StockID")["QuarterlyVolatility"].shift(1)
df_dv["QuarterlyVolatility_Lag2"] = df_dv.groupby("StockID")["QuarterlyVolatility"].shift(2)
df_dv["QuarterlyVolatility_Lag3"] = df_dv.groupby("StockID")["QuarterlyVolatility"].shift(3)
myfn.describe_data(df_dv, "Dependent Variable Data (Quarterly)", n = 10)

# IV: Operating Performance
df_op_quarterly = pd.merge(
    df_eps_quarterly,
    df_revenue_quarterly,
    left_on = ["StockID", "Year", "Quarter"],
    right_on = ["StockID", "Year", "Quarter"]
)
df_op_quarterly = pd.merge(
    df_op_quarterly,
    df_roe_roa_grossmargin_quarterly,
    left_on = ["StockID", "Year", "Quarter"],
    right_on = ["StockID", "Year", "Quarter"]
)
df_op_quarterly = pd.merge(
    df_op_quarterly,
    df_company_category,
    on = "StockID",
    how = "left"
)
df_op_quarterly.dropna(inplace = True)
myfn.describe_data(df_op_quarterly, "Operating Performance Data (Quarterly)")

# IV: Transaction Data
cols = ["Year", "Quarter", "StockID", "TradingVolume", "TradingMoney", "TradingTurnover", "Spread"]
df_transaction_quarterly = df_transaction_quarterly[cols].copy()
myfn.describe_data(df_transaction_quarterly, "Transaction Data (Quarterly)")

# IV: Technical Index
myfn.describe_data(df_tech_idx_quarterly, "Technology Index Data (Quarterly)")

df_tech_idx_features_QoQ = myfn.calculate_time_series_features(df_tech_idx_quarterly, target_cols = ["VIX", "SOX", "DJIA", "IXIC", "SP500", "FnG"], method = "QoQ")
myfn.describe_data(df_tech_idx_features_QoQ, "Technology Index Features (QoQ) (Quarterly)")
df_tech_idx_features_YoY = myfn.calculate_time_series_features(df_tech_idx_quarterly, target_cols = ["VIX", "SOX", "DJIA", "IXIC", "SP500", "FnG"], method = "YoY")
myfn.describe_data(df_tech_idx_features_YoY, "Technology Index Features (YoY) (Quarterly)")


# IV: Market Index
df_market_idx_quarterly = pd.merge(
    df_cpi_quarterly,
    df_unemployment_rate_quarterly,
    left_on = ["Year", "Quarter"],
    right_on = ["Year", "Quarter"]
)
df_market_idx_quarterly = pd.merge(
    df_market_idx_quarterly,
    df_rediscount_rate_quarterly,
    left_on = ["Year", "Quarter"],
    right_on = ["Year", "Quarter"]
)
df_market_idx_quarterly.dropna(inplace = True)
myfn.describe_data(df_market_idx_quarterly, "Market Index Data (Quarterly)")

df_market_idx_features_QoQ = myfn.calculate_time_series_features(df_market_idx_quarterly, target_cols = ["OverallIndex", "TotalUnempPct", "RediscountRate"], method = "QoQ")
myfn.describe_data(df_market_idx_features_QoQ, "Market Index Features (QoQ) (Quarterly)")
df_market_idx_features_YoY = myfn.calculate_time_series_features(df_market_idx_quarterly, target_cols = ["OverallIndex", "TotalUnempPct", "RediscountRate"], method = "YoY")
myfn.describe_data(df_market_idx_features_YoY, "Market Index Features (YoY) (Quarterly)")


# ==========================================
# Modeling Dataset
# ==========================================

print("##### Modeling Datasets #####")

df_op_lag = df_op_quarterly.copy()
df_op_lag["TargetYear"] = np.where(df_op_lag["Quarter"] == 4, df_op_lag["Year"] + 1, df_op_lag["Year"])
df_op_lag["TargetQuarter"] = np.where(df_op_lag["Quarter"] == 4, 1, df_op_lag["Quarter"] + 1)
op_cols = [c for c in df_op_lag.columns if c not in ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "SubCategory"]]
rename_esg = {c: f"{c}_last1" for c in op_cols}
df_op_lag = df_op_lag.rename(columns = rename_esg)
df_op_lag = df_op_lag[["StockID", "TargetYear", "TargetQuarter"] + list(rename_esg.values())]

df_esg_lag = df_esg_quarterly.copy()
df_esg_lag["TargetYear"] = np.where(df_esg_lag["Quarter"] == 4, df_esg_lag["Year"] + 1, df_esg_lag["Year"])
df_esg_lag["TargetQuarter"] = np.where(df_esg_lag["Quarter"] == 4, 1, df_esg_lag["Quarter"] + 1)
esg_cols = [c for c in df_esg_lag.columns if c not in ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "SubCategory"]]
rename_esg = {c: f"ESG_{c}_last1" for c in esg_cols}
df_esg_lag = df_esg_lag.rename(columns = rename_esg)
df_esg_lag = df_esg_lag[["StockID", "TargetYear", "TargetQuarter"] + list(rename_esg.values())]

df_market_QoQ_lag = df_market_idx_quarterly.copy()
df_market_QoQ_lag["TargetYear"] = np.where(df_market_QoQ_lag["Quarter"] == 4, df_market_QoQ_lag["Year"] + 1, df_market_QoQ_lag["Year"])
df_market_QoQ_lag["TargetQuarter"] = np.where(df_market_QoQ_lag["Quarter"] == 4, 1, df_market_QoQ_lag["Quarter"] + 1)
market_QoQ_cols = [c for c in df_market_QoQ_lag.columns if c not in ["Year", "Quarter", "TargetYear", "TargetQuarter"]]
rename_market_QoQ = {c: f"{c}_last1" for c in market_QoQ_cols}
df_market_QoQ_lag = df_market_QoQ_lag.rename(columns = rename_market_QoQ)
df_market_QoQ_lag = df_market_QoQ_lag[["TargetYear", "TargetQuarter"] + list(rename_market_QoQ.values())]

df_market_YoY_lag = df_market_idx_quarterly.copy()
df_market_YoY_lag["TargetYear"] = np.where(df_market_YoY_lag["Quarter"] == 4, df_market_YoY_lag["Year"] + 1, df_market_YoY_lag["Year"])
df_market_YoY_lag["TargetQuarter"] = np.where(df_market_YoY_lag["Quarter"] == 4, 1, df_market_YoY_lag["Quarter"] + 1)
market_YoY_cols = [c for c in df_market_YoY_lag.columns if c not in ["Year", "Quarter", "TargetYear", "TargetQuarter"]]
rename_market_YoY = {c: f"{c}_last1" for c in market_YoY_cols}
df_market_YoY_lag = df_market_YoY_lag.rename(columns = rename_market_YoY)
df_market_YoY_lag = df_market_YoY_lag[["TargetYear", "TargetQuarter"] + list(rename_market_YoY.values())]

transaction_cols = [c for c in df_transaction_quarterly.columns if c not in ["StockID", "Year", "Quarter", "SubCategory"]]
rename_transaction = {c: f"{c}_cur" for c in transaction_cols}
df_trans_cur = df_transaction_quarterly.rename(columns = rename_transaction)

df_tech_QoQ_cur = df_tech_idx_features_QoQ.copy().sort_values(by = ["Year", "Quarter"])
keep_tech_cols = ["Year", "Quarter"] + [col for col in df_tech_QoQ_cur.columns if "QoQ" in col]
df_tech_QoQ_cur = df_tech_QoQ_cur[keep_tech_cols]

df_tech_YoY_cur = df_tech_idx_features_YoY.copy().sort_values(by = ["Year", "Quarter"])
keep_tech_cols = ["Year", "Quarter"] + [col for col in df_tech_YoY_cur.columns if "YoY" in col]
df_tech_YoY_cur = df_tech_YoY_cur[keep_tech_cols]

# Merge all features to create the final modeling dataset
model_return_general = df_dv[["StockID", "Year", "Quarter", "QuarterlyReturn", "QuarterlyReturn_Lag1", "QuarterlyReturn_Lag2", "QuarterlyReturn_Lag3"]].copy()
model_volatility_general = df_dv[["StockID", "Year", "Quarter", "QuarterlyVolatility", "QuarterlyVolatility_Lag1", "QuarterlyVolatility_Lag2", "QuarterlyVolatility_Lag3"]].copy()

# Return
model_return_general = pd.merge(
    model_return_general,
    df_trans_cur,
    on = ["StockID", "Year", "Quarter"],
    how = "left"
)
model_return_general = pd.merge(
    model_return_general,
    df_op_lag,
    left_on = ["Year", "Quarter", "StockID"],
    right_on = ["TargetYear", "TargetQuarter", "StockID"],
    how = "left"
)
if HAS_ESG:
    model_return_general = pd.merge(
        model_return_general,
        df_esg_lag,
        left_on = ["Year", "Quarter", "StockID"],
        right_on = ["TargetYear", "TargetQuarter", "StockID"],
        how = "left"
    )
    esg_features = [col for col in df_esg_lag.columns if col not in ["TargetYear", "TargetQuarter", "StockID"]]
    model_return_general[esg_features] = model_return_general[esg_features].fillna(0)

# Volatility
model_volatility_general = pd.merge(
    model_volatility_general,
    df_trans_cur,
    on = ["StockID", "Year", "Quarter"],
    how = "left"
)
model_volatility_general = pd.merge(
    model_volatility_general,
    df_op_lag,
    left_on = ["Year", "Quarter", "StockID"],
    right_on = ["TargetYear", "TargetQuarter", "StockID"],
    how = "left"
)
if HAS_ESG:
    model_volatility_general = pd.merge(
        model_volatility_general,
        df_esg_lag,
        left_on = ["Year", "Quarter", "StockID"],
        right_on = ["TargetYear", "TargetQuarter", "StockID"],
        how = "left"
    )
    esg_features = [col for col in df_esg_lag.columns if col not in ["TargetYear", "TargetQuarter", "StockID"]]
    model_volatility_general[esg_features] = model_volatility_general[esg_features].fillna(0)


# Create two versions of the final dataset: one with QoQ features and another with YoY features, for later comparison
model_1 = pd.merge(
    model_return_general,
    df_market_QoQ_lag,
    left_on = ["Year", "Quarter"],
    right_on = ["TargetYear", "TargetQuarter"],
    how = "left"
)
model_1 = pd.merge(
    model_1,
    df_tech_QoQ_cur,
    on = ["Year", "Quarter"],
    how = "left"
    )
model_1 = myfn.clean_model(model_1)
print(f"Model 1 Created Successfully!!!!!")
print(f"Dataset shape: {model_1.shape}")
myfn.describe_data(model_1, "Model 1")

model_2 = pd.merge(
    model_return_general,
    df_market_YoY_lag,
    left_on = ["Year", "Quarter"],
    right_on = ["TargetYear", "TargetQuarter"],
    how = "left"
)
model_2 = pd.merge(
    model_2,
    df_tech_YoY_cur,
    on = ["Year", "Quarter"],
    how = "left"
    )
model_2 = myfn.clean_model(model_2)
print(f"Model 2 Created Successfully!!!!!")
myfn.describe_data(model_2, "Model 2")

model_3 = pd.merge(
    model_volatility_general,
    df_market_QoQ_lag,
    left_on = ["Year", "Quarter"],
    right_on = ["TargetYear", "TargetQuarter"],
    how = "left"
)
model_3 = pd.merge(
    model_3,
    df_tech_QoQ_cur,
    on = ["Year", "Quarter"],
    how = "left"
    )
model_3 = myfn.clean_model(model_3) 
print(f"Model 3 Created Successfully!!!!!")
myfn.describe_data(model_3, "Model 3")

model_4 = pd.merge(
    model_volatility_general,
    df_market_YoY_lag,
    left_on = ["Year", "Quarter"],
    right_on = ["TargetYear", "TargetQuarter"],
    how = "left"
)
model_4 = pd.merge(
    model_4,
    df_tech_YoY_cur,
    on = ["Year", "Quarter"],
    how = "left"
    )
model_4 = myfn.clean_model(model_4)
print(f"Model 4 Created Successfully!!!!!")
myfn.describe_data(model_4, "Model 4")

# ====================================================================================
# Mechine Learning Modeling
# ====================================================================================
print("##### Start Modeling #####")
parts = TRAIN_START_DATE.split("-")
train_start_year = model_1["Year"].min() if not model_1.empty else parts[0]
train_end_year = model_1["Year"].max() if not model_1.empty else "2026"

if HAS_ESG:
    template_file_name = f"ESG_{train_start_year}_{train_end_year}"
elif not HAS_ESG:
    template_file_name = f"NoESG_{train_start_year}_{train_end_year}"

models = {
    "Model 1": model_1,
    "Model 2": model_2,
    "Model 3": model_3,
    "Model 4": model_4
}

all_evaluations = []
all_importances = []

for label, df in models.items():
    eva_rf, imp_rf, fig_rf = myfn.run_random_forest(df, label, n_estimators = 100, max_depth = 5, min_samples_leaf = 10)
    eva_xgb, imp_xgb, fig_xgb = myfn.run_xgboost(df, label, n_estimators = 100, max_depth = 5, learning_rate = 0.05)
    eva_knn = myfn.run_knn(df, label, n_neighbors = 5, weights = "distance")
    eva_linear_dict = myfn.run_linear_regressions(df, label)             # 接收包含 OLS, Lasso, Ridge 的字典並拆解
    eva_ols = eva_linear_dict["OLS"]
    eva_lasso = eva_linear_dict["Lasso"]
    eva_ridge = eva_linear_dict["Ridge"]
    
    eva_cur = pd.DataFrame({
        "Model": [label] * 6,
        "Algorithm": ["Random Forest", "XGBoost", "KNN", "LR - OLS", "LR - Lasso", "LR - Ridge"],
        
        # Training
        "Training_MSE": [eva_rf["MSE"].iloc[0], eva_xgb["MSE"].iloc[0], eva_knn["MSE"].iloc[0], eva_ols["MSE"].iloc[0], eva_lasso["MSE"].iloc[0], eva_ridge["MSE"].iloc[0]],
        "Training_MAE": [eva_rf["MAE"].iloc[0], eva_xgb["MAE"].iloc[0], eva_knn["MAE"].iloc[0], eva_ols["MAE"].iloc[0], eva_lasso["MAE"].iloc[0], eva_ridge["MAE"].iloc[0]],
        "Training_R-Squared": [eva_rf["R-Squared"].iloc[0], eva_xgb["R-Squared"].iloc[0], eva_knn["R-Squared"].iloc[0], eva_ols["R-Squared"].iloc[0], eva_lasso["R-Squared"].iloc[0], eva_ridge["R-Squared"].iloc[0]],
        "Training_Adj R-Squared": [eva_rf["Adj-R-Squared"].iloc[0], eva_xgb["Adj-R-Squared"].iloc[0], eva_knn["Adj-R-Squared"].iloc[0], eva_ols["Adj-R-Squared"].iloc[0], eva_lasso["Adj-R-Squared"].iloc[0], eva_ridge["Adj-R-Squared"].iloc[0]],

        # Holdout
        "Holdout_MSE": [eva_rf["MSE"].iloc[1], eva_xgb["MSE"].iloc[1], eva_knn["MSE"].iloc[1], eva_ols["MSE"].iloc[1], eva_lasso["MSE"].iloc[1], eva_ridge["MSE"].iloc[1]],
        "Holdout_MAE": [eva_rf["MAE"].iloc[1], eva_xgb["MAE"].iloc[1], eva_knn["MAE"].iloc[1], eva_ols["MAE"].iloc[1], eva_lasso["MAE"].iloc[1], eva_ridge["MAE"].iloc[1]],
        "Holdout_R-Squared": [eva_rf["R-Squared"].iloc[1], eva_xgb["R-Squared"].iloc[1], eva_knn["R-Squared"].iloc[1], eva_ols["R-Squared"].iloc[1], eva_lasso["R-Squared"].iloc[1], eva_ridge["R-Squared"].iloc[1]],
        "Holdout_Adj R-Squared": [eva_rf["Adj-R-Squared"].iloc[1], eva_xgb["Adj-R-Squared"].iloc[1], eva_knn["Adj-R-Squared"].iloc[1], eva_ols["Adj-R-Squared"].iloc[1], eva_lasso["Adj-R-Squared"].iloc[1], eva_ridge["Adj-R-Squared"].iloc[1]]
    })

    imp_cur = pd.DataFrame({
        "Model": [label] * (len(imp_rf) + len(imp_xgb)),
        "Algorithm": ["Random Forest"] * len(imp_rf) + ["XGBoost"] * len(imp_xgb),
        "Rank": list(imp_rf["Rank"]) + list(imp_xgb["Rank"]),
        "Feature": list(imp_rf["Feature"]) + list(imp_xgb["Feature"]),
        "Importance": list(imp_rf["Importance"]) + list(imp_xgb["Importance"])
    })
    
    all_evaluations.append(eva_cur)
    all_importances.append(imp_cur)

    fig_rf.savefig(f"./figures/Imp_Fig_RF_{label}_{template_file_name}.png")
    fig_xgb.savefig(f"./figures/Imp_Fig_XGB_{label}_{template_file_name}.png")

final_eva = pd.concat(all_evaluations, ignore_index = True)
final_eva = final_eva.sort_values(by = ["Algorithm", "Model"]).reset_index(drop = True)
final_imp = pd.concat(all_importances, ignore_index = True)
final_imp = final_imp.sort_values(by = ["Algorithm", "Model", "Rank"]).reset_index(drop = True)

# Insert an empty row between different algorithms for better readability
empty_row = pd.DataFrame([[None] * len(final_eva.columns)], columns = final_eva.columns)
chunks = []
for _, group in final_eva.groupby("Algorithm", sort = False):
    chunks.append(group)
    chunks.append(empty_row)

final_eva = pd.concat(chunks, ignore_index = True).iloc[:-1]

final_eva = final_eva.fillna("")

eva_file_name = f"Eva_{template_file_name}.csv"
imp_file_name = f"Imp_{template_file_name}.csv"

final_eva.to_csv(f"./results/{eva_file_name}", index = False)
final_imp.to_csv(f"./results/{imp_file_name}", index = False)

model_1.to_csv(f"./models/Model1_{template_file_name}.csv", index = False)
model_2.to_csv(f"./models/Model2_{template_file_name}.csv", index = False)
model_3.to_csv(f"./models/Model3_{template_file_name}.csv", index = False)
model_4.to_csv(f"./models/Model4_{template_file_name}.csv", index = False)

print(f"\n{'=' * 5}", f"Evaluation Results", f"{'=' * 5}\n", sep = " ")
print(final_eva)