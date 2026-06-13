import os
import numpy as np
import pandas as pd
import config as myCfg
import src.function as myFn
from dotenv import load_dotenv
from fredapi import Fred
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Daily Transaction Data
def load_daily_transaction():
    print("Loading daily transaction data...")
    if myCfg.IS_LARGE_DATASET:
        path_transaction = "data/processed/FinMind_transaction_2000_2026.parquet"
    else:
        path_transaction = "data/processed/FinMind_transaction_2019_2025.parquet"

    df_daily_transaction = pd.read_parquet(path_transaction)
    df_daily_transaction["Date"] = df_daily_transaction["Date"].apply(myFn.convert_date)
    df_daily_transaction["StockID"] = df_daily_transaction["StockID"].astype(str).str.strip()
    df_daily_transaction["ClosingPrice"] = pd.to_numeric(df_daily_transaction["ClosingPrice"], errors = "coerce")
    df_daily_transaction["TradingVolume"] = pd.to_numeric(df_daily_transaction["TradingVolume"], errors = "coerce")
    df_daily_transaction["TradingMoney"] = pd.to_numeric(df_daily_transaction["TradingMoney"], errors = "coerce")
    df_daily_transaction["TradingTurnover"] = pd.to_numeric(df_daily_transaction["TradingTurnover"], errors = "coerce")
    df_daily_transaction["Spread"] = pd.to_numeric(df_daily_transaction["Spread"], errors = "coerce")
    df_daily_transaction = df_daily_transaction.dropna()
    df_daily_transaction = df_daily_transaction.sort_values(by = ["StockID", "Date"], ascending = True)

    if myCfg.IS_DEBUG:
        myFn.describe_data(df_daily_transaction, "Daily Transaction Data")
    return df_daily_transaction


# Company Category
def load_company_category():
    print("\nLoading company category data...")
    df_company_category = myFn.get_file_from_drive("https://docs.google.com/spreadsheets/d/1dJqfJVqryl2XhHoW9Fa01SXzxVkHpfnDQTaMF58Urh4/edit?gid=84991497#gid=84991497")
    df_company_category = df_company_category.rename(columns = {"Code": "StockID"})
    df_company_category["StockID"] = df_company_category["StockID"].astype(str).str.strip()
    df_company_category = df_company_category[["StockID", "SubCategory"]].copy()

    if myCfg.IS_DEBUG:
        myFn.describe_data(df_company_category, "Company Category Data")

    return df_company_category


# Tech Index (FRED & Github)

def load_tech_index():
    print("\nLoading technology index data from FRED and Fear & Greed Index from Github...")
    fred = Fred(api_key = myCfg.FRED_API_KEY)
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
    df_tech_idx["Date"] = df_tech_idx["Date"].apply(myFn.convert_date)
    df_tech_idx = df_tech_idx.dropna()

    if myCfg.IS_DEBUG:
        myFn.describe_data(df_tech_idx, "Technology Index Data")
    
    return df_tech_idx


# Macro (Rediscount, CPI, Unemp)
def load_rediscount_rate():
    print("\nLoading rediscount rate data...")
    df_rediscount_rate = myFn.get_file_from_drive("https://docs.google.com/spreadsheets/d/1qy3yNGYnj5vPCCCDd7XtzZ5wr2eDm-zw9ChmwgvuUAE/edit?gid=834914550#gid=834914550")
    df_rediscount_rate["Duration"] = df_rediscount_rate["Duration"].apply(myFn.convert_date)

    if myCfg.IS_DEBUG:
        myFn.describe_data(df_rediscount_rate, "Rediscount Rate Data")

    return df_rediscount_rate

def load_cpi():
    print("\nLoading CPI data...")

    df_cpi = myFn.get_file_from_drive("https://docs.google.com/spreadsheets/d/1CHlZ0XWqH0e1TlRbBu7t43tzK4Wu_pHcjIWwKB_Hv7g/edit?usp=sharing")
    df_cpi["Duration"] = df_cpi["Duration"].apply(myFn.convert_date)
    df_cpi = df_cpi[["Duration", "OverallIndex"]].copy()

    if myCfg.IS_DEBUG:
        myFn.describe_data(df_cpi, "CPI Data")

    return df_cpi


def load_unemployment_rate():
    print("\nLoading unemployment rate data...")
    df_unemployment_rate = myFn.get_file_from_drive("https://docs.google.com/spreadsheets/d/1OuZ-xTaNkT0nCx-eb0ABtThWDmaLYYRMTja9yJrqwYQ/edit?gid=606389627#gid=606389627")
    df_unemployment_rate["Duration"] = df_unemployment_rate["Duration"].apply(myFn.convert_date)

    if myCfg.IS_DEBUG:
        myFn.describe_data(df_unemployment_rate, "Unemployment Rate Data")

    return df_unemployment_rate


# Financial Data (EPS, Revenue, ROE&ROA&GrossMargin)

def load_eps():
    print("\nLoading EPS data...")
    df_eps = myFn.get_file_from_drive("https://drive.google.com/file/d/1TnuzhXPyIpJJtKM3vxjkF9d9GqQWN4mj/view?usp=sharing")
    df_eps["Duration"] = df_eps["Duration"].apply(myFn.convert_date)
    df_eps = df_eps[["StockID", "Duration", "EPS"]].copy()
    df_eps = df_eps.sort_values(by = ["StockID", "Duration"])
    
    if myCfg.IS_DEBUG:
        myFn.describe_data(df_eps, "EPS Data")

    return df_eps


def load_revenue():
    print("\nLoading revenue data...")
    df_revenue = myFn.get_file_from_drive("https://drive.google.com/file/d/11IFSInZP7sJmxYzxi8mPb6Akt4FoY5g0/view?usp=sharing")
    df_revenue["Duration"] = df_revenue["Duration"].apply(myFn.convert_date)

    if myCfg.IS_DEBUG:
        myFn.describe_data(df_revenue, "Revenue Data")

    return df_revenue


def load_roe_roa_grossmargin():
    print("\nLoading ROE, ROA, Gross Margin data...")
    df_roe_roa_grossmargin = myFn.get_file_from_drive("https://docs.google.com/spreadsheets/d/1ngHwal_vAQ4IQeON8vT0D0IMkzz3cOseiDROuU4KqK4/edit?pli=1&gid=1778576688#gid=1778576688")
    df_roe_roa_grossmargin["Duration"] = df_roe_roa_grossmargin["Duration"].apply(myFn.convert_date)

    if myCfg.IS_DEBUG:
        myFn.describe_data(df_roe_roa_grossmargin, "ROE, ROA, Gross Margin Data")

    return df_roe_roa_grossmargin


# ESG Data
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

def load_esg():
    df_esg = []

    for index, (esg_topic, file_url) in enumerate(esg_file_list.items()):
        print(f"\n[{index + 1}/{len(esg_file_list)}] Processing: {esg_topic}")

        df_temp = myFn.get_file_from_drive(file_url)
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

    if myCfg.IS_DEBUG:
        myFn.describe_data(df_esg, "ESG Data (Long Format)")

    print("ESG Data loaded and transformed successfully.")

    return df_esg