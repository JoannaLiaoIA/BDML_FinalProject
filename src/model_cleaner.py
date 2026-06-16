import numpy as np
import pandas as pd
import config as myCfg
import src.utility_function as myFn
import src.algo_runner as myRunner


def calculate_time_series_features(df: pd.DataFrame, target_cols: list, method: str) -> pd.DataFrame:
    """
    method = "QoQ": calculate quarter-over-quarter growth rate
    method = "YoY": calculate year-over-year growth rate
    """
    out = df.copy()
    
    hasStockID = "StockID" in out.columns
    
    if hasStockID:
        out = out.sort_values(by = ["StockID", "Year", "Quarter"]).reset_index(drop = True)
    else:
        out = out.sort_values(by = ["Year", "Quarter"]).reset_index(drop = True)

    for col in target_cols:
        if col not in out.columns:
            continue
            
        if method == "QoQ":
            if hasStockID:
                out[f"{col}_QoQ"] = out.groupby("StockID")[col].pct_change(periods = 1)
                # Lag features
                out[f"{col}_QoQ_Lag1"] = out.groupby("StockID")[col].shift(1)
                out[f"{col}_QoQ_Lag2"] = out.groupby("StockID")[col].shift(2)
                out[f"{col}_QoQ_Lag3"] = out.groupby("StockID")[col].shift(3)
            else:
                out[f"{col}_QoQ"] = out[col].pct_change(periods = 1)
                # Lag features
                out[f"{col}_QoQ_Lag1"] = out[col].shift(1)
                out[f"{col}_QoQ_Lag2"] = out[col].shift(2)
                out[f"{col}_QoQ_Lag3"] = out[col].shift(3)
                
        elif method == "YoY":
            if hasStockID:
                out[f"{col}_YoY"] = out.groupby("StockID")[col].pct_change(periods = 4)
                # Lag features
                out[f"{col}_YoY_Lag1"] = out.groupby("StockID")[col].shift(1)
                out[f"{col}_YoY_Lag2"] = out.groupby("StockID")[col].shift(2)
                out[f"{col}_YoY_Lag3"] = out.groupby("StockID")[col].shift(3)
            else:
                out[f"{col}_YoY"] = out[col].pct_change(periods = 4)
                # Lag features
                out[f"{col}_YoY_Lag1"] = out[col].shift(1)
                out[f"{col}_YoY_Lag2"] = out[col].shift(2)
                out[f"{col}_YoY_Lag3"] = out[col].shift(3)
        else:
            raise ValueError("Invalid method. Please choose \"QoQ\" or \"YoY\".")

    return out


def annual_to_quarterly(df: pd.DataFrame, target_cols: list, method: str = "divide") -> pd.DataFrame:
    """
    method = "divide": divide the annual value by 4 to get quarterly value.
    method = "duplicate": duplicate the annual value to all four quarters.
    """
    out = df.copy()
    
    parts = out["Duration"].astype(str).str.split("-")
    out["Year"] = parts.str[0].astype(int)

    seasons = [1, 2, 3, 4]
    out["Quarter"] = [seasons] * len(out)
    out = out.explode("Quarter")
    out["Quarter"] = out["Quarter"].astype(int)
    out["StockID"] = df["StockID"].astype(str).str.zfill(4)
    
    out[target_cols] = out[target_cols].apply(pd.to_numeric, errors = "coerce")
    
    if method == "divide":
        for col in target_cols:
            out[col] = out[col] / 4
    elif method == "duplicate":
        pass
    else:
        raise ValueError("Invalid method. Please choose \"divide\" or \"duplicate\".")
        
    final_cols = [ "StockID", "Year", "Quarter"] + target_cols
    out = out[final_cols].reset_index(drop = True)
    out = out.sort_values(by = ["StockID", "Year", "Quarter"])
    
    return out


def quarter_to_quarterly(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
    out = df.copy()

    df["Duration"] = df["Duration"].astype(str)
    parts = df["Duration"].str.split("-")

    out["Year"] = parts.str[0].astype(int)
    out["Month"] = parts.str[1].astype(int)
    out["Quarter"] = (parts.str[1].astype(int) - 1) // 3 + 1
    out["StockID"] = df["StockID"].astype(str).str.zfill(4)
    out[target_cols] = out[target_cols].apply(pd.to_numeric, errors = "coerce")

    out = out[out["Month"] % 3 == 0]

    final_cols = ["StockID", "Year", "Quarter"] + target_cols
    out = out[final_cols].reset_index(drop = True)
    out = out.sort_values(by = ["StockID", "Year", "Quarter"]).reset_index(drop = True)

    return out


def month_to_quarterly(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
    out = df.copy()

    out["Duration"] = out["Duration"].astype(str)
    parts = out["Duration"].str.split("-")

    out["Year"] = parts.str[0].astype(int)
    out["Quarter"] = ((parts.str[1].astype(int) - 1) // 3) + 1
    out[target_cols] = out[target_cols].apply(pd.to_numeric, errors = "coerce")

    group_cols = ["Year", "Quarter"]
    if "StockID" in out.columns:
        out["StockID"] = out["StockID"].astype(str).str.zfill(4)
        group_cols = ["StockID", "Year", "Quarter"]

    out = out.groupby(group_cols)[target_cols].mean().reset_index()

    final_cols = group_cols + target_cols
    out = out[final_cols]
    out = out.sort_values(by = ["Year", "Quarter"]).reset_index(drop = True)

    return out


def daily_to_quarterly(df: pd.DataFrame, target_cols: list, method = "mean") -> pd.DataFrame:
    """
    method = "mean": calculate the mean of daily values for each quarter
    method = "sum": calculate the sum of daily values for each quarter (suitable for TradingVolume, TradingMoney, TradingTurnover)
    """
    out = df.copy()

    parts = out["Date"].astype(str).str.split("-")

    out["Year"] = parts.str[0].astype(int)
    out["Quarter"] = ((parts.str[1].astype(int) - 1) // 3) + 1
    out[target_cols] = out[target_cols].apply(pd.to_numeric, errors = "coerce")

    group_cols = ["Year", "Quarter"]
    if "StockID" in out.columns:
        out["StockID"] = out["StockID"].astype(str).str.zfill(4)
        group_cols = ["StockID", "Year", "Quarter"]

    agg_dict = {}
    if method == "mean":
        for col in target_cols:
            agg_dict[col] = "mean"
    elif method == "sum":  
        for col in target_cols:
            agg_dict[col] = "sum"

    out = out.groupby(group_cols).agg(agg_dict).reset_index()

    return out


def clean_model(model: pd.DataFrame) -> pd.DataFrame:
    out = model.copy()

    out.replace([np.inf, -np.inf], np.nan, inplace = True)

    if myCfg.IS_DEBUG:
        print(out.isna().sum().sort_values(ascending = False).head(20))
    
    cols_to_drop = [c for c in model.columns if "Target" in c]
    out = out.drop(columns = cols_to_drop)
    out = out.dropna().reset_index(drop = True)
    
    return out


def add_dv_features(df: pd.DataFrame) -> pd.DataFrame:
    window_size = 4
    
    out = df.copy()
    
    out = out[["StockID", "Year", "Quarter", "ClosingPrice"]].copy()
    out["QuarterlyReturn"] = out.groupby("StockID")["ClosingPrice"].pct_change()
    out["QuarterlyReturn_Lag1"] = out.groupby("StockID")["QuarterlyReturn"].shift(1)
    out["QuarterlyReturn_Lag2"] = out.groupby("StockID")["QuarterlyReturn"].shift(2)
    out["QuarterlyReturn_Lag3"] = out.groupby("StockID")["QuarterlyReturn"].shift(3)

    out["QuarterlyLogReturn"] = np.log(out["ClosingPrice"] / out.groupby("StockID")["ClosingPrice"].shift(1))
    out["QuarterlyVolatility"] = out.groupby("StockID")["QuarterlyLogReturn"].rolling(window = window_size).std().reset_index(0, drop = True)
    out["QuarterlyVolatility_Lag1"] = out.groupby("StockID")["QuarterlyVolatility"].shift(1)
    out["QuarterlyVolatility_Lag2"] = out.groupby("StockID")["QuarterlyVolatility"].shift(2)
    out["QuarterlyVolatility_Lag3"] = out.groupby("StockID")["QuarterlyVolatility"].shift(3)

    if myCfg.IS_DEBUG:
        myFn.describe_data(out, "Dependent Variable Data (Quarterly)", n = 10)

    print("Dependent variable calculated successfully.")

    return out


def add_op_features(df_eps: pd.DataFrame, df_revenue: pd.DataFrame, df_roe_roa_grossmargin: pd.DataFrame, df_company_category: pd.DataFrame) -> pd.DataFrame:
    df_op = pd.merge(
        df_eps,
        df_revenue,
        left_on = ["StockID", "Year", "Quarter"],
        right_on = ["StockID", "Year", "Quarter"]
    )
    df_op = pd.merge(
        df_op,
        df_roe_roa_grossmargin,
        left_on = ["StockID", "Year", "Quarter"],
        right_on = ["StockID", "Year", "Quarter"]
    )
    df_op = pd.merge(
        df_op,
        df_company_category,
        on = "StockID",
        how = "left"
    )

    df_op.dropna(inplace = True)

    if myCfg.IS_DEBUG:
        myFn.describe_data(df_op, "Operating Performance Data (Quarterly)")

    print("Operating Performance Data loaded and transformed successfully.")

    return df_op


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    cols = ["Year", "Quarter", "StockID", "TradingVolume", "TradingMoney", "TradingTurnover", "Spread"]
    df_transaction_quarterly = df[cols].copy()
    
    if myCfg.IS_DEBUG:
        myFn.describe_data(df_transaction_quarterly, "Transaction Data (Quarterly)")

    print("Transaction Data loaded and transformed successfully.")

    return out


def add_tech_idx_features(df: pd.DataFrame) -> pd.DataFrame:
    df_tech_idx_features_QoQ = calculate_time_series_features(
        df,
        target_cols = ["VIX", "SOX", "DJIA", "IXIC", "SP500", "FnG"], method = "QoQ"
    )
    df_tech_idx_features_YoY = calculate_time_series_features(
        df,
        target_cols = ["VIX", "SOX", "DJIA", "IXIC", "SP500", "FnG"], method = "YoY"
    )

    if myCfg.IS_DEBUG:
        myFn.describe_data(df_tech_idx_features_QoQ, "Technology Index Features (QoQ) (Quarterly)")
        myFn.describe_data(df_tech_idx_features_YoY, "Technology Index Features (YoY) (Quarterly)")

    print("Technology Index features calculated successfully.")

    return df_tech_idx_features_QoQ, df_tech_idx_features_YoY


def add_market_idx_features(df_cpi: pd.DataFrame, df_unemployment_rate: pd.DataFrame, df_rediscount_rate: pd.DataFrame) -> pd.DataFrame:
    out = pd.merge(
        df_cpi,
        df_unemployment_rate,
        left_on = ["Year", "Quarter"],
        right_on = ["Year", "Quarter"]
    )
    out = pd.merge(
        out,
        df_rediscount_rate,
        left_on = ["Year", "Quarter"],
        right_on = ["Year", "Quarter"]
    )
    out.dropna(inplace = True)
    
    df_market_idx_features_QoQ = calculate_time_series_features(
        out,
        target_cols = ["OverallIndex", "TotalUnempPct", "RediscountRate"],
        method = "QoQ"
    )

    df_market_idx_features_YoY = calculate_time_series_features(
        out,
        target_cols = ["OverallIndex", "TotalUnempPct", "RediscountRate"],
        method = "YoY"
    )
    if myCfg.IS_DEBUG:
        myFn.describe_data(df_market_idx_features_QoQ, "Market Index Features (QoQ) (Quarterly)")
        myFn.describe_data(df_market_idx_features_YoY, "Market Index Features (YoY) (Quarterly)")

    print("Market Index Data loaded and transformed successfully.")

    return df_market_idx_features_QoQ, df_market_idx_features_YoY


def create_lag_features(df: pd.DataFrame, exclude_cols: list, prefix: str = "", suffix: str = "_last1"):
    """
    Shift the features to create lag features for the next quarter's prediction.
    """ 
    df_lag = df.copy()
    
    # 計算下一季的年份與季度
    df_lag["TargetYear"] = np.where(df_lag["Quarter"] == 4, df_lag["Year"] + 1, df_lag["Year"])
    df_lag["TargetQuarter"] = np.where(df_lag["Quarter"] == 4, 1, df_lag["Quarter"] + 1)
    
    # 找出需要改名的特徵欄位
    base_exclude = ["Year", "Quarter", "TargetYear", "TargetQuarter"] + exclude_cols
    feature_cols = [c for c in df_lag.columns if c not in base_exclude]
    
    # 重新命名 (加上 prefix 或 suffix)
    rename_dict = {c: f"{prefix}{c}{suffix}" for c in feature_cols}
    df_lag = df_lag.rename(columns=rename_dict)
    
    # 決定要保留的 Key (如果有 StockID 就保留，沒有就只留時間)
    base_keys = ["TargetYear", "TargetQuarter"]
    if "StockID" in df_lag.columns:
        base_keys.insert(0, "StockID")
        
    return df_lag[base_keys + list(rename_dict.values())]

def merge_base_features(df_base, df_trans_cur, df_op_lag, df_esg_lag, has_esg):
    """
    Merge the current quarter's transaction features, the previous quarter's operating performance features, and conditionally merge the previous quarter's ESG features into the base DataFrame.
    """
    df_merged = pd.merge(
        df_base,
        df_trans_cur,
        on = ["StockID", "Year", "Quarter"],
        how = "left"
    )
    
    df_merged = pd.merge(
        df_merged, df_op_lag, 
        left_on = ["Year", "Quarter", "StockID"], 
        right_on = ["TargetYear", "TargetQuarter", "StockID"], 
        how = "left"
    )
    
    if has_esg and df_esg_lag is not None:
        df_merged = pd.merge(
            df_merged, df_esg_lag, 
            left_on = ["Year", "Quarter", "StockID"], 
            right_on = ["TargetYear", "TargetQuarter", "StockID"], 
            how = "left"
        )
        
        # Fill NaN values in ESG features with 0 (assuming missing ESG data implies no ESG initiatives or disclosures)
        esg_features = [c for c in df_esg_lag.columns if c not in ["TargetYear", "TargetQuarter", "StockID"]]
        df_merged[esg_features] = df_merged[esg_features].fillna(0)
        
    return df_merged

def assemble_final_models(model_ret, model_vol, market_QoQ, market_YoY, tech_QoQ, tech_YoY):
    """
    1. Assemble the final modeling datasets for both Return and Volatility, each with QoQ and YoY market/tech features.
    2. Return a dictionary containing the 4 final datasets.
    """
    recipes = {
        "Model 1": {"base": model_ret, "market": market_QoQ, "tech": tech_QoQ},
        "Model 2": {"base": model_ret, "market": market_YoY, "tech": tech_YoY},
        "Model 3": {"base": model_vol, "market": market_QoQ, "tech": tech_QoQ},
        "Model 4": {"base": model_vol, "market": market_YoY, "tech": tech_YoY}
    }
    
    final_models = {}
    
    for model_name, parts in recipes.items():
        df_merged = pd.merge(
            parts["base"],
            parts["market"],
            left_on = ["Year", "Quarter"],
            right_on = ["TargetYear", "TargetQuarter"],
            how = "left"
        )
        
        df_merged = pd.merge(
            df_merged,
            parts["tech"],
            on = ["Year", "Quarter"],
            how = "left"
        )
        
        df_cleaned = clean_model(df_merged)
        final_models[model_name] = df_cleaned
                
        if myCfg.IS_DEBUG:
            myFn.describe_data(df_cleaned, model_name)

        print(f"{model_name} Created Successfully!!!!!")
            
    return final_models