import re
import io
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from unittest import result
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler

TOP_N_FEATURES = 50

def set_training_periods(training_start_date: str, training_end_date: str) -> None:
    global TRAINING_START_YEAR, TRAINING_END_YEAR, TRAINING_START_QUARTER, TRAINING_END_QUARTER

    TRAINING_START_YEAR = int(training_start_date.split("-")[0])
    TRAINING_END_YEAR = int(training_end_date.split("-")[0])

    TRAINING_START_QUARTER = (int(training_start_date.split("-")[1]) - 1) // 3 + 1
    TRAINING_END_QUARTER = (int(training_end_date.split("-")[1]) - 1) // 3 + 1


def set_holdout_periods(holdout_start_date: str, holdout_end_date: str) -> None:
    global HOLDOUT_START_YEAR, HOLDOUT_END_YEAR, HOLDOUT_START_QUARTER, HOLDOUT_END_QUARTER
    
    HOLDOUT_START_YEAR = int(holdout_start_date.split("-")[0])
    HOLDOUT_END_YEAR = int(holdout_end_date.split("-")[0])

    HOLDOUT_START_QUARTER = (int(holdout_start_date.split("-")[1]) - 1) // 3 + 1
    HOLDOUT_END_QUARTER = (int(holdout_end_date.split("-")[1]) - 1) // 3 + 1


def set_top_n_features(n: int = 50) -> None:
    global TOP_N_FEATURES
    TOP_N_FEATURES = n


def set_esg_large_dataset_flags(esg_flag: bool, large_dataset_flag: bool) -> None:
    global has_ESG, is_large_dataset
    has_ESG = esg_flag
    is_large_dataset = large_dataset_flag

def describe_data(df: pd.DataFrame, title: str, n: int = 3) -> None:
    """
    Print the head and tail n rows, then check "infinite", "NA" and the datatype of each columns.
    """

    title_str = f"| This is {title} |"
    num = len(title_str)

    print("\n")
    print("+", "-" * (num - 2), "+", sep = "")
    print(title_str)
    print("+", "-" * (num - 2), "+", sep = "")

    print("\n", f"| Dataset Shape: {df.shape}", "\n", sep = "")

    print(f"========== first {n} rows ==========")
    print(df.head(n), "\n")

    print(f"========== last {n} rows ==========")
    print(df.tail(n), "\n")

    print(f"========== infinite check ==========")
    print(df.isin([np.inf, -np.inf]).sum(), "\n")

    print(f"========== NA check ==========")
    print(df.isna().sum(), "\n")

    print(f"========== info of data ==========")
    df.info()
    print("\n")


def convert_date(date_str: str):
    if pd.isna(date_str) or str(date_str).strip() == "":
        return pd.NaT
    
    def check_year(y: int) -> int:
        return y + 1911 if y < 200 else y

    s = str(date_str).strip()
    s = s.split(" ")[0]

    # x 年 x 月
    if "年" in s and "月" in s:
        match = re.search(r"(\d+)\s*年\s*(\d+)\s*月", s)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            year = check_year(year)
            return f"{year:04d}-{month:02d}-00"
        
    # x 年
    elif "年" in s:
        match = re.search(r"(\d+)\s*年", s)
        if match:
            year = int(match.group(1))
            year = check_year(year)
            return f"{year:04d}-00-00"
        
    # yymmdd or yyymmdd or yyyymmdd e.g. 920912, 1110912, 20210912
    elif s.isdigit() and (len(s) >= 6 and len(s) <= 7):
        year = int(s[:-4])
        month = int(s[-4:-2])
        day = int(s[-2:])
        year = check_year(year)
        return f"{year:04d}-{month:02d}-{day:02d}"

    # yy or yyy or yyyy
    elif s.isdigit() and (len(s) >= 2 and len(s) <= 4):
        year = int(s)
        year = check_year(year)
        return f"{year:04d}-00-00"
    
    # yyyMmm or yyyyMmm e.g. 111M09, 2021M09
    elif "M" in s:
        year = int(s[:-3])
        month = int(s[-2:])
        year = check_year(year)
        return f"{year:04d}-{month:02d}-00"
    
    # yyy/mm/dd or yyyy/mm/dd or yyy/mm or yyyy/mm e.g. 111/09/12, 2021/09/12
    # yyy-mm-dd or yyyy-mm-dd or yyy-mm or yyyy-mm e.g. 111-09-12, 2021-09-12
    elif "/" in s or "-" in s:
        s = s.replace("/", "-")
        parts = s.split("-")

        year = int(parts[0])
        month = int(parts[1]) if len(parts) >= 2 else 0
        day = int(parts[2]) if len(parts) >= 3 else 0
        year = check_year(year)
        return f"{year:04d}-{month:02d}-{day:02d}"
    
    return "Invalid Date Format"


# def save_to_sheets(df: pd.DataFrame, url: str, worksheet_name: str) -> None:
    # df_clean = df.copy()
    # df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    # df_clean = df_clean.fillna("")
    # df_clean = df_clean.astype(str)
    # df_clean = df_clean.replace(["nan", "NaT", "None", "<NA>"], "")

    # try:
    #     sh = gc.open_by_url(url)
    # except NameError:
    #     print("> 錯誤：找不到 gc (Google Client) 通行證。")
    #     return

    # try:
    #     ws = sh.worksheet(worksheet_name)
    #     sh.del_worksheet(ws)
    #     print(f"> 刪除舊有工作表: "{worksheet_name}"")
    # except gspread.exceptions.WorksheetNotFound:
    #     pass

    # new_ws = sh.add_worksheet(
    #     title=worksheet_name,
    #     rows=str(len(df_clean) + 1),
    #     cols=str(len(df_clean.columns))
    # )

    # header = df_clean.columns.tolist()
    # rows = df_clean.values.tolist()
    # all_data = [header] + rows

    # new_ws.update(
    #     range_name="A1",
    #     values=all_data,
    #     value_input_option="USER_ENTERED"
    # )
    # print(f"> 成功上傳至 \"{worksheet_name}\"，共包含 {len(df)} 筆資料")


def get_file_from_drive(file_url: str) -> pd.DataFrame:
    """
    Download a CSV file from Google Drive or Google Sheets and return it as a DataFrame.
    """
    if not file_url:
        raise ValueError("File url cannot be empty")

    target_url = file_url
    params = None
    is_drive_file = False

    if "docs.google.com/spreadsheets" in file_url:
        print("Detected Google Sheets. Converting to CSV export URL...")
        if any(keyword in file_url for keyword in ["/edit", "/view", "htmlview"]):
            target_url = re.sub(r"/(edit|view|htmlview).*$", "/export?format=csv", file_url)
        elif "export" in file_url and "format=csv" not in file_url:
            target_url = re.sub(r"format=[^&]+", "format=csv", file_url)
            if "format=csv" not in target_url:
                target_url += "&format=csv"

    elif "drive.google.com/file" in file_url:
        print("Detected Google Drive CSV file. Preparing to fetch...")
        file_id_match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", file_url)
        if file_id_match:
            file_id = file_id_match.group(1)
            target_url = "https://drive.google.com/uc"
            params = {"id": file_id, "export": "download", "confirm": "true"}
            is_drive_file = True
        else:
            raise ValueError("Failed to parse Google Drive file ID")

    else:
        print("Detected standard CSV URL...")

    print(f"Fetching data from {target_url}...")
    
    try:
        # 使用 Session 來保持連線與 Cookies (這對繞過警告很重要)
        session = requests.Session()
        response = session.get(target_url, params = params, stream = True, timeout = 30)
        
        # 檢查是否遇到「檔案過大無法掃描病毒」的警告
        if is_drive_file:
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    print("Large file detected. Bypassing virus scan warning...")
                    # 將抓到的 token 塞進參數裡再次請求
                    params["confirm"] = value
                    response = session.get(target_url, params = params, stream = True, timeout = 30)
                    break

        response.raise_for_status()
        
        # 再次確認最終抓下來的是不是 HTML
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            raise PermissionError("Downloaded content is STILL HTML.")

        # 讀取大檔案可能會花一點時間
        df = pd.read_csv(io.BytesIO(response.content), encoding = "utf-8-sig")
        print("Download finished.")
        return df
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error occurred while fetching the file: {e}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse CSV. The file format might be incorrect: {e}")


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
    method = "divide": divide the annual value by 4 to get quarterly value
    method = "duplicate": duplicate the annual value to all four quarters
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
    print(out.isna().sum().sort_values(ascending = False).head(20))
    
    cols_to_drop = [c for c in model.columns if "Target" in c]
    out = out.drop(columns = cols_to_drop)
    out = out.dropna().reset_index(drop = True)
    
    return out


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def print_evaluation(model_name: str, dataset_name: str, y_true: np.ndarray, y_pred: np.ndarray, p: int) -> pd.DataFrame:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    n = len(y_true)
    # 確保分母不為 0 (樣本數必須大於特徵數 + 1)
    if n > p + 1:
        adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    else:
        adj_r2 = np.nan # 或做其他預設處理

    title = f"{model_name} - {dataset_name} - {TRAINING_START_YEAR}Q{TRAINING_START_QUARTER} to {HOLDOUT_END_YEAR}Q{HOLDOUT_END_QUARTER}"

    eva = pd.DataFrame({
        "Model": [title],
        "MSE": [f"{mse:.6f}"],
        "MAE": [f"{mae:.6f}"],
        "R-Squared": [f"{r2:.6f}"],
        "Adj-R-Squared": [f"{adj_r2:.6f}"]
    })
    
    print("=" * 5, title, "=" * 5, sep = " ")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R-Squared: {r2:.6f}")
    print(f"Adj-R-Squared: {adj_r2:.6f}")
    
    return eva


def print_feature_importance(model_name: str, model, features: list, plot: bool = True) -> pd.DataFrame:
    # 1. 計算特徵重要度與排序
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:TOP_N_FEATURES] 
    
    top_features = [features[i] for i in indices]
    top_importances = importances[indices]
    
    # 2. 準備 DataFrame 所需的欄位資料
    ranks = list(range(1, len(top_features) + 1))
    cumulative_importances = np.cumsum(top_importances) # 一次性計算累計重要度

    # 3. Terminal 列印輸出
    # 這裡假設您的環境中已經有定義 TRAINING_START_YEAR 等全域變數
    title_str = f"| {model_name} - Top {TOP_N_FEATURES} Feature Importances - {TRAINING_START_YEAR}Q{TRAINING_START_QUARTER} to {HOLDOUT_END_YEAR}Q{HOLDOUT_END_QUARTER} |"
    cnt = len(title_str)
    print("+", "-" * (cnt - 2), "+", sep="")
    print(title_str)
    print("+", "-" * (cnt - 2), "+", sep="")

    for rank, feat, imp, cum_imp in zip(ranks, top_features, top_importances, cumulative_importances):
        print(f"  {rank:2d}. {feat:<50} : {imp:.4f}  (cumulative_importance: {cum_imp:>6.4%})")
    print("\n")
    
    df_importance = pd.DataFrame({
        "Model": model_name,  # 改回乾淨的 model_name，方便後續合併或 groupby
        "Rank": ranks,
        "Feature": top_features,
        "Importance": top_importances,
        "Cumulative_Importance": cumulative_importances
    })
    
    # 4. 繪製特徵重要性圖表 (Horizontal Bar Chart)
    fig = None
    if plot:
        # 使用物件導向寫法，建立 fig (畫布) 與 ax (座標軸)
        fig, ax = plt.subplots(figsize = (10, max(8, TOP_N_FEATURES * 0.3)))
        
        # 將 ax 傳遞給 seaborn
        sns.barplot(
            x = "Importance", 
            y = "Feature", 
            data = df_importance, 
            palette = "viridis",
            ax = ax
        )
        
        # 使用 ax.set_* 來設定標題與標籤
        if has_ESG and is_large_dataset:
            ax.set_title(f"{model_name} - Top {TOP_N_FEATURES} Feature Importances (ESG & Large Dataset)", fontsize = 14, pad = 15)
        elif has_ESG:
            ax.set_title(f"{model_name} - Top {TOP_N_FEATURES} Feature Importances (ESG Dataset)", fontsize = 14, pad = 15)
        elif is_large_dataset:
            ax.set_title(f"{model_name} - Top {TOP_N_FEATURES} Feature Importances (Large Dataset)", fontsize = 14, pad = 15)
        else:
            ax.set_title(f"{model_name} - Top {TOP_N_FEATURES} Feature Importances", fontsize = 14, pad = 15)
        ax.set_xlabel("Importance Score", fontsize = 12)
        ax.set_ylabel("Features", fontsize = 12)
        ax.grid(axis = 'x', linestyle = '--', alpha = 0.7)
        
        fig.tight_layout()
    
    return df_importance, fig


def run_random_forest(model: pd.DataFrame, label: str, n_estimators: int = 200, max_depth: int = 5, min_samples_leaf: int = 10) -> None:
    start_time = datetime.now()

    print(f"\n{'=' * 5}", f"Running Random Forest for {label}", f"{'=' * 5}\n", sep = " ")
    rf = model.dropna().reset_index(drop = True)
    if "QuarterlyReturn" in rf.columns:
        exclude_cols = ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "QuarterlyReturn", "SubCategory_last1"]
    elif "QuarterlyVolatility" in rf.columns:
        exclude_cols = ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "QuarterlyVolatility", "SubCategory_last1"]
    features = [col for col in rf.columns if col not in exclude_cols]

    training_mask = (rf["Year"] >= TRAINING_START_YEAR) & ((rf["Year"] < TRAINING_END_YEAR) | ((rf["Year"] == TRAINING_END_YEAR) & (rf["Quarter"] <= TRAINING_END_QUARTER)))
    holdout_mask = (rf["Year"] == HOLDOUT_START_YEAR) & (rf["Quarter"] >= HOLDOUT_START_QUARTER)

    training_data = rf[training_mask]
    holdout_data = rf[holdout_mask]

    if "QuarterlyReturn" in rf.columns:
        X_train, y_train = training_data[features], training_data["QuarterlyReturn"]
        X_test, y_test = holdout_data[features], holdout_data["QuarterlyReturn"]
    elif "QuarterlyVolatility" in rf.columns:
        X_train, y_train = training_data[features], training_data["QuarterlyVolatility"]
        X_test, y_test = holdout_data[features], holdout_data["QuarterlyVolatility"]

    print(f"[RF {label}] Training Set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"[RF {label}] Holdout Set: X_test {X_test.shape}, y_test {y_test.shape}")

    # 4. 建立與訓練隨機森林模型
    # 參數建議：先限制 max_depth 避免嚴重過擬合，n_jobs=-1 可以讓 CPU 全速運轉
    result = RandomForestRegressor(
        n_estimators = n_estimators,            # 樹的數量
        max_depth = max_depth,
        min_samples_leaf = min_samples_leaf,    # 葉節點最少樣本數
        random_state = 42,                      # Random seed
        n_jobs = -1              
    )

    print(f"[RF {label}] Training...")
    result.fit(X_train, y_train)

    print(f"[RF {label}] Predicting...")
    y_pred_train = result.predict(X_train)
    y_pred_test = result.predict(X_test)

    print(f"[RF {label}] Done!!!!!")
    importance, fig = print_feature_importance(f"RF {label}", result, features, plot = True)
    print(f"[RF {label}] Exe Time: {(datetime.now() - start_time).total_seconds():.2f} seconds\n")

    train_evaluation = print_evaluation(f"RF {label}", "Training Set", y_train, y_pred_train, len(features))
    test_evaluation = print_evaluation(f"RF {label}", "Holdout Set", y_test, y_pred_test, len(features))
    evaluation = pd.concat([train_evaluation, test_evaluation], ignore_index = True)
    # evaluation.to_csv(f"RF_{label}_Evaluation.csv", index = False)

    return evaluation, importance, fig


def run_xgboost(model: pd.DataFrame, label: str, n_estimators: int = 200, max_depth: int = 6, learning_rate: float = 0.05) -> None:
    start_time = datetime.now()
    print(f"\n{'=' * 5}", f"Running XGBoost for {label}", f"{'=' * 5}\n", sep = " ")
    xgb_df = model.dropna().reset_index(drop = True)
    
    if "QuarterlyReturn" in xgb_df.columns:
        exclude_cols = ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "QuarterlyReturn", "SubCategory_last1"]
        target_col = "QuarterlyReturn"
    elif "QuarterlyVolatility" in xgb_df.columns:
        exclude_cols = ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "QuarterlyVolatility", "SubCategory_last1"]
        target_col = "QuarterlyVolatility"
        
    features = [col for col in xgb_df.columns if col not in exclude_cols]

    training_mask = (xgb_df["Year"] >= TRAINING_START_YEAR) & ((xgb_df["Year"] < TRAINING_END_YEAR) | ((xgb_df["Year"] == TRAINING_END_YEAR) & (xgb_df["Quarter"] <= TRAINING_END_QUARTER)))
    holdout_mask = (xgb_df["Year"] == HOLDOUT_START_YEAR) & (xgb_df["Quarter"] >= HOLDOUT_START_QUARTER)

    training_data = xgb_df[training_mask]
    holdout_data = xgb_df[holdout_mask]

    X_train, y_train = training_data[features], training_data[target_col]
    X_test, y_test = holdout_data[features], holdout_data[target_col]

    print(f"[XGB {label}] Training Set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"[XGB {label}] Holdout Set: X_test {X_test.shape}, y_test {y_test.shape}")

    # 建立與訓練 XGBoost 模型
    result = XGBRegressor(
        n_estimators = n_estimators,
        max_depth = max_depth,
        learning_rate = learning_rate,
        random_state = 42,
        n_jobs = -1
    )

    print(f"[XGB {label}] Training...")
    result.fit(X_train, y_train)

    print(f"[XGB {label}] Predicting...")
    y_pred_train = result.predict(X_train)
    y_pred_test = result.predict(X_test)

    print(f"[XGB {label}] Done!!!!!")
    importance, fig = print_feature_importance(f"XGB {label}", result, features, plot = True)
    print(f"[XGB {label}] Exe Time: {(datetime.now() - start_time).total_seconds():.2f} seconds\n")
    
    train_evaluation = print_evaluation(f"XGB {label}", "Training Set", y_train, y_pred_train, len(features))
    test_evaluation = print_evaluation(f"XGB {label}", "Holdout Set", y_test, y_pred_test, len(features))
    evaluation = pd.concat([train_evaluation, test_evaluation], ignore_index = True)
    # evaluation.to_csv(f"XGB_{label}_Evaluation.csv", index = False)

    return evaluation, importance, fig


def run_knn(model: pd.DataFrame, label: str, n_neighbors: int = 5, weights: str = "distance") -> KNeighborsRegressor:
    start_time = datetime.now()

    print(f"\n{'=' * 5}", f"Running KNN for {label}", f"{'=' * 5}\n", sep = " ")
    knn_df = model.dropna().reset_index(drop = True)
    
    if "QuarterlyReturn" in knn_df.columns:
        exclude_cols = ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "QuarterlyReturn", "SubCategory_last1"]
        target_col = "QuarterlyReturn"
    elif "QuarterlyVolatility" in knn_df.columns:
        exclude_cols = ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "QuarterlyVolatility", "SubCategory_last1"]
        target_col = "QuarterlyVolatility"
        
    features = [col for col in knn_df.columns if col not in exclude_cols]

    training_mask = (knn_df["Year"] >= TRAINING_START_YEAR) & ((knn_df["Year"] < TRAINING_END_YEAR) | ((knn_df["Year"] == TRAINING_END_YEAR) & (knn_df["Quarter"] <= TRAINING_END_QUARTER)))
    holdout_mask = (knn_df["Year"] == HOLDOUT_START_YEAR) & (knn_df["Quarter"] >= HOLDOUT_START_QUARTER)

    training_data = knn_df[training_mask]
    holdout_data = knn_df[holdout_mask]

    X_train, y_train = training_data[features], training_data[target_col]
    X_test, y_test = holdout_data[features], holdout_data[target_col]

    print(f"[KNN {label}] Training Set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"[KNN {label}] Holdout Set: X_test {X_test.shape}, y_test {y_test.shape}")

    # ⚠️ 特徵標準化 (對 KNN 極度重要)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 建立與訓練 KNN 模型
    result = KNeighborsRegressor(
        n_neighbors = n_neighbors,
        weights = weights,
        n_jobs = -1
    )

    print(f"[KNN {label}] Training...")
    result.fit(X_train_scaled, y_train)

    print(f"[KNN {label}] Predicting...")
    y_pred_train = result.predict(X_train_scaled)
    y_pred_test = result.predict(X_test_scaled)

    print(f"[KNN {label}] Done!!!!!")
    print(f"[KNN {label}] Exe Time: {(datetime.now() - start_time).total_seconds():.2f} seconds\n")
    
    # ⚠️ KNN 沒有 feature_importances_，所以不要呼叫 print_feature_importance
    
    train_evaluation = print_evaluation(f"KNN {label}", "Training Set", y_train, y_pred_train, len(features))
    test_evaluation = print_evaluation(f"KNN {label}", "Holdout Set", y_test, y_pred_test, len(features))
    evaluation = pd.concat([train_evaluation, test_evaluation], ignore_index = True)
    # evaluation.to_csv(f"KNN_{label}_Evaluation.csv", index = False)

    return evaluation


def run_linear_regressions(model: pd.DataFrame, label: str) -> dict:
    start_time = datetime.now()

    print(f"\n{'=' * 5}", f"Running Linear Regressions (OLS, Lasso, Ridge) for {label}", f"{'=' * 5}\n", sep = " ")
    lr_df = model.dropna().reset_index(drop = True)
    
    if "QuarterlyReturn" in lr_df.columns:
        exclude_cols = ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "QuarterlyReturn", "SubCategory_last1"]
        target_col = "QuarterlyReturn"
    elif "QuarterlyVolatility" in lr_df.columns:
        exclude_cols = ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "QuarterlyVolatility", "SubCategory_last1"]
        target_col = "QuarterlyVolatility"
        
    features = [col for col in lr_df.columns if col not in exclude_cols]

    training_mask = (lr_df["Year"] >= TRAINING_START_YEAR) & ((lr_df["Year"] < TRAINING_END_YEAR) | ((lr_df["Year"] == TRAINING_END_YEAR) & (lr_df["Quarter"] <= TRAINING_END_QUARTER)))
    holdout_mask = (lr_df["Year"] == HOLDOUT_START_YEAR) & (lr_df["Quarter"] >= HOLDOUT_START_QUARTER)

    training_data = lr_df[training_mask]
    holdout_data = lr_df[holdout_mask]

    X_train, y_train = training_data[features], training_data[target_col]
    X_test, y_test = holdout_data[features], holdout_data[target_col]

    print(f"[LR {label}] Training Set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"[LR {label}] Holdout Set: X_test {X_test.shape}, y_test {y_test.shape}")

    # ⚠️ 特徵標準化 (對 Lasso 和 Ridge 來說是絕對必要的)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    linear_algorithms = {
        "OLS": LinearRegression(n_jobs = -1),
        "Lasso": Lasso(alpha = 0.005, random_state = 42),
        "Ridge": Ridge(alpha = 1.0, random_state = 42)
    }

    evaluations = {}

    for name, reg_model in linear_algorithms.items():
        print(f"[{name} {label}] Training...")
        reg_model.fit(X_train_scaled, y_train)

        print(f"[{name} {label}] Predicting...\n")
        y_pred_train = reg_model.predict(X_train_scaled)
        y_pred_test = reg_model.predict(X_test_scaled)

        train_evaluation = print_evaluation(f"{name} {label}", "Training Set", y_train, y_pred_train, len(features))
        test_evaluation = print_evaluation(f"{name} {label}", "Holdout Set", y_test, y_pred_test, len(features))

        evaluations[name] = pd.concat([train_evaluation, test_evaluation], ignore_index = True)

    print(f"[LR {label}] Done!!!!!")
    print(f"[LR {label}] Exe Time: {(datetime.now() - start_time).total_seconds():.2f} seconds\n")
    
    return evaluations