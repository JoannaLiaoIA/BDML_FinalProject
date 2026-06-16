import numpy as np
import pandas as pd
import config as myCfg
import seaborn as sns
import src.data_loader as myLoader
import src.algo_runner as myRunner
import src.model_cleaner as myCleaner
import src.utility_function as myFn
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler


def print_evaluation(sample: myFn.Sample, dataset_name: str, y_true: np.ndarray, y_pred: np.ndarray, p: int) -> pd.DataFrame:
    """
    1. Print evaluation metrics for a given model and dataset. 
    2. Return a DataFrame containing the evaluation results.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    title = f"{sample.sample_label} - {dataset_name}"
    
    n = len(y_true)
    if n > p + 1:
        adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    else:
        adj_r2 = np.nan

    eva = pd.DataFrame({
        "Algorithm": [sample.algorithm],
        "Model": [sample.model_label],
        "Dataset": [dataset_name],
        "Parameters": [", ".join(f"{k} = {v}" for k, v in sample.parameters.items())],
        "MSE": [f"{mse:.6f}"],
        "MAE": [f"{mae:.6f}"],
        "R-Squared": [f"{r2:.6f}"],
        "Adj-R-Squared": [f"{adj_r2:.6f}"]
    })
    
    print("=" * 10, title, "=" * 10, sep = " ")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R-Squared: {r2:.6f}")
    print(f"Adj-R-Squared: {adj_r2:.6f}")
    
    return eva



def print_feature_importance(sample: myFn.Sample, result: dict, features: list, plot: bool = True) -> pd.DataFrame:
    """
    1. Print the feature importances of a given model.
    2. Return a DataFrame containing the results.
    """
    # 1. 計算特徵重要度與排序
    importances = result.feature_importances_
    indices = np.argsort(importances)[::-1][:myCfg.TOP_N_FEATURES] 
    
    top_features = [features[i] for i in indices]
    top_importances = importances[indices]
    
    # 2. 準備 DataFrame 所需的欄位資料
    ranks = list(range(1, len(top_features) + 1))
    cumulative_importances = np.cumsum(top_importances)

    # 3. Terminal 列印輸出
    title_str = f"| {sample.sample_label} - Top {myCfg.TOP_N_FEATURES} Feature Importances|"
    cnt = len(title_str)
    print("+", "-" * (cnt - 2), "+", sep = "")
    print(title_str)
    print("+", "-" * (cnt - 2), "+", sep = "")

    for rank, feat, imp, cum_imp in zip(ranks, top_features, top_importances, cumulative_importances):
        print(f"  {rank:2d}. {feat:<50} : {imp:.4f}  (cumulative importance: {cum_imp:>6.4%})")
    
    df_importance = pd.DataFrame({
        "Algorithm": [sample.algorithm] * len(top_features),
        "Model": [sample.model_label] * len(top_features),
        "Parameters": [", ".join(f"{k} = {v}" for k, v in sample.parameters.items())] * len(top_features),
        "Rank": ranks,
        "Feature": top_features,
        "Importance": top_importances,
        "Cumulative_Importance": cumulative_importances
    })
    
    # 4. 繪製特徵重要性圖表 (Horizontal Bar Chart)
    fig = None
    if plot:
        fig, ax = plt.subplots(figsize = (10, max(8, myCfg.TOP_N_FEATURES * 0.3)))
        
        # 將 ax 傳遞給 seaborn
        sns.barplot(
            x = "Importance", 
            y = "Feature", 
            data = df_importance,
            hue = "Feature",
            palette = "viridis",
            legend = False,
            ax = ax
        )
        
        # 使用 ax.set_* 來設定標題與標籤
        if myCfg.HAS_ESG and myCfg.IS_LARGE_DATASET:
            ax.set_title(f"{sample.sample_label}\n\n -- Top {myCfg.TOP_N_FEATURES} Feature Importances (ESG)", fontsize = 14, pad = 15)
        elif myCfg.HAS_ESG:
            ax.set_title(f"{sample.sample_label}\n\n -- Top {myCfg.TOP_N_FEATURES} Feature Importances (ESG Dataset)", fontsize = 14, pad = 15)
        elif myCfg.IS_LARGE_DATASET:
            ax.set_title(f"{sample.sample_label}\n\n -- Top {myCfg.TOP_N_FEATURES} Feature Importances (NO ESG)", fontsize = 14, pad = 15)
        else:
            ax.set_title(f"{sample.sample_label}\n\n -- Top {myCfg.TOP_N_FEATURES} Feature Importances", fontsize = 14, pad = 15)
            
        ax.set_xlabel("Importance Score", fontsize = 12)
        ax.set_ylabel("Features", fontsize = 12)
        ax.grid(axis = 'x', linestyle = '--', alpha = 0.7)
        
        fig.tight_layout()
    
    return df_importance, fig


def run_random_forest(sample: myFn.Sample) -> tuple:
    """
    1. Run Random Forest regression on the given model dataset with specified hyperparameters.
    2. Return a tuple containing the evaluation DataFrame, feature importance DataFrame, and the importance plot figure.
    """
    start_time = datetime.now()
    print(f"\n\033[94mRunning Random Forest for {sample.sample_label}...\033[0m")

    rf = sample.model.dropna().reset_index(drop = True)
    cols_to_drop = [c for c in sample.model.columns if "Quarter_" in c]

    if "QuarterlyReturn" in rf.columns:
        exclude_cols = ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "QuarterlyReturn", "SubCategory_last1"]
        exclude_cols += cols_to_drop
    elif "QuarterlyVolatility" in rf.columns:
        exclude_cols = ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "QuarterlyVolatility", "SubCategory_last1"]
        exclude_cols += cols_to_drop
    features = [col for col in rf.columns if col not in exclude_cols]

    training_mask = (rf["Year"] >= myCfg.TRAINING_START_YEAR) & ((rf["Year"] < myCfg.TRAINING_END_YEAR) | ((rf["Year"] == myCfg.TRAINING_END_YEAR) & (rf["Quarter"] <= myCfg.TRAINING_END_QUARTER)))
    holdout_mask = (rf["Year"] == myCfg.HOLDOUT_START_YEAR) & (rf["Quarter"] >= myCfg.HOLDOUT_START_QUARTER)

    training_data = rf[training_mask]
    holdout_data = rf[holdout_mask]

    if "QuarterlyReturn" in rf.columns:
        X_train, y_train = training_data[features], training_data["QuarterlyReturn"]
        X_test, y_test = holdout_data[features], holdout_data["QuarterlyReturn"]
    elif "QuarterlyVolatility" in rf.columns:
        X_train, y_train = training_data[features], training_data["QuarterlyVolatility"]
        X_test, y_test = holdout_data[features], holdout_data["QuarterlyVolatility"]

    print(f"[{sample.sample_label}] Training Set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"[{sample.sample_label}] Holdout Set: X_test {X_test.shape}, y_test {y_test.shape}")

    # Initialize and train Random Forest model
    result = RandomForestRegressor(
        n_estimators = sample.parameters["n_estimators"],
        max_depth = sample.parameters["max_depth"],
        min_samples_leaf = sample.parameters["min_samples_leaf"],
        max_features = sample.parameters["max_features"],
        random_state = 42,
        n_jobs = -1              
    )

    print(f"[{sample.sample_label}] Training...")
    result.fit(X_train, y_train)

    print(f"[{sample.sample_label}] Predicting...")
    y_pred_train = result.predict(X_train)
    y_pred_test = result.predict(X_test)

    print(f"[{sample.sample_label}] Done!!!!! Exe Time: {(datetime.now() - start_time).total_seconds():.2f} sec")

    importance, fig = print_feature_importance(sample, result, features, plot = True)
    train_evaluation = print_evaluation(sample, "Training Set", y_train, y_pred_train, len(features))
    test_evaluation = print_evaluation(sample, "Holdout Set", y_test, y_pred_test, len(features))
    evaluation = pd.concat([train_evaluation, test_evaluation], ignore_index = True)

    return evaluation, importance, fig


def run_xgboost(sample: myFn.Sample) -> None:
    start_time = datetime.now()
    print(f"\n\033[94mRunning XGBoost for {sample.sample_label}...\033[0m")

    xgb_df = sample.model.dropna().reset_index(drop = True)
    n_estimators = sample.parameters["n_estimators"]
    max_depth = sample.parameters["max_depth"]
    learning_rate = sample.parameters["learning_rate"]

    if "QuarterlyReturn" in xgb_df.columns:
        exclude_cols = ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "QuarterlyReturn", "SubCategory_last1"]
        target_col = "QuarterlyReturn"
    elif "QuarterlyVolatility" in xgb_df.columns:
        exclude_cols = ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "QuarterlyVolatility", "SubCategory_last1"]
        target_col = "QuarterlyVolatility"
        
    features = [col for col in xgb_df.columns if col not in exclude_cols]

    training_mask = (xgb_df["Year"] >= myCfg.TRAINING_START_YEAR) & ((xgb_df["Year"] < myCfg.TRAINING_END_YEAR) | ((xgb_df["Year"] == myCfg.TRAINING_END_YEAR) & (xgb_df["Quarter"] <= myCfg.TRAINING_END_QUARTER)))
    holdout_mask = (xgb_df["Year"] == myCfg.HOLDOUT_START_YEAR) & (xgb_df["Quarter"] >= myCfg.HOLDOUT_START_QUARTER)

    training_data = xgb_df[training_mask]
    holdout_data = xgb_df[holdout_mask]

    X_train, y_train = training_data[features], training_data[target_col]
    X_test, y_test = holdout_data[features], holdout_data[target_col]

    print(f"[{sample.sample_label}] Training Set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"[{sample.sample_label}] Holdout Set: X_test {X_test.shape}, y_test {y_test.shape}")

    # 建立與訓練 XGBoost 模型
    result = XGBRegressor(
        n_estimators = n_estimators,
        max_depth = max_depth,
        learning_rate = learning_rate,
        random_state = 42,
        n_jobs = -1
    )

    print(f"[{sample.sample_label}] Training...")
    result.fit(X_train, y_train)

    print(f"[{sample.sample_label}] Predicting...")
    y_pred_train = result.predict(X_train)
    y_pred_test = result.predict(X_test)

    print(f"[{sample.sample_label}] Done!!!!! Exe Time: {(datetime.now() - start_time).total_seconds():.2f} sec")
    
    importance, fig = print_feature_importance(sample, result, features, plot = True)
    train_evaluation = print_evaluation(sample, "Training Set", y_train, y_pred_train, len(features))
    test_evaluation = print_evaluation(sample, "Holdout Set", y_test, y_pred_test, len(features))
    evaluation = pd.concat([train_evaluation, test_evaluation], ignore_index = True)

    return evaluation, importance, fig, result, list(X_train.columns)


def run_knn(sample: myFn.Sample) -> KNeighborsRegressor:
    start_time = datetime.now()

    print(f"\n\033[94mRunning KNN for {sample.sample_label}...\033[0m")
    knn_df = sample.model.dropna().reset_index(drop = True)
    
    if "QuarterlyReturn" in knn_df.columns:
        exclude_cols = ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "QuarterlyReturn", "SubCategory_last1"]
        target_col = "QuarterlyReturn"
    elif "QuarterlyVolatility" in knn_df.columns:
        exclude_cols = ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "QuarterlyVolatility", "SubCategory_last1"]
        target_col = "QuarterlyVolatility"
        
    features = [col for col in knn_df.columns if col not in exclude_cols]

    training_mask = (knn_df["Year"] >= myCfg.TRAINING_START_YEAR) & ((knn_df["Year"] < myCfg.TRAINING_END_YEAR) | ((knn_df["Year"] == myCfg.TRAINING_END_YEAR) & (knn_df["Quarter"] <= myCfg.TRAINING_END_QUARTER)))
    holdout_mask = (knn_df["Year"] == myCfg.HOLDOUT_START_YEAR) & (knn_df["Quarter"] >= myCfg.HOLDOUT_START_QUARTER)

    training_data = knn_df[training_mask]
    holdout_data = knn_df[holdout_mask]

    X_train, y_train = training_data[features], training_data[target_col]
    X_test, y_test = holdout_data[features], holdout_data[target_col]

    print(f"[{sample.sample_label}] Training Set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"[{sample.sample_label}] Holdout Set: X_test {X_test.shape}, y_test {y_test.shape}")

    # ⚠️ 特徵標準化 (對 KNN 極度重要)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 建立與訓練 KNN 模型
    result = KNeighborsRegressor(
        n_neighbors = sample.parameters["n_neighbors"],
        weights = sample.parameters["weights"],
        n_jobs = -1
    )

    print(f"[{sample.sample_label}] Training...")
    result.fit(X_train_scaled, y_train)

    print(f"[{sample.sample_label}] Predicting...")
    y_pred_train = result.predict(X_train_scaled)
    y_pred_test = result.predict(X_test_scaled)

    print(f"[{sample.sample_label}] Done!!!!! Exe Time: {(datetime.now() - start_time).total_seconds():.2f} sec")
    
    train_evaluation = print_evaluation(sample, "Training Set", y_train, y_pred_train, len(features))
    test_evaluation = print_evaluation(sample, "Holdout Set", y_test, y_pred_test, len(features))
    evaluation = pd.concat([train_evaluation, test_evaluation], ignore_index = True)

    return evaluation


def run_linear_regression(sample: myFn.Sample) -> dict:
    start_time = datetime.now()

    print(f"\n\033[94mRunning Linear Regressions {sample.algorithm} for {sample.sample_label}...\033[0m")
    lr_df = sample.model.dropna().reset_index(drop = True)
    
    if "QuarterlyReturn" in lr_df.columns:
        exclude_cols = ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "QuarterlyReturn", "SubCategory_last1"]
        target_col = "QuarterlyReturn"
    elif "QuarterlyVolatility" in lr_df.columns:
        exclude_cols = ["StockID", "Year", "Quarter", "TargetYear", "TargetQuarter", "QuarterlyVolatility", "SubCategory_last1"]
        target_col = "QuarterlyVolatility"
        
    features = [col for col in lr_df.columns if col not in exclude_cols]

    training_mask = (lr_df["Year"] >= myCfg.TRAINING_START_YEAR) & ((lr_df["Year"] < myCfg.TRAINING_END_YEAR) | ((lr_df["Year"] == myCfg.TRAINING_END_YEAR) & (lr_df["Quarter"] <= myCfg.TRAINING_END_QUARTER)))
    holdout_mask = (lr_df["Year"] == myCfg.HOLDOUT_START_YEAR) & (lr_df["Quarter"] >= myCfg.HOLDOUT_START_QUARTER)

    training_data = lr_df[training_mask]
    holdout_data = lr_df[holdout_mask]

    X_train, y_train = training_data[features], training_data[target_col]
    X_test, y_test = holdout_data[features], holdout_data[target_col]

    print(f"[{sample.sample_label}] Training Set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"[{sample.sample_label}] Holdout Set: X_test {X_test.shape}, y_test {y_test.shape}")

    # ⚠️ 特徵標準化 (對 Lasso 和 Ridge 來說是絕對必要的)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if sample.algorithm == "OLS":
        model = LinearRegression(n_jobs=-1, **sample.parameters)
    elif sample.algorithm == "Lasso":
        model = Lasso(random_state=42, **sample.parameters)
    elif sample.algorithm == "Ridge":
        model = Ridge(random_state=42, **sample.parameters)
    else:
        raise ValueError(f"Not supported algorithm: {sample.algorithm}")
    
    print(f"[{sample.sample_label}] Training...")
    model.fit(X_train_scaled, y_train)

    print(f"[{sample.sample_label}] Predicting...\n")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_evaluation = print_evaluation(sample, "Training Set", y_train, y_pred_train, len(features))
    test_evaluation = print_evaluation(sample, "Holdout Set", y_test, y_pred_test, len(features))

    evaluation = pd.concat([train_evaluation, test_evaluation], ignore_index = True)

    print(f"[{sample.sample_label}] Done!!!!!")
    print(f"[{sample.sample_label}] Exe Time: {(datetime.now() - start_time).total_seconds():.2f} sec")

    return evaluation