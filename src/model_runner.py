from pyexpat import features
import re
import io
import joblib
import numpy as np
import pandas as pd
import config as myCfg
import seaborn as sns
import src.function as myFn
import src.data_loader as myLoader
import src.model_runner as myRunner
import src.model_cleaner as myCleaner
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler


def assemble_final_models(model_ret, model_vol, market_QoQ, market_YoY, tech_QoQ, tech_YoY):
    """
    1. Assemble the final modeling datasets for both Return and Volatility, each with QoQ and YoY market/tech features.
    2. Return a dictionary containing the 4 final datasets.
    """
    recipes = {
        "Model1": {"base": model_ret, "market": market_QoQ, "tech": tech_QoQ},
        "Model2": {"base": model_ret, "market": market_YoY, "tech": tech_YoY},
        "Model3": {"base": model_vol, "market": market_QoQ, "tech": tech_QoQ},
        "Model4": {"base": model_vol, "market": market_YoY, "tech": tech_YoY}
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
        
        df_cleaned = myCleaner.clean_model(df_merged)
        final_models[model_name] = df_cleaned
        
        print(f"{model_name} Created Successfully!!!!!")
        if myCfg.IS_DEBUG:
            myCleaner.describe_data(df_cleaned, model_name)
            
    return final_models

def print_evaluation(model_name: str, dataset_name: str, y_true: np.ndarray, y_pred: np.ndarray, p: int) -> pd.DataFrame:
    """
    1. Print evaluation metrics for a given model and dataset. 
    2. Return a DataFrame containing the evaluation results.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    tmp_dataset_name = dataset_name.replace(" ", "_").replace(',', '').replace(")", "").replace("(", "")
    title = f"{model_name} - {dataset_name}"
    
    n = len(y_true)
    if n > p + 1:
        adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    else:
        adj_r2 = np.nan

    eva = pd.DataFrame({
        "Model": [model_name],
        "Dataset": [tmp_dataset_name],
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



def print_feature_importance(model_name: str, result: dict, features: list, plot: bool = True) -> pd.DataFrame:
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
    title_str = f"| {model_name} - Top {myCfg.TOP_N_FEATURES} Feature Importances|"
    cnt = len(title_str)
    print("+", "-" * (cnt - 2), "+", sep = "")
    print(title_str)
    print("+", "-" * (cnt - 2), "+", sep = "")

    for rank, feat, imp, cum_imp in zip(ranks, top_features, top_importances, cumulative_importances):
        print(f"  {rank:2d}. {feat:<50} : {imp:.4f}  (cumulative importance: {cum_imp:>6.4%})")
    
    df_importance = pd.DataFrame({
        "Model": model_name,
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
            ax.set_title(f"{model_name}\n\n -- Top {myCfg.TOP_N_FEATURES} Feature Importances (ESG)", fontsize = 14, pad = 15)
        elif myCfg.HAS_ESG:
            ax.set_title(f"{model_name}\n\n -- Top {myCfg.TOP_N_FEATURES} Feature Importances (ESG Dataset)", fontsize = 14, pad = 15)
        elif myCfg.IS_LARGE_DATASET:
            ax.set_title(f"{model_name}\n\n -- Top {myCfg.TOP_N_FEATURES} Feature Importances (NO ESG)", fontsize = 14, pad = 15)
        else:
            ax.set_title(f"{model_name}\n\n -- Top {myCfg.TOP_N_FEATURES} Feature Importances", fontsize = 14, pad = 15)
            
        ax.set_xlabel("Importance Score", fontsize = 12)
        ax.set_ylabel("Features", fontsize = 12)
        ax.grid(axis = 'x', linestyle = '--', alpha = 0.7)
        
        fig.tight_layout()
    
    return df_importance, fig


def run_random_forest(model: pd.DataFrame, label: str, n_estimators: int = 200, max_depth: int = 5, min_samples_leaf: int = 10, max_features = 1.0) -> tuple:
    """
    1. Run Random Forest regression on the given model dataset with specified hyperparameters.
    2. Return a tuple containing the evaluation DataFrame, feature importance DataFrame, and the importance plot figure.
    """
    start_time = datetime.now()
    print(f"\n\033[94mRunning Random Forest for {label}...\033[0m")

    rf = model.dropna().reset_index(drop = True)
    cols_to_drop = [c for c in model.columns if "Quarter_" in c]

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

    print(f"[RF {label}] Training Set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"[RF {label}] Holdout Set: X_test {X_test.shape}, y_test {y_test.shape}")

    # Initialize and train Random Forest model
    result = RandomForestRegressor(
        n_estimators = n_estimators,
        max_depth = max_depth,
        min_samples_leaf = min_samples_leaf,
        max_features = max_features,
        random_state = 42,
        n_jobs = -1              
    )

    print(f"[RF {label}] Training...")
    result.fit(X_train, y_train)

    print(f"[RF {label}] Predicting...")
    y_pred_train = result.predict(X_train)
    y_pred_test = result.predict(X_test)

    print(f"[RF {label}] Done!!!!!")
    print(f"[RF {label}] Exe Time: {(datetime.now() - start_time).total_seconds():.2f} sec")

    importance, fig = print_feature_importance(f"RF {label}", result, features, plot = True)
    train_evaluation = print_evaluation(f"RF {label}", "Training Set", y_train, y_pred_train, len(features))
    test_evaluation = print_evaluation(f"RF {label}", "Holdout Set", y_test, y_pred_test, len(features))
    evaluation = pd.concat([train_evaluation, test_evaluation], ignore_index = True)

    return evaluation, importance, fig


def run_xgboost(model: pd.DataFrame, label: str, n_estimators: int = 200, max_depth: int = 6, learning_rate: float = 0.05) -> None:
    start_time = datetime.now()
    print(f"\n\033[94mRunning XGBoost for {label}...\033[0m")
    xgb_df = model.dropna().reset_index(drop = True)
    
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
    print(f"[XGB {label}] Exe Time: {(datetime.now() - start_time).total_seconds():.2f} sec")
    
    importance, fig = print_feature_importance(f"XGB {label}", result, features, plot = True)
    train_evaluation = print_evaluation(f"XGB {label}", "Training Set", y_train, y_pred_train, len(features))
    test_evaluation = print_evaluation(f"XGB {label}", "Holdout Set", y_test, y_pred_test, len(features))
    evaluation = pd.concat([train_evaluation, test_evaluation], ignore_index = True)
    # evaluation.to_csv(f"XGB_{label}_Evaluation.csv", index = False)

    return evaluation, importance, fig, result, list(X_train.columns)


def run_knn(model: pd.DataFrame, label: str, n_neighbors: int = 5, weights: str = "distance") -> KNeighborsRegressor:
    start_time = datetime.now()

    print(f"\n\033[94mRunning KNN for {label}...\033[0m")
    knn_df = model.dropna().reset_index(drop = True)
    
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
    print(f"[KNN {label}] Exe Time: {(datetime.now() - start_time).total_seconds():.2f} sec")
    
    train_evaluation = print_evaluation(f"KNN {label}", "Training Set", y_train, y_pred_train, len(features))
    test_evaluation = print_evaluation(f"KNN {label}", "Holdout Set", y_test, y_pred_test, len(features))
    evaluation = pd.concat([train_evaluation, test_evaluation], ignore_index = True)

    return evaluation


def run_linear_regressions(model: pd.DataFrame, label: str) -> dict:
    start_time = datetime.now()

    print(f"\n\033[94mRunning Linear Regressions (OLS, Lasso, Ridge) for {label}...\033[0m")
    lr_df = model.dropna().reset_index(drop = True)
    
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
    print(f"[LR {label}] Exe Time: {(datetime.now() - start_time).total_seconds():.2f} sec")
    
    return evaluations