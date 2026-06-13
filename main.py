import os
import json
import joblib
import numpy as np
import pandas as pd
import config as myCfg
import src.function as myFn
import src.data_loader as myLoader
import src.model_runner as myRunner
import src.model_cleaner as myCleaner
from datetime import datetime
from sklearn.model_selection import ParameterGrid


def main():
    print("\n##### Initializing #####\n")
    print(f"Target period: {myCfg.START_DATE} to {myCfg.END_DATE}")
    print(f"Training period: {myCfg.TRAIN_START_DATE} to {myCfg.TRAIN_END_DATE}")
    print(f"Holdout period: {myCfg.HOLDOUT_START_DATE} to {myCfg.HOLDOUT_END_DATE}")
    print()

    print(f"Debug mode: {'ON' if myCfg.IS_DEBUG else 'OFF'}")
    print(f"Using ESG data: {'YES' if myCfg.HAS_ESG else 'NO'}")
    print(f"Using Technology Index data: {'YES' if myCfg.HAS_TECH_IDX else 'NO'}")
    print(f"Using large dataset: {'YES' if myCfg.IS_LARGE_DATASET else 'NO'}")
    print(f"Using backup regressors: {'YES' if myCfg.HAS_BACKUP_REGRESSORS else 'NO'}")
    print()

    print(f"Using template file name: {myCfg.TEMPLATE_FILE_NAME}")
    print()

    print(f"Initialization complete!")
    
    # +-----------+
    # | Load data |
    # +-----------+
    print("\n##### Loading Data #####\n")
    df_daily_transaction = myLoader.load_daily_transaction()
    df_company_category = myLoader.load_company_category()
    df_tech_idx = myLoader.load_tech_index()
    df_rediscount_rate = myLoader.load_rediscount_rate()
    df_cpi = myLoader.load_cpi()
    df_unemployment_rate = myLoader.load_unemployment_rate()
    df_eps = myLoader.load_eps()
    df_revenue = myLoader.load_revenue()
    df_roe_roa_grossmargin = myLoader.load_roe_roa_grossmargin()
    df_esg = myLoader.load_esg()

    
    # +---------------------------+
    # | Convert to Quarterly Data |
    # +---------------------------+
    print("\n##### Converting to Quarterly Data #####\n")

    # Annually -> Quarterly
    df_esg_quarterly = myCleaner.annual_to_quarterly(
        df = df_esg,
        target_cols = [col for col in df_esg.columns if col not in ["Duration", "StockID"]],
        method = "duplicate"
    )
    print("Annually data converted to quarterly successfully.")

    # Quarterly -> Quarterly
    df_eps_quarterly = myCleaner.quarter_to_quarterly(
        df = df_eps,
        target_cols = ["EPS"]
    )
    df_revenue_quarterly = myCleaner.quarter_to_quarterly(
        df = df_revenue,
        target_cols = ["Last3mCumulativeRevenue", "Last3mCumulativeRevenueGrowthRate"]
    )
    df_roe_roa_grossmargin_quarterly = myCleaner.quarter_to_quarterly(
        df = df_roe_roa_grossmargin,
        target_cols = ["ROE", "ROA", "GrossMargin"]
    )
    print("Quarterly data converted successfully.")

    # Monthly -> Quarterly
    df_cpi_quarterly = myCleaner.month_to_quarterly(
        df = df_cpi,
        target_cols = ["OverallIndex"]
    )
    df_unemployment_rate_quarterly = myCleaner.month_to_quarterly(
        df = df_unemployment_rate,
        target_cols = ["TotalUnempPct"]
    )
    df_rediscount_rate_quarterly = myCleaner.month_to_quarterly(
        df = df_rediscount_rate,
        target_cols = ["RediscountRate"]
    )
    print("Monthly data converted to quarterly successfully.")

    # Daily -> Quarterly
    df_daily_transaction_tmp_sum = myCleaner.daily_to_quarterly(
        df = df_daily_transaction,
        target_cols = ["TradingVolume", "TradingMoney", "TradingTurnover"],
        method = "sum"
    )
    df_daily_transaction_tmp_mean = myCleaner.daily_to_quarterly(
        df = df_daily_transaction,
        target_cols = ["ClosingPrice", "Spread"],
        method = "mean"
    )
    df_transaction_quarterly = pd.merge(
        df_daily_transaction_tmp_sum,
        df_daily_transaction_tmp_mean,
        on = ["Year", "Quarter", "StockID"],
        how = "inner")

    df_tech_idx_quarterly = myCleaner.daily_to_quarterly(
        df = df_tech_idx,
        target_cols = ["VIX", "SOX", "DJIA", "IXIC", "SP500", "FnG"],
        method = "mean"
    )
    print("Daily data converted to quarterly successfully.")
    
    # +-----------------------------------------------+
    # | Split, Merge and Extract Time Series Features |
    # +-----------------------------------------------+
    print("\n##### Extracting Time Series Features and Merging #####\n")

    # DV: y_{t + 1}
    df_dv = myCleaner.add_dv_features(
        df = df_transaction_quarterly
    )

    # IV: Operating Performance
    df_op_quarterly = myCleaner.add_op_features(
        df_eps = df_eps_quarterly,
        df_revenue = df_revenue_quarterly,
        df_roe_roa_grossmargin = df_roe_roa_grossmargin_quarterly,
        df_company_category = df_company_category
    )

    # IV: Transaction Dataransfer_df_tran
    df_transaction_quarterly = myCleaner.add_transaction_features(
        df = df_transaction_quarterly
    )

    # IV: Technical Index
    df_tech_idx_features_QoQ, df_tech_idx_features_YoY = myCleaner.add_tech_idx_features(df_tech_idx_quarterly)

    # IV: Market Index
    df_market_idx_features_QoQ, df_market_idx_features_YoY = myCleaner.add_market_idx_features(
        df_cpi = df_cpi_quarterly,
        df_unemployment_rate = df_unemployment_rate_quarterly,
        df_rediscount_rate = df_rediscount_rate_quarterly
    )

    # +------------------+
    # | Modeling Dataset |
    # +------------------+
    print("\n##### Modeling Datasets #####\n")

    # Lag
    df_op_lag = myCleaner.create_lag_features(
        df = df_op_quarterly,
        exclude_cols = ["StockID", "SubCategory"]
    )
    df_esg_lag = myCleaner.create_lag_features(
        df = df_esg_quarterly, 
        exclude_cols = ["StockID", "SubCategory"],
        prefix = "ESG_") if myCfg.HAS_ESG else None
    df_market_QoQ_lag = myCleaner.create_lag_features(
        df = df_market_idx_features_QoQ,
        exclude_cols = []
    )
    df_market_YoY_lag = myCleaner.create_lag_features(
        df = df_market_idx_features_YoY,
        exclude_cols = []
    )

    # Current
    trans_cols = [c for c in df_transaction_quarterly.columns if c not in ["StockID", "Year", "Quarter", "SubCategory"]]
    df_trans_cur = df_transaction_quarterly.rename(columns = {c: f"{c}_cur" for c in trans_cols})

    keep_tech_QoQ = ["Year", "Quarter"] + [c for c in df_tech_idx_features_QoQ.columns if "QoQ" in c]
    df_tech_QoQ_cur = df_tech_idx_features_QoQ.sort_values(by = ["Year", "Quarter"])[keep_tech_QoQ]

    keep_tech_YoY = ["Year", "Quarter"] + [c for c in df_tech_idx_features_YoY.columns if "YoY" in c]
    df_tech_YoY_cur = df_tech_idx_features_YoY.sort_values(by = ["Year", "Quarter"])[keep_tech_YoY]

    # Base DVs
    model_ret = df_dv[["StockID", "Year", "Quarter", "QuarterlyReturn", "QuarterlyReturn_Lag1", "QuarterlyReturn_Lag2", "QuarterlyReturn_Lag3"]].copy()
    model_vol = df_dv[["StockID", "Year", "Quarter", "QuarterlyVolatility", "QuarterlyVolatility_Lag1", "QuarterlyVolatility_Lag2", "QuarterlyVolatility_Lag3"]].copy()

    model_ret = model_ret.join(pd.get_dummies(model_ret["Quarter"], prefix = "Quarter", drop_first = True, dtype = int))
    model_vol = model_vol.join(pd.get_dummies(model_vol["Quarter"], prefix = "Quarter", drop_first = True, dtype = int))

    # Merge all features
    model_return_general = myCleaner.merge_base_features(
        df_base = model_ret,
        df_trans_cur = df_trans_cur,
        df_op_lag = df_op_lag,
        df_esg_lag = df_esg_lag,
        has_esg = myCfg.HAS_ESG
    )
    model_volatility_general = myCleaner.merge_base_features(
        df_base = model_vol,
        df_trans_cur = df_trans_cur,
        df_op_lag = df_op_lag,
        df_esg_lag = df_esg_lag,
        has_esg = myCfg.HAS_ESG
    )

    # +---------------------------+
    # | Assemble Modeling Dataset |
    # +---------------------------+
    print("\n##### Assembling Modeling Datasets #####\n")
    models_dict = myRunner.assemble_final_models(
        model_ret = model_return_general,
        model_vol = model_volatility_general,
        market_QoQ = df_market_QoQ_lag,
        market_YoY = df_market_YoY_lag,
        tech_QoQ = df_tech_QoQ_cur,
        tech_YoY = df_tech_YoY_cur
    )

    # +----------------+
    # | Run Algorithms |
    # +----------------+
    print("\n##### Running Algorithms #####\n")

    all_evaluations = []
    all_importances = []

    # Experiment Tracking
    run_time = datetime.now().strftime("%y%m%d_%H%M")
    exp_name = f"{myCfg.TEMPLATE_FILE_NAME}_{run_time}"
    exp_dir = f"./experiments/{exp_name}"

    os.makedirs(exp_dir, exist_ok = True)
    os.makedirs(f"{exp_dir}/models", exist_ok = True)
    os.makedirs(f"{exp_dir}/figures", exist_ok = True)

    for label, df in models_dict.items():
        print(f"\nTraining {label}...")

        # --- 1. Random Forest (使用 Grid Search) ---
        for params in ParameterGrid(myCfg.RF_GRID):
            # 動態產生帶有參數的標籤，例如 "Model 1 [n_estimators=100, max_depth=5]"
            param_str = ", ".join(f"{k} = {v}" for k, v in params.items())
            algo_label = f"{label} ({param_str} - {df["Year"].min()}Q{df["Quarter"].min()} to {df["Year"].max()}Q{df["Quarter"].max()})"
            temp_label = f"{algo_label.replace(" ", "_").replace(',', '')}"
            
            # 使用 **params 進行字典解包
            eva_rf, imp_rf, fig_rf = myRunner.run_random_forest(
                model = df,
                label = algo_label,
                n_estimators = params["n_estimators"],
                max_depth = params["max_depth"],
                min_samples_leaf = params["min_samples_leaf"],
                max_features = params["max_features"])
            
            all_evaluations.append(eva_rf)
            all_importances.append(imp_rf)
            fig_rf.savefig(
                f"{exp_dir}/figures/RF_{temp_label}.jpg",
                format = "jpg",
                bbox_inches = 'tight'
            )

        # --- 2. XGBoost (使用 Grid Search) ---
        for params in ParameterGrid(myCfg.XGB_GRID):
            param_str = ", ".join(f"{k} = {v}" for k, v in params.items())
            algo_label = f"{label} ({param_str}) - {df["Year"].min()}Q{df["Quarter"].min()} to {df["Year"].max()}Q{df["Quarter"].max()})"
            temp_label = f"{algo_label.replace(" ", "_").replace(',', '')}"
            
            eva_xgb, imp_xgb, fig_xgb, xgb_result, xgb_cols = myRunner.run_xgboost(
                model = df,
                label = algo_label,
                n_estimators = params["n_estimators"],
                max_depth = params["max_depth"],
                learning_rate = params["learning_rate"]
            )

            all_evaluations.append(eva_xgb)
            all_importances.append(imp_xgb)
            fig_xgb.savefig(
                f"{exp_dir}/figures/XGB_{temp_label}.jpg",
                format = "jpg",
                bbox_inches = 'tight'
            )
            if myCfg.HAS_BACKUP_REGRESSORS:
                joblib.dump(
                    xgb_result,
                    f"{exp_dir}/models/XGB_{temp_label}.pkl"
                )
                joblib.dump(
                    xgb_cols,
                    f"{exp_dir}/models/XGB_{temp_label}_cols.pkl"
                )

        # --- 3. KNN (不調參，維持單次執行) ---
        eva_knn = myRunner.run_knn(
            model = df,
            label = f"{label} - {df['Year'].min()}Q{df['Quarter'].min()} to {df['Year'].max()}Q{df['Quarter'].max()}",
            n_neighbors = 5,
            weights = "distance"
        )
        all_evaluations.append(eva_knn)

        # --- 4. Linear Regressions (不調參，維持單次執行) ---
        eva_lr_dict = myRunner.run_linear_regressions(
            model = df,
            label = f"{label} - {df['Year'].min()}Q{df['Quarter'].min()} to {df['Year'].max()}Q{df['Quarter'].max()}"
        )
        all_evaluations.append(eva_lr_dict["OLS"])
        all_evaluations.append(eva_lr_dict["Lasso"])
        all_evaluations.append(eva_lr_dict["Ridge"])

    print("\nAggregating evaluation metrics...")
    final_evaluation = pd.concat(all_evaluations, ignore_index = True)
    final_importance = pd.concat(all_importances, ignore_index = True)

    final_evaluation.to_csv(f"{exp_dir}/Eva_{myCfg.TEMPLATE_FILE_NAME}.csv", index = False)
    final_importance.to_csv(f"{exp_dir}/Imp_{myCfg.TEMPLATE_FILE_NAME}.csv", index = False)

    config_dict = {key: getattr(myCfg, key) for key in dir(myCfg) if key.isupper()}
    
    with open(f"{exp_dir}/config.json", "w", encoding = "utf-8") as f:
        json.dump(config_dict, f, indent = 4, ensure_ascii = False)

    print(f"Experiment results successfully saved to: {exp_dir}")
    print("\nPipeline execution finished successfully!!!!!")

if __name__ == "__main__":
    main()