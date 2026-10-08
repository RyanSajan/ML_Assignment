import sys, os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings("ignore")

ROLL_NUMBER = "IMT2024036"
DATA_DIRECTORY = "."
OUTPUT_DIRECTORY = "."

CONFIG = {
    1: {"degrees": range(1, 11)},
    2: {"degrees": range(1, 21)}
}

RIDGE_ALPHAS = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10] 
LASSO_ALPHAS = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2]
ELASTICNET_ALPHAS = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2]
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]

NUM_FOLDS = 5

def create_regression_pipeline(model_type: str, degree: int, alpha: float, l1_ratio: float = 0.5):
    if model_type == "linear":
        estimator = LinearRegression()
    elif model_type == "ridge":
        estimator = Ridge(alpha=alpha) 
    elif model_type == "lasso":
        estimator = Lasso(alpha=alpha, max_iter=10000)
    elif model_type == "elasticnet":
        estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    return make_pipeline(
        PolynomialFeatures(degree, include_bias=False), 
        StandardScaler(), 
        estimator
    )

def evaluate_hyperparameters(X: np.ndarray, y: np.ndarray, model_type: str, degree: int, alpha: float, l1_ratio: float = 0.5) -> tuple:
    mse_scores, r2_scores = [], []
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)
    
    for train_index, val_index in kfold.split(X):
        pipeline = create_regression_pipeline(model_type, degree, alpha, l1_ratio)
        pipeline.fit(X[train_index], y[train_index])
        predictions = pipeline.predict(X[val_index])
        
        mse_scores.append(mean_squared_error(y[val_index], predictions))
        r2_scores.append(r2_score(y[val_index], predictions))
        
    return np.mean(mse_scores), np.mean(r2_scores)

def plot_cv_results(results_df: pd.DataFrame, dataset_id: int, output_directory: str):
    plt.figure(figsize=(8, 5))
    plt.plot(results_df['degree'], results_df['cv_mse'], marker='o', linestyle='-', color='b', label='CV MSE')
    
    best_row = results_df.loc[results_df.cv_mse.idxmin()]
    plt.axvline(x=best_row.degree, color='r', linestyle='--', label=f'Best Degree ({int(best_row.degree)})')
    
    plt.title(f'Cross-Validation MSE vs. Polynomial Degree (Var {dataset_id})', fontsize=12, fontweight='bold')
    plt.xlabel('Polynomial Degree', fontsize=10)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plot_filename = f"{output_directory}/optimal_degree_{dataset_id}.png"
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    plt.close()

def process_dataset(dataset_id: int, roll_number: str, data_directory: str, output_directory: str):
    train_file = f"{data_directory}/{roll_number}_train_var{dataset_id}.csv"
    test_file = f"{data_directory}/{roll_number}_test_var{dataset_id}.csv"
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    feature_columns = [col for col in train_df.columns if col != "y"]
    X_train = train_df[feature_columns].values
    y_train = train_df["y"].values
    X_test = test_df[feature_columns].values

    print(f"\n=============================================")
    print(f" Processing Dataset: var{dataset_id} ({X_train.shape[0]} rows, {len(feature_columns)} features)")
    print(f"=============================================")
    
    search_records = []

    for degree in CONFIG[dataset_id]["degrees"]:
        best_model_config = None

        grid = [("linear", 0.0, 0.0)] + [("ridge", alpha, 0.0) for alpha in RIDGE_ALPHAS]

        if (dataset_id == 1 and degree <= 6) or (dataset_id == 2 and degree <= 9): 
            grid += [("lasso", alpha, 1.0) for alpha in LASSO_ALPHAS] + \
                    [("elasticnet", alpha, l1_ratio) for alpha in ELASTICNET_ALPHAS for l1_ratio in L1_RATIOS]

        for model_type, alpha, l1_ratio in grid:
            cv_mse, cv_r2 = evaluate_hyperparameters(X_train, y_train, model_type, degree, alpha, l1_ratio)
            if best_model_config is None or cv_mse < best_model_config[3]:
                best_model_config = (model_type, alpha, l1_ratio, cv_mse, cv_r2)

        pipeline = create_regression_pipeline(best_model_config[0], degree, best_model_config[1], best_model_config[2])
        pipeline.fit(X_train, y_train)
        train_predictions = pipeline.predict(X_train)

        search_records.append({
            "degree": degree, 
            "model": best_model_config[0], 
            "alpha": best_model_config[1], 
            "l1_ratio": best_model_config[2],
            "cv_mse": best_model_config[3], 
            "cv_r2": best_model_config[4],
            "train_mse": mean_squared_error(y_train, train_predictions), 
            "train_r2": r2_score(y_train, train_predictions)
        })
        print(f" Degree {degree:2d}| Model: {best_model_config[0]:10s} | CV R2 : {best_model_config[4]:.5g} | CV MSE: {best_model_config[3]:.5g}")
        
    results_df = pd.DataFrame(search_records)
    results_df.to_csv(f"{output_directory}/degree_search_var{dataset_id}.csv", index=False)

    plot_cv_results(results_df, dataset_id, output_directory)
    
    best_row = results_df.loc[results_df.cv_mse.idxmin()]

    print(f"\n Optimal Configuration Found for var{dataset_id}:")
    print(f"      1. Best Degree   : {int(best_row.degree)}")
    print(f"      2. Best Model    : {best_row.model.upper()}")
    print(f"      3. Best Alpha    : {best_row.alpha:g}")
    print(f"      4. Best L1 Ratio : {best_row.l1_ratio:.2f}")
    print(f"      5. Best CV MSE   : {best_row.cv_mse:.5g}")
    print(f"      6. Best CV R²    : {best_row.cv_r2:.5f}\n")
    
    final_pipeline = create_regression_pipeline(best_row.model, int(best_row.degree), best_row.alpha, best_row.l1_ratio)
    final_pipeline.fit(X_train, y_train)
    test_predictions = final_pipeline.predict(X_test)
    
    pd.DataFrame({"y": test_predictions}).to_csv(f"{output_directory}/{roll_number}_pred_var{dataset_id}.csv", index=False)

def main():
    for dataset_id in (1, 2):
        process_dataset(dataset_id, ROLL_NUMBER, DATA_DIRECTORY, OUTPUT_DIRECTORY)

if __name__ == "__main__":
    main()