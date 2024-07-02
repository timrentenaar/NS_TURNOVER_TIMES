
# General libraries  
import pandas as pd 
import numpy as np 

# Modelling
import os
import json
import optuna
import joblib
import lightgbm as lgb 
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Create a folder for saving the outputs
output_dir = r"C:\Users\Gebruiker\Documents\CODE\Master\Thesis\Local_Testing\output\turn1200"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


"""
LOAD AND SPLIT DATA
"""
df = pd.read_csv(r"df_large.csv", low_memory = False)

# split data into 0-600, 600-1200, 1200-1800, and 1800 plus parts
df_0_600 = df[df["REALIZED_TURNOVER_TIME"] <= 600]
df_600_1200 = df[(df["REALIZED_TURNOVER_TIME"] > 600) & (df["REALIZED_TURNOVER_TIME"] <= 1200)]
df_1200_1800 = df[(df["REALIZED_TURNOVER_TIME"] > 1200) & (df["REALIZED_TURNOVER_TIME"] <= 1800)]
df_1800_2500 = df[df["REALIZED_TURNOVER_TIME"] > 1800]

"""
MODELLING
"""
def objective(trial, X, y, X_train, X_val, y_train, y_val):
    params = {    
        # Fixed Params
        "metric": "mean_absolute_error",
        "verbosity": -1,
        "random_state" : trial.suggest_categorical("random_state", [1]),
     
        # Training Params, fixed after intial optimization
        "n_estimators": trial.suggest_categorical("n_estimators", [500]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.01]),
                
        # Data Sampling
        'feature_fraction': trial.suggest_float('feature_fraction', 1.0, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.8, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0,12),
        
        # Tree Shape
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "num_leaves": trial.suggest_int("num_leaves", 8, 512),
        
        # Tree Growth
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 1000),
        "min_gain_to_split": trial.suggest_int("min_gain_to_split", 0, 20),
        
        # Regularization
        "lambda_l1" : trial.suggest_int("lambda_l1", 0, 100, step = 5),
        "lambda_l2" : trial.suggest_int("lambda_l2", 0, 100, step = 5)
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    
    model = LGBMRegressor(**params)
    
    mae_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train_fold, y_train_fold, 
                  categorical_feature = "auto",
                  eval_set = [(X_val, y_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=50)]
                 )
        y_pred = model.predict(X_test_fold)
        
        mae = round(mean_absolute_error(y_test_fold, y_pred), 2)
        mae_scores.append(mae)
    
    avg_mae = np.mean(mae_scores)
    
    return avg_mae

def optimize_model(dataframe, trials, study):
    df = dataframe
    y = df["REALIZED_TURNOVER_TIME"]
    X = df[["DALUREN","CUM_DISTANCE_M", "STATION", "COMBINE", "SPLIT", "ROLLINGSTOCK_TYPE", "NUMBER_CARRIAGES", 
            "HOUR_sin", "HOUR_cos", "DAY_OF_WEEK_sin", "DAY_OF_WEEK_cos", "PLAN_TURNOVER_TIME", "DRIVER_CHANGE"]]

    # Initial split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Further split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # Pass X and y as well as train and validation sets to the objective function
    study.optimize(lambda trial: objective(trial, X, y, X_train, X_val, y_train, y_val), n_trials=trials)
    
    # Plot and save study information
    history_fig = optuna.visualization.plot_optimization_history(study)
    history_fig.write_image(os.path.join(output_dir, 'optimization_history.png'))
    history_fig.show()
    importances_fig = optuna.visualization.plot_param_importances(study)
    importances_fig.write_image(os.path.join(output_dir, 'param_importances.png'))
    importances_fig.show()
    parallel_fig = optuna.visualization.plot_parallel_coordinate(study, params=["bagging_fraction", "bagging_freq", "feature_fraction", "lambda_l1", "lambda_l2", "max_depth", "min_data_in_leaf", "min_gain_to_split", "num_leaves"])
    parallel_fig.write_image(os.path.join(output_dir, 'parallel_coordinate.png'))
    parallel_fig.show()

    # Save study 
    joblib.dump(study, r"C:\Users\Gebruiker\Documents\CODE\Master\Thesis\Local_Testing\study1200.pkl")

    # Train the model with the best parameters
    best_params = study.best_params
    model = LGBMRegressor(**best_params)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate MAE and RMSE
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Save the best parameters and metrics
    results = {
        "best_params": best_params,
        "mae": mae,
    }
    
    with open(os.path.join(output_dir, 'best_params_and_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)

    return study.best_params

# Load Existing study
study = joblib.load(r"C:\Users\Gebruiker\Documents\CODE\Master\Thesis\Local_Testing\study1200.pkl")

#Continue the study, assumes there is an existing study
optimize_model(df_600_1200, 49, study)


