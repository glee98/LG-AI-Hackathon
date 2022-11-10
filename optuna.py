
from statistics import mean
from tkinter import Y
from catboost import CatBoostRegressor
import optuna 
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor

from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor

import numpy as np

import pandas as pd

train_df = pd.read_csv('./train.csv').drop(columns=['ID','X_04','X_23','X_47','X_48','X_10','X_11','X_34','X_36'])
X = train_df.iloc[:,1:57]

def objective(trial: Trial) -> float:
    params = {
        
        "n_estimators" : trial.suggest_int('n_estimators', 500, 3000),
        'max_depth':trial.suggest_int("max_depth", 4, 16),
        'min_child_weight': trial.suggest_loguniform("min_child_weight", 10, 2000),
        'gamma':trial.suggest_float("gamma", 0.1, 1.0, log=True),
        'learning_rate': trial.suggest_loguniform("learning_rate", 0.001, 0.4),
        'colsample_bytree':trial.suggest_discrete_uniform('colsample_bytree',0.5, 1, 0.1),
        'nthread' : -1,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.005),
        'random_state': 42,
       

      #x8 x9 x49 표준편차 큼
    }

    X_tr, X_val, y_tr, y_val = train_test_split(X, Y, test_size=0.3)
    

    model = MultiOutputRegressor(xgb.XGBRegressor(**params))
    
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_tr,y_tr), (X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )
    
  
    cat_pred = model.predict(X_val)
    log_score = mean_absolute_error(y_val, cat_pred)

    return log_score


output = pd.DataFrame()
test_df = pd.read_csv('./test.csv').drop(columns=['ID','X_04','X_23','X_47','X_48','X_10','X_11','X_34','X_36'])
id = test_df.iloc[:,:1]
test_df = test_df.iloc[:,1:]

for i in range(14):
    Y = train_df.iloc[:,57+i:58+i]

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name = "trial 6",
        direction = "minimize",
        sampler=sampler,
    )

    study.optimize(objective, n_trials=50)
    print("Best Score: ", study.best_value)
    print("Best trial: ", study.best_trial.params)


    cat_p = study.best_trial.params
    cat= MultiOutputRegressor(xgb.XGBRegressor(**cat_p))

    skf = StratifiedKFold(5)

    preds = []
    for tr_id, val_id in skf.split(X,Y):
        X_tr = X.iloc[tr_id]
        y_tr = Y.iloc[tr_id]

        cat.fit(X_tr, y_tr, verbose=0)
        
        pred = cat.predict(test_df)
        preds.append(pred)
    cat_pred = np.mean(preds, axis=0)

    print(cat_pred)