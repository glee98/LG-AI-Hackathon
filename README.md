# LG-AI-Hackathon

## LG AImers 자율주행 센서의 안테나 성능 예측 AI 경진대회

### Initial Goal
-공정 데이터를 활용하여 Radar 센서의 안테나 성능 예측을 위한 AI 모델 개발

#### Used Language
- Python

#### Used Library
```c
import pandas as pd
import random
import os
import numpy as np
from statistics import mean
from tkinter import Y

from catboost import CatBoostRegressor
import optuna 
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
```

#### Code Review
- Seed 고정 및 NRMSE 값 계산식 setup
```c
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHINHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) #set 고정

def lg_nrmse(gt, preds):
    all_nrmse = []
    for idx in range(1,15): # ignore 'ID'
        rmse = mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:14])
    print(score)
 ```
- 학습 데이터에서 불필요한 Feature 제거 및 결측치 확인
```c
train_df = pd.read_csv('./train.csv').drop(columns=['ID','X_04','X_23','X_47','X_48','X_10','X_11','X_34','X_36'])

train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature

print(train_x.isnull().any())
print(train_y.isnull().any())
```
- Optuna를 이용해 최적의 Hyperprameter 값 도출
```c
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
```
```c
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
    ```

