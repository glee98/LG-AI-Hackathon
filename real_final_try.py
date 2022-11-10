import pandas as pd
import random
import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import xgboost as xgb
#from xgboost import XGBClassifier


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHINHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) #set 고정

def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(1,15): # ignore 'ID'
        rmse = mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:14])
    print(score)


train_df = pd.read_csv('./train.csv').drop(columns=['ID','X_04','X_23','X_47','X_48','X_10','X_11','X_34','X_36'])

train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature

#결측치 확인 - x feature, y feature 둘 다 없음
print(train_x.isnull().any())
print(train_y.isnull().any())

#Linear Regression Model setup
LR = MultiOutputRegressor(LinearRegression()).fit(train_x, train_y)
#preds= LR.predict(test_x)


#train test_x value
test_x = pd.read_csv('./test.csv').drop(columns=['ID','X_04','X_23','X_47','X_48','X_10','X_11','X_34','X_36'])
#preds= LR.predict(test_x)

#test_y= pd.read_csv('./test.csv').drop(columns=['ID'])
#preds_y = LR.predict(test_y)

#Feature X_01 확인
"""
plt.figure(figsize=(30,5))
plt.plot(train_x['X_01'], label='X_01') 
#plt.legend()
plt.show()
"""
"-----------------"


"""
xgb_model = XGBClassifier(n_estimators=100)
params = {'max_depth':[5,7], 'min_child_weight':[1,3], 'colsample_bytree':[0.5,0.75]}
gridcv = GridSearchCV(xgb_model, param_grid=params, cv=3)
gridcv.fit(train_x, train_y, early_stopping_rounds=30, eval_metric='auc')
xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=7, min_child_weight=1, colsample_bytree=0.75, reg_alpha=0.03)
"""

model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=2214, learning_rate=0.011852778535192848, gamma = 0.13330690180214358, subsample= 0.85, colsample_bytree = 0.9, max_depth=13, 
                                    min_child_weight = 14.54774756029115, reg_lambda =1.224618131945341, reg_alpha =0.00046437900749889116)).fit(train_x, train_y)
#default learning_rate = 0.08


#model = MultiOutputRegressor(SVR(kernel = 'rbf', C=1000, gamma = 1)).fit(train_x ,train_y)
preds = model.predict(test_x)


#submission

submit = pd.read_csv('./sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]
print('Done.')

submit.to_csv('./submit_latest7_dropXval7.csv', index=False)


