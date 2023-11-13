import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
import optuna
from optuna.samplers import TPESampler
from optuna import create_study


def load_data(csv_file, x_name, y_name):
    df = pd.read_csv(csv_file)
    X = df[x_name].values
    Y = df[y_name].values
    return X, Y

def preprocess_data(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, Y_train, Y_test

def root_mean_squared_error(Y_test, Y_pred):
    return np.sqrt(mean_squared_error(Y_test, Y_pred))

def xgboost_grid_search(X_train, Y_train):
    xgb_regressor = XGBRegressor()
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1],
        'colsample_bytree': [0.8, 0.9, 1]
    }
    grid_search = GridSearchCV(xgb_regressor, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
    grid_search.fit(X_train, Y_train)
    print("Best parameters found: ", grid_search.best_params_)
    return grid_search.best_estimator_

def xgboost_grid_search_expanded(X_train, Y_train):
    xgb_regressor = XGBRegressor()
    param_grid = {
        'n_estimators': [10, 50, 100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.001, 0.01, 0.1, 0.2],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    }
    grid_search = GridSearchCV(xgb_regressor, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
    grid_search.fit(X_train, Y_train)
    print("Best parameters found: ", grid_search.best_params_)
    return grid_search.best_estimator_

def xgboost_randomized_search(X_train, Y_train):
    xgb_regressor = XGBRegressor()
    param_dist = {
        'n_estimators': range(10, 500),
        'max_depth': range(3, 10),
        'learning_rate': [0.001, 0.01, 0.1, 0.2],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    }
    random_search = RandomizedSearchCV(xgb_regressor, param_dist, scoring='neg_mean_squared_error', n_iter=100, cv=5, verbose=1)
    random_search.fit(X_train, Y_train)
    print("Best parameters found: ", random_search.best_params_)
    return random_search.best_estimator_

def objective(trial, X_train, Y_train):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }
    
    xgb_regressor = XGBRegressor(**params)
    xgb_regressor.fit(X_train, Y_train)
    
    Y_pred = xgb_regressor.predict(X_train)
    rmse = root_mean_squared_error(Y_train, Y_pred)
    return rmse

def xgboost_bayesian_optimization(X_train, Y_train):
    study = create_study(sampler=TPESampler(), direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, Y_train), n_trials=100)
    
    best_params = study.best_params
    print("Best parameters found: ", best_params)
    
    xgb_regressor = XGBRegressor(**best_params)
    xgb_regressor.fit(X_train, Y_train)
    return xgb_regressor

def xgb_train(X_train, Y_train, method, method_name):
    
    print("XGBoost Regression:")
    xgb_model = method(X_train, Y_train)
    
    # 保存模型到文件
    with open('%s.pkl'%method_name, 'wb') as f:
        pickle.dump(xgb_model, f)

def evaluate_and_plot(model, X, Y, y_name, model_name):

    Y_pred = model.predict(X)
    mse = mean_squared_error(Y, Y_pred)
    rmse = root_mean_squared_error(Y, Y_pred)
    print("Mean Squared Error: ", mse)
    print("Root Mean Squared Error: ", rmse)
    
    # 绘制散点图
    plt.scatter(Y, Y_pred, label='Predictions')
    
    # 绘制斜率为1的直线
    min_val = min(min(Y), min(Y_pred))
    max_val = max(max(Y), max(Y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r', label='Ideal fit')

    # 获取当前坐标轴
    ax = plt.gca()

    # 在右下角显示评价指标
    ax.text(0.95, 0.05, f'MSE: {mse:.2f}\nRMSE: {rmse:.2f}',
            fontsize=12, ha='right', va='bottom', transform=ax.transAxes)
    
    # 设置图表信息
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{y_name}_{model_name}')
    plt.legend()
    
    # 保存图表为JPG文件
    plt.savefig(f'{y_name}_{model_name}_regression.jpg', format='jpg')

    # 显示图表
    plt.show()

if __name__ == '__main__':
    
    # 加载数据
    csv_file = '200Case.csv'
    x_name = ['Bx', 'DT', 'Rc','SR','SOI','DOI','SA']
    y_name = 'ISFC'
    X, Y = load_data(csv_file, x_name, y_name)
    X_train, X_test, Y_train, Y_test = preprocess_data(X, Y)

    #训练
    #xgb_train(X_train, Y_train, xgboost_grid_search, 'xgboost_grid_search')
    #xgb_train(X_train, Y_train, xgboost_grid_search_expanded, 'xgboost_grid_search_expanded')
    #xgb_train(X_train, Y_train, xgboost_randomized_search, 'xgboost_randomized_search')
    #xgb_train(X_train, Y_train, xgboost_bayesian_optimization, 'xgboost_bayesian_optimization')

    # 加载已保存的模型
    with open('xgboost_randomized_search.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    # 分析数据    
    X_full = np.vstack((X_train, X_test))
    Y_full = np.hstack((Y_train, Y_test))
    evaluate_and_plot(xgb_model, X_full, Y_full, y_name, 'xgboost_randomized_search_1')