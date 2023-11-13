import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel


def load_data(csv_file, x_name, y_name):
    df = pd.read_csv(csv_file)
    X = df[x_name].values
    Y = df[y_name].values
    return X, Y

def preprocess_data(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    #scaler = MinMaxScaler()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, Y_train, Y_test

def preprocess_data_feture_selection(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    #scaler = MinMaxScaler()
    lasso = LassoCV(cv=5)
    lasso.fit(X_train, Y_train)
    feature_selector = SelectFromModel(lasso, prefit=True)
    X_train = feature_selector.transform(X_train)
    X_test = feature_selector.transform(X_test)
    return X_train, X_test, Y_train, Y_test

def root_mean_squared_error(Y_test, Y_pred):
    return np.sqrt(mean_squared_error(Y_test, Y_pred))

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

def random_forest_randomized_search(X_train, Y_train):
    rf_regressor = RandomForestRegressor()
    param_dist = {
        'n_estimators': range(10, 500),
        'max_depth': range(3, 20),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    random_search = RandomizedSearchCV(rf_regressor, param_dist, scoring='neg_mean_squared_error', n_iter=100, cv=5, verbose=1)
    random_search.fit(X_train, Y_train)
    print("Best parameters found: ", random_search.best_params_)
    return random_search.best_estimator_

def svm_randomized_search(X_train, Y_train):
    svm_regressor = SVR()
    param_dist = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': range(1, 6),
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto'] + list(np.logspace(-4, 0, num=5)),
        'coef0': [-1, 0, 1],
        'shrinking': [True, False]
    }
    random_search = RandomizedSearchCV(svm_regressor, param_dist, scoring='neg_mean_squared_error', n_iter=100, cv=5, verbose=1)
    random_search.fit(X_train, Y_train)
    print("Best parameters found: ", random_search.best_params_)
    return random_search.best_estimator_

def svm_randomized_search_expand(X_train, Y_train):
    svm_regressor = SVR()
    param_dist = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': range(1, 6),
        'C': np.logspace(-3, 3, num=7),  # 扩大C的搜索范围
        'gamma': ['scale', 'auto'] + list(np.logspace(-6, 0, num=7)),  # 扩大gamma的搜索范围
        'coef0': [-1, 0, 1],
        'shrinking': [True, False]
    }
    random_search = RandomizedSearchCV(svm_regressor, param_dist, scoring='neg_mean_squared_error', n_iter=200, cv=5, verbose=1) # 增加迭代次数为200
    random_search.fit(X_train, Y_train)
    print("Best parameters found: ", random_search.best_params_)
    return random_search.best_estimator_

def xgb_train(X_train, Y_train, method, method_name):
       
    reg_model = method(X_train, Y_train)
    
    # 保存模型到文件
    with open('%s.pkl'%method_name, 'wb') as f:
        pickle.dump(reg_model, f)

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
    X_train, X_test, Y_train, Y_test = preprocess_data_feture_selection(X, Y)

    #训练及保存模型
    #xgb_train(X_train, Y_train, xgboost_grid_search_expanded, 'xgboost_grid_search_expanded')
    #xgb_train(X_train, Y_train, random_forest_randomized_search, 'random_forest_randomized_search')
    xgb_train(X_train, Y_train, svm_randomized_search, 'svm_randomized_search')

    # 加载已保存的模型
    with open('svm_randomized_search.pkl', 'rb') as f:
        reg_model = pickle.load(f)

    # 分析数据    
    X_full = np.vstack((X_train, X_test))
    Y_full = np.hstack((Y_train, Y_test))

    #最后一个为保留图片名称
    evaluate_and_plot(reg_model, X_full, Y_full, y_name, 'svm_randomized_search')