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

    # 加载已保存的模型
    with open('xgboost_grid_search_expanded.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    # 分析数据    
    X_full = np.vstack((X_train, X_test))
    Y_full = np.hstack((Y_train, Y_test))
    evaluate_and_plot(xgb_model, X_full, Y_full, y_name, 'xgboost_grid_search_expanded')