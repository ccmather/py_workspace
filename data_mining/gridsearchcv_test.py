#-*- coding: utf-8 -*-
#使用randomforest对iris数据集进行分类
#使用gridsearchcv寻找最优参数

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_data():
    iris = load_iris()
    # print(iris)
    return iris

def data_train(iris_datas):
    rf = RandomForestClassifier()
    parameters = {'randomforestclassifier__n_estimators': range(1,11)}  #pipeline使用时注意key的写法，name__

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('randomforestclassifier', rf)
    ])

    clf = GridSearchCV(param_grid= parameters, estimator = pipe)
    clf.fit(iris_datas.data, iris_datas.target)
    best_score = clf.best_score_
    best_param = clf.best_params_
    return best_score, best_param

if __name__ == "__main__":
    iris_data = load_data()
    best_score,best_param = data_train(iris_data)
    print("最优分数：", best_score)
    print("最优参数：", best_param)