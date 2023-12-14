import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pre_process import pre_process
from model import model
from judge import judge

data_path = os.path.dirname(__file__) + "/../data/"
train = pd.read_csv(data_path + "train.csv")
test = pd.read_csv(data_path + "test.csv")
ans = pd.read_csv(data_path + "ans.csv")

methods_pre_process = [
    "Fillna",
    # "Standardize",
    "KMeans",
    "Rot_15",
    "Rot_30",
    "Dist_Rwanda",
]
methods_model = [
    "CatBoostRegressor",
    # "RadiusNeighborsRegressor",
    # "KNeighborsRegressor",
    # "RandomForestRegressor",
    # "AdaBoostRegressor",
    # "LinearRegression",
    # "SupportVectorRegressor",
    # "DecisionTreeRegressor",
    # "XGBoostRegressor",
]
methods_judge = ["mean_squared_error"]


def run(methods_pre_process, method_model, method_judge):
    p = pre_process(train, test)
    train_csv, test_csv = p.preprocess(methods_pre_process)

    m = model(train_csv, test_csv)
    result = m.get_result(method_model)

    j = judge(result, ans)
    score = j.get_score(method_judge)

    print("method_pre_process: ", methods_pre_process)
    print("method_model: ", method_model)
    print("method_judge: ", method_judge)
    print("score: ", score)
    print("")


if __name__ == "__main__":
    # run('Brute', 'K-Mean', 'mean_squared_error')
    for method_model in methods_model:
        for method_judge in methods_judge:
            run(methods_pre_process, method_model, method_judge)
