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

methods_pre_process = ["Brute"]
methods_model = [
    "CatBoostRegressor",
    "RadiusNeighborsRegressor",
    "KNeighborsRegressor",
    "RandomForestRegressor",
]
methods_judge = ["mean_squared_error"]


def run(methods_pre_process, methods_model, methods_judge):
    # print("method_pre_process: ", methods_pre_process)
    p = pre_process(train, test)
    train_csv, test_csv = p.preprocsee(methods_pre_process)

    # print("method_model: ", methods_model)
    m = model(train_csv, test_csv)
    result = m.get_result(methods_model)

    # print("method_judge: ", methods_judge)
    j = judge(result, ans)
    score = j.get_score(methods_judge)

    print("method_pre_process: ", methods_pre_process)
    print("method_model: ", methods_model)
    print("method_judge: ", methods_judge)
    print("score: ", score)
    print("")


if __name__ == "__main__":
    # run('Brute', 'K-Mean', 'mean_squared_error')
    for method_pre_process in methods_pre_process:
        for method_model in methods_model:
            for method_judge in methods_judge:
                run(method_pre_process, method_model, method_judge)
