import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import preprocessing
from catboost import CatBoostRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import RadiusNeighborsRegressor

class model:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.test = test

    def get_result(self, method):
        if method == 'K-Mean' :
            self.K_Mean()
        else :
            raise Exception("model: No such method")
        return self.ans
    
    def K_Mean(self) -> dict:
        # 聚类
        train = self.train.loc[:,["latitude", "longitude", "year", "week_no", "emission"]]
        test = self.test.loc[:,["latitude", "longitude", "year", "week_no"]]

        k_model = KMeans(n_clusters=6)
        train_x = train.groupby(by=['latitude', 'longitude'], as_index=False)['emission'].mean()
        k_model.fit(train_x)
        k_Means = k_model.predict(train_x)
        train_x['k_Means'] = k_Means

        train = train.merge(train_x[['latitude', 'longitude', 'k_Means']], on=['latitude', 'longitude'])
        test = test.merge(train_x[['latitude', 'longitude', 'k_Means']], on=['latitude', 'longitude'])

        # predict
        cat_params = {
            'n_estimators': 799, 
            'learning_rate': 0.09180872710592884,
            'depth': 8, 
            'l2_leaf_reg': 1.0242996861886846, 
            'subsample': 0.38227256755249117, 
            'colsample_bylevel': 0.7183481537623551,
            'random_state': 42,
            "silent": True,
        }

        clusters = train["k_Means"].unique()

        for i in clusters:
            train_c = train[train["k_Means"] == i]
            test_c = test[test["k_Means"] == i].drop(columns = ["k_Means"])
            X = train_c.drop(columns = ["emission", "k_Means"])
            y = train_c["emission"].copy()

            catboost_reg = CatBoostRegressor(**cat_params)
            catboost_reg.fit(X, y)
            catboost_pred = catboost_reg.predict(test_c)

            test.loc[test["k_Means"] == i, "emission"] = catboost_pred

        # CatBoostRegressor = 
        self.ans = test.loc[:,["emission"]]
