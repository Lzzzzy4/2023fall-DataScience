from multiprocessing import process
from matplotlib import scale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import cluster, optimize

from sklearn import preprocessing
from sklearn.cluster import KMeans
from catboost import CatBoostRegressor
from sklearn.feature_selection import RFE
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler


class model:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.test = test

    def get_result(self, model):
        assert hasattr(self, model), f"pre_process: No such model {model}"
        getattr(self, model)()

        self.FixPrediction()
        return self.ans

    def CatBoostRegressor(self):
        train = self.train.loc[:, ["latitude", "longitude", "year", "week_no", "emission", "k_Means"]]
        test = self.test.loc[:, ["latitude", "longitude", "year", "week_no", "k_Means"]]
        n_clusters = train["k_Means"].unique()
        cat_params = {
            # "n_estimators": 250,
            # "learning_rate": 0.95,
            "n_estimators": 799,
            "learning_rate": 0.09180872710592884,
            "depth": 8,
            "l2_leaf_reg": 1.0242996861886846,
            "subsample": 0.38227256755249117,
            "colsample_bylevel": 0.7183481537623551,
            "random_state": 42,
            "silent": True,
        }
        for i in n_clusters:
            train_c = train[train["k_Means"] == i]
            test_c = test[test["k_Means"] == i].drop(columns=["k_Means"])
            X = train_c.drop(columns=["emission", "k_Means"])
            y = train_c["emission"].copy()

            catboost_reg = CatBoostRegressor(**cat_params)
            catboost_reg.fit(X, y)
            if 'emission' in test_c.columns:
                test_c = test_c.drop(columns=['emission'])
            catboost_pred = catboost_reg.predict(test_c)

            test.loc[test["k_Means"] == i, "emission"] = catboost_pred

        self.ans = test.loc[:, ["emission"]]

    def RadiusNeighborsRegressor(self):
        train = self.train.loc[:, ["latitude", "longitude", "week_no", "emission"]]
        test = self.test.loc[:, ["latitude", "longitude", "week_no"]]
        X = train.drop(columns=["emission"])
        y = train["emission"].copy()

        rnr = RadiusNeighborsRegressor(radius=0)
        rnr.fit(X, y)
        rnr_pred = rnr.predict(test)

        test["emission"] = rnr_pred * 1.1

        self.ans = test.loc[:, ["emission"]]

    def KNeighborsRegressor(self):
        train = self.train.loc[:, ["latitude", "longitude", "week_no", "emission"]]
        test = self.test.loc[:, ["latitude", "longitude", "week_no"]]
        X = train.drop(columns=["emission"])
        y = train["emission"].copy()

        knr = KNeighborsRegressor(n_neighbors=3)
        knr.fit(X, y)
        knr_pred = knr.predict(test)

        test["emission"] = knr_pred * 1.1

        self.ans = test.loc[:, ["emission"]]

    def RandomForestRegressor(self):
        # train = self.train[self.train['year'] != 2020]
        train = self.train.loc[:, ["latitude", "longitude", "year", "week_no", "emission"]]
        test = self.test.loc[:, ["latitude", "longitude", "year", "week_no"]]
        X = train.drop(columns=["emission"])
        y = train["emission"].copy()

        rfr = RandomForestRegressor(min_samples_leaf=6)
        rfr.fit(X, y)
        rfr_pred = rfr.predict(test)

        test["emission"] = rfr_pred * 1.1

        self.ans = test.loc[:, ["emission"]]

    def AdaBoostRegressor(self):
        train = self.train.loc[:, ["latitude", "longitude", "year", "week_no", "emission", "k_Means"]]
        test = self.test.loc[:, ["latitude", "longitude", "year", "week_no", "k_Means"]]
        X = train.drop(columns=["emission"])
        y = train["emission"].copy()

        abr = AdaBoostRegressor()
        abr.fit(X, y)
        abr_pred = abr.predict(test)

        test["emission"] = abr_pred * 1.1

        self.ans = test.loc[:, ["emission"]]

    def LinearRegression(self):
        train = self.train.loc[:, ["latitude", "longitude", "year", "week_no", "emission", "k_Means"]]
        test = self.test.loc[:, ["latitude", "longitude", "year", "week_no", "k_Means"]]
        clusters = train["k_Means"].unique()
        for i in clusters:
            train_c = train[train["k_Means"] == i]
            test_c = test[test["k_Means"] == i].drop(columns=["k_Means"])
            X = train_c.drop(columns=["emission", "k_Means"])
            y = train_c["emission"].copy()

            lr = LinearRegression(n_jobs=-1)
            lr.fit(X, y)
            if 'emission' in test_c.columns:
                test_c = test_c.drop(columns=['emission'])
            lr_pred = lr.predict(test_c)

            test.loc[test["k_Means"] == i, "emission"] = lr_pred

        self.ans = test.loc[:, ["emission"]]

    def SupportVectorRegressor(self):
        train = self.train.loc[:, ["latitude", "longitude", "year", "week_no", "emission", "k_Means"]]
        test = self.test.loc[:, ["latitude", "longitude", "year", "week_no", "k_Means"]]
        X = train.drop(columns=["emission"])
        y = train["emission"].copy()
        # scaler = StandardScaler().fit(X)
        # X = scaler.transform(X)
        # test_ = scaler.transform(test)

        svr = SVR(max_iter=1000, C=1.0, epsilon=0.1)
        svr.fit(X, y)
        svr_pred = svr.predict(test)

        test["emission"] = svr_pred
        self.ans = test.loc[:, ["emission"]]

    def DecisionTreeRegressor(self):
        train = self.train.loc[:, ["latitude", "longitude", "year", "week_no", "emission"]]
        test = self.test.loc[:, ["latitude", "longitude", "year", "week_no"]]
        X = train.drop(columns=["emission"])
        y = train["emission"].copy()

        dtr = DecisionTreeRegressor(min_samples_leaf=3, min_samples_split=4)
        dtr.fit(X, y)
        dtr_pred = dtr.predict(test)

        test["emission"] = dtr_pred * 1.072

        self.ans = test.loc[:, ["emission"]]

    def XGBoostRegressor(self):
        train = self.train.loc[:, ["latitude", "longitude", "year", "week_no", "emission", "k_Means"]]
        test = self.test.loc[:, ["latitude", "longitude", "year", "week_no", "k_Means"]]
        X = train.drop(columns=["emission"])
        y = train["emission"].copy()

        xgb = XGBRegressor()
        xgb.fit(X, y)
        xgb_pred = xgb.predict(test)

        test["emission"] = xgb_pred * 1.1

        self.ans = test.loc[:, ["emission"]]

    def FixPrediction(self):
        train = self.train
        test = self.test
        ans = self.ans
        ans[ans['emission'] < 0] = 0
        zero_emissions = train.groupby(['latitude', 'longitude'])['emission'].mean().to_frame()
        zero_emissions = zero_emissions[zero_emissions['emission'] == 0]
        mask = test.apply(lambda x: (x['latitude'], x['longitude']) in zero_emissions.index, axis=1)
        ans.loc[mask, "emission"] = 0
