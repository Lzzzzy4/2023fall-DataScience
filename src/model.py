from multiprocessing import process
from re import S
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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


class model:
    def __init__(self, train, test):
        # use_feature = ["latitude", "longitude", "year", "week_no", "k_Means"]
        # self.train = train.loc[:, use_feature + ["emission"]]
        # self.test = test.loc[:, use_feature]
        self.train = train
        self.test = test

    def get_result(self, model):
        assert hasattr(self, model), f"pre_process: No such model {model}"
        getattr(self, model)()

        self.FixPrediction()
        return self.ans

    def CatBoostRegressor(self):
        # train = self.train.drop(columns="ID_LAT_LON_YEAR_WEEK")
        # test = self.test.drop(columns="ID_LAT_LON_YEAR_WEEK")
        train = self.train.loc[:, ["latitude", "longitude", "week_no", "emission", "k_Means", "year"]]
        test = self.test.loc[:, ["latitude", "longitude", "week_no", "k_Means", "year"]]
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
            "silent": True,
        }
        for i in n_clusters:
            train_c = train[train["k_Means"] == i]
            test_c = test[test["k_Means"] == i].drop(columns=["k_Means"])
            if "emission" in test_c.columns:
                test_c = test_c.drop(columns=["emission"])
            X = train_c.drop(columns=["emission", "k_Means"])
            y = train_c["emission"].copy()

            cbg = CatBoostRegressor(silent=True, learning_rate=0.1, depth=8)
            cbg_cv_scores = list()
            kf = KFold(n_splits=3, shuffle=True)
            for _, (train_ix, test_ix) in enumerate(kf.split(X)):
                X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
                Y_train, Y_test = y.iloc[train_ix], y.iloc[test_ix]
                cbg_md = cbg.fit(X_train, Y_train)
                cbg_pred = cbg_md.predict(X_test)
                cbg_score_fold = mean_squared_error(Y_test, cbg_pred, squared=False)
                cbg_cv_scores.append(cbg_score_fold)
            print(
                "CBG Mean oof RMSE score for cluster",
                i,
                " is ==>",
                np.mean(cbg_cv_scores),
            )

            cbg.fit(X, y)
            catboost_pred = cbg.predict(test_c)

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
        test["emission"] = rnr_pred * 1.07
        self.ans = test.loc[:, ["emission"]]

    def KNeighborsRegressor(self):
        train = self.train.loc[:, ["latitude", "longitude", "week_no", "emission"]]
        test = self.test.loc[:, ["latitude", "longitude", "week_no"]]
        X = train.drop(columns=["emission"])
        y = train["emission"].copy()

        knr = KNeighborsRegressor(n_neighbors=3)

        # knr_cv_scores = list()
        # kf = KFold(n_splits=3, shuffle=True)
        # for i, (train_ix, test_ix) in enumerate(kf.split(X)):
        #     X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        #     Y_train, Y_test = y.iloc[train_ix], y.iloc[test_ix]
        #     knr_md = knr.fit(X_train, Y_train)
        #     knr_pred = knr_md.predict(X_test)
        #     knr_score_fold = mean_squared_error(Y_test, knr_pred, squared=False)
        #     knr_cv_scores.append(knr_score_fold)
        # print("KNR Mean oof RMSE score is ==>", np.mean(knr_cv_scores))

        knr.fit(X, y)
        knr_pred = knr.predict(test)
        test["emission"] = knr_pred * 1.07
        self.ans = test.loc[:, ["emission"]]

    def RandomForestRegressor(self):
        # train = self.train[self.train['year'] != 2020]
        train = self.train.loc[:, ["latitude", "longitude", "week_no", "emission", "year"]]
        test = self.test.loc[:, ["latitude", "longitude", "week_no", "year"]]
        X = train.drop(columns=["emission"])
        y = train["emission"].copy()

        # scores = cross_val_score(RandomForestRegressor(), X, y, cv=3, scoring="neg_root_mean_squared_error")
        # print(scores)
        # print("RMSE: %0.2f (+/- %0.2f)" % (-scores.mean(), scores.std() * 2))

        rfr = RandomForestRegressor()
        rf_cv_scores = list()
        kf = KFold(n_splits=3, shuffle=True)
        for i, (train_ix, test_ix) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            Y_train, Y_test = y.iloc[train_ix], y.iloc[test_ix]
            rf_md = rfr.fit(X_train, Y_train)
            rf_pred = rf_md.predict(X_test)
            rf_score_fold = mean_squared_error(Y_test, rf_pred, squared=False)
            rf_cv_scores.append(rf_score_fold)
        print("RF Mean oof RMSE score is ==>", np.mean(rf_cv_scores))

        rfr.fit(X, y)
        rfr_pred = rfr.predict(test)
        test["emission"] = rfr_pred * 1.06
        self.ans = test.loc[:, ["emission"]]

    def AdaBoostRegressor(self):
        train = self.train.loc[
            :,
            [
                "latitude",
                "longitude",
                "year",
                "week_no",
                "emission",
                "rot_30_x",
                "k_Means",
            ],
        ]
        test = self.test.loc[
            :, ["latitude", "longitude", "year", "week_no", "rot_30_x", "k_Means"]
        ]
        # train = self.train
        # test = self.test
        X = train.drop(columns=["emission"])
        y = train["emission"].copy()
        abr = AdaBoostRegressor()
        abr_cv_scores = list()
        kf = KFold(n_splits=3, shuffle=True)
        for i, (train_ix, test_ix) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            Y_train, Y_test = y.iloc[train_ix], y.iloc[test_ix]
            abr_md = abr.fit(X_train, Y_train)
            abr_pred = abr_md.predict(X_test)
            abr_score_fold = mean_squared_error(Y_test, abr_pred, squared=False)
            abr_cv_scores.append(abr_score_fold)
        print("ABR Mean oof RMSE score is ==>", np.mean(abr_cv_scores))

        abr.fit(X, y)
        abr_pred = abr.predict(test)
        test["emission"] = abr_pred * 1.1
        self.ans = test.loc[:, ["emission"]]

    def LinearRegression(self):
        train = self.train.loc[
            :, ["latitude", "longitude", "week_no", "emission", "k_Means"]
        ]
        test = self.test.loc[:, ["latitude", "longitude", "week_no", "k_Means"]]
        clusters = train["k_Means"].unique()
        for i in clusters:
            train_c = train[train["k_Means"] == i]
            test_c = test[test["k_Means"] == i].drop(columns=["k_Means"])
            X = train_c.drop(columns=["emission", "k_Means"])
            y = train_c["emission"].copy()
            lr = LinearRegression(n_jobs=-1)

            lr_cv_scores = list()
            kf = KFold(n_splits=10, shuffle=True)
            for _, (train_ix, test_ix) in enumerate(kf.split(X)):
                X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
                Y_train, Y_test = y.iloc[train_ix], y.iloc[test_ix]
                lr_md = lr.fit(X_train, Y_train)
                lr_pred = lr_md.predict(X_test)
                lr_score_fold = mean_squared_error(Y_test, lr_pred, squared=False)
                lr_cv_scores.append(lr_score_fold)
            print(
                "LR Mean oof RMSE score for cluster",
                i,
                " is ==>",
                np.mean(lr_cv_scores),
            )

            lr.fit(X, y)
            if "emission" in test_c.columns:
                test_c = test_c.drop(columns=["emission"])
            lr_pred = lr.predict(test_c)
            test.loc[test["k_Means"] == i, "emission"] = lr_pred

        test["emission"] *= 1.06
        self.ans = test.loc[:, ["emission"]]

    def SupportVectorRegressor(self):
        train = self.train.loc[
            :, ["latitude", "longitude", "week_no", "emission", "k_Means"]
        ]
        test = self.test.loc[:, ["latitude", "longitude", "week_no", "k_Means"]]
        # n_clusters = train["k_Means"].unique()
        # for i in n_clusters:
        #     train_c = train[train["k_Means"] == i]
        #     test_c = test[test["k_Means"] == i].drop(columns=["k_Means"])
        #     if "emission" in test_c.columns:
        #         test_c = test_c.drop(columns=["emission"])
        #     X = train_c.drop(columns=["emission", "k_Means"])
        #     y = train_c["emission"].copy()

        #     scaler = StandardScaler().fit(X)
        #     X = scaler.transform(X)
        #     test_c = scaler.transform(test_c)

        #     svr = SVR(max_iter=1000)
        #     svr.fit(X, y)
        #     svr_pred = svr.predict(test_c)

        #     test.loc[test["k_Means"] == i, "emission"] = svr_pred

        X = train.drop(columns=["emission"])
        y = train["emission"].copy()
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        test_ = scaler.transform(test)

        svr = SVR(kernel="rbf", max_iter=1000)
        svr.fit(X, y)
        svr_pred = svr.predict(test_)
        test["emission"] = svr_pred

        self.ans = test.loc[:, ["emission"]]

    def DecisionTreeRegressor(self):
        train = self.train.loc[
            :,
            [
                "latitude",
                "longitude",
                "year",
                "week_no",
                "emission",
                "k_Means",
                "rot_30_x",
                # "rot_15_x",
            ],
        ]
        test = self.test.loc[
            :,
            [
                "latitude",
                "longitude",
                "year",
                "week_no",
                "k_Means",
                "rot_30_x",
                # "rot_15_x",
            ],
        ]
        X = train.drop(columns=["emission"])
        y = train["emission"].copy()

        dtr = DecisionTreeRegressor(min_samples_leaf=3, min_samples_split=4)

        dtr_cv_scores = list()
        kf = KFold(n_splits=10, shuffle=True)
        for i, (train_ix, test_ix) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            Y_train, Y_test = y.iloc[train_ix], y.iloc[test_ix]
            dtr_md = dtr.fit(X_train, Y_train)
            dtr_pred = dtr_md.predict(X_test)
            dtr_score_fold = mean_squared_error(Y_test, dtr_pred, squared=False)
            dtr_cv_scores.append(dtr_score_fold)
        print("DTR Mean oof RMSE score is ==>", np.mean(dtr_cv_scores))

        dtr.fit(X, y)
        dtr_pred = dtr.predict(test)
        test["emission"] = dtr_pred * 1.072
        self.ans = test.loc[:, ["emission"]]

    def XGBoostRegressor(self):
        # train = self.train.drop(columns="ID_LAT_LON_YEAR_WEEK")
        # test = self.test.drop(columns="ID_LAT_LON_YEAR_WEEK")
        train = self.train.loc[
            :,
            [
                "latitude",
                "longitude",
                "week_no",
                "emission",
                "k_Means",
                "year",
                "rot_30_y",
                "rot_30_x",
                "rot_15_y",
                "rot_15_x",
            ],
        ]
        test = self.test.loc[
            :,
            [
                "latitude",
                "longitude",
                "week_no",
                "k_Means",
                "year",
                "rot_30_y",
                "rot_30_x",
                "rot_15_y",
                "rot_15_x",
            ],
        ]
        X = train.drop(columns=["emission"])
        y = train["emission"].copy()

        xgb = XGBRegressor()
        xgb_cv_scores = list()
        kf = KFold(n_splits=5, shuffle=True)
        for i, (train_ix, test_ix) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            Y_train, Y_test = y.iloc[train_ix], y.iloc[test_ix]
            xgb_md = xgb.fit(X_train, Y_train)
            xgb_pred = xgb_md.predict(X_test)
            xgb_score_fold = mean_squared_error(Y_test, xgb_pred, squared=False)
            xgb_cv_scores.append(xgb_score_fold)
        print("XGB Mean oof RMSE score is ==>", np.mean(xgb_cv_scores))

        xgb.fit(X, y)
        xgb_pred = xgb.predict(test)
        test["emission"] = xgb_pred * 1.06
        self.ans = test.loc[:, ["emission"]]

    def FixPrediction(self):
        train = self.train
        test = self.test
        ans = self.ans
        ans[ans["emission"] < 0] = 0
        zero_emissions = (
            train.groupby(["latitude", "longitude"])["emission"].mean().to_frame()
        )
        zero_emissions = zero_emissions[zero_emissions["emission"] == 0]
        mask = test.apply(
            lambda x: (x["latitude"], x["longitude"]) in zero_emissions.index, axis=1
        )
        ans.loc[mask, "emission"] = 0
