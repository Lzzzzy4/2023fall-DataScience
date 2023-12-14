from math import e
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class pre_process:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.test = test

    def preprocsee(self, method: str):
        for m in method:
            if m == "KMeans":
                self.KMeans()
            elif m == "Rot_15":
                self.Rot_15()
            elif m == "Rot_30":
                self.Rot_30()
            else:
                raise Exception("pre_process: No such method")

        return (self.train, self.test)

    def KMeans(self):
        train = self.train.loc[
            :, ["latitude", "longitude", "year", "week_no", "emission"]
        ]
        test = self.test.loc[:, ["latitude", "longitude", "year", "week_no"]]

        k_model = KMeans(n_clusters=6, n_init=10)
        train_x = train.groupby(by=["latitude", "longitude"], as_index=False)[
            "emission"
        ].mean()
        k_model.fit(train_x)
        k_Means = k_model.predict(train_x)
        train_x["k_Means"] = k_Means

        train = train.merge(
            train_x[["latitude", "longitude", "k_Means"]], on=["latitude", "longitude"]
        )
        test = test.merge(
            train_x[["latitude", "longitude", "k_Means"]], on=["latitude", "longitude"]
        )
        self.train = train
        self.test = test

    def Rot_15(self):
        train = self.train
        test = self.test

        train["rot_15_y"] = train["latitude"] * np.cos(np.pi / 12) + train[
            "longitude"
        ] * np.sin(np.pi / 12)
        train["rot_15_x"] = train["longitude"] * np.cos(np.pi / 12) - train[
            "latitude"
        ] * np.sin(np.pi / 12)
        test["rot_15_y"] = test["latitude"] * np.cos(np.pi / 12) + test[
            "longitude"
        ] * np.sin(np.pi / 12)
        test["rot_15_x"] = test["longitude"] * np.cos(np.pi / 12) - test[
            "latitude"
        ] * np.sin(np.pi / 12)

        self.train = train
        self.test = test

    def Rot_30(self):
        train = self.train
        test = self.test

        train["rot_30_y"] = train["latitude"] * np.cos(np.pi / 6) + train[
            "longitude"
        ] * np.sin(np.pi / 6)
        train["rot_30_x"] = train["longitude"] * np.cos(np.pi / 6) - train[
            "latitude"
        ] * np.sin(np.pi / 6)
        test["rot_30_y"] = test["latitude"] * np.cos(np.pi / 6) + test[
            "longitude"
        ] * np.sin(np.pi / 6)
        test["rot_30_x"] = test["longitude"] * np.cos(np.pi / 6) - test[
            "latitude"
        ] * np.sin(np.pi / 6)

        self.train = train
        self.test = test
