import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class pre_process:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.test = test

    def preprocess(self, method: str):
        for m in method:
            if m == "KMeans":
                self.KMeans()
            elif m == "Rot_15":
                self.Rot_15()
            elif m == "Rot_30":
                self.Rot_30()
            elif m == "Dist_Rwanda":
                self.Dist_Rwanda()
            elif m == "Fillna":
                self.Fillna()
            elif m == "Standardize":
                self.Standardize()
            elif m == "process_2020_drop":
                self.process_2020_drop()
            elif m == "process_2020_fix":
                self.process_2020_fix()
            elif m == "process_2020_addfeatrue":
                self.process_2020_addfeatrue()
            else:
                raise Exception("pre_process: No such method")

        return (self.train, self.test)

    def KMeans(self):
        train = self.train
        test = self.test

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
        train["rot_15_x"] = train["longitude"] * np.cos(np.pi / 12) + train[
            "latitude"
        ] * np.sin(np.pi / 12)
        test["rot_15_y"] = test["latitude"] * np.cos(np.pi / 12) + test[
            "longitude"
        ] * np.sin(np.pi / 12)
        test["rot_15_x"] = test["longitude"] * np.cos(np.pi / 12) + test[
            "latitude"
        ] * np.sin(np.pi / 12)

    def Rot_30(self):
        train = self.train
        test = self.test

        train["rot_30_y"] = train["latitude"] * np.cos(np.pi / 6) + train[
            "longitude"
        ] * np.sin(np.pi / 6)
        train["rot_30_x"] = train["longitude"] * np.cos(np.pi / 6) + train[
            "latitude"
        ] * np.sin(np.pi / 6)
        test["rot_30_y"] = test["latitude"] * np.cos(np.pi / 6) + test[
            "longitude"
        ] * np.sin(np.pi / 6)
        test["rot_30_x"] = test["longitude"] * np.cos(np.pi / 6) + test[
            "latitude"
        ] * np.sin(np.pi / 6)

    def Dist_Rwanda(self):
        rwanda_center = (-1.9607, 29.9707)
        train = self.train
        test = self.test

        train["dist_rwanda"] = np.sqrt(
            (train["latitude"] - rwanda_center[0]) ** 2
            + (train["longitude"] - rwanda_center[1]) ** 2
        )
        test["dist_rwanda"] = np.sqrt(
            (test["latitude"] - rwanda_center[0]) ** 2
            + (test["longitude"] - rwanda_center[1]) ** 2
        )

    def Fillna(self):
        train = self.train
        test = self.test

        good_col = "Ozone_solar_azimuth_angle"
        train[good_col] = train.groupby(["year"])[good_col].ffill().bfill()
        test[good_col] = test.groupby(["year"])[good_col].ffill().bfill()

        numeric_cols = train.columns.drop("ID_LAT_LON_YEAR_WEEK")
        train[numeric_cols] = train[numeric_cols].fillna(train[numeric_cols].mean())

        numeric_cols = test.columns.drop("ID_LAT_LON_YEAR_WEEK")
        test[numeric_cols] = test[numeric_cols].fillna(test[numeric_cols].mean())

    def process_2020_drop(self):
        train = self.train
        self.train = train[
            (train.year == 2019)
            | (train.year == 2020) & (train.week_no <= 10)
            | (train.year == 2021) & (train.week_no > 10)
        ]

    def process_2020_fix(self):
        train = self.train
        avg_emission_non_virus = train[train['year'].isin((2019,2021))].groupby('week_no')['emission'].mean()
        avg_emission_virus = train[train['year'] == 2020].groupby('week_no')['emission'].mean()
        ratios_for_weeks = avg_emission_non_virus/avg_emission_virus
        train.loc[train['year'] == 2020, 'emission'] *= train['week_no'].map(ratios_for_weeks)
        self.train = train

    def process_2020_addfeatrue(self):
        self.train['is2020'] = (self.train['year'] == 2020)
        self.test['is2020'] = False

    def Standardize(self):
        train = self.train
        test = self.test
        scaler = StandardScaler()

        numeric_cols = train.columns.drop(["ID_LAT_LON_YEAR_WEEK", "emission"])
        train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
        numeric_cols = test.columns.drop("ID_LAT_LON_YEAR_WEEK")
        test[numeric_cols] = scaler.transform(test[numeric_cols])
