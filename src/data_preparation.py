import pandas as pd
import numpy as np
from pandas_datareader.data import DataReader
from datetime import datetime
import matplotlib.pyplot as plt

class DataPreparation():
    def __init__(self, df):
        self.df = df

    def compute_log_GNP(self):
        print("Data Cleaning and Data Standardizing in progress ...")
        self.df["GR_GNP"] = np.log(self.df['GNP']).diff() * 100
        self.df["DATE"] = pd.to_datetime(self.df["DATE"])
        self.df = self.df.iloc[1:]

    def get_crisis_period(self):
        self.crisis = DataReader("USREC",
                           "fred",
                           start=datetime(1947, 1, 1),
                           end=datetime(2023, 10, 1))

        self.crisis.index = pd.to_datetime(self.crisis.index)
        print("Data Cleaned ✅")

    def prepare_data(self):
        self.compute_log_GNP()
        self.get_crisis_period()
        self.df = self.df.merge(self.crisis, how = "left", on = "DATE")
        self.df = self.df.rename(columns={"USREC": "crise"})

    def plot_GNP_crisis(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.df["DATE"], self.df["GR_GNP"], label='GR_GNP')

        for period in self.crisis.loc[self.crisis['USREC'] == 1].index.to_series().groupby((self.crisis['USREC'] == 1).cumsum()).agg(
                ['first', 'last']).itertuples():
            plt.axvspan(period.first, period.last, color='red', alpha=0.3)

        plt.title("GNP with recessions periods")
        plt.xlabel("Date")
        plt.ylabel("GR_GNP")
        plt.legend()
        plt.show()