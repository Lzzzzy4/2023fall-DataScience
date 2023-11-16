import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pre_process import pre_process
from model import model

data_path = os.path.dirname(__file__) + '/../data/'
train = pd.read_csv(data_path + "train.csv")
test = pd.read_csv(data_path + "test.csv")

if __name__ == '__main__':
    pp = pre_process(train)
    train = pp.procsee()
    m = model(train, test)
    m.train()
    result = m.test()
