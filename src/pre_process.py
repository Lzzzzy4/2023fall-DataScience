import numpy as np
import pandas as pd
class pre_process:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.test = test

    def preprocsee(self, method: str):
        if method == 'Brute' :
            self.Brute()
        else :
            raise Exception("pre_process: No such method")
        
        return (self.train, self.test)
    
    def Brute(self):
        pass
