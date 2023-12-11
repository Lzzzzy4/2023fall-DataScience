
from sklearn.metrics import mean_squared_error
import pandas as pd
class judge:
    def __init__(self, test: pd.DataFrame, ans: pd.DataFrame) :
        # 不需要id 只保留分数即可
        self.ans = ans
        self.test = test

    def get_score(self, method: str) :
        if method == 'mean_squared_error' :
            return self.mean_squared_error()
        else :
            raise Exception("socre: No such method")
        
    def mean_squared_error(self) :
        rmse = mean_squared_error(self.test["emission"], self.ans["emission"], squared=False)
        return rmse