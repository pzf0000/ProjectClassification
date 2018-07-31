class BaseInput:
    """
    基本输入接口
    """
    def read(self, **kwargs):
        return self.input(**kwargs)


class BaseOutput:
    """
    基本输出接口
    """
    def write(self):
        return self.output()


class BaseIO(BaseInput, BaseOutput):
    pass


class IO(BaseIO):
    pass


import numpy as np
import pandas as pd


def load_dataset(filename="data.npy"):
    try:
        return np.load(filename)
    except:
        try:
            return pd.read_csv(filename)
        except:
            raise ValueError("Can not open file.")