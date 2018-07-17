import pandas as pd
import numpy as np


def load_dataset(filename):
    try:
        return np.load(filename)
    except:
        try:
            return pd.read_csv(filename)
        except:
            raise ValueError("Can not open file.")

