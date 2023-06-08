import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("out/cow50.csv")

# calculate mae
data["abs_err"] = np.abs(data["obs"] - data["pre"])

data["abs_err"].mean()

# correlation

plt.scatter(data["obs"], data["pre"])
data.corr()

