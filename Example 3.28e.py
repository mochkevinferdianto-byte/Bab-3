# Example 3.28e LazyPredict regression plot

import matplotlib.pyplot as plt
import pandas as pd

models = pd.read_csv("models.csv", index_col=0)

plt.figure(figsize=(10, 5))
plt.plot(models.index, models["R-Squared"], "-s")
plt.show()