# Example 3.3 Python SVM Iris CSV Classifications
import pandas as pd
from sklearn import svm

df = pd.read_csv("iris.csv")
X = df.values[:, :2]
s = df["species"]
d = {label: index for index, label in enumerate(sorted(set(s)))}
y = [d[label] for label in s]

clf = svm.SVC()
clf.fit(X, y)

# Predict the flower for a given sepal length and width
p = clf.predict([[5.4, 3.2]])
print(p)
