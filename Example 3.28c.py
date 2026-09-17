# Example 3.28c - The LazyPredict.ipynb Program (Part 3)

from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset Iris
iris = load_iris()

X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=123
)

# LazyPredict classification
clf = LazyClassifier(verbose=0, ignore_warnings=True)

models, predictions = clf.fit(
    X_train,
    X_test,
    y_train,
    y_test
)

# Plot - dibuat sama seperti buku
plt.figure(figsize=(10, 5))
plt.plot(models.index, models["Accuracy"])
plt.show()