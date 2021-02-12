import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import XOR
epochs = 50000
lr = 0.01
rng = np.random.RandomState(0)
X = rng.randn(300, 2)
y = np.array(np.logical_xor(X[:, 0] > 0, X[:, 1] > 0),dtype=int)
model =XOR.XOR(2,4,1,lr, epochs)
model.fit(X,y)
predict=model.predict(X)
print(predict)
fig = plt.figure(figsize=(10,8))
fig = plot_decision_regions(X=X, y=y, clf=model, legend=2)
plt.title("XOR From Scrutch")
plt.show()
