from pynndescent import NNDescent
from pynndescent import *
from sklearn.datasets import load_iris

data = load_iris()

index = NNDescent(data.data, tree_init=False, n_neighbors=5, verbose=True)

result = index.query([data.data[0]], k=3)

print(result)

residual = index.residual
print(residual[0])