import pandas as pd
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data # Sepal and pedal negth and width

# Normalize Data ----
from sklearn.preprocessing import scale
x = scale(X)

# Prinicpal Components Analysis ----
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
PC = pca.fit(x)

# Factor Analysis ----
from sklearn.preprocessing import scale
from sklearn.decomposition import FactorAnalysis
data_normal = scale(dat)
fa = FactorAnalysis(n_components = 10)
fa = fa.fit(data_normal)
