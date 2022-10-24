
# Replace Outliers with the Median ----
# Just Change the Function to what you want to replace identified outliers with like mean ----
from sklearn.datasets import load_iris
iris = load_iris()
import pandas as pd
import numpy as np
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['Species'])
names = list(df.columns)
rang = range(0,len(names))
for i in rang:
  missing_col = [names[i]]
  median = float(df[missing_col].median())
  df[missing_col] = np.where(df[missing_col] > median, median, df[missing_col])


df.head()






