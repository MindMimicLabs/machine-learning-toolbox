# - subset variables ----
from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df.head()
data = df[['sepal length (cm)']] # conditionally select the variables to remove from the dataframe. 
print(data.head())
