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
  for i in missing_col:
    df.loc[df.loc[:,i].isnull(),i]=df.loc[:,i].mean()
    
df.head()
