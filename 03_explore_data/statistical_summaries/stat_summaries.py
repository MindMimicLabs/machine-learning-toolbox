# Stat summaries 
# creation of DataFrame
import pandas as pd
import numpy as np
#Create a Dictionary of series
d = {'Name':pd.Series(['Alisa','Bobby','Cathrine','Madonna','Rocky','Sebastian','Jaqluine',
   'Rahul','David','Andrew','Ajay','Teresa']),
   'Age':pd.Series([26,27,25,24,31,27,25,33,42,32,51,47]),
   'Score':pd.Series([89,87,67,55,47,72,76,79,44,92,99,69])}
#Create a DataFrame
df = pd.DataFrame(d)
# summary statistics
print df.describe(include='all')
