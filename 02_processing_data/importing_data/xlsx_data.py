# - Import XLSX Files ----
import pandas as pd # import the module 

# set working directory 
# os.getcwd() ~ get the current directory...
# os.chdir() ~ set the working directory...

df = pd.read_excel (r'./data/Texting.xlsx')
print (df) 
