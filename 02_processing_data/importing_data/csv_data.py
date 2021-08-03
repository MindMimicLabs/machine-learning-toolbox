# - Import CSV Files ----
import pandas as pd # import the module 

# set working directory 
# os.getcwd() ~ get the current directory...
# os.chdir() ~ set the working directory...

df = pd.read_csv (r'./data/text.csv')
print(df.head) 
string_list = c(df[,'text'])
for i in range(len(string_list)):
  string_list[i] = string_list[i].lower()
print(string_list)
