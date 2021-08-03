# - NLP in Python ----
import pandas as pd # import the module 
import os
# set working directory 
# os.getcwd() ~ get the current directory...
# os.chdir() ~ set the working directory...

df = pd.read_csv (r'./data/text.csv')
print (df) 

# Convert Letter Casing to Lower ----
string_list = list(df['text'])
for i in range(len(string_list)):
  string_list[i] = string_list[i].lower()
print(string_list[1])

# Remove all Numbers from Text ----
import re 
def remove(list):
    string_list = ''.join(i for i in list if not i.isdigit()) 
    return string_list
for i in range(len(string_list)):
  string_list[i] = remove(string_list[i])
print(string_list[1])

# Remove all puncuation from Text ----
import string
def punctuation(list): 
  punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
  for ele in list:  
    if ele in punc:  
        string_list = list.replace(ele, "")  
  return list
for i in range(len(string_list)):
  string_list[i] = punctuation(string_list[i])
print(string_list[1])

# Remove all stopwords from text ----
import nltk
import stop_words
from stop_words import get_stop_words
from nltk.corpus import stopwords

stop_words = list(get_stop_words('en'))         #About 900 stopwords
nltk_words = list(stopwords.words('english')) #About 150 stopwords
stop_words.extend(nltk_words)

output = [w for w in string_list if not w in stop_words]
print(output[1])
