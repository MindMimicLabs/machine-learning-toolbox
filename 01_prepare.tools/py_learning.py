# Basics in Programming in Python ----
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
# - [-] - Variables ----
x = 1 # store a integer/number into the env. 
# ... notice the variable is stored in the R env as a "value".
# ...  this is because the type of value stored. 
# ... note, all types of values/objects can be stored. 
print(x) # print the stored variable.
a = 5 # store a value in an object...
b = 5 # store a value in an object...
c = a + b # perform mathematical operations 
string = "Adding 5+5= "
print(list(string, c)) # to bind objects, they have to be contained in an object. 
# ... think of it like storing smaller objects within objects, ...
# ... or better yet piping data into objects.
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
# - [-] - Lists ----
list = list(1,2,3) # a c() function creates a list, really a vector, but mainly the samething at least for this class. 
emp_list = [] # empty lists may be created to fill with data later on in the code process.
print(list)
print(emp_list)
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
# - [-] - List Indexing ----
list[1] # index the 1st object within the list. 
list[1:3] # index the 1st to the 3rd object contained in the list. 
# ... not the order of the container may be altered and will ultimately change the index 
# ... positions of certain variables within the object.
list[0::3] # Empty index means to the beginning/end
list [-1] # note negations work...
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
# - [-] - Matrices ----
import numpy as np 
x = list(1,2,3)
matrix = np.matrix(x) # contain a list in a matrix...
print(matrix)
# or 
x = list(1,2,3)
y = list(4,5,6)
matrix = np.matrix(x,y) # bind two lists together to make a matrix...
print(matrix)
# or 
x = list(1,2,3,4,5,6)
matrix = np.matrix(x) # create a matrix specifying the # of cols and rows...
print(matrix)
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
# - [-] - Dataframes ----
import pandas as pd
x = list(1,2,3)
y = list(4,5,6)
z = list("mike", "paul", "steve") # list of strings...
df = pd.DataFrame("measure"=x,"age"=y,"name"=z) # store the lists in a df...
print(df)
# note that a df is a double object, which means that it contains multi. objects...
# A df contains both a matrix and lists...
# A df contains rows and columns, but the columns are also lists that may be indexed...
print(df[,list(1,2)]) # subset by cols...
print(df[1:2,]) # subset by rows...
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
# - [-] - Time Series Frame ----
x = list(1,2,3) # store a list of values...
y = list(4,5,6) # store a list of values...
z = list("02/27/92", "02/27/92", "01/14/92") # list of dates...
df = pd.to_datetime(x,y,z)
print(ts)
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
# - [-] - Functions ----
x = list(1,2,3) # store a list of values...
y = list(4,5,6) # store a list of values...
def df(x, y): # simply make a list of variable holders after the function...
   return pd.DataFrame("measure"=x,"age"=y) # store the lists in a df...
df = df(x,y) # execute the function, just feed the actually objects in place of the holders...
print(df)
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
# - [-] - Loops ----
x = list(1,2,3) # store a list of values...
z = list("mike", "paul", "steve") # list of strings...
df = pd.DataFrame(x,z) # frame the lists...
for i in range(3): # everything indented after the ':' is the function...
   print('Looping %d' % i, df[i,1], df[i,2])
