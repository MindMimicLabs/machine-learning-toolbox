# - Data Type Conversions ----

# Numerical Conversion 
x = "1"
str(x)
x = as.numeric(x)
str(x)

# Factor Conversion 
x = 1
str(x)
x = as.factor(x)
str(x)

# Character Conversion
x = 1
str(x)
x = as.character(x)
str(x)

# Date Conversion 
x = "01-11-2018"
str(x)
x = as.Date(x,  tryFormats = c("%m-%d-%Y"))
str(x)
