#!/usr/bin/env python
# coding: utf-8

# In[1]:


# libraries
import matplotlib.pyplot as plt
import numpy as np

# create data
values=np.cumsum(np.random.randn(1000,1))

# use the plot function
plt.plot(values)

