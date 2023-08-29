#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Sample graph

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2)
y = np.sin(4 * np.pi * x) * np.exp(-5 * x)

plt.fill(x, y, 'r')
plt.grid(True)
plt.show()


# In[5]:


import pandas, numpy
s = pandas.Series([10, 23, 19, 15, 56, 15, 41])
print(s)


# In[6]:


s.sum(), s.min(), s.max(), s.mean()


# In[9]:


get_ipython().run_cell_magic('time', '', 's = list(numpy.random.randn(1000000))\n')


# In[11]:


import pandas
df = pandas.DataFrame(
    [['Boston', '10.1.1.1', 10, 2356, 0.100],
     ['Boston', '10.1.1.2', 23, 16600, 0.112],
     ['Boston', '10.1.1.15', 19, 22600, 0.085],
     ['SanFran', '10.38.5.1', 15, 10550, 0.030],
     ['SanFran', '10.38.8.2', 56, 35000, 0.020],
     ['London', '192.168.4.6', 15, 3400, 0.130],
     ['London', '192.168.5.72', 41, 55000, 0.120]],
     columns = ['location', 'ip', 'pkts', 'bytes', 'rtt'])


# In[12]:


df.dtypes


# In[13]:


df


# In[14]:


df['location']


# In[15]:


df['pkts']


# In[16]:


df[:2]


# In[17]:


df[2:4]


# In[18]:


df[df['location'] == 'Boston']


# In[19]:


df[df['pkts'] < 20]


# In[20]:


df[(df['location'] == 'Boston') & (df['pkts'] < 20)]


# In[ ]:




