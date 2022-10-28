#!/usr/bin/env python
# coding: utf-8

# ## `BERCHMANS KEVIN S`

# ## Department of Data Science - Data and Visual Analytics Lab 
# 
# 

# ## `Red Wine Quality Data Analysis using NumPy Part-II `

# #### Import necessary modules 

# In[1]:


import numpy as np


# In[2]:


wines = np.genfromtxt("winequality-red.csv", delimiter=";", skip_header=1)


# In[3]:


wines


# ### NumPy Aggregation Methods 

# #### Find sum of all residual sugar values 

# In[4]:


wines[:,3].sum()


# #### Find sums of every feature value. There are 12 features altogether 
# 

# In[5]:


wines.sum(axis=0)


# #### Find sum of every row 
# 

# In[6]:


wines.sum(axis=1)


# #### What is its size? 
# 

# In[7]:


wines.sum(axis=1).shape


# #### What is the maximum residual sugar value in red wines data? 
# 

# In[8]:


wines[:,3].astype('int')


# #### find its maximum residual sugar value 
# 

# In[9]:


wines[:,3].astype('int').max()


# #### What is the minimum residual sugar value in red wines data? 
# 

# In[10]:


wines[:,3].astype('int').min()


# #### What is the average residual sugar value in red wines data?

# In[11]:


wines[:,3].mean()


# #### What is 25 percentile residual sugar value? 

# In[12]:


np.percentile(wines[:,3], 25)


# #### What is 75 percentile residual sugar value? 

# In[13]:


np.percentile(wines[:,3], 75)


# #### Find the average of each feature value 
# 

# In[14]:


wines.mean(axis=0)


# ### NumPy Array Comparisons 
# 
# 

# #### Show all wines with quality > 5 

# In[15]:


wines[:, -1] > 5


# #### Show all wines with quality > 7 

# In[16]:


wines[:, -1] > 7


# #### check if any wines value is True for the condition quality > 7 
# 

# In[17]:


wines[0,-1] > 7


# #### Show first 3 rows where wine quality > 7, call it high_quality 

# In[18]:


high_quality = wines[:, -1] > 7


# In[19]:


high_quality


# #### Show only top 3 rows and all columns of high_quality wines data 
# 

# In[20]:


wines[high_quality,:][:3,:]


# #### Show wines with a lot of alcohol > 10 and high wine quality > 7 

# In[21]:


high_quality_and_alcohol = (wines[:,-2] > 10) & (wines[:,-1] > 7)


# #### show only alcohol and wine quality columns 

# In[22]:


wines[high_quality_and_alcohol,10:]


# ### Combining NumPy Arrays 
# 
# #### Combine red wine and white wine data 
# 

# #### Open white wine dataset 

# In[23]:


white_wines = np.genfromtxt("winequality-white.csv", delimiter=";", skip_header=1) 


# #### Show size of white_wines 
# 

# In[24]:


white_wines.shape


# #### combine wines and white_wines data frames using vstack and call it all_wines 
# 

# In[25]:


all_wines = np.vstack((wines, white_wines))


# In[26]:


all_wines.shape


# #### Combine wines and white_wines data frames using concatenate method 
# 

# In[27]:


data3 = np.concatenate((wines, white_wines), axis=0)


# In[28]:


data3.shape


# ### Matrix Operations and Reshape 
# 
# #### Find Transpose of wines and print its size 
# 

# In[29]:


np.transpose(wines).shape


# #### Convert wines data into 1D array 
# 

# In[30]:


wines.ravel()


# In[31]:


wines.ravel().shape


# #### Reshape second row of wines into a 2-dimensional array with 2 rows and 6 columns

# In[32]:


wines[1, :].reshape((2,6))


# ### Sort alcohol column Ascending Order 
# 

# In[33]:


sorted_alcohol = np.sort(wines[:, -2])


# In[34]:


sorted_alcohol


# #### Make sorting to take place in-place 

# In[35]:


wines[:, -2].sort()


# #### Show top 10 rows 
# 

# In[36]:


wines[:, -2]


# #### Sort alcohol column Descending Order 
# 

# In[37]:


sorted_alcohol_desc = np.sort(wines[:,-2]) [::-1]


# In[38]:


sorted_alcohol_desc


# #### Will original data be modified?. Check top 10 rows 
# 

# In[39]:


wines[:, -2]


# In[ ]:




