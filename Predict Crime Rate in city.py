#!/usr/bin/env python
# coding: utf-8

# In[98]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import prophet


# In[96]:



chicago_df_2 = pd.read_csv('H:\\Project3\\Chicago_Crimes_2008_to_2011.csv', error_bad_lines = False)
chicago_df_3 = pd.read_csv('H:\\Project3\\Chicago_Crimes_2012_to_2017.csv', error_bad_lines = False)


# In[97]:


chicago_df_1.shape


# In[5]:


chicago_df_2.shape


# In[6]:


chicago_df_3.shape


# In[8]:


chicago_df_1.head(10)


# In[9]:


chicago_df = pd.concat([chicago_df_1, chicago_df_2, chicago_df_3])


# In[11]:


chicago_df.shape


# In[12]:


chicago_df.head()


# In[13]:


chicago_df.tail()


# In[11]:


plt.figure(figsize = (10,10))
sns.heatmap(chicago_df.isnull(), cbar = False, cmap= 'YlGnBu')


# In[15]:


chicago_df_1 = pd.read_csv('H:\\Project3\\Chicago_Crimes_2005_to_2007.csv', error_bad_lines = False)


# In[16]:


chicago_df_1


# In[17]:


chicago_df_1.shape


# In[19]:


plt.figure(figsize=(10,10))
sns.heatmap(chicago_df_1.isnull(), cbar = False)


# In[23]:


chicago_df_1.drop(['Unnamed: 0', 'ID', 'Case Number', 'IUCR', 'X Coordinate', 'Y Coordinate','Year','FBI Code','Updated On','Beat', 'Ward', 'Community Area','Location','District','Latitude', 'Longitude'], inplace = True, axis = 1)


# In[24]:


chicago_df_1


# In[26]:


chicago_df_1.Date = pd.to_datetime(chicago_df_1.Date, format = '%m/%d/%Y %I:%M:%S %p')


# In[28]:


chicago_df_1.Date


# In[42]:


chicago_df_1.index = pd.DatetimeIndex(chicago_df_1.Date)


# In[43]:


chicago_df_1


# In[ ]:





# In[31]:


chicago_df_1['Primary Type'].value_counts()


# In[32]:


chicago_df_1['Primary Type'].value_counts().iloc[:15]


# In[34]:


order_data = chicago_df_1['Primary Type'].value_counts().iloc[:15].index


# In[38]:


#showing plots
plt.figure(figsize = (15,10))
sns.countplot(y = 'Primary Type', data = chicago_df_1, order = order_data)


# In[40]:


plt.figure(figsize= (15,10))
sns.countplot(y= 'Location Description', data=chicago_df_1, order = chicago_df_1['Location Description'].value_counts().iloc[:15].index)


# In[44]:


chicago_df_1.resample('Y').size()


# In[75]:


plt.plot(chicago_df_1.resample('Y').size())
plt.title('Crime Count Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')


# In[51]:


plt.plot(chicago_df_1.resample('M').size())
plt.title('Crime Count Per Month')
plt.xlabel('Months')
plt.ylabel('Number of Crimes')


# In[52]:


plt.plot(chicago_df_1.resample('Q').size())
plt.title('Crime Count Per Quarter')
plt.xlabel('Quarter')
plt.ylabel('Number of Crimes')


# In[55]:


#Preparing The Data
chicago_prophet = chicago_df_1.resample('M').size().reset_index()


# In[56]:


chicago_prophet


# In[57]:


chicago_prophet.columns = ['Date', 'Crime Count']


# In[58]:


chicago_prophet


# In[59]:


chicago_prophet_df_final = chicago_prophet.rename(columns = {'Date': 'ds', 'Crime Count':'y'})


# In[60]:


chicago_prophet_df_final


# In[ ]:




