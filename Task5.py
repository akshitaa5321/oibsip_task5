#!/usr/bin/env python
# coding: utf-8

# # SALES PREDICTION

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV,cross_val_score,KFold


# In[7]:


df = pd.read_csv('Advertising.csv')
df.head()


# In[8]:


print('Rows:',df.shape[0])
print('columns:',df.shape[1])


# In[9]:


df.info()


# In[10]:


df.dtypes


# In[11]:


df.size


# In[12]:


df.describe()


# In[13]:


df.isna().sum()


# In[14]:


df.duplicated().sum()


# In[15]:


df[:2]


# In[16]:


df.head()


# In[17]:


plt.boxplot(df,vert=False,data = df,labels=df.columns)
plt.show()


# In[18]:


df[:5]


# In[19]:


def sc():
    l=df.columns
    for i in range(len(l)-1):
        for j in l:
            s = plt.scatter(j,'Sales',data=df)
    return 's'
sc()


# In[20]:


sns.distplot(df['Newspaper'])


# In[21]:


sns.distplot(df['Radio'])


# In[22]:


df.drop(columns='Unnamed: 0',axis=1,inplace=True)


# In[23]:


x=df.iloc[:,:-1]
x


# In[24]:


y = df.iloc[:,-1:]
y


# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test  = train_test_split(x,y,test_size=0.3,random_state=43)


# In[26]:


x_train,y_train


# In[27]:


x_test,y_test


# In[28]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
model = LinearRegression()


# In[29]:


model.fit(x_train,y_train)


# In[30]:


ypred=model.predict(x_test)
ypred


# In[31]:


model.score(x_train,y_train)*100


# In[32]:


model.score(x_test,y_test)*100


# In[33]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[34]:


mean_squared_error(y_test,ypred)


# In[35]:


mean_absolute_error(y_test,ypred)


# In[36]:


r2_score(y_test,ypred)*100


# In[37]:


rmse = np.sqrt(mean_squared_error(y_test,ypred))
rmse


# In[38]:


cv = KFold(n_splits=5,shuffle=True, random_state=0)
cv


# In[39]:


scores=cross_val_score(model,x,y,cv=cv,n_jobs=-1)
finalscore=np.mean(scores)
finalscore


# In[40]:


from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[41]:


column_trans=make_column_transformer((OneHotEncoder(sparse=False),[]),remainder='passthrough')
scaler=StandardScaler()
oe=OrdinalEncoder()


# In[42]:


#Random Forest Regression Model

from sklearn.ensemble import RandomForestRegressor
r=RandomForestRegressor(n_estimators=10,random_state=0)
pipe=make_pipeline(column_trans,scaler,r)
pipe.fit(x_train,y_train)
y_pred_r=pipe.predict(x_test)
r2_score(y_test,y_pred_r)

