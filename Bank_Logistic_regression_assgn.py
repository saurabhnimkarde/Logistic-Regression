#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


# In[3]:


bank = pd.read_csv("bank-full.csv",sep = ';')


# In[4]:


bank.head()


# In[5]:


bank.info()


# In[6]:


bank.shape


# In[7]:


col = ['default','housing','loan','y']
def conv(x):
    return x.map({'yes':1,'no':0})
bank[col] = bank[col].apply(conv)


# In[8]:


bank.head()


# In[9]:


bank_1 = pd.get_dummies(bank,columns = ['job','marital','education','contact','month','poutcome'])


# In[10]:


bank_1


# In[11]:


bank_1.shape


# In[12]:


X = bank_1.iloc[:,0:47]
Y = bank_1.iloc[:,-1]


# In[13]:


import warnings
warnings.filterwarnings("ignore")


# In[14]:


classifier = LogisticRegression()
classifier.fit(X,Y)


# In[15]:


y_pred = classifier.predict_proba(X)


# In[16]:


y_pred


# In[17]:


y_pred = classifier.predict(X)


# In[18]:


y_pred


# In[19]:


confusion_matrix = confusion_matrix(y_pred,Y)


# In[20]:


confusion_matrix


# In[21]:


print(classification_report(Y,y_pred))


# In[22]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(Y, classifier.predict_proba (X)[:,-1])

auc = roc_auc_score(Y, y_pred)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='blue', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')


# In[ ]:




