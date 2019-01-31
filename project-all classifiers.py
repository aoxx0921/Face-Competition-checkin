#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle
from sklearn.ensemble import BaggingClassifier
from mlxtend.classifier import StackingClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score
import warnings


warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


df = pd.read_csv("data_1x1.xlsx")


# In[8]:


train = df.drop('Unnamed: 0', axis = 1)
train.head()
train.shape


# In[9]:


th = 300
df_01 = train[(train['x']>=0.5)&(train['x']<=1)&
                     (train['y']>=0)&(train['y']<=1)]     


###### oversampling #######

place_counts = df_01.place_id.value_counts()
mask = (place_counts[df_01.place_id.values]<=th).values
#df_01 = df_01.loc[mask]
df_02 = df_01.loc[mask]

print(df_02.head())
print(df_01.shape)
print(df_02.shape)


# In[10]:


frames = [df_01, df_02]

new_data = pd.concat(frames)


# In[11]:


print(new_data.head())
print(new_data.shape)

place_counts = new_data.place_id.value_counts()
mask = (place_counts[new_data.place_id.values]>=100).values
#df_01 = df_01.loc[mask]
df_03 = new_data.loc[mask]

print(df_03.shape)


# In[12]:


# Check how how frequently different locations appear
#df_placecounts = df_01["place_id"].value_counts()

#counts, bins = np.histogram(df_placecounts.values, bins=50)
#binsc = bins[:-1] + np.diff(bins)/2.

#plt.figure(3, figsize=(12,6))
#plt.bar(binsc, counts/(counts.sum()*1.0), width=np.diff(bins)[0])
#plt.grid(True)
#plt.xlabel("Number of place occurances")
#plt.ylabel("Fraction")
#plt.title("Train")
#plt.show()


# In[13]:


# Check how how frequently different locations appear
df_01 = df_03
df_placecounts = df_01["place_id"].value_counts()

with pd.option_context('display.max_rows', None, 'display.max_columns', len(df_placecounts)):
    print(len(df_placecounts) )


# In[14]:


#fig,ax = plt.subplots(figsize=(15,200))

# Example data

#y_val = df_placecounts.index
#y_pos = np.arange(len(df_placecounts))
#x_val = df_placecounts


#ax.barh(y_pos, x_val,align='edge',color='green', ecolor='black')
#ax.grid(True)
#ax.set_yticks(y_pos)
#ax.set_yticklabels(y_val)
#ax.invert_yaxis()  # labels read top-to-bottom
#ax.set_xlabel('occurence')
#ax.set_title('Occurence of place_id')

#plt.show()


# In[15]:


#feature engineering
df_01["hour"] = (df_01["time"]%(60*24))/60.
df_01["dayofweek"] = np.ceil((df_01["time"]%(60*24*7))/(60.*24))
df_01["dayofmonth"] = np.ceil((df_01["time"]%(60*24*30))/(60.*24))
df_01["month"] = (df_01['time']//43200)%12+1
df_01["dayofyear"] = np.ceil((df_01["time"]%(60*24*365))/(60.*24))
df_01["mean_x_y"] = (df_01["x"] + df_01["y"])/2.
df_01["x/y"] = (df_01["x"]/(df_01["y"]+0.01**10))


# In[16]:


df_01.head()


# In[17]:


y = df_01['place_id'].values
x = df_01.drop('place_id', axis = 1)
x = x.drop('row_id', axis = 1)
x = x.drop('time', axis = 1)

x.head(5)


# In[18]:


#from sklearn.cluster import KMeans
#kmeans = KMeans(n_clusters=500) 
#kmeans.fit(x)
#kmeans_pred = kmeans.predict(x)

#x["cluster"] = kmeans_pred
#x.head(5)


# In[19]:


#check if there is any nan or infinite value
print(np.all(np.isfinite(x)))
print(np.all(np.isnan(x)))


# In[20]:


#y-labels need to be encoded
le = preprocessing.LabelEncoder()
labels = le.fit_transform(y)


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size = 0.33, random_state=42)
scaler = StandardScaler()  
scaler.fit_transform(x)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)


# In[ ]:


#Naive Bayes

classifier = GaussianNB()
classifier.fit(X_train, y_train)

classifier.score(X_test,y_test)


# In[ ]:


#DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# In[ ]:


#Random Forest

clf_RF = RandomForestClassifier(n_estimators = 100, max_depth = 25, class_weight = "balanced")
clf_RF.fit(X_train, y_train)
clf_RF.score(X_test,y_test)


# In[ ]:


#KNN

clf_KNN = KNeighborsClassifier()  
clf_KNN.fit(X_train, y_train)

clf_KNN.score(X_test,y_test)


# In[ ]:


#AdaBoostClassifier

AdaBoost = AdaBoostClassifier(RandomForestClassifier(max_depth=20),n_estimators=20,learning_rate=1.5,algorithm="SAMME")
AdaBoost.fit(X_train,y_train)
AdaBoost.score(X_test,y_test)


# In[ ]:


#knn = KNeighborsClassifier(n_neighbors = 100 , weights = 'distance', metric = 'manhattan', n_jobs = 1, p = 2)

model1 = DecisionTreeClassifier(criterion = 'entropy', random_state = 100, max_depth = 20, min_samples_leaf = 5)
model2 = GaussianNB()
#model3 = KNeighborsClassifier(n_neighbors = 10 , weights = 'distance', metric = 'manhattan', n_jobs = 1, p = 2)

rf = RandomForestClassifier(n_estimators = 65, criterion = 'entropy', random_state = 42, max_depth = 20)

sclf = StackingClassifier(classifiers=[model1, model2], meta_classifier = rf, use_probas=True)

print('3-fold cross validation:')
for model, label in zip([model1, model2, sclf], ['dtree', 'nb', 'stacking']):
    scores = model_selection.cross_val_score(model, x, labels, cv = 3, scoring = 'accuracy')
    print(scores)


# In[23]:


# perception 
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver = 'lbfgs', hidden_layer_sizes=(50,10), activation = 'logistic')
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
mlp.score(X_test,y_test)


# In[ ]:




