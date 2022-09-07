#!/usr/bin/env python
# coding: utf-8

# # Name:- Abhishek Ramesh Pawar
# ## Diabetes Data Analysis
# #### Exposy Data Labs Intern (Jan 2022)
# 
# #### import libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# #### Data Collection

# In[3]:


df = pd.read_csv("C:\\Users\\ap585\\Downloads\\diabetes.csv")
df.head()


# In[7]:


df.columns


# In[9]:


#Dimension of dataset
df.shape


# In[10]:


df.describe()


# In[12]:


#counting values of outcome from dataset for 0 means non diabetic and 1 means diabetic
sns.countplot(x='Outcome',data=df)


# In[13]:


df['Outcome'].value_counts()


# In[18]:


#Correlation matrix to show relation between two variables
corr_mat=df.corr()
sns.heatmap(corr_mat, annot=True)


# #### Data Cleaning

# In[15]:


#check for null or empty value in dataset
df.isnull().sum()


# In[16]:


#Independent matrix
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[17]:


x.shape


# In[18]:


x[0] #refering tp column 1 in data


# In[19]:


y


# #### Exploratory Data Analysis

# In[21]:


#presentation Of Glucose
fig = plt.figure(figsize =(15,7))
sns.distplot(df['Glucose'][df['Outcome'] == 1])
plt.xticks([i for i in range(0,201,15)], rotation = 45)
plt.ylabel("Glucose Count", fontsize = 16)
plt.title("Glucose For Diabetic", fontsize = 20)


# In[12]:


#insulin for diabetic
fig = plt.figure(figsize =(15,7))
sns.distplot(df['Insulin'][df['Outcome'] == 1], color='Grey')
plt.xticks()
plt.xlabel('Insulin', fontsize = 16)
plt.ylabel("Density", fontsize = 16)
plt.title("Insulin For Diabetic", fontsize = 20)


# In[8]:


#Body Mass for diabetic
fig = plt.figure(figsize =(15,7))
sns.distplot(df['BMI'][df['Outcome'] == 1])
plt.xticks()
plt.xlabel('BMI', fontsize = 16)
plt.ylabel("Density", fontsize = 16)
plt.title("BMI For Diabetic", fontsize = 20)


# In[11]:


#PedigreeFunction for diabetic
fig = plt.figure(figsize =(15,7))
sns.distplot(df['DiabetesPedigreeFunction'][df['Outcome'] == 1], color= 'Black')
plt.xticks([i*0.15 for i in range(1,12)])

plt.title("Diabetes Pedigree Function", fontsize = 20)


# In[17]:


#AGE for diabetic
fig = plt.figure(figsize =(15,7))
sns.distplot(df['Age'][df['Outcome'] == 1])
plt.xticks([i*0.15 for i in range(1,12)])
plt.xlabel('Age', fontsize = 16)
plt.ylabel("Density", fontsize = 16)
plt.title("AGE", fontsize = 20)


# #### Removing Unnessary Columns
# 

# In[5]:


x = df.drop(['Pregnancies','BloodPressure','SkinThickness','Outcome'],axis = 1)
y = df.iloc[:,-1]  


# #### Train And Test set

# In[6]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# In[22]:


#80% of dataset after removing unnecceasry data
x_train.shape 


# In[23]:


#20% of original dataset
x_test.shape


# #### Feature Scaling

# In[27]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[25]:


x_train


# #### Model Building- K Nearest Neighbor

# In[8]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 25, metric = 'minkowski')
knn.fit(x_train, y_train)


# In[9]:


#predecting data from KNeighbors
knn_y_pred = knn.predict(x_test)


# In[10]:


knn_y_pred


# #### Confusion Matrix 

# In[11]:


from sklearn.metrics import confusion_matrix
knn_cm = confusion_matrix(y_test, knn_y_pred)
sns.heatmap(knn_cm, annot=True)


# In[12]:


print("Correct:", sum(knn_y_pred==y_test))
print("Incorrect:", sum(knn_y_pred != y_test))
print("Accurancy:", sum(knn_y_pred ==y_test)/len(knn_y_pred))


# In[15]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,knn_y_pred)


# #### Simple Vector Mahine(SVM) 

# In[16]:


from sklearn.svm import SVC
svc = SVC(kernel = 'linear',random_state=0)
svc.fit(x_train, y_train)


# In[17]:


svc_y_pred = svc.predict(x_test)


# In[18]:


svc_cm = confusion_matrix(y_test,svc_y_pred)
print(svc_cm)


# In[19]:


print("Correct:", sum(svc_y_pred==y_test))
print("Incorrect:", sum(svc_y_pred != y_test))
print("Accurancy:", sum(svc_y_pred ==y_test)/len(knn_y_pred))


# #### Naive Bias

# In[21]:


from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)


# In[22]:


nb_y_pred = nb_classifier.predict(x_test)


# In[23]:


nb_cm = confusion_matrix(nb_y_pred, y_test)
print(nb_cm)


# In[24]:


print("Correct:", sum(nb_y_pred==y_test))
print("Incorrect:", sum(nb_y_pred != y_test))
print("Accurancy:", sum(nb_y_pred ==y_test)/len(nb_y_pred))


# #### Saving the classifier

# In[25]:


import pickle
pickle.dump(svc, open('classifier.pkl', 'wb'))


# In[28]:


pickle.dump(sc, open('sc.pkl', 'wb'))

