#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd

train_df = pd.read_csv('C:/Users/hp/OneDrive/Desktop/dataset1.csv')
train_df.info()


# In[10]:


import pandas as pd
myFile=pd.read_csv('C:/Users/hp/OneDrive/Desktop/dataset.csv')
print(myFile)


# In[11]:


import numpy as np # linear algebra
import pandas as pd # data processing
import warnings# warning filter


# In[12]:


import matplotlib.pyplot as plt 
import seaborn as sns


# In[13]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[15]:



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[16]:


sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)


# In[17]:


warnings.filterwarnings("ignore")


# In[19]:


tr_df=pd.read_csv("C:/Users/hp/OneDrive/Desktop/dataset1.csv")
print(tr_df)


# In[20]:


te_df=pd.read_csv("C:/Users/hp/OneDrive/Desktop/dataset.csv")
print(te_df)


# In[21]:


print(f"training set (row, col): {tr_df.shape}\n\ntesting set (row, col): {te_df.shape}")


# In[22]:


tr_df.info(verbose=True, null_counts=True)


# In[23]:


tr_df.describe()


# In[24]:


tr_df.drop('Loan_ID',axis=1,inplace=True)
te_df.drop('Loan_ID',axis=1,inplace=True)
print(f"training set (row, col): {tr_df.shape}\n\ntesting set (row, col): {te_df.shape}")


# In[25]:


tr_df.isnull().sum().sort_values(ascending=False)


# In[26]:


print("Before filling missing values\n\n","#"*50,"\n")
null_cols = ['Credit_History', 'Self_Employed', 'LoanAmount','Dependents', 'Loan_Amount_Term', 'Gender', 'Married']


# In[27]:


for col in null_cols:
    print(f"{col}:\n{tr_df[col].value_counts()}\n","-"*50)
    tr_df[col] = tr_df[col].fillna(
    tr_df[col].dropna().mode().values[0] )   


# In[28]:


tr_df.isnull().sum().sort_values(ascending=False)
print("After filling missing values\n\n","#"*50,"\n")
for col in null_cols:
    print(f"\n{col}:\n{tr_df[col].value_counts()}\n","-"*50)


# In[29]:


num = tr_df.select_dtypes('number').columns.to_list()
cat = tr_df.select_dtypes('object').columns.to_list()


# In[30]:


loan_num =  tr_df[num]


# In[31]:


loan_cat = tr_df[cat]


# In[32]:


total = float(len(tr_df[cat[-1]]))
plt.figure(figsize=(8,10))
sns.set(style="whitegrid")
ax = sns.countplot(tr_df[cat[-1]])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format(height/total),ha="center") 
plt.show()


# In[33]:


for i in loan_num:
    plt.hist(loan_num[i])
    plt.title(i)
    plt.show()


# In[34]:


for i in cat[:-1]: 
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    sns.countplot(x=i ,hue='Loan_Status', data=tr_df ,palette='plasma')
    plt.xlabel(i, fontsize=14)


# In[35]:


sns.heatmap(tr_df.corr() ,cmap='cubehelix_r')


# In[36]:


to_numeric = {'Male': 1, 'Female': 2,
'Yes': 1, 'No': 2,
'Graduate': 1, 'Not Graduate': 2,
'Urban': 3, 'Semiurban': 2,'Rural': 1,
'Y': 1, 'N': 0,
'3+': 3}


# In[37]:


tr_df = tr_df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)
te_df = te_df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)


# In[38]:


Dependents_ = pd.to_numeric(tr_df.Dependents)
Dependents__ = pd.to_numeric(te_df.Dependents)


# In[39]:


tr_df.drop(['Dependents'], axis = 1, inplace = True)
te_df.drop(['Dependents'], axis = 1, inplace = True)


# In[40]:


tr_df = pd.concat([tr_df, Dependents_], axis = 1)
te_df = pd.concat([te_df, Dependents__], axis = 1)


# In[41]:


print(f"training set (row, col): {tr_df.shape}\n\ntesting set (row, col): {te_df.shape}\n")
print(tr_df.info(), "\n\n", te_df.info())


# In[42]:


sns.heatmap(tr_df.corr() ,cmap='cubehelix_r')


# In[43]:


corr = tr_df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[44]:


y = tr_df['Loan_Status']
X = tr_df.drop('Loan_Status', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[45]:


DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)

y_predict = DT.predict(X_test)


# In[46]:


print(classification_report(y_test, y_predict))


# In[47]:


DT_SC = accuracy_score(y_predict,y_test)
print(f"{round(DT_SC*100,2)}% Accurate")


# In[48]:


DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)

y_predict = DT.predict(X_test)


# In[49]:


print(classification_report(y_test, y_predict))


# In[50]:


DT_SC = accuracy_score(y_predict,y_test)
print(f"{round(DT_SC*100,2)}% Accurate")


# In[51]:


from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)


# In[52]:


y_predict = DT.predict(X_test)


# In[53]:


print(classification_report(y_test, y_predict))


# In[54]:


DT_SC = accuracy_score(y_predict,y_test)
print(f"{round(DT_SC*100,2)}% Accurate")


# In[55]:


Decision_Tree=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
Decision_Tree.to_csv("Dection Tree.csv")  


# In[56]:


from sklearn.ensemble import RandomForestClassifier



RF = RandomForestClassifier()
RF.fit(X_train, y_train)

y_predict = RF.predict(X_test)


# In[57]:


print(classification_report(y_test, y_predict))


# In[58]:


RF_SC = accuracy_score(y_predict,y_test)
print(f"{round(RF_SC*100,2)}% Accurate")


# In[59]:


Random_Forest=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
Random_Forest.to_csv("Random Forest.csv")   


# In[60]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train, y_train)

y_predict = LR.predict(X_test)


# In[61]:


print(classification_report(y_test, y_predict))


# In[62]:


LR_SC = accuracy_score(y_predict,y_test)
print('accuracy is',accuracy_score(y_predict,y_test))


# In[63]:


score = [DT_SC,RF_SC,LR_SC]
Models = pd.DataFrame({
    'n_neighbors': ["Decision Tree","Random Forest", "Logistic Regression"],
    'Score': score})
Models.sort_values(by='Score', ascending=False)


# In[ ]:




