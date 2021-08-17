#!/usr/bin/env python
# coding: utf-8

# # Part One: Get all of the libraries and prepare the data connection in Snowflake

# In[3]:


#  Load the libraries for the model

import graphviz
from graphviz import Digraph
from IPython.display import display
from IPython.display import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from pandas import DataFrame
import pandas as pd
import pydot
import pydotplus
import seaborn as sns
from sklearn.externals.six import StringIO 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split #for decision tree object
from sklearn.tree import DecisionTreeClassifier #for checking testing results
from sklearn.metrics import classification_report, confusion_matrix #for visualizing tree 
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
import snowflake
from snowflake import connector
import tempfile


# In[9]:


# Connection to snowflake

ctx = snowflake.connector.connect (
                 user = 'MPETERSON',
                 password = 'Welcome1@',
                 account = 'cradlepoint',
                 warehouse = 'PROD_MGMT_XS',
                 database = 'PROD_MGMT_DB',
                 schema = 'ML'
                  )
print("Connection is good to go!")


# In[10]:


cs = ctx.cursor()
print("Recieved the cursor object!")


# In[11]:


try:
    cs.execute("select * FROM PROD_MGMT_DB.CASE_DATA.CASE_FEATURE_DATA WHERE SALESFORCE_CASE_CREATED_DATE >= '2021-01-01' AND PENDO_FLAG = 1")
    df = cs.fetch_pandas_all()
    df.info()
    print("---------")
finally:
    cs.close()
    print("These are the results of the Query")


# In[84]:


df = pd.read_csv("CASE_DATA_CLEANSED.csv")


# In[85]:


df = df[df['PENDO_FLAG']== 1]


# In[86]:


df


# # Part Two: Feature Engineering 

# In[102]:


decision_ds = df[[
# 'SFDC_ACCOUNT_ID',
# 'PENDO_FLAG',
# 'SHIPPING_COUNTRY',
# 'CASE_MONTH',
# 'EVENT_MONTH',
'ACCOUNT_AGE_IN_MONTHS',
'SALESFORCE_CASE_CARRIER_C',
'NUM_EVENTS',
'NUM_MINUTES',
'SALESFORCE_CASE_COUNT',
'AVG_ROUTER_AGE_IN_MONTHS',
'SALESFORCE_CASE_STATUS',
'SALESFORCE_CASE_ORIGIN',
'CHURNED_ACCOUNT_FLAG',
'CURRENT_ROUTER_COUNT',
'SUBSCRIBED_ROUTER_COUNT',
'SUBSCRIBED_AVG_ROUTER_COUNT',
'SUBSCRIBED_MAX_ROUTER_COUNT',
'LOGIN_COUNT',
'TOTAL_MINUTES']]


# In[103]:


decision_ds.head()


# In[104]:


decision_ds.info()


# In[105]:


decision_ds.isnull().any()


# In[106]:


decision_ds.fillna("Unknown", inplace = True)


# In[107]:


decision_ds['SALESFORCE_CASE_STATUS'].unique()


# In[108]:


decision_ds['SALESFORCE_CASE_ORIGIN'].unique()


# In[109]:


decision_ds['SALESFORCE_CASE_CARRIER_C'].unique()


# In[110]:


x = decision_ds['CHURNED_ACCOUNT_FLAG']


# In[111]:


y = decision_ds['SALESFORCE_CASE_COUNT']


# In[112]:


np.corrcoef(x, y)


# In[113]:


decision_ds.info()


# In[114]:


decision_ds_coded = pd.get_dummies(decision_ds, columns=["SALESFORCE_CASE_CARRIER_C", "SALESFORCE_CASE_STATUS", "SALESFORCE_CASE_ORIGIN"], prefix=["SF_Carrier", "SF_Status", "SF_Origin"])


# In[115]:


decision_ds_coded.head()


# In[116]:


decision_ds_coded.info()


# # Part Three: Split the data

# In[21]:


print(sorted(decision_ds_coded))


# In[22]:


feature_cols = ['CURRENT_ROUTER_COUNT','ACCOUNT_AGE_IN_MONTHS','AVG_ROUTER_AGE_IN_MONTHS','LOGIN_COUNT','SUBSCRIBED_AVG_ROUTER_COUNT','SUBSCRIBED_MAX_ROUTER_COUNT','NUM_MINUTES','NUM_EVENTS','TOTAL_MINUTES','SUBSCRIBED_ROUTER_COUNT','SF_Carrier_Not Provided','SF_Carrier_Verizon','SF_Origin_Phone','SF_Origin_SaaS', 'SF_Carrier_T-Mobile','SF_Origin_Portal']


# In[23]:


feature_cols


# In[24]:


X = decision_ds_coded[feature_cols]


# In[25]:


y = decision_ds_coded.CHURNED_ACCOUNT_FLAG


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# # Part Four: Build the Model and Visualize Model

# In[27]:


clf = DecisionTreeClassifier()


# In[28]:


clf = clf.fit(X_train,y_train)


# In[29]:


y_pred = clf.predict(X_test)


# In[30]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[31]:


dot_data = StringIO()


# In[32]:


dot_data = export_graphviz(clf, out_file=None,
                filled=True, rounded=True, feature_names = feature_cols, special_characters=True, class_names=['0','1'])


# In[33]:


feat_importance = clf.tree_.compute_feature_importances(normalize=False)


# In[34]:


feat_importance


# In[35]:


feat_importance_df = DataFrame (feat_importance)


# In[36]:


feat_importance_df.rename(columns={feat_importance_df.columns[0]:'importance'}, inplace=True)


# In[37]:


feature_cols_df = DataFrame (feature_cols)


# In[38]:


feature_cols_df.rename(columns={feature_cols_df.columns[0]:'feature'}, inplace=True)


# In[39]:


feature_importance_df = pd.concat([feat_importance_df,feature_cols_df], axis=1)


# In[40]:


feature_importance_df


# In[41]:


chart = sns.catplot(x ='importance', y ='feature', data = feature_importance_df, kind ='bar')
chart.set_xticklabels(rotation=90, horizontalalignment='right')
chart.fig.set_size_inches(15,15)


# In[42]:


graph = graphviz.Source(dot_data)


# In[43]:


graph


# In[44]:


graph.render("graph.jpeg")


# In[45]:


tree.plot_tree(clf) 


# # Part Five: Testing the Model

# In[46]:


print(classification_report(y_test, y_pred))


# In[47]:


print('confusion matrix {}'.format(pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['Predicted Positive', 'Predicted Negative'],
    index=['True Positive', 'True Negative']
)))


# In[48]:


y_pred = clf.predict(X_test)


# # Section Two (Bonus): Decision Tree Regression Modeling

# # Part One: Split Data

# In[117]:


# Use the dataset from classification

feature_cols = ['CURRENT_ROUTER_COUNT','ACCOUNT_AGE_IN_MONTHS','AVG_ROUTER_AGE_IN_MONTHS','LOGIN_COUNT','SUBSCRIBED_AVG_ROUTER_COUNT','SUBSCRIBED_MAX_ROUTER_COUNT','NUM_MINUTES','NUM_EVENTS','TOTAL_MINUTES','SUBSCRIBED_ROUTER_COUNT','SF_Carrier_Not Provided','SF_Carrier_Verizon','SF_Origin_Phone','SF_Origin_SaaS', 'SF_Carrier_T-Mobile','SF_Origin_Portal']


# In[118]:


feature_cols


# In[119]:


X = decision_ds_coded[feature_cols]


# In[66]:


X = X[1:5000]


# In[120]:


X


# In[121]:


y = decision_ds_coded.SALESFORCE_CASE_COUNT


# In[70]:


y = y[1:5000]


# In[122]:


y


# In[123]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# # Part Two: Build the Model and Visualize Model

# In[124]:


DtReg = DecisionTreeRegressor(random_state=0)


# In[125]:


DtReg.fit(X_train, y_train)


# In[126]:


dot_data = StringIO()


# In[127]:


dot_data = export_graphviz(DtReg, out_file=None,
                filled=True, rounded=True, feature_names = feature_cols, special_characters=True)


# In[128]:


feat_importance = DtReg.tree_.compute_feature_importances(normalize=False)


# In[129]:


feat_importance


# In[130]:


feat_importance_df = DataFrame (feat_importance)


# In[131]:


feat_importance_df.rename(columns={feat_importance_df.columns[0]:'importance'}, inplace=True)


# In[132]:


feature_cols_df = DataFrame (feature_cols)


# In[133]:


feature_cols_df.rename(columns={feature_cols_df.columns[0]:'feature'}, inplace=True)


# In[134]:


feature_importance_df = pd.concat([feat_importance_df,feature_cols_df], axis=1)


# In[135]:


feature_importance_df


# In[136]:


chart = sns.catplot(x ='importance', y ='feature', data = feature_importance_df, kind ='bar')
chart.set_xticklabels(rotation=90, horizontalalignment='right')
chart.fig.set_size_inches(15,15)


# In[137]:


graph = graphviz.Source(dot_data)


# In[138]:


graph 


# In[88]:


tree.plot_tree(DtReg) 


# # Part Three: Testing the Model and Performing Descriptive Statistics

# In[139]:


df_with_case_count = decision_ds_coded[['SALESFORCE_CASE_COUNT','CURRENT_ROUTER_COUNT','ACCOUNT_AGE_IN_MONTHS','AVG_ROUTER_AGE_IN_MONTHS','LOGIN_COUNT','SUBSCRIBED_AVG_ROUTER_COUNT','SUBSCRIBED_MAX_ROUTER_COUNT','NUM_MINUTES','NUM_EVENTS','TOTAL_MINUTES','SUBSCRIBED_ROUTER_COUNT','SF_Carrier_Not Provided','SF_Carrier_Verizon','SF_Origin_Phone','SF_Origin_SaaS', 'SF_Carrier_T-Mobile','SF_Origin_Portal']]


# In[140]:


df_not_with_case_count  = decision_ds_coded[['CURRENT_ROUTER_COUNT','ACCOUNT_AGE_IN_MONTHS','AVG_ROUTER_AGE_IN_MONTHS','LOGIN_COUNT','SUBSCRIBED_AVG_ROUTER_COUNT','SUBSCRIBED_MAX_ROUTER_COUNT','NUM_MINUTES','NUM_EVENTS','TOTAL_MINUTES','SUBSCRIBED_ROUTER_COUNT','SF_Carrier_Not Provided','SF_Carrier_Verizon','SF_Origin_Phone','SF_Origin_SaaS', 'SF_Carrier_T-Mobile','SF_Origin_Portal']]


# In[141]:


# Create a dataset that is for comparison

df_with_case_count


# In[142]:


# Create a dataset that is for predicting on

df_not_with_case_count


# In[143]:


plt.figure(figsize=(20, 20))
heatmap = sns.heatmap(df_with_case_count.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')


# In[144]:


plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(df_with_case_count.corr()[['SALESFORCE_CASE_COUNT']].sort_values(by='SALESFORCE_CASE_COUNT', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Case Count', fontdict={'fontsize':18}, pad=16)
plt.savefig('heatmap_case_count.png', dpi=300, bbox_inches='tight')


# In[145]:


y_hats = DtReg.predict(X)


# In[146]:


y_hats  = pd.DataFrame(y_hats)


# In[147]:


y_hats


# In[152]:


df


# In[153]:


# Export prediction dataframe out of Python

y_hats.to_csv('d1.csv')


# In[156]:


# Export original dataframe out of Python

decision_ds_coded[5001:33936].to_csv('d2.csv')


# In[104]:


y_hats = DtReg.predict(X)


# In[105]:


y_hats = DtReg.predict_proba(X)


# In[108]:


df[df['PENDO_FLAG']== 1]


# In[ ]:




