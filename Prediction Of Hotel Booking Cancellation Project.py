#!/usr/bin/env python
# coding: utf-8

# # Prediction Of Hotel Booking Cancellation
# 
# Name : Mrunmayee Prasad Bhosale
# 
# Subject : Machine Learning
# 
# Roll No : 16
# 
# Project : Prediction Of Hotel Booking Cancellation
#     

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data= pd.read_csv('hotel_bookings.csv')
data.head()


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


data.isnull().sum()


# In[6]:


def data_clean(df):
    df.fillna(0,inplace=True)
    print(df.isnull().sum())


# In[7]:


data_clean(data)


# In[8]:


list= ['adults', 'children', 'babies']
for i in list:
    print("{} has unique value as {}".format(i,data[i].unique()))


# In[9]:


len(data[data['adults']==0])


# In[10]:


filter=(data['children']==0) & (data['adults']==0) & (data['babies']==0)
data[filter]


# In[11]:


data_ori= data[~filter]


# In[12]:


data_ori.shape


# In[13]:


data_ori.head()


# # Which Countries do the guests come from?
# Spatial Analysis

# In[14]:


country_wise_data=data_ori[data_ori['is_canceled']==0]['country'].value_counts().reset_index()
country_wise_data.columns=['country','No of guests']
country_wise_data


# In[98]:


pip install folium


# In[99]:


import folium
from folium.plugins import HeatMap


# In[100]:


basemap=folium.Map()


# In[101]:


country_wise_data.dtypes


# In[102]:


import plotly.express as px


# In[103]:


map_guest = px.choropleth(country_wise_data,
                    locations=country_wise_data['country'],
                    color=country_wise_data['No of guests'], 
                    hover_name=country_wise_data['country'], 
                    title="Home country of guests")
map_guest.show()


# # Highest number of guests are from Portugal and other countries in Europe
# How much do guests pay for a room per night?

# In[ ]:


data2=data_ori[data_ori['is_canceled']==0]


# In[ ]:


plt.figure(figsize=(12, 8))
sns.boxplot(x="reserved_room_type",
            y="adr",
            hue="hotel",
            data=data2)
plt.title("Price of room types per night and person", fontsize=16)
plt.xlabel("Room type", fontsize=16)
plt.ylabel("Price [EUR]", fontsize=16)
plt.legend(loc="upper right")
plt.ylim(0, 600)
plt.show()


# # This figure shows the average price per room, depending on its type and the standard deviation. Note that due to data anonymization rooms with the same type letter may not necessarily be the same across hotels.

# # How does the price per night vary over the year?

# In[15]:


data_resort = data[(data["hotel"] == "Resort Hotel") & (data["is_canceled"] == 0)]
data_city = data[(data["hotel"] == "City Hotel") & (data["is_canceled"] == 0)]


# In[16]:


data_resort.head()


# In[17]:


resort_hotel=data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
resort_hotel


# In[18]:


city_hotel=data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel


# In[19]:


final=resort_hotel.merge(city_hotel,on='arrival_date_month')
final.columns=['month','price_for_resort','price_for_city_hotel']
final


# # now we will observe over here is month column is not in order, & if we will visualise we will get improper conclusion so very first we have to provide right hierarchy to the month column

# In[20]:


get_ipython().system('pip install sort-dataframeby-monthorweek')


# In[21]:


pip install sorted-months-weekdays


# In[22]:


import sort_dataframeby_monthorweek as sd


# In[23]:


def sort_data(df,colname):
    return sd.Sort_Dataframeby_Month(df,colname)


# In[24]:


final=sort_data(final,'month')
final


# In[104]:


px.line(final, x='month', y=['price_for_resort','price_for_city_hotel'], title='Room price per night over the Months')


# # Conclusion:- This clearly shows that the prices in the Resort hotel are much higher during the summer (no surprise here). The price of the city hotel varies less and is most expensive during spring and autumn.

# # Which are the most busy month or in which months Guests are high?

# In[26]:


data_resort.head()


# In[27]:


rush_resort=data_resort['arrival_date_month'].value_counts().reset_index()
rush_resort.columns=['month','no of guests']
rush_resort


# In[28]:


rush_city=data_city['arrival_date_month'].value_counts().reset_index()
rush_city.columns=['month','no of guests']
rush_city


# In[29]:


final_rush=rush_resort.merge(rush_city,on='month')
final_rush.columns=['month','no of guests in resort','no of guest in city hotel']
final_rush


# In[30]:


final_rush=sort_data(final_rush,'month')
final_rush


# In[31]:


final_rush.dtypes


# In[32]:


final_rush.columns


# In[105]:


px.line(data_frame=final_rush, x='month', y=['no of guests in resort','no of guest in city hotel'], title='Total no of guests per Months')


# # Conclusion
# The City hotel has more guests during spring and autumn, when the prices are also highest. In July and August there are less visitors, although prices are lower.
# 
# Guest numbers for the Resort hotel go down slighty from June to September, which is also when the prices are highest. Both hotels have the fewest guests during the winter.

# # How long do people stay at the hotels?

# In[34]:


filter=data['is_canceled']==0
clean_data=data[filter]


# In[35]:


clean_data.head()


# In[36]:


clean_data["total_nights"] = clean_data["stays_in_weekend_nights"] + clean_data["stays_in_week_nights"]


# In[37]:


clean_data.head()


# In[38]:


stay=clean_data.groupby(['total_nights','hotel']).agg('count').reset_index()
stay=stay.iloc[:,0:3]
stay.head()


# In[39]:


stay=stay.rename(columns={'is_canceled':'Number of stays'})
stay.head()


# In[40]:


plt.figure(figsize=(20, 8))
sns.barplot(x = "total_nights", y = "Number of stays" , hue="hotel",
            hue_order = ["City Hotel", "Resort Hotel"], data=stay)


# # Select important Features using Co-relation

# In[41]:


data.head()


# In[42]:


co_relation=data.corr()
co_relation


# In[43]:


co_relation=data.corr()["is_canceled"]
co_relation


# In[44]:


co_relation.abs().sort_values(ascending=False)


# In[45]:


co_relation.abs().sort_values(ascending=False)[1:]


# In[46]:


data.columns


# In[47]:


data.groupby("is_canceled")["reservation_status"].value_counts()


# In[48]:


list_not=['days_in_waiting_list','arrival_date_year']


# In[49]:


num_features=[col for col in data.columns if data[col].dtype!='O' and col not in list_not]
num_features


# In[50]:


cat_not=['arrival_date_year', 'assigned_room_type', 'booking_changes', 'reservation_status', 'country','days_in_waiting_list']


# In[51]:


cat_features=[col for col in data.columns if data[col].dtype=='O' and col not in cat_not]
cat_features


# In[52]:


data_cat=data[cat_features]


# In[53]:


data_cat.head()


# In[54]:


import warnings
from warnings import filterwarnings
filterwarnings("ignore")


# In[55]:


data_cat['reservation_status_date']=pd.to_datetime(data_cat['reservation_status_date'])


# In[56]:


data_cat['year']=data_cat['reservation_status_date'].dt.year
data_cat['month']=data_cat['reservation_status_date'].dt.month
data_cat['day']=data_cat['reservation_status_date'].dt.day


# In[57]:


data_cat.head()


# In[58]:


data_cat.drop('reservation_status_date',axis=1,inplace=True)


# In[59]:


data_cat['cancellation']=data['is_canceled']


# In[60]:


data_cat.columns


# # Feature Encoding
# Perform Mean Encoding Technique

# In[61]:


cols=data_cat.columns[0:8]
cols


# In[62]:


for col in cols:
    print(data_cat.groupby([col])['cancellation'].mean())
    print('\n')


# In[63]:


for col in cols:
    print(data_cat.groupby([col])['cancellation'].mean().to_dict())
    print('\n')


# In[64]:


df=data_cat.copy()


# In[65]:


for col in cols:
    dict=data_cat.groupby([col])['cancellation'].mean().to_dict()
    data_cat[col]=data_cat[col].map(dict)


# In[66]:


data_cat.head(20)


# In[67]:


dataframe=pd.concat([data_cat,data[num_features]],axis=1)


# In[68]:


dataframe.head()


# In[69]:


dataframe.drop(['cancellation'],axis=1,inplace=True)


# In[70]:


dataframe.shape


# # Handle Outliers

# In[71]:


sns.distplot(dataframe['lead_time'])


# In[73]:


import numpy as np

def handle_outlier(col):
    dataframe[col]=np.log1p(dataframe[col])


# In[74]:


handle_outlier('lead_time')


# In[75]:


sns.distplot(dataframe['lead_time'].dropna())


# In[76]:


sns.distplot(dataframe['adr'])


# In[77]:


dataframe.isnull().sum()


# In[78]:


dataframe.dropna(inplace=True)


# In[79]:


y=dataframe['is_canceled']
x=dataframe.drop('is_canceled',axis=1)


# # Feature Importance

# In[80]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[81]:


feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) 
feature_sel_model.fit(x,y)


# In[82]:


feature_sel_model.get_support()


# In[83]:


cols=x.columns


# In[84]:


selected_feat = cols[(feature_sel_model.get_support())]


# In[85]:


print('total features: {}'.format((x.shape[1])))
print('selected features: {}'.format(len(selected_feat)))


# In[86]:


selected_feat


# In[87]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,random_state=0)


# In[88]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)


# In[89]:


y_pred=logreg.predict(x_test)


# In[90]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[91]:


from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
score


# In[92]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(logreg,x,y,cv=10)


# In[93]:


score


# In[94]:


score.mean()


# In[95]:


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[96]:


models = []
models.append(('LogisticRegression', LogisticRegression()))
models.append(('Naive Bayes',GaussianNB()))
models.append(('RandomForest', RandomForestClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))


# In[97]:


for name, model in models:
    print(name)
    model.fit(x_train, y_train)
    
    # Make predictions.
    predictions = model.predict(x_test)

    # Compute the error.
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(predictions, y_test))

    from sklearn.metrics import accuracy_score
    print(accuracy_score(predictions,y_test))
    print('\n')


# # Conclusion:
# Random Forest gives the highest accuracy

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




