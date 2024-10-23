#!/usr/bin/env python
# coding: utf-8

# # Project Python Foundations: FoodHub Data Analysis
# 
# **Marks: 60**

# ### Context
# 
# The number of restaurants in New York is increasing day by day. Lots of students and busy professionals rely on those restaurants due to their hectic lifestyles. Online food delivery service is a great option for them. It provides them with good food from their favorite restaurants. A food aggregator company FoodHub offers access to multiple restaurants through a single smartphone app.
# 
# The app allows the restaurants to receive a direct online order from a customer. The app assigns a delivery person from the company to pick up the order after it is confirmed by the restaurant. The delivery person then uses the map to reach the restaurant and waits for the food package. Once the food package is handed over to the delivery person, he/she confirms the pick-up in the app and travels to the customer's location to deliver the food. The delivery person confirms the drop-off in the app after delivering the food package to the customer. The customer can rate the order in the app. The food aggregator earns money by collecting a fixed margin of the delivery order from the restaurants.
# 
# ### Objective
# 
# The food aggregator company has stored the data of the different orders made by the registered customers in their online portal. They want to analyze the data to get a fair idea about the demand of different restaurants which will help them in enhancing their customer experience. Suppose you are hired as a Data Scientist in this company and the Data Science team has shared some of the key questions that need to be answered. Perform the data analysis to find answers to these questions that will help the company to improve the business. 
# 
# ### Data Description
# 
# The data contains the different data related to a food order. The detailed data dictionary is given below.
# 
# ### Data Dictionary
# 
# * order_id: Unique ID of the order
# * customer_id: ID of the customer who ordered the food
# * restaurant_name: Name of the restaurant
# * cuisine_type: Cuisine ordered by the customer
# * cost_of_the_order: Cost of the order
# * day_of_the_week: Indicates whether the order is placed on a weekday or weekend (The weekday is from Monday to Friday and the weekend is Saturday and Sunday)
# * rating: Rating given by the customer out of 5
# * food_preparation_time: Time (in minutes) taken by the restaurant to prepare the food. This is calculated by taking the difference between the timestamps of the restaurant's order confirmation and the delivery person's pick-up confirmation.
# * delivery_time: Time (in minutes) taken by the delivery person to deliver the food package. This is calculated by taking the difference between the timestamps of the delivery person's pick-up confirmation and drop-off information

# ### Let us start by importing the required libraries

# In[1]:


# import libraries for data manipulation
import numpy as np
import pandas as pd
# import libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns


# ### Understanding the structure of the data

# In[3]:


# read the data
df = pd.read_csv('foodhub_order.csv')
# returns the first 5 rows
df.head()


# In[ ]:


# show a chart by Cuisine_type and the cost of order ? also show if their cost of
#order diffeent for week vs weekday.
# make sure it make in professional. Because your shariing this to the CEO of Seamless


# In[31]:


df['cuisine_type'].value_counts().head()


# In[32]:


df.groupby(by=['cuisine_type','day_of_the_week'])['cost_of_the_order'].mean().reset_index() 


# In[53]:


dem = df.groupby(by=['cuisine_type','day_of_the_week'])['cost_of_the_order'].mean().reset_index().sort_values(by='cost_of_the_order',ascending=False)#.head(10)
dem = dem.rename(columns = ({'day_of_the_week':'Day Of The Week'}))
dem 


# In[68]:


def plot_bar_plot_(df,feature1,feature2):
    color = ['#fb8500','#5a189a']
    plt.figure(figsize=(18,7))
    ax = sns.barplot(data=dem,x='cuisine_type',
                y='cost_of_the_order',
                palette=color,
                ci=False,
                hue='Day Of The Week')
    for p in ax.patches:
        ax.annotate("{:,.1f}".format(p.get_height()),
                     (p.get_x() + p.get_width()/2,p.get_height()),
                     ha ='left',
                     va='center',
                     size=11,
                     xytext=(-12,8),
                     textcoords = 'offset points')
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.suptitle('Cuisine Type Cost Of Order By Week vs Weekend',size=22)
    plt.title('July 2014 - July 2015',size=18)
    plt.ylabel('Cost Of Order',size=16,fontweight='bold')
    plt.xlabel('Cuisine Type',size=16,fontweight='bold')
    #plt.legend(labels=['Day Of The Week'])
    plt.show()


# In[69]:


def plot_bar_plot_(df,feature1,feature2,feature3,tile):
    color = ['#fb8500','#5a189a']
    plt.figure(figsize=(18,7))
    ax = sns.barplot(data=dem,x=feature1,
                y=feature2,
                palette=color,
                ci=False,
                hue=feature3)
    for p in ax.patches:
        ax.annotate("{:,.1f}".format(p.get_height()),
                     (p.get_x() + p.get_width()/2,p.get_height()),
                     ha ='left',
                     va='center',
                     size=11,
                     xytext=(-12,8),
                     textcoords = 'offset points')
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.suptitle(tile ,size=22)
    plt.title('July 2014 - July 2015',size=18)
    plt.ylabel('Cost Of Order',size=16,fontweight='bold')
    plt.xlabel('Cuisine Type',size=16,fontweight='bold')
    #plt.legend(labels=['Day Of The Week'])
    plt.show()


# In[167]:


pwd 


# In[70]:


plot_bar_plot_(dem,'cuisine_type','cost_of_the_order','Day Of The Week',)


# #### Observations:
# 
# The DataFrame has 9 columns as mentioned in the Data Dictionary. Data in each row corresponds to the order placed by a customer.

# ### **Question 1:** Write the code to check the shape of the dataset and write your observations based on that. (0.5 mark)

# In[55]:


# check the shape of the dataset
df.shape


# #### Observations:
# 
# * The DataFrame has 1898 rows and 9 columns.

# ### Question 2: Write the observations based on the below output from the info() method. (0.5 mark)

# In[56]:


# use info() to print a concise summary of the DataFrame
df.info()


# #### Observations:
# * There are a total of 1898 non-null observations in each of the columns.
# 
# * The dataset contains 9 columns: 4 are of integer type ('order_id', 'customer_id', 'food_preparation_time', 'delivery_time'), 1 is of floating point type ('cost_of_the_order') and 4 are of the general object type ('restaurant_name', 'cuisine_type', 'day_of_the_week', 'rating').
# 
# * Total memory usage is approximately 133.6 KB.
# 
# 

# ### Question 3: 'restaurant_name', 'cuisine_type', 'day_of_the_week' are object types. Write the code to convert the mentioned features to 'category' and write your observations on the same. (0.5 mark)

# In[58]:


# coverting "objects" to "category" reduces the data space required to store the dataframe
# write the code to convert 'restaurant_name', 'cuisine_type', 'day_of_the_week' into categorical data
df.restaurant_name = df.restaurant_name.astype('category')     # Convert restaurant name from object to category
df.cuisine_type = df.cuisine_type.astype('category')           # Convert cuisine type from object to category
df.day_of_the_week = df.day_of_the_week.astype('category')     # Convert day of the week from object to category

# use info() to print a concise summary of the DataFrame
df.info()


# #### Observations:
# 
# * 'restaurant_name', 'cuisine_type' and 'day_of_the_week' are now converted into categorical values.
# 
# * Total memory usage has decreased now.
# 

# ### **Question 4:** Write the code to find the summary statistics and write your observations based on that. (1 mark)

# In[60]:


# get the summary statistics of the numerical data
df.describe()


# #### Observations:
# 
# * Order ID and Customer ID are just identifiers for each order.
# 
# * The cost of an order ranges from 4.47 to 35.41 dollars, with an average order costing around 16 dollars and a standard deviation of 7.5 dollars. The cost of 75% of the orders are below 23 dollars. This indicates that most of the customers prefer low-cost food compared to the expensive ones.
# 
# * Food preparation time ranges from 20 to 35 minutes, with an average of around 27 minutes and a standard deviation of 4.6 minutes. The spread is not very high for the food preparation time.
# 
# * Delivery time ranges from 15 to 33 minutes, with an average of around 24 minutes and a standard deviation of 5 minutes. The spread is not too high for delivery time either. 
# 

# ### **Question 5:** How many orders are not rated? (0.5 mark)

# In[10]:


df['rating'].value_counts(dropna=False)


# In[62]:


df['rating'].value_counts(1)*100


# In[63]:


df.head()


# In[64]:


df['cost_of_the_order'].hist()


# #### Observations:
# 
# * There are 736 orders that are not rated.

# ### Exploratory Data Analysis (EDA)

# ### Univariate Analysis

# ### **Question 6:** Explore all the variables and provide observations on the distributions of all the relevant variables in the dataset. (5 marks)

# In[65]:


df['food_preparation_time'].value_counts()


# In[66]:


sns.histplot(x='cost_of_the_order',data=df)


# In[67]:


# function to plot a boxplot and a histogram along the same scale.
def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="#DE3163"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="#2988FA",
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="#BF2611", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram
    


# In[71]:


# function to create labeled barplots

def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 5, 5))
    else:
        plt.figure(figsize=(n + 5, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=19,
            xytext=(1, -18),
            textcoords="offset points",
            horizontalalignment='right'
        )  # annotate the percentage

    plt.show()  # show the plot
    

    
# 


# #### Order ID

# In[72]:


# check unique order ID
#df['order_id'].value_counts()


# #### Observations:
# 
# * There are 1898 unique orders. As mentioned earlier, 'order_id' is just an identifier for the orders.

# #### Customer ID

# In[73]:


# check unique customer ID
df['customer_id'].value_counts().shape


# #### Observations:
# 
# * There are 1200 unique customers. Though 'customer_id' is just a variable to identify customers, we can see that there are some customers who have placed more than one order.
# 
# * Let's check the top 5 customers' IDs who have ordered most frequently.

# In[74]:


df.head(2)


# In[75]:


df['customer_id'].value_counts()[:6]


# #### Observations:
# 
# * Customer with ID 52832 has ordered 13 times.

# #### Restaurant name

# In[77]:


# check unique restaurant name
df['restaurant_name'].value_counts().shape


# #### Observations:
# 
# * There are 178 unique restaurants in the dataset.
# 
# * Let's check the number of orders that get served by the restaurants.

# In[80]:


df['restaurant_name'].value_counts().head(10).reset_index()


# #### Observations:
# 
# * The restaurant that has received maximum number of orders is Shake Shack 

# #### Cuisine type

# In[81]:


# check unique cuisine type
df['cuisine_type'].value_counts().shape


# In[88]:


df['cuisine_type'].value_counts().reset_index().sort_values(by='cuisine_type',ascending=Flase)


# In[84]:


df['cuisine_type'].value_counts().plot(kind='bar',width=0.7)


# In[28]:


labeled_barplot(df, 'cuisine_type', perc=True)


# #### Observations:
# 
# * There are 14 unique cuisines in the dataset.
# 
# * The distribution of cuisine types show that cuisine types are not equally distributed. 
# 
# * The most frequent cuisine type is American followed by Japanese and Italian.
# 
# * Vietnamese appears to be the least popular of all the cuisines.

# #### Cost of the order

# In[89]:


histogram_boxplot(df, 'cost_of_the_order')


# #### Observations:
# 
# * The average cost of the order is greater than the median cost indicating that the distribution for the cost of the order is right-skewed.
# 
# * The mode of the distribution indicates that a large chunk of people prefer to order food that costs around 10-12 dollars.
# 
# * There are few orders that cost greater than 30 dollars. These orders might be for some expensive meals.

# #### Day of the week

# In[90]:


# check the unique values
df['day_of_the_week'].value_counts()


# In[91]:


labeled_barplot(df, 'day_of_the_week', perc=True)


# #### Observations:
# 
# * The 'day_of_the_week' columns consists of 2 unique values - Weekday and Weekend
# * The distribution shows that around 71% of all orders are placed on weekends.

# #### Rating

# In[92]:


# check the unique values
df['rating'].value_counts()


# In[93]:


labeled_barplot(df, 'rating', perc=True)


# #### Observations:
# 
# * The distribution of 'rating' shows that the most frequent rating category is 'not given' (around 39%), followed by a rating of 5 (around 31%).
# 
# * Only 10% orders have been rated 3.

# #### Food Preparation time

# In[94]:


histogram_boxplot(df, 'food_preparation_time', bins = 16)


# #### Observations:
# 
# * The average food preparation time is almost equal to the median food preparation time indicating that the distribution is nearly symmetrical.
# 
# * The food preparation time is pretty evenly distributed between 20 and 35 minutes.
# 
# * There are no outliers in this column.

# #### Delivery time

# In[95]:


histogram_boxplot(df, 'delivery_time')


# #### Observations:
# 
# * The average delivery time is a bit smaller than the median delivery time indicating that the distribution is a bit left-skewed.
# 
# * Comparatively more number of orders have delivery time between 24 and 30 minutes.
# 
# * There are no outliers in this column.

# ### Question 7: Write the code to find the top 5 restaurants that have received the highest number of orders. (1 mark)

# In[96]:


# Get top 5 restaurants with highest number of orders
df['restaurant_name'].value_counts()[:5] # you  could also do .head(5)


# In[97]:


plt.figure(figsize=(7,5))
df['restaurant_name'].value_counts()[:5].sort_values().plot(kind='barh',rot='horizontal',width=0.85)
plt.xlabel('Restaurant Counts',size=14)
plt.ylabel('Restaurant  Name',size=14)
plt.title('Top Five Restaurant In New York City',size=17)
plt.show()



# In[99]:


#print('Almost {} % of the orders in the dataset are from these restaurants'.format(round(res['restaurant_name'].sum()/df.shape[0],3)*100))


# In[100]:


#print('Total order {} in this data'.format(round(res['restaurant_name'].sum()/df.shape[0],3)*100))


# #### Observations:
# 
# * Top 5 popular restaurants that have received the highest number of orders **'Shake Shack', 'The Meatball Shop', 'Blue Ribbon Sushi', 'Blue Ribbon Fried Chicken' and 'Parm'**. 
# 
# * Almost 33% of the orders in the dataset are from these restaurants.
# 

# ### Question 8: Write the code to find the most popular cuisine on weekends. (1 mark)

# In[101]:


# Get most popular cuisine on weekends
df_weekend = df[df['day_of_the_week'] == 'Weekend']
df_weekend['cuisine_type'].value_counts() 

plt.figure(figsize=(7,5))
df_weekend['cuisine_type'].value_counts().sort_values().plot(kind='barh',rot='horizontal',width=0.85)
#plt.xlabel('Restaurant Counts',size=14)
#plt.ylabel('Restaurant  Name',size=14)
#plt.title('Top Five Restaurant In New York City',size=17)
plt.show()


# In[102]:


# Get most popular cuisine on weekends
df_weekend = df[df['day_of_the_week'] == 'Weekend']
wkd_pop = df_weekend['cuisine_type'].value_counts().reset_index()


wkd_pop = wkd_pop.rename(columns={'index':'cuisine','cuisine_type':'counts'})
wkd_pop


# In[103]:


# Get most popular cuisine on weekends
df_weekend = df[df['day_of_the_week'] == 'Weekend']
wkd_pop = df_weekend['cuisine_type'].value_counts().reset_index()
wkd_pop = wkd_pop.rename(columns={'index':'cuisine','cuisine_type':'counts'})
top = wkd_pop.head(4)
top 


# In[104]:


top['counts'].sum()/wkd_pop['counts'].sum()


# In[105]:


top_orders = round(top['counts'].sum()/wkd_pop['counts'].sum(),3)
top_orders


# In[106]:


print('Over 80% of the orders are for American, Japanese, Italian and Chinese cuisines. Thus, it seems that these cuisines are quite popular among customers of FoodHub.',top_orders)


# In[107]:


plt.figure(figsize=(8,6))
sns.barplot(y='cuisine',x='counts',data=wkd_pop,color = '#1165BF',order=wkd_pop['cuisine'])
plt.title('The most popular cuisine type on weekends is American',size=18)
plt.xlabel('Order count')
plt.ylabel('Cuisine')
plt.rcParams['axes.spines.right']=False # remove the plot border
plt.rcParams['axes.spines.left']=False # remove the plot border
plt.rcParams['axes.spines.top']=False # remove the plot border
plt.rcParams['axes.spines.bottom']=True


# #### Observations:
# 
# * The most popular cuisine type on weekends is American.
# 

# ### Question 9: Write the code to find the number of total orders where the cost is above 20 dollars. What is the percentage of such orders in the dataset?  (1 mark)

# In[108]:


df.head(2)


# In[109]:


# Get orders that cost above 20 dollars
df_greater_than_20 = df[df['cost_of_the_order'] > 20]

# Calculate the number of total orders where the cost is above 20 dollars

print('The number of total orders that cost above 20 dollars is:', df_greater_than_20.shape[0])

# Calculate percentage of such orders in the dataset
percentage = (df_greater_than_20.shape[0] / df.shape[0]) * 100

print("Percentage of orders above 20 dollars:", round(percentage, 2), '%')


# In[372]:


len(df)


# In[110]:


df.shape


# In[111]:


# Using Python String format() Method
print('All together there are {} % orders that cost over $20'.format(round(percentage, 2)))


# #### Observations:
# 
# * There are a total of 555 orders that cost above 20 dollars.
# 
# * The percentage of such orders in the dataset is around 29.24%.
# 

# ### Question 10: Write the code to find the mean delivery time based on this dataset. (1 mark)

# In[51]:


# get the mean delivery time
print('The mean delivery time for this dataset is', round(df['delivery_time'].mean(), 2), 'minutes')


# In[112]:


# Python String format() Method
print('On average it takes {} minutes to deliver food to customers'.format(round(df['delivery_time'].mean(),2)))


# #### Observations:
# 
# * The mean delivery time is around 24.16 minutes.
# 

# ### Question 11: Suppose the company has decided to give a free coupon of 15 dollars to the customer who has spent the maximum amount on a single order. Write the code to find the ID of the customer along with the order details. (1 mark)

# In[113]:


df[df['cost_of_the_order'] == df['cost_of_the_order'].max()]


# In[54]:


#df['cost_of_the_order'] == df['cost_of_the_order'].max()


# In[116]:


df[df['cost_of_the_order'] == df['cost_of_the_order'].max()].T  


# #### Observations:
# 
# * The customer_id of the customer who has spent the maximum amount on a single order is '62359'.
# 
# * The order details are:
# 
# >  The order_id is '1477814'. 
# 
# > The customer ordered at 'Pylos' which is a Mediterranean restaurant.
# 
# > The cost of the order was around 35 dollars.
# 
# > The order was placed on a weekend.
# 
# > The food preparation time and delivery time for the order were 21 minutes and 29 minutes respectively.
# 
# > The rating given by the customer is 4.
# 

# ### Multivariate Analysis

# ### Question 12: Perform bivariate/multivariate analysis to explore relationships between the important variables in the dataset. (7 marks)

# #### Cuisine vs Cost of the order

# In[117]:


# Relationship between cost of the order and cuisine type
plt.figure(figsize=(15,7))
sns.boxplot(x = "cuisine_type", y = "cost_of_the_order", data = df, palette = 'PuBu')
plt.xticks(rotation = 60)
plt.show()


# In[118]:


# Relationship between cost of the order and cuisine type

plt.figure(figsize=(15,7))
sns.boxplot(x = "cuisine_type", y = "cost_of_the_order", 
            data = df, 
            color = '#227CFF',
            saturation=0.6,linewidth=0.9,)
plt.xticks(rotation = 60)
plt.show()


# #### Observations:
# 
# * Vietnamese and Korean cuisines cost less compared to other cuisines.
# * The boxplots for Italian, American, Chinese, Japanese cuisines are quite similar. This indicates that the quartile costs for these cuisines are quite similar.
# * Outliers are present for the cost of Korean, Mediterranean and Vietnamese cuisines.
# * French and Spanish cuisines are costlier compared to other cuisines.

# #### Cuisine vs Food Preparation time

# In[30]:


# Relationship between food preparation time and cuisine type
plt.figure(figsize=(15,7))
colrs = ['']
sns.boxplot(x = "cuisine_type", y = "food_preparation_time", data = df, palette = colrs)
plt.xticks(rotation = 60)
plt.show()


# In[119]:


# Relationship between food preparation time and cuisine type
plt.figure(figsize=(15,7))
sns.boxplot(x = "cuisine_type", y = "food_preparation_time", data = df, color = '#048F1D')
plt.xticks(rotation = 60)
plt.title('The median food preparation time lies between 24 and 30 minutes for all the cuisines',size=18)
plt.xlabel('Cuisine Type',size=14)
plt.ylabel('Food Prep Time',size=14)
plt.show()


# #### Observations:
# 
# * Food preparation time is very consistent for most of the cuisines. 
# * The median food preparation time lies between 24 and 30 minutes for all the cuisines.
# * Outliers are present for the food preparation time of Korean cuisine.
# * Korean cuisine takes less time compared to the other cuisines.
# 

# #### Day of the Week vs Delivery time

# In[120]:


# Relationship between day of the week and delivery time
plt.figure(figsize=(15,7))
sns.boxplot(x = "day_of_the_week", y = "delivery_time", data = df, palette = 'PuBu')
plt.xticks(rotation = 60)
plt.show()


# In[125]:


# Relationship between day of the week and delivery time
plt.figure(figsize=(6,5))
# Create an array with the colors you want to use
colors = ["#048F1D", "#4374B3"]
# Set your custom color palette
sns.set_palette(sns.color_palette(colors))
sns.boxplot(x = "day_of_the_week", y = "delivery_time", data = df)
#plt.xticks(rotation = 180)
plt.title('Higher Delivery Time On Week Day Compared TO Weekend')
plt.show()


# #### Observations:
# 
# * The delivery time for all the orders over the weekends is less compared to weekdays. This could be due to the dip in traffic over the weekends.
# 

# #### Revenue generated by the restaurants

# In[126]:


df.head(2)


# In[127]:


#plt.figure(figsize = (15, 7))
print('Shake Shack is the most popular restaurant that has received the highest number of orders.')

df.groupby(['restaurant_name'])['cost_of_the_order'].sum().sort_values(ascending = False)[:14]


# In[128]:


#plt.figure(figsize = (15, 7))
df.groupby(['restaurant_name'])['cost_of_the_order'].sum().sort_values(ascending = False)[:14].reset_index()


# #### Observations:
# 
# * The above 14 restaurants are generating more than 500 dollars revenue.
# 

# #### Rating vs Delivery time

# In[129]:


df.head(2)


# In[211]:


# Relationship between rating and delivery time
plt.figure(figsize=(10, 5))
sns.pointplot(x = 'rating', y = 'delivery_time', data = df)
plt.show()


# In[131]:


df.groupby(by=['rating'])['delivery_time'].mean().reset_index()


# #### Observations:
# 
# * It is possible that delivery time plays a role in the low-rating of the orders.
# 

# In[132]:


plt.figure(figsize=(10, 5))
sns.lineplot(
    data=df,
    x="rating", y="delivery_time", hue="day_of_the_week", ci=False,
    markers=True, dashes=False
)
plt.title('')
plt.show()


# #### Rating vs Food preparation time

# In[133]:


# Relationship between rating and food preparation time
plt.figure(figsize=(9, 5.5))
sns.pointplot(x = 'rating', y = 'food_preparation_time', data = df)
plt.show()


# #### Observations:
# 
# * It seems that food preparation time does not play a role in the low-rating of the orders.
# 

# #### Rating vs Cost of the order

# In[134]:


# Relationship between rating and cost of the order
plt.figure(figsize=(15, 7))
sns.pointplot(x = 'rating', y = 'cost_of_the_order', data = df)
plt.show()


# #### Observations
# 
# * It seems that high-cost orders have been rated well and low-cost orders have not been rated.

# #### Correlation among variables

# In[136]:


# plot the heatmap 
col_list = ['cost_of_the_order', 'food_preparation_time', 'delivery_time']
plt.figure(figsize=(3, 3))
sns.heatmap(df[col_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="coolwarm")
plt.show()


# #### Observations:
# 
# * There is no correlation between cost of the order, delivery time and food preparation time.
# 

# In[137]:


df.head(2)


# In[ ]:


df.sort_values()


# In[379]:


# function to create labeled barplots
#import pdb;pdb.set_trace()

def hlabeled_barplot2(data, feature, perc=False, n=None, orient = 'h'):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count  = data[feature].nunique()
    counts = data[feature].value_counts()
    howTallOrWide = 5
    
    if n is None:
            num_categories = count + 1
    else:
            num_categories = n + 1
    
    if orient == 'h':
        
        height = num_categories    
        width = howTallOrWide
    elif orient == 'v':
        
        width = num_categories
        height = howTallOrWide
    
    plt.figure(figsize=(width, height))

    plt.xticks(rotation=90, fontsize=15)

    if orient == 'h':
        ax = sns.barplot(data=data,
                         y=counts.index,
                         x=counts.values,
                         color='#E90F93',
                         order = counts.index[:n].sort_values(ascending=False)
                         )
    elif orient =='v':
        ax = sns.countplot(
            data=data,
            x=feature,
            color="#E90F93",
            order=counts.index[:n].sort_values(ascending=False)
            )
    
    for p in ax.patches:
        if orient == 'h':
            x = p.get_x() + p.get_width()  # width of the plot
            y = p.get_y() + p.get_height()/2  # height of the plot
            barThickness = p.get_width()
        elif orient == 'v':
            x = p.get_x() + p.get_width()/2 # width of the plot
            y = p.get_y() + p.get_height()  # height of the plot
            barThickness = p.get_height()
        
        if perc == True:
            label = "{:.1f}%".format(
                100 * barThickness / total
            )  # percentage of each class of the category
        else:
            label = barThickness  # count of each level of the category
   
        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=15,
            xytext=(25, 1), # change the position of the percent value on top of barpot
            textcoords="offset points",
        )  # annotate the percentage
    plt.show()  # show the plot

    

#labeled_barplot2(df, 'restaurant_name', perc=True, orient='v', n=5) orient = v is normal


# In[380]:


hlabeled_barplot2(df,'rating',perc=True)


# ### Question 13: Suppose the company wants to provide a promotional offer in the advertisement of the restaurants. The condition to get the offer is that the restaurants must have a rating count of more than 50 and the average rating should be greater than 4. Write the code to find the restaurants fulfilling the criteria to get the promotional offer. (2 marks)

# In[139]:


df.info()


# In[144]:


df_rated['restaurant_name'].value_counts()


# In[145]:


df['rating'].value_counts()


# In[146]:


df['rating'].sample(10)


# In[147]:


# filter the rated restaurants
df_rated = df[df['rating'] != 'Not given'].copy()
# convert rating column from object to integer
df_rated['rating'] = df_rated['rating'].astype('int')
# create a dataframe that contains the restaurant names with their rating counts
df_rating_count = df_rated.groupby(['restaurant_name'])['rating'].count().sort_values(ascending = False).reset_index()
df_rating_count.head()


# In[148]:


rest_names = df_rating_count[df_rating_count['rating']>50]['restaurant_name']
rest_names


# In[149]:


# df_rated  excludes  rest that are  not rated 
df_avg_rat_g4 = df_rated[df_rated['restaurant_name'].isin(rest_names)].copy()
df_avg_rat_g4 # this gives everything. Lets group  and filter 

# Lot of time ratings are not given so we see many NaN
df_avg_rat_g4.groupby(by='restaurant_name')['rating'].mean().sort_values(ascending = False).reset_index().head(5)


# In[ ]:


# So, use cat.remove_unused_categories() to remove NaN and get only the numeric values 
#https://stackoverflow.com/questions/48064965/drop-unused-categories-using-groupby-on-categorical-variable-in-pandas


# In[69]:


# get the restaurant names that have rating count more than 50
rest_names = df_rating_count[df_rating_count['rating'] > 50]['restaurant_name']
# filter to get the data of restaurants that have rating count more than 50
df_mean_4 = df_rated[df_rated['restaurant_name'].isin(rest_names)].copy()
# find the mean rating of the restaurants
df_mean_4.groupby(df_mean_4['restaurant_name'].cat.remove_unused_categories())['rating'].mean().sort_values(ascending = False).reset_index()


# #### Observations:
# 
# * The restaurants fulfilling the criteria to get the promotional offer are: **'The Meatball Shop', 'Blue Ribbon Fried Chicken',  'Shake Shack' and 'Blue Ribbon Sushi'**.
# 

# ### Question 14: Suppose the company charges the restaurant 25% on the orders having cost greater than 20 dollars and 15% on the orders having cost greater than 5 dollars. Write the code to find the net revenue generated on all the orders given in the dataset. (2 marks)

# In[150]:


df.head(2)


# In[151]:


#if order > $20 charge 25% 
#if order > $5 charge 15% 
#else charge 0%
#get  revenue on all orders i.e sum()
#solution: create new col tht has charges 
# we can solve this two ways 


# In[155]:


# Method 1 using list compression
# syntax:  [expression for item in iterable if some_condtion]
# how to read syntax ?
# --> give me all expression such that expression is an elements of iterable and expression has some condtion

df['foodhu_charge'] = [order_cost * 0.25 if order_cost > 20 else 
                       order_cost * 0.15 if order_cost > 5 else
                       0 
                       for order_cost in df['cost_of_the_order']]


# In[156]:


# Methos 2 using lambda function 
df['extr_charge'] = df['cost_of_the_order'].apply(lambda x: x*0.25 if x >20 else x*0.15 if x > 5 else 0)


# In[157]:


df.head(3)


# In[158]:


df.head(2)


# In[159]:


# now get  sum  charges  
df['foodhu_charge'].sum()
#  print it  using  .formate
print()


# In[344]:


df['extr_charge'].sum()


# In[382]:


df.head()


# In[39]:


# add a new column to the dataframe df that stores the company charges
df['foodhub_charge'] = [order_cost * 0.25 if order_cost > 20 else
                        order_cost * 0.15 if order_cost > 5 else
                        0
                        for order_cost in df['cost_of_the_order']]
# get the total revenue and print it
print('The net revenue is around', round(df['foodhub_charge'].sum(), 2), 'dollars')


# #### Observations:
# 
# * The net revenue generated on all the orders given in the dataset is around 6166.3 dollars.
# 

# ### Question 15: Suppose the company wants to analyze the total time required to deliver the food. Write the code to find out the percentage of orders that have more than 60 minutes of total delivery time. (2 marks)
# 
# Note: The total delivery time is the summation of the food preparation time and delivery time. 

# In[160]:


df.head(2)


# In[161]:


# add a new column to the dataframe df to store the total delivery time
df['total_time'] = df['food_preparation_time'] + df['delivery_time']


# In[162]:


df.head(2)


# In[163]:


#  find out the percentage of orders that have more than 60 minutes of total time
df[df['total_time'] > 60].shape[0]/df.shape[0]*100


# In[164]:


df.shape


# In[40]:


# add a new column to the dataframe df to store the total delivery time
df['total_time'] = df['food_preparation_time'] + df['delivery_time']

# find the percentage of orders that have more than 60 minutes of total delivery time
print ('The percentage of orders that have more than 60 minutes of total delivery time is',
       round(df[df['total_time'] > 60].shape[0] / df.shape[0] * 100, 2),'%')


# #### Observations:
# 
# * Approximately 10.54 % of the total orders have more than 60 minutes of total delivery time.
# 

# ### Question 16: Suppose the company wants to analyze the delivery time of the orders on weekdays and weekends. Write the code to find the mean delivery time on weekdays and weekends. Write your observations on the results. (2 marks)

# In[165]:


df.head(2)


# In[166]:


df['day_of_the_week'].value_counts()


# In[41]:


# get the mean delivery time on weekdays and print it
print('The mean delivery time on weekdays is around', 
      round(df[df['day_of_the_week'] == 'Weekday']['delivery_time'].meana()),
     'minutes')

# get the mean delivery time on weekends and print it
print('The mean delivery time on weekends is around', 
      round(df[df['day_of_the_week'] == 'Weekend']['delivery_time'].mean()),
     'minutes')


# #### Observations:
# 
# * The mean delivery time on weekdays is around 28 minutes whereas the mean delivery time on weekends is around 22 minutes.
# 
# * This could be due to the dip of traffic volume in the weekends.

# ### Conclusion and Recommendations
# 

# ### **Question 17:** Write the conclusions and business recommendations derived from the analysis. (3 marks)

# ### Conclusions:
# 
# 
# * Around 80% of the orders are for ***American***
# , Japanese, Italian and Chinese cuisines. Thus, it seems that these cuisines are quite popular among customers of FoodHub. 
# Shake Shack 1000 is the $most popular restaurant that has received the highest number of orders.
# * Order volumes increase on the weekends compared to the weekdays.
# * Delivery time over the weekends is less compared to the weekdays. This could be due to the dip in traffic volume over the weekends.
# * Around 39% of the orders have not been rated.

# ### Business Recommendations:
# 
# * FoodHub should integrate with restaurants serving American, Japanese, Italian and Chinese cuisines as these cuisines are very popular among FoodHub customers. 
# 
# * FoodHub should provide promotional offers to top-rated popular restaurants like Shake Shack that serve most of the orders. 
# 
# * As the order volume is high during the weekends, more delivery persons should be employed during the weekends to ensure timely delivery of the order. Weekend promotional offers should be given to the customers to increase the food orders during weekends.
# 
# * Customer Rating is a very important factor to gauge customer satisfaction. The company should investigate the reason behind the low count of ratings. They can redesign the rating page in the app and make it more interactive to lure the customers to rate the order. 
# 
# * Around 11% of the total orders have more than 60 minutes of total delivery time. FoodHub should try to minimize such instances in order to avoid customer dissatisfaction. They can provide some reward to the punctual delivery persons.
