# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 18:58:46 2021

@author: Pranav Maruwada
"""
#Importing requisite packages
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
#import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#Preparing header of app
st.image("used-car-valuation.png")
st.title("Buy, Sell or Get Great Prices for used cars!")
st.write("Our website's basic and premium assessments recommend a fair market price for used cars. When purchasing a used car, it is critical to understand the car's current fair market value. Through our website, you may get cost of any pre-owned car produced by any notable brand.")

#Loading requisite data
@st.cache(allow_output_mutation=True, show_spinner=False)
def load_data():
     dataset = pd.read_excel("vehicles.xlsx")
     return dataset

#Function to clean data and remove outliers
@st.cache(allow_output_mutation=True, show_spinner=False)
def cleanData():
#Data cleaning - Missing Values & Outliers
    dataset=load_data()
    dataset = dataset.drop(['url', 'region_url', 'VIN', 'image_url', 'description', 'posting_date', 'paint_color', 'title_status', 'size', 'type', 'region', 'county', 'state'], axis = 1)
    dataset = dataset.dropna()
    col_names = dataset.keys()[[1, 8]]
    for i in col_names:
        mu = dataset[i].mean()
        sd = dataset[i].std()
        ind = []
        for j in dataset[i]:
            z = (j - mu)/sd
            if(z > 3 or z < -3):
                index = int(dataset.index[dataset[i] == j][0])
                ind.append(index)
        dataset = dataset.drop(ind)
    df = dataset.copy()
    def outlier_remover(df,st):
        count=0
        list1=[]
        for i in df[st]:
            if(abs(((i-(df[st].mean()))/(df[st].std())))>3):
                list1.append(df.index[count])   
            count+=1
        df1=df.drop(list1)
        df2=df1.reset_index()
        del(df1)
        return (df2)
    dfr1 = outlier_remover(df, 'price')
    dfr2 = outlier_remover(dfr1, 'odometer')
    del(dfr1)
    dfr3 = dfr2.loc[(dfr2['price'] >= 10000)]
    del(dfr2)
    dfr3 = dfr3.iloc[:, 2:]
    return dfr3

#Function to convert all list objects to string
def listToString(oldList):
    newList = []
    for i in oldList:
        newList.append(str(i))
    return(newList)
    
#Function to convert list values to proper case
def properCase(oldList):
    newList = []
    for i in oldList:
        if isinstance(i, str):
            newList.append(i.title())
        else:
            newList.append(i)
    return(newList)

#Function to select unique objects from list
def uniqueValue(oldList):
    tempList = np.array(listToString(oldList))
    newList = np.unique(tempList)
    return(newList)     

#Function for price prediction option
def predictPrice():
    st.title("Check Price of Used Cars!")
    dfr3 = cleanData()
     
    
    #Creating user selectible lists
    manuf = uniqueValue(dfr3['manufacturer'])
    manuf = properCase(manuf)
    manufval = st.selectbox('Select a Manufacturer', manuf)
    
    modelfilterdata = dfr3.loc[dfr3['manufacturer'] == manufval.lower()]
    modelfilter = modelfilterdata['model']
    model = uniqueValue(modelfilter)
    model = properCase(model)
    modval = st.selectbox('Select a Model', model)
    
    if (modval.isdigit()):
        modval = int(modval)
    else:
        modval = modval.lower()
    yearfilterdata = modelfilterdata.loc[modelfilterdata['model'] == modval]
    yearfilter = yearfilterdata['year']
    year = uniqueValue(yearfilter)
    yearval = st.selectbox('Select year of manufacturing', year)
    
    condition = uniqueValue(dfr3['condition'])
    condition = properCase(condition)
    condval = st.selectbox('Select condition of vehicle', condition)
    
    cylfilterdata = yearfilterdata.loc[yearfilterdata['year'] == yearval.astype(np.int)]
    cylfilter = cylfilterdata['cylinders']
    cylinders = uniqueValue(cylfilter)
    cylinders = properCase(cylinders)
    cylval = st.selectbox('Select no. of cylinders', cylinders)
    
    drivfilterdata = cylfilterdata.loc[cylfilterdata['cylinders'] == cylval.lower()]
    drivfilter = drivfilterdata['drive']
    drive = uniqueValue(drivfilter)
    drive = properCase(drive)
    drival = st.selectbox('Select drive type', drive)
    
    fuelfilterdata = drivfilterdata.loc[drivfilterdata['drive'] == drival.lower()]
    fuelfilter = fuelfilterdata['fuel']
    fuel = uniqueValue(fuelfilter)
    fuel = properCase(fuel)
    fuelval = st.selectbox('Select fuel type', fuel)
    
    transfilter = fuelfilterdata.loc[fuelfilterdata['fuel'] == fuelval.lower()]['transmission']
    transmission = uniqueValue(transfilter)
    transmission = properCase(transmission)
    transval = st.selectbox('Select transmission type', transmission)
    
    st.write('Enter odometer reading (total distance driven)')
    odoval = st.text_input('Your odometer reading here:')
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    #Modeling
    @st.cache(allow_output_mutation=True, show_spinner=False)
    def modelPrice():
        reslist = []
        dumcols = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'transmission', 'drive']
        dumdata = pd.get_dummies(dfr3, columns = dumcols, drop_first=True)
        regdata = dumdata.drop(['id', 'lat', 'long'], axis = 1)
        x = regdata.iloc[:, 1:]
        keys = x.keys()
        y = regdata['price']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
        model = Ridge()
        reg = model.fit(x_train, y_train)
        y_pred = reg.predict(x_test)
        vals = reg.coef_
        inter = reg.intercept_
        coefdict = dict(zip(keys,vals))
        predval = inter
        if (bool(manufval)):
            key = 'manufacturer_'+str(str(manufval).lower())
            if key in coefdict.keys():
                val = coefdict.get(key)
                predval += val
        if (bool(modval)):
            key = 'model_'+str(str(modval).lower())
            if key in coefdict.keys():
                val = coefdict.get(key)
                predval += val  
        if (bool(yearval)):
            key = 'year'
            if key in coefdict.keys():
                val = coefdict.get(key)
                predval += (val * int(yearval))
        if (bool(condval)):
            key = 'condition_'+str(str(condval).lower())
            if key in coefdict.keys():
                val = coefdict.get(key)
                predval += val 
        if (bool(cylval)):
            key = 'cylinders_'+str(str(cylval).lower())
            if key in coefdict.keys():
                val = coefdict.get(key)
                predval += val 
        if (bool(drival)):
            key = 'drive_'+str(str(drival).lower())
            if key in coefdict.keys():
                val = coefdict.get(key)
                predval += val
        if (bool(fuelval)):
            key = 'fuel_'+str(str(fuelval).lower())
            if key in coefdict.keys():
                val = coefdict.get(key)
                predval += val
        if (bool(transval)):
            key = 'transmission_'+str(str(transval).lower())
            if key in coefdict.keys():
                val = coefdict.get(key)
                predval += val
        if (bool(odoval)):
            key = 'odometer'
            if key in coefdict.keys():
                val = coefdict.get(key)
                predval += (val * int(odoval))
        reslist.append(predval)
        reslist.append(y_test)
        reslist.append(y_pred)
        return reslist
    
    if st.button('Predict Price Now!'):
        result = modelPrice()
        price = result[0]
        y_test = result[1]
        y_pred = result[2]
        if(price < 0):
            st.write('Could not predict price! Try again by selecting all / more of the above fields or by giving proper values.')
        else:
            st.write('The price you will get for your car is: USD %s' %round(price, 2))
            st.write('Our prediction accuracy as shown by metrics like R2 score and a plot: ')
            r2 = r2_score(y_test, y_pred) * 100
            st.write('R2 score = ' + str(round(r2, 2)) + '%')
            sns.regplot(x = y_pred, y = y_test)
            st.pyplot()
            
    #Inferences
    st.write('To know more about current used car market trends and statistics, abd about how our prediction works, click the button below!')
    if st.button('For Inquisitive Users!'):
        
        st.write('It can be seen in Plot 1 below that the majority of the used autos are around $20,000. Furthermore, we can observe that there are still a significant number of cars that cost more than $20,000.')
        sns.distplot(dfr3['price'])
        st.pyplot()
    
        st.write('As seen in Plot 2 below, the most popular cars in the used car market are those that are excellent, like new, and in good shape. Salvage automobiles are rapidly catching up to these three categories in terms of popularity. As a result, estimating a car\'s price just based on the kind or condition of the vehicle is difficult.')
        dfr3['condition'].value_counts().plot(kind = 'bar')
        st.pyplot()
    
        st.write('People pay close attention to the odometer value on a used car when purchasing it. As seen in Plot 3 below, the odometer has a substantial impact on the price of an automobile.')
        sns.scatterplot(data = dfr3, x = 'odometer', y = 'price')
        st.pyplot()
        
        st.write('From Plot 4 below, we see that the number of outliers for excellent condition class are quite high, indicating that people demand exhorbitant prices for excellent condition cars')
        sns.boxplot(x = 'condition', y = 'price', data = dfr3)
        st.pyplot()
    
        st.write('This does not imply that only cars with low mileage are sold. As seen in Plot 5 below, high odometer autos have buyers depending on price.')
        sns.distplot(dfr3['odometer'])
        st.pyplot()
    
        st.write('Another key determinant on the used automobile market is the car\'s manufacturer. According to Plot 6 below, Ford and Chevrolet are the two most powerful manufacturers in North America. As big automakers, Toyota and Nissan follow the order. It may be stated that American cars account for a significant portion of the used car market.')
        dfr3['manufacturer'].value_counts().plot(kind = 'bar')
        st.pyplot()
   
        st.write('It\'s critical to understand what factors influence a car\'s condition while appraising it. Plot 7 shows that four-wheel-drive vehicles are more durable and reliable. In terms of numbers, it is clear that 4wd vehicles are the most popular. In addition, 4wd has the largest percentage of automobiles in "excellent," "like new," and "good" condition.')
        pvt = pd.pivot_table(dfr3, index = ['drive'], columns = ['condition'], aggfunc = {'condition':'count'})
        pvt.plot(kind = 'bar')
        st.pyplot()
    
        st.write('The average odometers for all drivetrain types are so close to one other for all quantiles, as seen in Plot 8. This also implies that in the secondhand car market, drive type may have no bearing on the odometer.')
        sns.boxplot(x = 'drive', y = 'odometer', data = dfr3)
        st.pyplot()
        
#Function for buying old cars
#@st.cache(allow_output_mutation=True, show_spinner=False)
def buySellUsed():
    st.title("Buy Used Cars online at great prices!")
    data = cleanData()
    manuf = uniqueValue(data['manufacturer'])
    manuf = properCase(manuf)
    manufval = st.selectbox('Select a Manufacturer', manuf)
    modelfilterdata = data.loc[data['manufacturer'] == manufval.lower()]
    modelfilter = modelfilterdata['model']
    model = uniqueValue(modelfilter)
    model = properCase(model)
    modval = st.selectbox('Select a Model', model)
    if (modval.isdigit()):
        modval = int(modval)
    else:
        modval = modval.lower()
    set_price = st.slider("Price Range", min_value=int(data['price'].min()), max_value=int(data['price'].max()), step=50, value=int(data['price'].min()))
    datafilter = data.loc[(data['price']<int(set_price)) & (data['manufacturer'] == manufval.lower()) & (data['model'] == modval)]
    if datafilter.empty:
        st.write('Sorry! Car with this spec is not available with anyone')
    else:
        fig = px.scatter_mapbox(datafilter, lat = 'lat', lon = 'long', size = 'price')
        fig.update_layout(mapbox_style = 'open-street-map')
        st.plotly_chart(fig)

#Main Page
my_button = st.sidebar.radio("What do you want to do today?", ('Predict Price of used cars', 'Buy & Sell Portal'))

if my_button == 'Predict Price of used cars':
    predictPrice()
elif my_button == 'Buy & Sell Portal':
    buySellUsed()
