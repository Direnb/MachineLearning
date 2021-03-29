#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 10:14:00 2021

@author: direny
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('odev_tenis.csv')

pd.read_csv("veriler.csv")
#test
#print(veriler)
outlook = veriler.iloc[:,0:1].values
temp = veriler.iloc[:,1:2].values
hum = veriler.iloc[:,2:3].values


#encoder: Kategorik -> Numeric

from sklearn import preprocessing

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()

le = preprocessing.LabelEncoder()

windy = veriler.iloc[:,-2]
play = veriler.iloc[:,-1]

windy = le.fit_transform(windy)
play = le.fit_transform(play)

#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=outlook, index = range(14), columns = ['sunny','overcast','rainy'])
temp = pd.DataFrame(data =temp,index = range(14), columns = ['temp']  )
hum =pd.DataFrame(data =hum,index = range(14), columns = ['hum'] )

sonuc2 = pd.DataFrame(data = windy , index = range(14), columns = ['windy']) 
sonuc3 =pd.DataFrame(data=play , index = range(14), columns= ['play']) 



#print(sonuc2)
#sonuc2 = veriler.iloc[:,1:]
#concat - dataframe birlestirme islemi

s=pd.concat([sonuc,sonuc2, sonuc3], axis=1)
s2 = pd.concat([s, temp, hum], axis = 1)

#print(s2)


#Bağımsız değişken belirleme - kolon tahin ettirme


from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s2.iloc[:,:-1],hum,test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print(y_pred)










