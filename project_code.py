#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 08:52:02 2019

@author: mohamed
"""
"""
1- check the dataste by using (info,describe, describe object, isnull,value_counts() to check the count of very unique vlaue)
2- found difference types of NAN vlues such as (NNNN, UUUU, NN,XX ,U,...)
3- difine variable called missing_values included all kinds of NAN vlaues, then when I read dataset file add na_values=  missing_values
4-I drop NAN values in the all dataset
5- Start analysis
"""

import os
os.getcwd()
os.chdir('/Volumes/Work/Data Science/Python/Project/Proj_one/')
import numpy as np
import pandas as pd 
from pandas import Series, DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt


plt.interactive(False)

#1- Read data file:
# C_MNTH  C_WDAY  C_HOUR  C_VEHS  C_CONF  C_RCFG  C_WTHR  C_RSUR  \ C_RALN  C_TRAF    V_ID  V_TYPE  V_YEAR    P_ID   P_SEX   P_AGE  P_PSN  P_ISEV  P_SAFE  P_USER  

# Making a list of missing value types and solve the repeated values in the same column

missing_values = ["NNNN", "UUUU", "XXXX","UU", "XX","QQ","NN", "U", "X", "Q", "N"]
Collision_2000= pd.read_csv("Data_use_1.csv", na_values = missing_values)

Collision_2000.info()
Collision_2000.head(10)
Collision_2000.shape 
Collision_2000.ndim
Collision_2000.size  # numbers of elements 

Collision_2000.dropna()

# Check Values

Collision_2000['C_WDAY'].value_counts()
Collision_2000['C_HOUR'].value_counts()
Collision_2000['C_MNTH'].value_counts()
Collision_2000['C_MNTH'].hist(bins=23)


Collision_2000.describe(include='all')

Collision_2000.boxplot(column='C_MNTH')
plt.show()

Collision_2000['P_SEX'].value_counts()

Collision_2000['P_SEX'].hist(bins=5)
plt.show()


#-------------------------------------------------------------------

#Collisions by weekday
by_weekday = Collision_2000.groupby('C_WDAY')['C_SEV'].count() #--->  Relation between WeekDay and Collision severity C_SEV
by_weekday.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
plot2 = by_weekday.plot(kind='bar',title='Collisions by week day', color='turquoise')
plt.show()
#--------------------------------------------

#Collisions by hour
plt.figure(figsize=(20,15))
by_hour = Collision_2000.groupby('C_HOUR')['C_SEV'].count() #--->  Relation between Hour and Collision severity C_SEV

plt.title('Collisions by week hour',fontsize=18)
plt.xlabel('Numbers of Collision', fontsize=16)
plt.ylabel('C_HOUR', fontsize=16)

#plt.legend(by_hour,loc=3)
plot3 = by_hour.plot(kind='barh',width=0.8,color='y')

plt.show()

#--------------------------------------------
#Collisions by Month
by_Month = Collision_2000.groupby('C_MNTH')['C_SEV'].count() #--->  Relation between WeekDay and Collision severity C_SEV
by_Month.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul','Aug','Sep','Oct','Nov','Dec']

plt.title('Collisions by Month',fontsize=18)
plot4 = by_Month.plot(kind='bar', color='indianred')
plt.show()
#---------------------------------------    
#Gender & Collision Hour

# 1- Creating  new dataset Generating gender_by_Hour by keeping the using elements

gender_by_Hour =  Collision_2000.loc[:,['C_HOUR','P_SEX']]
print(gender_by_Hour)

# 2- define men and women as 0 and 1 and replace M=0 , F=1

gender_by_Hour.P_SEX.replace({'M':0,'F':1},inplace=True)

# drop any missing values and Clean the missing variable by droping the row which contains any missing variables

gender_by_Hour.dropna(inplace=True)

num = gender_by_Hour.groupby(['P_SEX']).count()
print(num)

gender_by_Hour1= gender_by_Hour.loc[:,['P_SEX','C_HOUR']].groupby(['C_HOUR','P_SEX']).size().unstack()
gender_by_Hour1

gender_by_Hour1.plot(rot=0,color=['g','r'],figsize=(15,5),
         title='Gender and Collision hour')
plt.legend(['0.Male','1.Female'])
plt.xticks(range(0,24),range(0,24))
plt.grid();
plt.show()

#-----------------------------------

# Medical treatment required only 3 values
gender_Medical_Required= Collision_2000.loc[:,['P_SEX','P_ISEV']].groupby(['P_SEX','P_ISEV']).size().unstack()
gender_Medical_Required

gender_Medical_Required.plot(kind='bar',rot=0, color=['wheat','lightcoral','lightslategray'], figsize=(8,3),
         title='Gender and Medical treatment required')
plt.legend(['1.No Injury','2.Injury','3.Fatality'])
plt.xticks(range(0,2),['0.Male','1.Female'])
plt.grid(axis='y');

plt.show()
#---------------------
#Total number of vehicles involved in collision in regards to weekday

car_involved= Collision_2000.loc[:,['C_WDAY','P_ISEV']]
num_Car_involve = car_involved.loc[:,['C_WDAY','C_VEHS']]

weekday = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
w_day = num_Car_involve.groupby('C_WDAY')['C_VEHS'].size()

#Visualize the result
w_day.plot(kind='barh', color='c',figsize=(10,6),rot =0,
           title='Total Number of vehicles involved in collision by Weekday')
plt.yticks(range(0,8),weekday)
plt.ylabel('Weekday');
plt.show()

#----------------------------------

#Total number of vehicles involved in collision in regards to day Hours

car_involved= Collision_2000.loc[:,['C_HOUR','P_ISEV']]
num_Car_involve = car_involved.loc[:,['C_HOUR','C_VEHS']]

Day_hours = num_Car_involve.groupby('C_HOUR')['C_VEHS'].size()

#Visualize the result


plt.ylabel('Number of vehicles involved')
plot7 = Day_hours.plot(kind='bar',width=0.8,color='y',rot =0,figsize=(12,15),  
                       title='Total Number of vehicles involved in collision by Day Hours')
plt.show()

#------------------------------------
#Collisions by Vehicale types
by_V_Type = Collision_2000.groupby('V_TYPE')['C_SEV'].count() #--->  Relation between Vehicale types and Collision severity C_SEV
by_V_Type.index = ['light Duty', 'cargo van', 'Other trucks and vans', 'Unit trucks ', 'Road tractor', 
                   'School bus', 'Smaller school bus','Urban and Intercity Bus','Motorcycle and moped ',
                   'Off road vehicles','Bicycle','Purpose-built motorhome ','Farm equipment','Construction equipment',
                   'Fire engine','Snowmobile','Street car']
plot5 = by_V_Type.plot(kind='bar',title='Collisions by Vehicale types', color='darkgray')
plt.show()
