# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:41:12 2022

@author: m_irf
"""
import os
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import pickle
from sklearn.neighbors import RadiusNeighborsClassifier
import gpxpy.geo as geo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
path_dir='android data'
arr = os.listdir(path_dir)
android_data=pd.read_csv(path_dir + "/" +arr[0],nrows=1)
for f in arr:
    android_data=android_data.append(pd.read_csv(path_dir+"/"+f))
android_data=android_data[(android_data['mnc']==3) & (android_data['pre_connection_type']==15)  & (android_data['post_connection_type']==15)] ## filter mobily LTE
android_data=android_data[android_data['client_city']=="Riyadh"] ## filter riyadh city
android_data=android_data[android_data['location_type']==1] # filter locations shared by user
allowed_ci=[i for i in range(10,121,10)]
X_train=android_data[~android_data['gsm_cell_id'].isnull()]
X_train['site']=X_train['gsm_cell_id'].apply(lambda x:str(int(x/256)))
X_train['cell']=X_train['gsm_cell_id'].apply(lambda x:x%256)
X_train=X_train[X_train['cell'].apply(lambda x:x in allowed_ci)]
X_train=X_train[['test_id','client_latitude','client_longitude','site','cell']]
dict_ci={}
for i in range(10,121,10):
    if(i in [10,40,70,100]):
        dict_ci[i]="A"
    if(i in [20,50,80,110]):
        dict_ci[i]="B"
    if(i in [30,60,90,120]):
        dict_ci[i]="C"
X_train['y']=X_train['site'].apply(lambda x:str(x))+X_train['cell'].apply(lambda x:dict_ci[x])
X_train['y']=X_train['y'].apply(lambda x:x[2:])
X_train['y']=X_train['site'].apply(lambda x:str(x))+X_train['cell'].apply(lambda x:dict_ci[x])
X_train['y']=X_train['y'].apply(lambda x:x[2:].strip("0"))
def custom_distance(x,y):
    return geo.haversine_distance(x[1],x[2],y[1],y[2])
x_tr, x_te = train_test_split(X_train,test_size = .2, random_state = 17)
y_train=x_tr['y']
y_test=x_te['y']
x_tr=x_tr[['test_id','client_latitude', 'client_longitude']]
x_te=x_te[['test_id','client_latitude', 'client_longitude']]
test_count=y_test.shape[0]
clf_final=RadiusNeighborsClassifier(radius=100,metric=custom_distance,outlier_label=0)
clf_final.fit(x_tr,y_train)
pickle.dump(clf_final, open("clf_final", "wb")) 