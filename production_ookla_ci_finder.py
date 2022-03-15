# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:18:55 2022

@author: m_irf
"""
import streamlit as st
import pickle
import pandas as pd
import gpxpy.geo as geo

st.title("Welcome to OOKLA ios CellId Prediction ")
def custom_distance(x,y):
    return geo.haversine_distance(x[1],x[2],y[1],y[2])
with open('clf_final','rb') as f:
    clf_final=pickle.load(f)
input_file=st.file_uploader("Please Select ios file",type=".csv")   
if(input_file is not None):
    ios_data=pd.read_csv(input_file)
    ios_data=ios_data[(ios_data['mnc']==3) & (ios_data['pre_connection_type']==12)  & (ios_data['post_connection_type']==12)] ## filter mobily LTE
    ios_data=ios_data[ios_data['client_city']=="Riyadh"] ## filter riyadh city
    ios_data=ios_data[ios_data['location_type']==1] # filter locations shared by user
    x_query=ios_data[['test_id','client_latitude','client_longitude']]
    x_query['cell_id']=clf_final.predict(x_query)
    found=x_query[x_query['cell_id']!=0].shape[0]
    result=pd.merge(ios_data,x_query,how='left',left_on='test_id',right_on='test_id' )
    styler = x_query[x_query['cell_id']!=0].style.hide_index()
    st.success(f"Cell Ids found for {found} samples")
    st.download_button("Press to Download",result.to_csv().encode('utf-8'),mime="csv",file_name="ios_with_ci.csv")
    st.write(styler.to_html(), unsafe_allow_html=True)

