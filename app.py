import  streamlit as st
import pickle
import numpy as np

#import model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

#brand laptop
company = st.selectbox('Brand', df['Company'].unique())

#tipe laptop
type = st.selectbox('Type', df['TypeName'].unique())

#ram
ram = st.selectbox('RAM', [2,4,6,8,12,16,24,32,64])

#weight
weight = st.number_input('weight')

#Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

#IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

#screensize
screen_size = st.number_input('Screen Size')

#resolution
resolution = st.selectbox('Screen Resolution', ['1366x768', '1600x900', '1920x1080','1920x1200','2160x1440','2256x1504','2304x1440','2400x1600','2560x1440','2560x1600','2736x1824','2880x1800','3200x1800','3840x2060','3840x2160'])

#cpu
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

#hdd
hdd = st.selectbox('HDD', [0,128,256,512,1024,2048])

#ssd
ssd = st.selectbox('SSD', [0,128,256,512,1024,2048])

#gpu
gpu = st.selectbox('GPU', df['Gpu Brand'].unique())

#os
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    #query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    #split nilai ppi dan diubah ke nilai integer

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    #prediksi harga laptop
    query = query.reshape(1,12)
    st.title("Predicted price in Dollar : $ " + str(int(np.exp(pipe.predict(query)[0]))))