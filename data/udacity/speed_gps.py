import numpy as np
import pandas as pd
import csv
import random
from random import randint
import os, sys, importlib
from geopy.distance import great_circle

def cal_distance(df):
    for i in range(1, len(df)):
        df.at[i,"distance"] = great_circle((df.at[i,'lat'],df.at[i,'long']), (df.at[i-1,'lat'],df.at[i-1,'long'])).km
    df['distance_cum'] = df["distance"].cumsum()
    return df

def find_speed(df, time_interval = 1):    
    df['offset'] = 0.05 #secs
    df['dspan'] = df['tspan'] = df['speed_gps'] = df['diff'] = None
    
    for N in range(20,len(df)):
        # if N == 0: continue
        # backi = N
        # totOffset = df.at[N,'offset']
        # while totOffset <= time_interval*60 :
        #     backi -= 1
        #     totOffset += df.at[backi,'offset']
        #     if backi == 0: break
        backi = N - 20 # average speed of 1 secs interval
        if backi < 0: continue
        tspan = df.at[N,'timestamp'] - df.at[backi,'timestamp'] #nanoseconds
        df.at[N,'tspan'] = tspan
        # distance = df.at[N,'distance_cum'] - df.at[backi,'distance_cum']
        distance = great_circle((df.at[backi,'lat'],df.at[backi,'long']), (df.at[N,'lat'],df.at[N,'long'])).meters
        df.at[N,'dspan'] = round(distance,3)
        if distance == 0:
            speed = df.at[N, 'speed_gps'] = 0
        else:
            speed = round((distance)/1000 / (tspan/10**9/3600)/3.6,4) # 3.6 is conversion from km/h to m/s
            df.at[N, 'speed_gps'] = speed
        speed_diff = round(df.at[N,'speed'] - speed, 2)
        df.at[N,'diff'] = speed_diff
    return df

if __name__ == "__main__":
    csv_input = pd.read_csv("Y_train.csv")    
    print("caculating distance")
    csv_input = cal_distance(csv_input)
    print("caculating speed")
    csv_input = find_speed(csv_input)
    csv_input.to_csv("Y_train_GPS.csv", index=False)
