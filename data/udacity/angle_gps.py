import numpy as np
import pandas as pd
import csv
import random
from random import randint
import os, sys, importlib
import pyproj
import math
from sklearn.metrics import mean_squared_error 

def find_angle(df,i=140, x=6):    
    df['angle_gps'] = None
    geodesic = pyproj.Geod(ellps='WGS84')
    
    # for N in range(21,40):
    for N in range(int(i/2),len(df)):
        # start = (df.at[N-i,'long'],df.at[N-i,'lat']) # gps cordinates 1 sec ago
        # middle = (df.at[N-(i/2),'long'],df.at[N-(i/2),'lat']) # gps cordinates 0.5 sec ago
        # end = (df.at[N,'long'],df.at[N,'lat']) # gps cordinates now
        if (N + (i/2)) >= len(df): 
            break
        start = (df.at[N-(i/2),'long'],df.at[N-(i/2),'lat'])
        middle = (df.at[N,'long'],df.at[N,'lat'])
        end = (df.at[N+(i/2),'long'],df.at[N+(i/2),'lat'])
        
        angle1 = geodesic.inv(start[0],start[1],middle[0],middle[1])[0]
        angle2 = geodesic.inv(middle[0],middle[1],end[0],end[1])[0]
        bearing = math.radians((angle1 - angle2))
        mean = df['angle_gps'].iloc[N-x:N-1].mean(axis=0)
        # print(mean)
        # remove unexpectedly large value
        if abs(bearing) > 2.5:   
            df.at[N,'angle_gps'] = mean
        # remove anomalies    
        elif (N > i+x) and abs(bearing) > abs(mean) + 0.2:
            df.at[N,'angle_gps'] = mean
        else:
            df.at[N,'angle_gps'] = bearing   
    return df

if __name__ == "__main__":
    df = pd.read_csv("Y_train_attack_abrupt.csv")    

    print("caculating angle")
    # df = find_angle(df)
    # df.to_csv("Y_all_GPS.csv", index=False)
    for i in [140]:
    # for i in range(0,10,1):
    # for i in np.arange(0.0, 1.0, 0.1):
        df_i = find_angle(df,i)
        df_i = df_i[df_i['angle_gps'].notna()]
        realVals = df_i.angle
        predictedVals = df_i.angle_gps
        mse = mean_squared_error(realVals, predictedVals)
        print("value: ", i, "\t mse: ", mse)
        # df_i.to_csv("test_GPS.csv", index=False)