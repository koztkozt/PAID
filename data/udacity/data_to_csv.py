# importiong the modules
import pandas as pd
import numpy as np
import math
import speed_gps
import angle_gps

df = pd.read_csv("./interpolated.csv", index_col=False, engine="python")
# select images from center camera only
df = df[df["frame_id"] == "center_camera"]
# drop HMB3 cos data are interpolated
df = df.drop(df[(df.timestamp <= 1479425721885864960 ) & (df.timestamp >= 1479425441183206144)].index) 

# edit filename
df["filename"] = df["filename"].str[7:]

df["angle"] = pd.to_numeric(df["angle"])
df["angle_convert_org"] = df["angle"]
# road is curve if angle is more than 0.3
df["curve"] = np.where(abs(df['angle']) > 0.3, True, False)

df.index = [x for x in range(0, len(df.values))]

# calculate speed using GPS values
df = speed_gps.cal_distance(df)
df = speed_gps.find_speed(df)

# calculate steering angle using GPS values
df = angle_gps.find_angle(df)

print(df.tail(n=10))

# save file
df.to_csv("data.csv", index=False)
