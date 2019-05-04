# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 20:46:16 2019

@author: xiazizhe
"""

import pandas as pd
import numpy as np
from itertools import product
from functools import partial
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
import multiprocessing

import environment

def regression(df, indeps, deps):
    m = LinearRegression()
    X = df[indeps].values
    y = df[deps].values
    m.fit(X,y)
    pred = m.predict(X)
    mse = mean_squared_error(pred, y)
    return [m.coef_, m.intercept_, mse, m.score(X,y)]
    
def get_manhatten_distance(df, birthplace):
    df['manhatten_distance'] = (df['pickup_longitude_bin'] - birthplace[0]).apply(np.abs)\
                              + (df['pickup_latitude_bin'] - birthplace[1]).apply(np.abs)
    return df

def backride_reg(file, birthplace):
    dtypes = {'reward':float,
              'trip_distance':float,
              'trip_time_in_secs':int,
              'passenger_count':int,
              'month':int,
              'day':int,
              'time':int,
              'pickup_longitude_bin':int,
              'pickup_latitude_bin':int, 
              'dropoff_longitude_bin':int,
              'dropoff_latitude_bin':int}
    df = pd.read_csv(file,dtype=dtypes)
    df = get_manhatten_distance(df, birthplace)
    reg_time = regression(df, ['manhatten_distance'], ['trip_time_in_secs'])
    reg_distance = regression(df, ['manhatten_distance'], ['trip_distance'])
    return reg_time, reg_distance

def backride_analysis(data_dir, birthplace):
    regs_time, regs_distance = [], []
    for f in Path(data_dir).glob('*'):
        reg_time, reg_distance = backride_reg(f, birthplace)
        regs_time.append(reg_time)
        regs_distance.append(reg_distance)
    regs_time = pd.DataFrame(regs_time, columns=['coef_mdist','intercept','mse','r2'])
    regs_distance = pd.DataFrame(regs_distance, columns=['coef_mdist','intercept','mse','r2'])
    return regs_time, regs_distance

def get_traffic_at_times(location, agent):
    agent._location = location
    _ = agent.reset()
    traffic_at_times = agent.get_traffic_at_times()
    return traffic_at_times

if __name__ == '__main__':
    output_dir = './../output/'
    data_dir = './../input/'
    ##### Predict back ride time and distance
#    birthplace = (165,1356)
#    regs_time, regs_distance = backride_analysis(data_dir, birthplace)
#    regs_time.to_csv(output_dir+'time_on_mdist.csv', index=False)
#    regs_distance.to_csv(output_dir+'distance_on_mdist.csv', index=False)
    
    ##### check location / time traffic status
    driver = environment.NYCTaxiEnv()
    ### traffic at all times at some location
    locations = [(165,1356)] + [(100*(i-1), 900+100*i) for i,j in product(range(10),range(10))]
    traffic = []
    # Single process
#    for location in locations:
#        driver._location = location
#        _ = driver.reset()
#        traffic_at_times = driver.get_traffic_at_times()
#        traffic.append(traffic_at_times)
#    traffic_df = pd.concat(traffic, axis=1)
    # Multi process
    p = multiprocessing.Pool(processes=12)
    f = partial(get_traffic_at_times, agent=driver)
    traffic = p.map(f, locations)
    traffic_df = pd.concat(traffic, axis=1)
    try:
        traffic_df = traffic_df[locations]
    except:
        traffic_df = traffic_df
    traffic_df.to_csv('traffic_at_times.csv')
    ### traffic at all locations at some time
#    for t in range(9):
#        driver._time = t * 10000
#        driver.reset()
#        traffic_at_locations = driver.get_traffic_at_locations()
#        traffic_at_locations.to_csv(output_dir+'traffic_at_locations_'+str(t)+'.csv', index=False)
    
    
    