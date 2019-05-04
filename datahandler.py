# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:09:26 2019

@author: xiazizhe

Used to engineer the features and read data from the raw dataset
"""

'''
features: original variables, adding datetime variables
weather

'''


import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from pathlib import Path
import multiprocessing
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torchnet
from torchnet.dataset import ListDataset, ConcatDataset, ShuffleDataset

###############################################################################
##### data reader
def read_df(data_file):
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
    df = pd.read_csv(data_file,dtype=dtypes)
    df['origin'] = list(zip(df['pickup_longitude_bin'], df['pickup_latitude_bin']))
    df['destination'] = list(zip(df['dropoff_longitude_bin'], df['dropoff_latitude_bin']))
    df = df.drop(columns = ['pickup_longitude_bin','pickup_latitude_bin','dropoff_longitude_bin','dropoff_latitude_bin'])
    return df.sort_values(by=['time'])

##### data preprocessing
def feature_engineering(f, save_to_dir='./../input/'):
    df = process_df(f)
    df.to_csv(save_to_dir+f.stem+'.csv', index=False)

def process_df(data_file):
    '''
    full_columns = ['medallion','hack_license','pickup_datetime',
                    'passenger_count','trip_time_in_secs', 'trip_distance', 
                    'pickup_longitude','pickup_latitude', 'dropoff_longitude',
                    'dropoff_latitude','fare_amount','tip_amount',
                    'tolls_amount','total_amount','reward']
    '''
    use_columns = ['reward','pickup_datetime','passenger_count',
                   'trip_time_in_secs', 'trip_distance','pickup_longitude',
                   'pickup_latitude', 'dropoff_longitude','dropoff_latitude']
    skip_cols = ['medallion', 'hack_license','fare_amount','tip_amount',
                 'tolls_amount','total_amount']
    dtypes = {'reward':float,
              'passenger_count':int,
              'trip_time_in_secs':int, 
              'trip_distance':float, 
              'pickup_longitude': float, 
              'pickup_latitude':float, 
              'dropoff_longitude':float, 
              'dropoff_latitude':float}
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    datetime_columns = ['pickup_datetime']
#    locations = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    df = pd.read_csv(data_file,low_memory=False,usecols=(lambda x: x not in skip_cols),\
         dtype=dtypes, parse_dates=datetime_columns, date_parser=dateparse)
    df = df[use_columns]
    df = add_features(df)
#    features = ['trip_time_in_secs','trip_distance','average_speed',] + \
#               [col + '_bin' for col in locations] + \
#               ['monthofyear_is_' + str(i) for i in range(12)] + \
#               ['dayofweek_is_' + str(i) for i in range(7)] + \
#               ['hourofday_is_' + str(i) for i in range(24)]
#               ['timefracofday_sin', 'timefracofday_cos']
    features = ['reward','trip_distance','trip_time_in_secs','passenger_count',
                'month','day','time',
                'pickup_longitude_bin', 'pickup_latitude_bin', 
                'dropoff_longitude_bin', 'dropoff_latitude_bin']
    '''
    not in use: 'pickup_datetime','dropoff_datetime',
    'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'
    '''
    df = df[features]
    return df

def add_features(df):
#    df['average_speed'] = df['trip_distance']/df['trip_time_in_secs']
    df['month'] = df['pickup_datetime'].dt.month.astype(int)
    df['day'] = df['pickup_datetime'].dt.weekday.astype(int)
    df['hour'] = df['pickup_datetime'].dt.hour.astype(int)
    df['minute'] = df['pickup_datetime'].dt.minute.astype(int)
    df['second'] = df['pickup_datetime'].dt.second.astype(int)
    df['time'] = df.second + df.minute * 60 + df.hour * 3600
#    ### one-hot month
#    for i in range(12):
#        month = 'monthofyear_is_' + str(i)
#        df[month] = (df['pickup_month']==i).astype(int)
#    ### one-hot day of week
#    for i in range(7):
#        day = 'dayofweek_is_' + str(i)
#        df[day] = (df['pickup_day']==i).astype(int)
#    ### one-hot hour of day
#    for i in range(24):
#        hour = 'hourofday_is_' + str(i)
#        df[hour] = (df['pickup_hour']==i).astype(int)
#    ### circle representation time of day
#    df['timefracofday'] = (df['pickup_datetime'].dt.hour * 3600 + df['pickup_datetime'].dt.minute * 60 + df['pickup_datetime'].dt.second) / (24*60*60)
#    df['timefracofday_sin'], df['timefracofday_cos'] = np.sin(df['timefracofday'] * math.tau), np.cos(df['timefracofday'] * math.tau)
#    ###normalize location
#    locations = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
#    for col in locations:
#        try:
#            df[col+'_n'] = (df[col]-df[col].mean())/df[col].var()
#        except:
#            print('problem for ' + col)
#            df[col+'_n'] = (df[col]-df[col][0]) * 1000
    ### bin location
    df = bin_location(df)
    return df

def bin_location(df, bin_size=(10,10)):
    '''
    bin size is the size of the binned location in meters
    be aware of the longitude and latitude differences
    '''
    longitude_bin, latitude_bin = bin_size[0]/111030, bin_size[1]/85390
    longitudes = ['pickup_longitude','dropoff_longitude']
    latitudes = ['pickup_latitude','dropoff_latitude']
    for col in longitudes:
        df[col + '_bin'] = ((df[col] + 74.0)/longitude_bin).apply(np.floor)
    for col in latitudes:
        df[col + '_bin'] = ((df[col] - 40.6)/latitude_bin).apply(np.floor)
#    df['origin'] = list(zip(df['pickup_longitude_bin'], df['pickup_latitude_bin']))
#    df['destination'] = list(zip(df['dropoff_longitude_bin'], df['dropoff_latitude_bin']))
    return df

def get_tensor(data_file):
    try:
        return torch.load(data_file)
    except:
        df = process_df(data_file)
        tensor = torch.from_numpy(df.values).double()
        path = Path(data_file)
        save_to = './../input_tensor/' + path.parent.stem + '/' + path.stem
        torch.save(tensor, save_to + '.pt')
        return tensor

def get_data(data_file):
    try:
        tensor = process_df(data_file)
    except:
        tensor = get_tensor(data_file)
    return tensor

###############################################################################
##### Data generator construction
def get_data_generator(path, params):
    list_dataset = ListDataset(os.listdir(path), get_data, path) #list.txt contain list of datafiles, one per line
    concat_dataset = ConcatDataset(list_dataset)
    generator = DataLoader(dataset=concat_dataset, **params)#, collate_fn=batchify) #This will load data when needed, in parallel, up to <num_workers> thread.
    n_features = len(concat_dataset[0])-1
    return generator, n_features
###############################################################################
##### Data processing, saved for future uses

###############################################################################
##### Raw data file manipulatio (one time operation)
def split_train_test(test_frac, data_dir, train_dir, test_dir):
    for f in Path(data_dir).glob('*'):
        df = pd.read_csv(open(f,'r'))
        train, test = train_test_split(df, test_size=test_frac)
        train.to_csv(train_dir + f.stem, index=False)
        test.to_csv(test_dir + f.stem, index=False)

def clear_df(df):
    ### Drop na data, already dropped in previous section
#    columns = ['trip_time_in_secs', 'pickup_datetime', 'dropoff_datetime',
#               'trip_distance', 'passenger_count', 'pickup_longitude', 
#               'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
#    df = df.dropna(subset = columns)
    ### Bad location data
    longitude_columns = ['pickup_longitude', 'dropoff_longitude']
    latitudde_columns = ['pickup_latitude', 'dropoff_latitude']
    for col in longitude_columns:
        df = df[(-74.3<df[col]) & (df[col]<-73.6)]
    for col in latitudde_columns:
        df = df[(40.48<df[col]) & (df[col]<40.93)]
    ### Passagers no more than 8
    df = df[df.passenger_count <= 8]
    ### Zero traveling time or distance or taxi fare
    df = df[~((df.trip_distance == 0) | (df.trip_time_in_secs == 0) | (df.total_amount == 0))]
    return df

def clear_data(data_dir, save_to_dir):
    for f in Path(data_dir).glob('*'):
        df = pd.read_csv(open(f,'r'))
        df = clear_df(df)
        df.to_csv(save_to_dir + f.stem + '.csv', index=False)

def get_sample(data_dir, frac):
    paths = [f for f in Path(data_dir).glob('*') if f.is_file()]
    sample = pd.concat([pd.read_csv(open(f,'r')).sample(frac=frac/len(paths), replace=False, random_state=1) for f in paths])
    return sample

#def perform_preprocessing():
#    columns1 = ['medallion', ' hack_license', ' vendor_id', ' rate_code', ' store_and_fwd_flag', ' pickup_datetime', ' dropoff_datetime', ' passenger_count', ' trip_time_in_secs', ' trip_distance', ' pickup_longitude', ' pickup_latitude', ' dropoff_longitude', ' dropoff_latitude']
#    columns = ['medallion', 'hack_license', 'vendor_id', 'rate_code', 'store_and_fwd_flag', 'pickup_datetime', 'dropoff_datetime', 'passenger_count', 'trip_time_in_secs', 'trip_distance', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
#    dict1 = dict(zip(columns1, columns))
#    for f in Path('./../raw_data/').glob('*'):
#        if f.is_file():
#            df = pd.read_csv(open(f,'r'))
#            try:
#                df = df.rename(index=str, columns=dict1)
#            except:
#                df = df
#            print(list(df.columns))
#            try:
#                df = clear_df(df)
#            except:
#                print('clearing failed ' + f.stem)
#            df.to_csv('./../raw_data/trip_data/' + f.stem + '.csv', index=False)
#    split_train_test(test_frac=0.05, data_dir='./../raw_data/trip_data/', train_dir='./../input/train/', test_dir='./../input/test/')
def raw_input_preprocessing(data_dir, save_to_dir):
    columns1 = ['medallion', ' hack_license', ' vendor_id', ' rate_code',
                ' store_and_fwd_flag', ' pickup_datetime', ' dropoff_datetime',
                ' passenger_count', ' trip_time_in_secs', ' trip_distance',
                ' pickup_longitude', ' pickup_latitude', ' dropoff_longitude',
                ' dropoff_latitude']
    columns10 = ['medallion', 'hack_license', 'vendor_id', 'rate_code', 
                 'store_and_fwd_flag', 'pickup_datetime', 'dropoff_datetime',
                 'passenger_count', 'trip_time_in_secs', 'trip_distance',
                 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                 'dropoff_latitude']
    columns1drop = ['vendor_id', 'rate_code', 'store_and_fwd_flag','dropoff_datetime']
    columns2 = ['medallion', ' hack_license', ' vendor_id', ' pickup_datetime',
                ' payment_type', ' fare_amount', ' surcharge', ' mta_tax', 
                ' tip_amount', ' tolls_amount', ' total_amount']
    columns20 = ['medallion', 'hack_license', 'vendor_id', 'pickup_datetime', 
                 'payment_type', 'fare_amount', 'surcharge', 'mta_tax', 
                 'tip_amount', 'tolls_amount', 'total_amount']
    columns2drop = ['vendor_id', 'payment_type','surcharge', 'mta_tax']
    dict1 = dict(zip(columns1, columns10))
    dict2 = dict(zip(columns2, columns20))
    dropna_columns = ['medallion','hack_license','pickup_datetime',
                      'passenger_count','trip_time_in_secs','trip_distance', 
                      'pickup_longitude','pickup_latitude','dropoff_longitude',
                      'dropoff_latitude','fare_amount','tip_amount',
                      'tolls_amount','total_amount'] # 'rewards'
    for i in range(1,13):
        df_trip = pd.read_csv(data_dir + 'trip_data_' + str(i) + '.csv')
        df_fare = pd.read_csv(data_dir + 'trip_fare_' + str(i) + '.csv')
        initial_length = len(df_trip)
        try:
            df_trip = df_trip.rename(index=str, columns=dict1)
        except:
            df_trip = df_trip
        df_trip = df_trip.drop(columns=columns1drop)
        try:
            df_fare = df_fare.rename(index=str, columns=dict2)
        except:
            df_fare = df_fare
        df_fare = df_fare.drop(columns=columns2drop)
        print(list(df_trip.columns))
        print(list(df_fare.columns))
        df = pd.merge(df_trip, df_fare, how='outer',\
                      left_on=['medallion', 'hack_license', 'pickup_datetime'],\
                      right_on=['medallion', 'hack_license', 'pickup_datetime'])
        df = df.dropna(subset=dropna_columns)
        df['reward'] = df['fare_amount'] + df['tip_amount']
        intermediate_length = len(df)
        try:
            df = clear_df(df)
        except:
            print('clearing failed ' + str(i))
        final_length = len(df)
        print(initial_length, intermediate_length, final_length)
        df.to_csv(save_to_dir + 'taxi_' + str(i-1) + '.csv', index=False)
        
###############################################################################
##### previous feature engineering
#def get_datetime_features(df):
#    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
#    try:
#        df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
#    except:
#        pass
#    df["month"]=df["pickup_datetime"].dt.month
#    df["week"]=df["pickup_datetime"].dt.week
#    df["dayofweek"]=df["pickup_datetime"].dt.dayofweek
#    df["hour"]=df["pickup_datetime"].dt.hour
#    df.sort_values(by='pickup_datetime', inplace=True)
#    return df
#
#def get_weather_features(df):
##    full_weather_features = ['pickup_datetime', 'tempm', 'tempi', 'dewptm', 'dewpti', 'hum', 'wspdm',
##       'wspdi', 'wgustm', 'wgusti', 'wdird', 'wdire', 'vism', 'visi',
##       'pressurem', 'pressurei', 'windchillm', 'windchilli', 'heatindexm',
##       'heatindexi', 'precipm', 'precipi', 'conds', 'icon', 'fog', 'rain',
##       'snow', 'hail', 'thunder', 'tornado']
#    # due to lack of data: 'wgustm','windchillm','heatindexm','precipm',
#    # textual description: 'conds', 'icon', 
#    weather_features = ['pickup_datetime','tempm','dewptm','hum', 'wspdm',
#       'wdird','vism','pressurem','fog','rain','snow','hail','thunder','tornado']
#    df_weather = df[weather_features]
#    df_weather.drop_duplicates(subset='pickup_datetime', keep='last')\
#              .sort_values(by='pickup_datetime', inplace=True)
#    return df_weather
#
#def get_features(raw_data_dir='./../raw_data/', saveto_dir='./../input/'):
#    # read df
#    df_train = pd.read_csv(raw_data_dir + 'nyc_taxi_trip_duration/train.csv')
#    df_test = pd.read_csv(raw_data_dir + 'nyc_taxi_trip_duration/test.csv')
#    df_weather = pd.read_csv(raw_data_dir + 'nyc_weather/weather.csv')
#    # parse datetimes
#    df_train, df_test = get_datetime_features(df_train), get_datetime_features(df_test)
#    df_weather['pickup_datetime'] = pd.to_datetime(df_weather["pickup_datetime"])
#    # get weather features
#    df_weather = get_weather_features(df_weather)
#    # merge and get all the features
#    df_train = pd.merge_asof(df_train,df_weather, on="pickup_datetime", tolerance=pd.Timedelta('31m'), direction='nearest')
#    df_test = pd.merge_asof(df_test,df_weather, on="pickup_datetime", tolerance=pd.Timedelta('31m'), direction='nearest')
#    df_train.to_csv(saveto_dir+'train.csv', index=False)
#    df_test.to_csv(saveto_dir+'test.csv', index=False)
#    df_train_for_test = df_train[0:1000]
#    df_train_for_test.to_csv(saveto_dir+'train_for_test.csv', index=False)

if __name__ == '__main__':
#    pass
    ##### data clearing and preprocessing #####
#    raw_input_preprocessing('./../raw_data/','./../raw_data/combined_data/')
    ##### feature engineering and input generating #####
    data_dir = './../raw_data/combined_data/'
    save_to_dir = './../input/'
    paths = (f for f in Path(data_dir).glob('*') if f.is_file())
    ### single process
#    for f in paths:
#        feature_engineering(f, save_to_dir)
    ### multi process
    p = multiprocessing.Pool(processes=12)
    p.map(feature_engineering, paths) # use the default save_to_dir
###############################################################################
#    dataset = torchnet.dataset.ListDataset(['./../input_csv/trip_data_testing.csv'], process_df) #list.txt contain list of datafiles, one per line
#    data = torch.utils.data.DataLoader(dataset=torchnet.dataset.ConcatDataset(dataset), batch_size=50, num_workers=0)#, collate_fn=batchify) #This will load data when needed, in parallel, up to <num_workers> thread.
#    for x in data:
#        print(x.size())
#    pass
    
    
    