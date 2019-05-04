# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:38:41 2019

@author: XIA Zizhe
"""

import pandas as pd
import numpy as np
import torch
import multiprocessing

import datahandler as dh

class NYCTaxiEnv(object):
    
    def __init__(self, birthpoint=(165,1356), time=0, day=0, month=3,\
                 data_months=3, filters={'location':20,'time':900}, \
                 randomness={'reward':0.03, 'time':0.03, 'distance':0.03}, 
                 data_dir='./../input/', multiprocessing=False):
        ### action space
        self._action_space = [0,1,2]
        self._action_space_meanings = {0: 'accept the current offer', 
                                       1: 'reject the current offer and wait',
                                       2: 'reject the current offer and back'}
        ### initial state
        self._birthpoint = birthpoint ### birthpoint to be returned to
        self._location = birthpoint ### location: tuples of binned long&la, default at time sq
        self._time = time ### time: seconds of day passed, integer 0 - 86400-1
        self._month = month ### month: month of year, 0-11
        self._day = day ### day: day of week, 0-6
        ### ride offer generation params
        self._data_month = data_months%12 ### number of months to collect data from
        self._data_dir = data_dir ### store the data dir
        self._filters = filters ### filters for ride selection, location (in binned int) & time distance (in seconds int)
        self._randomness = randomness ### randomness float: add randomness in ride generation as std/total, set as 0 to exclude randomness
        self._multiprocessing = multiprocessing
        self._data_retrieved = False
        
    def get_action_space(self):
        return self._action_space
    
    def get_action_space_meanings(self):
        return self._action_space_meanings
    
#    def traffic_at_time(self, time, times_, traffic):
#        self._time = time
#        times_.append(time)
#        traffic.append(len(self.filter_rides()))
        
    def get_status(self):
        status = {'location':self._location,
                  'time':self._time,
                  'month':self._month,
                  'day':self._day}
        return status
        
    def collect_data(self):
        '''
        collect the data on request to save memory
        '''
        if not self._data_retrieved:
            month = self._month
            data_months = self._data_month
            data_dir = self._data_dir
            incremental_list = [0,1,-1,2,-2,3,-3,4,-4,5,-5,6]
            month_list = [(month + i)%12 for i in incremental_list[0:data_months]]
            dfs = {i: dh.read_df(data_dir+'/taxi_'+str(i)+'.csv') for i in month_list} ### need amendment
            self._dfs = dfs
            self._data_retrieved = True
        else:
            pass
    
    def get_traffic_at_times(self):
        '''
        get the traffic (num of rides) at the current location throughout the day
        '''
        location = self._location
        times = [i * 900 for i in range(96)]
        traffic = []
        times_ = []
        def do(time):
            self._time = time
            times_.append(time)
            traffic.append(len(self.filter_rides()))
        
        if self._multiprocessing:
            ### multi process
            p = multiprocessing.Pool(8)
            p.map(do, times)
        else:
            ### single process
            for time in times:
                do(time)
        return pd.DataFrame({'time':times_, location:traffic}).sort_values(by=['time']).set_index('time')
        
    def get_traffic_at_locations(self):
        '''
        get the traffic (num of rides) at the current time throughout the city
        '''
        time = self._time
        locations = np.concatenate([df.origin.unique() for m,df in self._dfs.items()])
        traffic = []
        locations_ = []
        def do(location):
            self._location = location
            locations_.append(location)
            traffic.append(len(self.filter_rides()))
            
        if self._multiprocessing:
            ### multi process
            p = multiprocessing.Pool(8)
            p.map(do, locations)
        else:
            ### single process
            for time in locations:
                do(time)
        return pd.DataFrame({'location':locations_, time:traffic}).sort_values(by=['location']).set_index('location')
    
    def reset(self):
        '''
        call this to initialize the episode, and collect the data
        i.e. randomize the date, set timer to zero, location to birthpoint
        need to call this before any action
        '''
        self._location = self._birthpoint ### reset location back to the birthpoint
        self._time = 0 ### reset time
        self.collect_data() ### collect data
        self.generate_rides()
        return self.observe()
        
    def filter_rides(self):
        '''
        filter all rides from the data (dfs) according to the filter
        call this only after collecting the data
        need to handle empty situation?? or be aware of that
        '''
        dfs = self._dfs
        location = self._location
        time = self._time
        day = self._day
        filters = self._filters
        f_location = lambda x: abs(x[0]-location[0]) + abs(x[1]-location[1]) <= filters['location']
        f_time = lambda x: np.logical_or((x - time).abs() <= filters['time'], (x - time).abs() >= 86400 - filters['time'])
        f_day = lambda x: x == day
        f = lambda df: df[np.logical_and.reduce((f_time(df.time), np.array([f_location(x) for x in df.origin],dtype=bool), f_day(df.day)))]
        filtered_rides = pd.concat((f(df) for m,df in dfs.items()))
        return filtered_rides
    
    def generate_rides(self):
        '''
        generate 3 rides corresponds to three action and store them
        ride {} keys: 'reward','trip_distance','trip_time_in_secs','passenger_count','destination'
        not return anything but store the rides in self._rides
        '''
        ### get params
        randomness = self._randomness
        filters = self._filters
        ### generate ride0: random ride from data
        filtered_rides = self.filter_rides()
        traffic = len(filtered_rides)
#        ride_probability = traffic / (traffic + 1000)
        ride_probability = min(1,traffic/(filters['time']/10 * filters['location']**2/1000))
        rand = np.random.random()
        if rand <= ride_probability and traffic != 0:
            rands = np.random.normal(size=3)
            sampled_ride = filtered_rides.sample()
#            print(sampled_ride)
            ride0 = {'reward':abs(sampled_ride.reward.iloc[0] * (1 + randomness['reward']*rands[0])),
                     'trip_distance':abs(sampled_ride.trip_distance.iloc[0] * (1 + randomness['distance']*rands[1])),
                     'trip_time_in_secs':int(abs(sampled_ride.trip_time_in_secs.iloc[0] * (1 + randomness['time']*rands[2]))),
                     'passenger_count':sampled_ride.passenger_count.iloc[0],
                     'destination':sampled_ride.destination.iloc[0]}
        else:
            ride0 = {'reward':0,
                     'trip_distance':0,
                     'trip_time_in_secs':60,
                     'passenger_count':1,
                     'destination':self._location}
        ### generate ride1: wait for another ride
        ride1 = {'reward':0,
                 'trip_distance':0,
                 'trip_time_in_secs':60,
                 'passenger_count':1,
                 'destination':self._location}
        
        ### generate ride2: back to the birthpoint
        rands = np.random.normal(size=2)
        birthpoint = self._birthpoint
        location = self._location
        manhatten_distance = sum(abs(a-b) for a,b in zip(birthpoint, location))
        trip_distance = 1.3 + 0.0041 * manhatten_distance ### from simple regression
        trip_time_in_secs = 580 + 0.47 * manhatten_distance
        ride2 = {'reward':0,
                 'trip_distance':abs(trip_distance * (1 + randomness['distance']*rands[0])),
                 'trip_time_in_secs':60 + int(abs(trip_time_in_secs * (1 + randomness['time']*rands[1]))),
                 'passenger_count':1,
                 'destination':birthpoint}
        ### store the rides
        rides = {0:ride0, 1:ride1, 2:ride2}
        self._rides = rides
    
    def step(self, action):
        '''
        make a step and get the reward
        update status, return new observation, reward, done
        '''
        ride = self._rides[action]
        t_start = self._time
        trip_time_in_secs = ride['trip_time_in_secs']
        self._time = t_start + trip_time_in_secs
        self._location = ride['destination']
        reward = ride['reward']
        done = self._time >= 86400
        if done:
            reward = reward * (86400 - t_start) / trip_time_in_secs
        else:
            self.generate_rides() ### need no more ride if done
        observation = self.observe()
        return observation, reward, done
    
    def observe(self):
        '''
        get the current state of the agent and the ride offers
        observation dict
        '''
        observation = {'status':self.get_status(),
                       'rides':self._rides}
        return tensorify_obs(observation)

def tensorify_obs(obs):
    '''
    transform the observation input into tensors
    IMPORTANT: omit the unobservable items such as trip time
    Note that some features not needed: info of r1, r2
    '''
    s,r0,r1,r2 = obs['status'],obs['rides'][0],obs['rides'][1],obs['rides'][2]
    x = [s['location'][0], s['location'][1], s['time'], #s['month'], s['day'],
         r0['trip_distance'],r0['passenger_count'],r0['destination'][0],r0['destination'][1],
#         r1['trip_distance'],r1['passenger_count'],r1['destination'][0],r1['destination'][1],
#         r2['trip_distance'],r2['passenger_count'],r2['destination'][0],r2['destination'][1]
         r2['trip_distance']]
    return torch.Tensor(x).double()
    
if __name__ == '__main__':
    driver = NYCTaxiEnv()
    observation = driver.reset()
    df = driver.filter_rides()
    print(len(df))
    df.to_csv('filtered.csv')
