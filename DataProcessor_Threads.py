#!/usr/bin/env python
import os
import sys
import warnings
import shutil
import time
import datetime
import calendar
import schedule
import pandas as pd
import numpy as np
import mysql.connector as mysql
from fastparquet import ParquetFile, write
import pickle
import boto3
import logging
import IBTrader
import socket
from threading import Thread

from aitrader_datagen import gen_inference_dset
from aitrader_model import gen_new_pred, load_all_the_things

s3 = boto3.resource('s3')
#s3bucketName = 'legio-data'
s3bucketName = 'legio-test'
bucket = s3.Bucket(name=s3bucketName)

#s3_StorageLocation = "database_data/"
s3_StorageLocation = "Legio/data/database_data/"

#localStorage = '/general/legiotrader/MarketData/'
localStorage = '/root/data/'

dbHostContainer = "ec2-3-104-151-3.ap-southeast-2.compute.amazonaws.com"
dbHostPort = "31330"
user = "root"
passwd = "root"

model = None
model_feat_map = None
model_norm_params = None
modelParamsPath = ""

baseConfigInfo = None
baseCode = ""
activeTrader = False
clientId=88888
ibGWID = ""
host=""
port=4002
ibConn = IBTrader.IBTrader()
cash_contractKey = 0
cfd_contractKey = 0

tickData = []
singleTickData = []
minuteData = []
hourData = []
currentBid =0
currentBidSize=0
currentAsk =0
currentAskSize=0
currentLast =0
currentLastSize=0

dataProcessed = True
lastSignal = "WAIT"

positions = {}
orders = {}

positionSize = 0
approvedPositionSize = 0
maxFillTime = 90

class IBPosition(object):
    symbol = ''
    contracttype = ''
    currentcontractkey = 0
    qty = 0
    avgfillprice = 0
    orderids = []        
    
    def __init__(self, symbol, contracttype, currentcontractkey):        
        self.symbol = symbol
        self.contracttype = contracttype
        self.currentcontractkey = currentcontractkey               
        
    def ProcessOrder(self, size, avgfillprice, orderid):        
        self.qty += size
        self.avgfillprice = self.CalculateTotalAvgFill(size, avgfillprice)
        self.orderids.append(orderid)
        
    def CalculateTotalAvgFill(self,size,avgfillprice):
        if self.qty + size == 0:
            return 0
        else:
            if self.avgfillprice == 0:
                return avgfillprice
            else:
                return (self.avgfillprice+avgfillprice)/2        
class IBOrder(object):
    symbol = ''
    contracttype = ''
    currentcontractkey = 0
    size = 0
    orderid = 0
    status = ''
    progressdatetime = {}
    ibtraderorder = {}
        
    def __init__(self,symbol,contracttype,currentcontractkey,size,orderid,sentdatetime):        
        self.symbol = symbol
        self.contracttype = contracttype
        self.currentcontractkey = currentcontractkey
        self.size = size        
        self.orderid = orderid
        self.status = 'CREATED'
        self.progressdatetime['CREATED'] = sentdatetime
        self.ibtraderorder = ''
    
    def UpdateOrder(self, status, updatedatetime):
        self.status = status
        self.progressdatetime[status] = updatedatetime  

def log_stage(stage_name):
    path = f'/general/legiotrader/MarketData/logs/{datetime.datetime.now().strftime("%Y%m%d--%H-%M-%S")}--{stage_name}'
    open(path, 'w').close()

def RegisterSchedule():
    schedule.clear('data-tasks')      
    
    #schedule.every(1).minutes.do(HourlyProcess).tag('data-tasks')
    #schedule.every(1).minutes.do(StoreMarketData).tag('data-tasks')

    schedule.every().day.at("22:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("22:30").do(StoreMarketData).tag('data-tasks')
    schedule.every().day.at("23:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("23:30").do(StoreMarketData).tag('data-tasks')  
    schedule.every().day.at("00:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("00:30").do(StoreMarketData).tag('data-tasks')
    schedule.every().day.at("01:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("01:30").do(StoreMarketData).tag('data-tasks')
    schedule.every().day.at("02:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("02:30").do(StoreMarketData).tag('data-tasks')
    schedule.every().day.at("03:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("03:30").do(StoreMarketData).tag('data-tasks')
    schedule.every().day.at("04:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("04:30").do(StoreMarketData).tag('data-tasks')
    schedule.every().day.at("05:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("05:30").do(StoreMarketData).tag('data-tasks')  
    schedule.every().day.at("06:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("06:30").do(StoreMarketData).tag('data-tasks')  
    schedule.every().day.at("07:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("07:30").do(StoreMarketData).tag('data-tasks')  
    schedule.every().day.at("08:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("08:30").do(StoreMarketData).tag('data-tasks')  
    schedule.every().day.at("09:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("09:30").do(StoreMarketData).tag('data-tasks')  
    schedule.every().day.at("10:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("10:30").do(StoreMarketData).tag('data-tasks')     
    schedule.every().day.at("11:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("11:30").do(StoreMarketData).tag('data-tasks')  
    schedule.every().day.at("12:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("12:30").do(StoreMarketData).tag('data-tasks')  
    schedule.every().day.at("13:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("13:30").do(StoreMarketData).tag('data-tasks')  
    schedule.every().day.at("14:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("14:30").do(StoreMarketData).tag('data-tasks')  
    schedule.every().day.at("15:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("15:30").do(StoreMarketData).tag('data-tasks')  
    schedule.every().day.at("16:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("16:30").do(StoreMarketData).tag('data-tasks')  
    schedule.every().day.at("17:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("17:30").do(StoreMarketData).tag('data-tasks')  
    schedule.every().day.at("18:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("18:30").do(StoreMarketData).tag('data-tasks')  
    schedule.every().day.at("19:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("19:30").do(StoreMarketData).tag('data-tasks')  
    schedule.every().day.at("20:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("20:30").do(StoreMarketData).tag('data-tasks')
    schedule.every().day.at("21:00").do(HourlyProcess).tag('data-tasks')
    schedule.every().day.at("21:30").do(StoreMarketData).tag('data-tasks')

def CheckDatetime(datetime):
    #log_stage('dt_check_START')
    if datetime.weekday() == 6:
        if datetime.hour >= 9:
            #log_stage('dt_check_END')
            return True
        else:
            #log_stage('dt_check_END')
            return False    
    elif datetime.weekday() == 4:
        if datetime.hour < 21:
            #log_stage('dt_check_END')
            return True
        else:
            #log_stage('dt_check_END')
            return False
    elif datetime.weekday() < 4:
        #log_stage('dt_check_END')
        return True
    elif datetime.weekday() == 5:
        #log_stage('dt_check_END')
        return False
    log_stage('dt_check_ERROR')

def RequestSpecificMarketData(endtime, timeframe, lookbackSecs):
    log_stage('data_req_specific_START')
    global ibConn
    global baseCode
    global cash_contractKey
    ibConn.historicalData = {}
        
    ibConn.requestHistoricalData(ibConn.contracts[cash_contractKey],resolution=timeframe,end_datetime="{}".format(endtime.strftime("%Y%m%d %H:%M:%S")),lookback=str(lookbackSecs))
    while not ibConn.dataReqComplete:
        #print(datetime.datetime.now().strftime("%Y%m%d %H:%M:%S"), end='\r')
        time.sleep(0.2)
    log_stage('data_req_specific_END')
    return ibConn.historicalData[baseCode+"_CASH"]

def HourlyProcess():
    log_stage('hourly_proc_START')
    global ibConn
    global baseCode
    global dbHostContainer
    global dbHostPort
    global user
    global passwd
    global hourData
    global minuteData
    global dataProcessed    
    global modelParamsPath
    global model_feat_map
    global model_norm_params
    global model
    global positionSize
    global approvedPositionSize
    global lastSignal

    ibConn.cancelHistoricalData()
    ibConn.historicalData = {}

    startTime = datetime.datetime.utcnow()
    processToTime = datetime.datetime(startTime.year, startTime.month, startTime.day, startTime.hour, 0, 0)    
    timeChunk = "3600 S"
    
    log_stage('hourly_proc_START_min_dl')
    timeframe = "1 min"
    minutes = RequestSpecificMarketData(processToTime, timeframe, timeChunk)
    minutes.drop(['V','OI','WAP'],1,inplace=True)
    minutes['Vol'] = 0
    minutes = minutes.sort_values('datetime').reset_index()
    minuteData = minutes
    log_stage('hourly_proc_END_min_dl')

    log_stage('hourly_proc_START_hour')
    timeframe = "1 hour"
    hours = RequestSpecificMarketData(processToTime, timeframe, timeChunk)
    hours.drop(['V','OI','WAP'],1,inplace=True)
    hours = hours.sort_values('datetime').reset_index()
    hourData = hours
    log_stage('hourly_proc_END_hour_dl')

    log_stage('hourly_proc_START_min_commit')
    if len(minutes) > 0:   
        for commitData in list(np.array_split(minutes,10)):
            vals = [tuple(x) for x in commitData.values]
            sql = "INSERT INTO "+baseCode+"_Min (Timestamp, Open, High, Low, Close, Vol) VALUES (%s, %s, %s, %s, %s, %s)"
            database = mysql.connect(host=dbHostContainer,port=dbHostPort,user=user,passwd=passwd,database=baseCode)
            dbConnection = database.cursor()
            dbConnection.executemany(sql, vals)
            database.commit()
        log_stage('hourly_proc_END_min_commit')
    else:
        #Adding error logging
        log_stage('hourly_proc_ERROR_mins')

    log_stage('hourly_proc_START_hour_commit') 
    if len(hours) > 0:
        for commitData in list(np.array_split(hours,10)):
            vals = [tuple(x) for x in commitData.values]
            sql = "INSERT INTO "+baseCode+"_Hour (Timestamp, Open, High, Low, Close) VALUES (%s, %s, %s, %s, %s)"
            database = mysql.connect(host=dbHostContainer,port=dbHostPort,user=user,passwd=passwd,database=baseCode)
            dbConnection = database.cursor()
            dbConnection.executemany(sql, vals)
            database.commit()
        log_stage('hourly_proc_END_hour_commit')
    else:
        #Adding error logging
        log_stage('hourly_proc_ERROR_hours')

    dataProcessed = True

    if activeTrader:
        log_stage('hourly_proc_START_trader')
        try:  
            new_obs = gen_inference_dset(inference_file_loc=modelParamsPath,feat_map=model_feat_map,norm_params=model_norm_params)        
            value = gen_new_pred(new_obs.values, model)

            last_price = hourData.tail(1)['C'].item()
            current_price = ((currentAsk-currentBid)/2)+currentBid
            expected_change = (last_price+value) - current_price
            thresh = 0
            action = ""        
            if abs(expected_change) < thresh:
                action = 'WAIT'
            elif abs(expected_change) > 0:
                action = 'BUY'
            elif abs(expected_change) < 0:
                action = 'SELL'
            else:
                from aitrader_errors import log_error            
                log_error('model_output_error', expected_change)
                log_stage('hourly_proc_ERROR_trader_bsw')
                action = 'WAIT'           

            if action != 'WAIT':
                log_stage('hourly_proc_START_trader_order')                     
                ProcessOrderSignal(action, False)
                log_stage('hourly_proc_END_trader_order')
            else:
                ## Record signal for the hour??
                log_stage('hourly_proc_Trade_Signal_not_traded')
                pass        
            lastSignal = action
        except Exception as e:
            log_stage('hourly_proc_ERROR_trader_except')
            print(e)


def ProcessOrderSignal(action, closeoutOrder):
    global positions
    global orders
    global approvedPositionSize
    global lastSignal
    global maxFillTime
    global ibConn
    global cash_contractKey
    global cfd_contractKey
    global baseCode
        
    positionSymbol = baseCode+"_"+ibConn.contractDetails(cfd_contractKey)['m_summary']['m_secType']
    
    if closeoutOrder:
        log_stage('process_order_signal_CloseoutOrder_Start')
        if action == "BUY":
            startTime = datetime.datetime.utcnow()
            filled = False
            closeoutOrder = ibConn.createOrder(quantity=positions[positionSymbol].qty, price=0)
            closeoutOrderEx = ibConn.placeOrder(contract=ibConn.contracts[cfd_contractKey], order=closeoutOrder)                
            if closeoutOrder.m_action == 'BUY':
                orders[closeoutOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                         closeoutOrder.m_totalQuantity,closeoutOrderEx, datetime.datetime.now())                
            elif closeoutOrder.m_action == 'SELL':
                orders[closeoutOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                         -closeoutOrder.m_totalQuantity,closeoutOrderEx, datetime.datetime.now())
            
            log_stage('process_order_signal_CloseoutOrder_created')      
            while not filled:
                if ibConn.orders[closeoutOrderEx]['status'] == 'FILLED':
                    orders[closeoutOrderEx].UpdateOrder("FILLED", datetime.datetime.now())
                    orders[closeoutOrderEx].ibtraderorder = ibConn.orders[closeoutOrderEx]
                    positions[positionSymbol].ProcessOrder(orders[closeoutOrderEx].size,
                                                   orders[closeoutOrderEx].ibtraderorder['avgFillPrice'],closeoutOrderEx)
                    filled = True
                    log_stage('process_order_signal_CloseoutOrder_filled')
                elif (datetime.datetime.utcnow() - startTime).seconds > maxFillTime:
                    ibConn.cancelOrder(closeoutOrderEx)
                    log_stage('process_order_signal_CloseoutOrder_cancelled')
                    filled = True
                time.sleep(0.05)
        elif action == "SELL":
            startTime = datetime.datetime.utcnow()
            filled = False            
            closeoutOrder = ibConn.createOrder(quantity= -positions[positionSymbol].qty, price=0)
            closeoutOrderEx = ibConn.placeOrder(contract=ibConn.contracts[cfd_contractKey], order=closeoutOrder)                
            if closeoutOrder.m_action == 'BUY':
                orders[closeoutOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                         closeoutOrder.m_totalQuantity,closeoutOrderEx, datetime.datetime.now())                
            elif closeoutOrder.m_action == 'SELL':
                orders[closeoutOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                         -closeoutOrder.m_totalQuantity,closeoutOrderEx, datetime.datetime.now())
            log_stage('process_order_signal_CloseoutOrder_created')

            while not filled:
                if ibConn.orders[closeoutOrderEx]['status'] == 'FILLED':
                    orders[closeoutOrderEx].UpdateOrder("FILLED", datetime.datetime.now())
                    orders[closeoutOrderEx].ibtraderorder = ibConn.orders[closeoutOrderEx]
                    positions[positionSymbol].ProcessOrder(orders[closeoutOrderEx].size,
                                                   orders[closeoutOrderEx].ibtraderorder['avgFillPrice'],closeoutOrderEx)                    
                    filled = True
                    log_stage('process_order_signal_CloseoutOrder_filled')
                elif (datetime.datetime.utcnow() - startTime).seconds > maxFillTime:
                    ibConn.cancelOrder(closeoutOrderEx)
                    log_stage('process_order_signal_CloseoutOrder_cancelled')            
                    filled = True
                time.sleep(0.05)
        log_stage('process_order_signal_CloseoutOrder_End')
    else:
        if positionSymbol in positions:
            if positions[positionSymbol].qty > 0:
                if action == "SELL":
                    log_stage('process_order_signal_longposition_SellOrder_start')
                    startTime = datetime.datetime.utcnow()
                    filled = False
                    closeoutOrder = ibConn.createOrder(quantity=-positions[positionSymbol].qty, price=0)
                    closeoutOrderEx = ibConn.placeOrder(contract=ibConn.contracts[cfd_contractKey], order=closeoutOrder)
                    log_stage('process_order_signal_longposition_SellOrder_closeout_created')
                    if closeoutOrder.m_action == 'BUY':
                        orders[closeoutOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                                 closeoutOrder.m_totalQuantity,closeoutOrderEx, datetime.datetime.now())               
                    elif closeoutOrder.m_action == 'SELL':
                        orders[closeoutOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                                 -closeoutOrder.m_totalQuantity,closeoutOrderEx, datetime.datetime.now())
                    while not filled:
                        if ibConn.orders[closeoutOrderEx]['status'] == 'FILLED':
                            orders[closeoutOrderEx].UpdateOrder("FILLED", datetime.datetime.now())
                            orders[closeoutOrderEx].ibtraderorder = ibConn.orders[closeoutOrderEx]
                            positions[positionSymbol].ProcessOrder(orders[closeoutOrderEx].size,
                                                           orders[closeoutOrderEx].ibtraderorder['avgFillPrice'],closeoutOrderEx)
                            filled = True
                            log_stage('process_order_signal_longposition_SellOrder_closeout_filled')
                        elif (datetime.datetime.utcnow() - startTime).seconds > maxFillTime:
                            ibConn.cancelOrder(closeoutOrderEx)                            
                            filled = True
                            log_stage('process_order_signal_longposition_SellOrder_closeout_cancelled')
                        time.sleep(0.05)

                    startTime = datetime.datetime.utcnow()
                    filled = False
                    newOrder = ibConn.createOrder(quantity=-approvedPositionSize, price=0)
                    newOrderEx = ibConn.placeOrder(contract=ibConn.contracts[cfd_contractKey], order=newOrder)
                    if newOrder.m_action == 'BUY':
                        orders[newOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                                 newOrder.m_totalQuantity,newOrderEx, datetime.datetime.now())                
                    elif newOrder.m_action == 'SELL':
                        orders[newOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                                 -newOrder.m_totalQuantity,newOrderEx, datetime.datetime.now())     
                    log_stage('process_order_signal_longposition_SellOrder_created')
                    while not filled:
                        if ibConn.orders[newOrderEx]['status'] == 'FILLED':
                            orders[newOrderEx].UpdateOrder("FILLED", datetime.datetime.now())
                            orders[newOrderEx].ibtraderorder = ibConn.orders[newOrderEx]
                            positions[positionSymbol].ProcessOrder(orders[newOrderEx].size,
                                                           orders[newOrderEx].ibtraderorder['avgFillPrice'],newOrderEx)
                            filled = True
                            log_stage('process_order_signal_longposition_SellOrder_filled')
                        elif (datetime.datetime.utcnow() - startTime).seconds > maxFillTime:
                            ibConn.cancelOrder(newOrderEx)                
                            filled = True
                            log_stage('process_order_signal_SellOrder_cancelled')
                        time.sleep(0.05)
                else:
                    ## Need to manage some sort of stop loss based on the current price. If we are already in a 
                    ## position but are holding, we don't want to lose any more than the bounds of an hour.
                    ##
                    ## Should also probably record signals ???
                    log_stage('process_order_signal_longposition_BuyOrder_not_required_alreadyLong')
                    pass
                log_stage('process_order_signal_SellOrder_end')
            
            elif positions[positionSymbol].qty < 0:
                if action == "BUY":
                    log_stage('process_order_signal_shortposition_BuyOrder_start')
                    startTime = datetime.datetime.utcnow()
                    filled = False
                    closeoutOrder = ibConn.createOrder(quantity=-positions[positionSymbol].qty, price=0)
                    closeoutOrderEx = ibConn.placeOrder(contract=ibConn.contracts[cfd_contractKey], order=closeoutOrder)                
                    if closeoutOrder.m_action == 'BUY':
                        orders[closeoutOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                                 closeoutOrder.m_totalQuantity,closeoutOrderEx, datetime.datetime.now())                
                    elif closeoutOrder.m_action == 'SELL':
                        orders[closeoutOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                                 -closeoutOrder.m_totalQuantity,closeoutOrderEx, datetime.datetime.now())
                    log_stage('process_order_signal_shortposition_BuyOrder_closeout_created')
                    while not filled:
                        if ibConn.orders[closeoutOrderEx]['status'] == 'FILLED':
                            orders[closeoutOrderEx].UpdateOrder("FILLED", datetime.datetime.now())
                            orders[closeoutOrderEx].ibtraderorder = ibConn.orders[closeoutOrderEx]
                            positions[positionSymbol].ProcessOrder(orders[closeoutOrderEx].size,
                                                           orders[closeoutOrderEx].ibtraderorder['avgFillPrice'],closeoutOrderEx)
                            filled = True
                            log_stage('process_order_signal_shortposition_BuyOrder_closeout_filled')
                        elif (datetime.datetime.utcnow() - startTime).seconds > maxFillTime:
                            ibConn.cancelOrder(closeoutOrderEx)
                            filled = True
                            log_stage('process_order_signal_shortposition_BuyOrder_closeout_cancelled')
                        time.sleep(0.05)

                    startTime = datetime.datetime.utcnow()
                    filled = False
                    newOrder = ibConn.createOrder(quantity=approvedPositionSize, price=0)
                    newOrderEx = ibConn.placeOrder(contract=ibConn.contracts[cfd_contractKey], order=newOrder)                
                    if newOrder.m_action == 'BUY':
                        orders[newOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                                 newOrder.m_totalQuantity,newOrderEx, datetime.datetime.now())                
                    elif newOrder.m_action == 'SELL':
                        orders[newOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                                 -newOrder.m_totalQuantity,newOrderEx, datetime.datetime.now())
                    log_stage('process_order_signal_shortposition_BuyOrder_created')
                    while not filled:
                        if ibConn.orders[newOrderEx]['status'] == 'FILLED':
                            orders[newOrderEx].UpdateOrder("FILLED", datetime.datetime.now())
                            orders[newOrderEx].ibtraderorder = ibConn.orders[newOrderEx]
                            positions[positionSymbol].ProcessOrder(orders[newOrderEx].size,
                                                           orders[newOrderEx].ibtraderorder['avgFillPrice'],newOrderEx)
                            filled = True
                            log_stage('process_order_signal_shortposition_BuyOrder_filled')
                        elif (datetime.datetime.utcnow() - startTime).seconds > maxFillTime:
                            ibConn.cancelOrder(newOrderEx)                
                            filled = True
                            log_stage('process_order_signal_shortposition_BuyOrder_cancelled')
                        time.sleep(0.05)
                else:
                    ## Need to manage some sort of stop loss based on the current price. If we are already in a 
                    ## position but are holding, we don't want to lose any more than the bounds of an hour.
                    ##
                    ## Should also probably record signals ???
                    log_stage('process_order_signal_shortposition_SellOrder_not_required_alreadyShort')
                    pass
                log_stage('process_order_signal_shortposition_BuyOrder_end')

            elif positions[positionSymbol].qty == 0:
                if action == "BUY":
                    log_stage('process_order_signal_noposition_BuyOrder_start')
                    startTime = datetime.datetime.utcnow()
                    filled = False
                    newOrder = ibConn.createOrder(quantity=approvedPositionSize, price=0)
                    newOrderEx = ibConn.placeOrder(contract=ibConn.contracts[cfd_contractKey], order=newOrder)                
                    if newOrder.m_action == 'BUY':
                        orders[newOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                                 newOrder.m_totalQuantity,newOrderEx, datetime.datetime.now())                
                    elif newOrder.m_action == 'SELL':
                        orders[newOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                                 -newOrder.m_totalQuantity,newOrderEx, datetime.datetime.now())     
                    log_stage('process_order_signal_noposition_BuyOrder_created')
                    while not filled:
                        if ibConn.orders[newOrderEx]['status'] == 'FILLED':
                            orders[newOrderEx].UpdateOrder("FILLED", datetime.datetime.now())
                            orders[newOrderEx].ibtraderorder = ibConn.orders[newOrderEx]
                            positions[positionSymbol].ProcessOrder(orders[newOrderEx].size,
                                                           orders[newOrderEx].ibtraderorder['avgFillPrice'],newOrderEx)
                            filled = True
                            log_stage('process_order_signal_noposition_BuyOrder_filled')
                        elif (datetime.datetime.utcnow() - startTime).seconds > maxFillTime:
                            ibConn.cancelOrder(newOrderEx)                
                            filled = True
                            log_stage('process_order_signal_noposition_BuyOrder_cancelled')
                        time.sleep(0.05)
                    log_stage('process_order_signal_noposition_BuyOrder_end')
                elif action == "SELL":
                    log_stage('process_order_signal_noposition_SellOrder_start')
                    startTime = datetime.datetime.utcnow()
                    filled = False
                    newOrder = ibConn.createOrder(quantity=-approvedPositionSize, price=0)
                    newOrderEx = ibConn.placeOrder(contract=ibConn.contracts[cfd_contractKey], order=newOrder)                
                    if newOrder.m_action == 'BUY':
                        orders[newOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                                 newOrder.m_totalQuantity,newOrderEx, datetime.datetime.now())                
                    elif newOrder.m_action == 'SELL':
                        orders[newOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                                 -newOrder.m_totalQuantity,newOrderEx, datetime.datetime.now())     
                    log_stage('process_order_signal_noposition_SellOrder_created')
                    while not filled:
                        if ibConn.orders[newOrderEx]['status'] == 'FILLED':
                            orders[newOrderEx].UpdateOrder("FILLED", datetime.datetime.now())
                            orders[newOrderEx].ibtraderorder = ibConn.orders[newOrderEx]
                            positions[positionSymbol].ProcessOrder(orders[newOrderEx].size,
                                                           orders[newOrderEx].ibtraderorder['avgFillPrice'],newOrderEx)
                            filled = True
                            log_stage('process_order_signal_noposition_SellOrder_filled')
                        elif (datetime.datetime.utcnow() - startTime).seconds > maxFillTime:
                            ibConn.cancelOrder(newOrderEx)                
                            filled = True
                            log_stage('process_order_signal_noposition_SellOrder_cancelled')
                        time.sleep(0.05)
                    log_stage('process_order_signal_noposition_SellOrder_end')
        else:
            positions[baseCode+"_"+ibConn.contractDetails(cfd_contractKey)['m_summary']['m_secType']] = IBPosition(baseCode,ibConn.contractDetails(cfd_contractKey)['m_summary']['m_secType'],cfd_contractKey)        
            if action == "BUY":
                log_stage('process_order_signal_firstposition_BuyOrder_start')
                startTime = datetime.datetime.utcnow()
                filled = False
                newOrder = ibConn.createOrder(quantity=approvedPositionSize, price=0)
                newOrderEx = ibConn.placeOrder(contract=ibConn.contracts[cfd_contractKey], order=newOrder)                
                if newOrder.m_action == 'BUY':
                        orders[newOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                                 newOrder.m_totalQuantity,newOrderEx, datetime.datetime.now())                
                elif newOrder.m_action == 'SELL':
                    orders[newOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                             -newOrder.m_totalQuantity,newOrderEx, datetime.datetime.now())     
                log_stage('process_order_signal_firstposition_BuyOrder_created')
                while not filled:
                    if ibConn.orders[newOrderEx]['status'] == 'FILLED':
                        orders[newOrderEx].UpdateOrder("FILLED", datetime.datetime.now())
                        orders[newOrderEx].ibtraderorder = ibConn.orders[newOrderEx]
                        positions[positionSymbol].ProcessOrder(orders[newOrderEx].size,
                                                       orders[newOrderEx].ibtraderorder['avgFillPrice'],newOrderEx)
                        filled = True
                        log_stage('process_order_signal_firstposition_BuyOrder_filled')
                    elif (datetime.datetime.utcnow() - startTime).seconds > maxFillTime:
                        ibConn.cancelOrder(newOrderEx)                
                        filled = True
                        log_stage('process_order_signal_firstposition_BuyOrder_cancelled')
                    time.sleep(0.05)
                log_stage('process_order_signal_firstposition_BuyOrder_end')
            elif action == "SELL":
                log_stage('process_order_signal_firstposition_SellOrder_start')
                startTime = datetime.datetime.utcnow()
                filled = False
                newOrder = ibConn.createOrder(quantity=-approvedPositionSize, price=0)
                newOrderEx = ibConn.placeOrder(contract=ibConn.contracts[cfd_contractKey], order=newOrder)                
                if newOrder.m_action == 'BUY':
                        orders[newOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                                 newOrder.m_totalQuantity,newOrderEx, datetime.datetime.now())                
                elif newOrder.m_action == 'SELL':
                    orders[newOrderEx] = IBOrder(baseCode, positionSymbol.split('_')[-1],cfd_contractKey,
                                             -newOrder.m_totalQuantity,newOrderEx, datetime.datetime.now())
                log_stage('process_order_signal_firstposition_SellOrder_created')
                while not filled:
                    if ibConn.orders[newOrderEx]['status'] == 'FILLED':
                        orders[newOrderEx].UpdateOrder("FILLED", datetime.datetime.now())
                        orders[newOrderEx].ibtraderorder = ibConn.orders[newOrderEx]
                        positions[positionSymbol].ProcessOrder(orders[newOrderEx].size,
                                                       orders[newOrderEx].ibtraderorder['avgFillPrice'],newOrderEx)
                        filled = True
                        log_stage('process_order_signal_firstposition_SellOrder_filled')
                    elif (datetime.datetime.utcnow() - startTime).seconds > maxFillTime:
                        ibConn.cancelOrder(newOrderEx)                
                        filled = True
                        log_stage('process_order_signal_firstposition_SellOrder_cancelled')
                    time.sleep(0.05)
                log_stage('process_order_signal_firstposition_SellOrder_end')
            
    print("Order Signal Processed")

def RecordTick(timeStamp, bid, bidSize, ask, askSize,last,lastSize):
    global tickData
    tickData.append([timeStamp,bid,bidSize,ask,askSize,last,lastSize])

def RecordSingleTick(timeStamp, price, size, tickType):
    global singleTickData
    singleTickData.append([timeStamp, price, size, tickType])

def StoreMarketData():
    log_stage('store_market_data_START')
    global tickData
    global singleTickData
    global minuteData
    global hourData
    global dataProcessed
    global localStorage
    global s3_StorageLocation
    global ibGWID
    
    hourlyFilename = baseCode+'_H_'+str((datetime.datetime.now()).strftime("%Y%m%d%H%M%S"))
    minsFilename = baseCode+'_M_'+str((datetime.datetime.now()).strftime("%Y%m%d%H%M%S"))
    ticksFilename = baseCode+'_T_'+str((datetime.datetime.now()).strftime("%Y%m%d%H%M%S"))
    singleTicksFilename = baseCode+'_S_'+str((datetime.datetime.now()).strftime("%Y%m%d%H%M%S"))
    ordersFilename = ibGWID+"_Orders_Latest.pkl"
    positionsFilename = ibGWID+"_Positions_Latest.pkl"
    
    with open(localStorage+'hour/'+hourlyFilename,"wb") as f:
        pickle.dump(hourData, f)
    with open(localStorage+'min/'+minsFilename,"wb") as f:
        pickle.dump(minuteData, f)
    with open(localStorage+'ticks/'+ticksFilename,"wb") as f:
        pickle.dump(tickData, f)
    with open(localStorage+'ticks/'+singleTicksFilename,"wb") as f:
        pickle.dump(singleTickData, f)
    with open(localStorage+"ordersandpositions/"+ordersFilename,"wb") as f:
        pickle.dump(orders, f)
    with open(localStorage+"ordersandpositions/"+positionsFilename,"wb") as f:
        pickle.dump(positions, f)
    log_stage('store_market_data_END_pkl')
    try:
        bucket.upload_file(localStorage+hourlyFilename, s3_StorageLocation+hourlyFilename)
        bucket.upload_file(localStorage+minsFilename, s3_StorageLocation+minsFilename)
        bucket.upload_file(localStorage+ticksFilename, s3_StorageLocation+ticksFilename)
        bucket.upload_file(localStorage+singleTicksFilename, s3_StorageLocation+singleTicksFilename)
        bucket.upload_file(localStorage+"OrdersAndPositions/"+ordersFilename, s3_StorageLocation+ordersFilename)
        bucket.upload_file(localStorage+"OrdersAndPositions/"+positionsFilename, s3_StorageLocation+positionsFilename)

        os.remove(localStorage+hourlyFilename)
        os.remove(localStorage+minsFilename)
        os.remove(localStorage+ticksFilename)
        os.remove(localStorage+singleTicksFilename)
        hourData = []
        minuteData = []
        tickData = []
        singleTickData = []
        log_stage('store_market_data_END_boto')
    except:
        log_stage('store_market_data_ERROR_boto')
        print("Error on data storage "+str((datetime.datetime.now()).strftime("%Y%m%d%H%M%S")))
    log_stage('store_market_data_END_FINAL')
            
def LiveDataThread(threadname):
    # Not logging due to excessive generation
    global ibConn
    global cash_contractKey
    global cfd_contractKey
    global currentBid
    global currentAsk
    global currentLast
    global currentBidSize
    global currentAskSize
    global currentLastSize
    
    ibConn.requestMarketData()
    time.sleep(5)

    while CheckDatetime(datetime.datetime.now()):
        if currentBid == 0 or currentBidSize == 0 or currentAsk == 0 or currentAskSize == 0: # or currentLast == 0 or currentLastSize == 0:
            currentBid = ibConn.marketData[cash_contractKey]['bid'].item()
            currentBidSize = ibConn.marketData[cash_contractKey]['bidsize'].item()
            currentAsk = ibConn.marketData[cash_contractKey]['ask'].item()
            currentAskSize = ibConn.marketData[cash_contractKey]['asksize'].item()
            RecordTick(datetime.datetime.utcnow(),currentBid,currentBidSize,currentAsk,currentAskSize,currentLast,currentLastSize)
            RecordSingleTick(datetime.datetime.utcnow(),currentBid,currentBidSize,'Bid')
            RecordSingleTick(datetime.datetime.utcnow(),currentAsk,currentAskSize,'Ask')
        
        elif currentBid != ibConn.marketData[cash_contractKey]['bid'].item():
            currentBid = ibConn.marketData[cash_contractKey]['bid'].item()
            RecordTick(datetime.datetime.utcnow(),currentBid,currentBidSize,currentAsk,currentAskSize,currentLast,currentLastSize)
            RecordSingleTick(datetime.datetime.utcnow(),currentBid,currentBidSize,'Bid')
        elif currentBidSize != ibConn.marketData[cash_contractKey]['bidsize'].item():
            currentBidSize = ibConn.marketData[cash_contractKey]['bidsize'].item()
            RecordTick(datetime.datetime.utcnow(),currentBid,currentBidSize,currentAsk,currentAskSize,currentLast,currentLastSize)
            RecordSingleTick(datetime.datetime.utcnow(),currentBid,currentBidSize,'Bid')
        
        elif currentAsk != ibConn.marketData[cash_contractKey]['ask'].item():
            currentAsk = ibConn.marketData[cash_contractKey]['ask'].item()
            RecordTick(datetime.datetime.utcnow(),currentBid,currentBidSize,currentAsk,currentAskSize,currentLast,currentLastSize)
            RecordSingleTick(datetime.datetime.utcnow(),currentAsk,currentAskSize,'Ask')
        elif currentAskSize != ibConn.marketData[cash_contractKey]['asksize'].item():
            currentAskSize = ibConn.marketData[cash_contractKey]['asksize'].item()
            RecordTick(datetime.datetime.utcnow(),currentBid,currentBidSize,currentAsk,currentAskSize,currentLast,currentLastSize)        
            RecordSingleTick(datetime.datetime.utcnow(),currentAsk,currentAskSize,'Ask')
        time.sleep(0.005)   

def ScheduleProcessor(threadname):    
    global ibConn
    global baseCode

    RegisterSchedule()

    while True:
        if ibConn.connected:
            schedule.run_pending()
        else:
            ibConn.connect(clientId=clientId, host=host, port=port)        
        time.sleep(0.2)

def PrintTicks(threadname):
    global currentBid
    global currentAsk
    global currentLast
    global currentBidSize
    global currentAskSize
    global currentLastSize
    global tickData
    global singleTickData
    global minuteData
    global hourData

    while CheckDatetime(datetime.datetime.now()):
        print(str(len(hourData))+"\t"+str(len(minuteData))+"\t"+str(len(tickData))+"\t"+str(len(singleTickData)))
        #print(str(currentBid)+"\t"+str(currentBidSize)+"\t"+str(currentAsk)+"\t"+str(currentAskSize))        
        time.sleep(2)

def LoadModel():
    log_stage('model_load_START')
    global baseCode
    global modelParamsPath
    global model_feat_map
    global model_norm_params
    global model
    
    model_feat_map, model_norm_params, model = load_all_the_things(modelParamsPath+baseCode+'/')
    log_stage('model_load_END')

def sql_read_ts(dbHost, dbPort, user, passwd, symbol, connect_timeout, query):
    log_stage('sql_read_ts_Start')
    database = mysql.connect(host=dbHost,port=dbPort,user=user,passwd=passwd,database=symbol,connect_timeout=connect_timeout)
    dbConnection = database.cursor()
    dbConnection.execute(query)
    log_stage('sql_read_ts_query_execute')
    df =  pd.DataFrame(dbConnection.fetchall())
    log_stage('sql_read_ts_data_retrieved')
    df.columns = ["Timestamp"]
    df = df.drop_duplicates(subset='Timestamp')
    log_stage('sql_read_ts_data_returned')
    return pd.to_datetime(df['Timestamp'])

def DatabaseFill(requestPeriod='6 W', timeperiod=600):
    global baseCode
    global dbHostContainer
    global dbHostPort
    global user
    global passwd
    
    log_stage('databasefill_start')
    starttime = datetime.datetime.now()
    print(starttime)
    print()    

    print(f"\nChecking {baseCode} for inconsistencies...", end='')
    # Find the start of the last hour.
    # Further lookbacks are incompatible with being included on a startup script...we may need a
    # modified system to recover from a multiple-hour failure.
    now = datetime.datetime.now()
    then = now - datetime.timedelta(hours=600)
    then_formatted = then.strftime("%Y-%m-%d %H:00:00")
    
    log_stage('databasefill_retrieve_timestamps_start')
    # Get all valid timestamps in our DB for the few hours
    query = f"SELECT Timestamp FROM {baseCode}_Hour WHERE Timestamp >= '{then_formatted}';"
    hour_ts = sql_read_ts(dbHostContainer,dbHostPort,user,passwd,baseCode, 2, query).tolist()
    query = f"SELECT Timestamp FROM {baseCode}_Min WHERE Timestamp >= '{then_formatted}';"
    min_ts = sql_read_ts(dbHostContainer,dbHostPort,user,passwd,baseCode, 3, query).tolist()
    log_stage('databasefill_retrieve_timestamps_end')
    
    # Grab data
    log_stage('databasefill_retrieve_request_minutes_start')
    ibConn.historicalData = {}
    minDF = RequestSpecificMarketData(datetime.datetime.utcnow(), "1 Min", requestPeriod)

    minDF.drop(['V','OI','WAP'],1,inplace=True)
    minDF = minDF.sort_values('datetime').reset_index()
    minDF['datetime'] = pd.to_datetime(minDF['datetime'])
    log_stage('databasefill_retrieve_request_minutes_end')

    # Filter existing values
    minDF = minDF.loc[~minDF['datetime'].isin(min_ts)].reset_index(drop=True)
    minDF['datetime'] = minDF['datetime'].astype(str)
    log_stage('databasefill_retrieve_filter_minutes_complete')

    minDF['Vol'] = 0

    minDF = minDF[['datetime', 'O', 'H', 'L', 'C', 'Vol']]
    
    # Store them mofos
    log_stage('databasefill_retrieve_store_minutes_start')
    for commitData in list(np.array_split(minDF,100)):
        vals = [tuple(x) for x in commitData.values]
        sql = "INSERT INTO "+baseCode+"_Min (Timestamp, Open, High, Low, Close, Vol) VALUES (%s, %s, %s, %s, %s, %s)"
        database = mysql.connect(host=dbHostContainer,port=dbHostPort,user=user,passwd=passwd,database=baseCode)
        dbConnection = database.cursor()
        dbConnection.executemany(sql, vals)
        database.commit()
    log_stage('databasefill_retrieve_store_minutes_end')
    # Rinse and repeat
    log_stage('databasefill_retrieve_request_hours_start')
    hourDF = RequestSpecificMarketData(datetime.datetime.utcnow(), "1 Hour", requestPeriod)

    hourDF.drop(['V','OI','WAP'],1,inplace=True)
    hourDF = hourDF.sort_values('datetime').reset_index()
    hourDF['datetime'] = pd.to_datetime(hourDF['datetime'])
    hourDF = hourDF.loc[~hourDF['datetime'].isin(hour_ts)].reset_index(drop=True)
    hourDF['datetime'] = hourDF['datetime'].astype(str)
    log_stage('databasefill_retrieve_request_hours_end')

    hourDF = hourDF[['datetime', 'O', 'H', 'L', 'C']]

    log_stage('databasefill_retrieve_store_hours_start')
    for commitData in list(np.array_split(hourDF,100)):
        vals = [tuple(x) for x in commitData.values]
        sql = "INSERT INTO "+baseCode+"_Hour (Timestamp, Open, High, Low, Close) VALUES (%s, %s, %s, %s, %s)"
        database = mysql.connect(host=dbHostContainer,port=dbHostPort,user=user,passwd=passwd,database=baseCode)
        dbConnection = database.cursor()
        dbConnection.executemany(sql, vals)
        database.commit()    
    
    log_stage('databasefill_retrieve_request_hours_end')

def main():
    global ibConn
    global baseCode
    global cash_contractKey 
    global cfd_contractKey
    
    print("Connecting to IB Gateway")
    print("Client ID: "+str(clientId))
    print("Host ID: "+str(host))
    ibConn = IBTrader.IBTrader()    
    ibConn.connect(clientId=clientId, host=host, port=port)
    time.sleep(2)

    ibConn.contracts = {}
    time.sleep(5)
    ibConn.contracts = {}        
    ibConn.createCashContract(baseCode[:3], currency=baseCode[3:])
    ibConn.createFXCFDContract(baseCode[:3], currency=baseCode[3:])
    cash_contractKey = int([contractId for contractId in ibConn.contracts if ibConn.contracts[contractId].m_exchange == 'IDEALPRO'][0])
    cfd_contractKey = int([contractId for contractId in ibConn.contracts if ibConn.contracts[contractId].m_exchange == 'SMART'][0])
    print("Adding: "+baseCode)
    print("Contracts Processing: "+str(len(ibConn.contracts)))
    #for contract in ibConn.contracts:
    #    print(contract)

    #print("Checking Database")
    #DatabaseFill()

    print("Loading Current Model")
    LoadModel()

    while not CheckDatetime(datetime.datetime.now()):
        print("Awaiting Market Opening Times", end="\r")
        time.sleep(60)
    
    liveMDThread = Thread(target=LiveDataThread, args=("LiveMDThread",))
    scheduleProcess = Thread(target=ScheduleProcessor, args=("ScheduleProcess",))
    #printTicks = Thread(target=PrintTicks, args=("PrintTicks",))
    
    liveMDThread.start()
    scheduleProcess.start()
    #printTicks.start()
    
    liveMDThread.join()
    scheduleProcess.join()
    #printTicks.join()
    
    print("Market Closed")
    print("Processor Shutting Down")
    ibConn.cancelHistoricalData()
    ibConn.cancelMarketData()
    ibConn.contracts = {}
    ibConn.disconnect()
    print("IB Connection Closed")

    StoreMarketData()

    print("Processor Complete")
    exit()

if __name__ == '__main__':    
    print("Initialising")
    boto3.set_stream_logger('', logging.INFO)

    ### Should this be here?
    warnings.filterwarnings("ignore")
    #os.makedirs('/logs/', exist_ok=False)

    print("Retrieving Market Symbol Code")
    baseConfigInfo = pd.read_csv("/BaseConfigInfo.csv")
    #baseConfigInfo = pd.read_csv("/general/legiotrader/DockerFiles/docker_env/LegioContainerFiles/BaseConfigInfo.csv")
    containerIP = socket.gethostbyname(socket.gethostname())
    #containerIP = "172.50.0.57"
    
    thisContainer = baseConfigInfo.loc[baseConfigInfo['IPAddress']==containerIP]
    baseCode = thisContainer['Symbol'].item()
    activeTrader = thisContainer['ActiveTrading'].item()
    clientId = thisContainer['IB Client ID'].item()
    ibGWID = thisContainer['IBAccount_Data'].item()
    modelParamsPath = thisContainer['ParamsPath'].item()
    approvedPositionSize = thisContainer['Position'].item()

    host = thisContainer['IBGatewayIP'].item()
    #host = "10.1.1.194"
    #host = "ec2-54-252-236-54.ap-southeast-2.compute.amazonaws.com"

    main()