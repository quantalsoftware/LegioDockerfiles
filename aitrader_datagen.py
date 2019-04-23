import mysql.connector as mysql
import numpy as np
import pandas as pd

# from dask import delayed, compute
from itertools import chain, product
from tenacity import retry
from tech_indicators import get_function_lookup

def dst_open_hrs_subset(temp):
    temp = temp.tz_convert('Australia/Sydney')
    syd_time = temp.index
    temp.loc[(syd_time.dayofweek==0)&(syd_time.hour==7), 'open_hrs'] = 1
    temp.loc[(syd_time.dayofweek==0)&(syd_time.hour==6), 'open_hrs'] = -99
    temp = temp.tz_convert('US/Eastern')
    nyc_time = temp.index
    temp.loc[(nyc_time.dayofweek==4)&(nyc_time.hour==17), 'open_hrs'] = 1
    temp.loc[(nyc_time.dayofweek==4)&(nyc_time.hour==18), 'open_hrs'] = -99
    temp = temp.tz_convert('UTC')
    temp['open_hrs'].interpolate(inplace=True)
    temp = temp[temp['open_hrs']>0]
    temp.drop('open_hrs', axis=1, inplace=True)
    return temp

#@retry
def sql_read(dbHost, dbPort, user, passwd, symbol, connect_timeout, query):
    try:
        #connection = mysql.connect(host=dbHost,port=dbPort,user=user,passwd=passwd,
        #                      database=symbol, connect_timeout=connect_timeout)
        #return pd.read_sql_query(query, connection, parse_dates= {'Timestamp': '%Y-%m-%d %H:%M:%S', 'TimeStamp': '%Y-%m-%d %H:%M:%S'})
        database = mysql.connect(host=dbHost,port=dbPort,user=user,passwd=passwd,database=symbol,connect_timeout=connect_timeout)

        dbConnection = database.cursor()
        dbConnection.execute(query)
        return pd.DataFrame(dbConnection.fetchall())
    except KeyboardInterrupt:
        print("Stopping code.")
        return 'kbi'
    except mysql.OperationalError:
        print("Timeout occured. Retrying connection...")
        """
        Logging code here
        """
        raise Exception
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        return 'other'

def readin(db_location, symbol, inference=True, timeperiod= '60T'):
    colnames = {'TimeStamp':'Timestamp', 'O':'Open', 'H':'High', 'L':'Low', 'C':'Close', 'V':'Volume'}
    aggnames = {'median':'mid', 'mean':'avg', 'std':'spread'}

    dbHost = db_location
    dbPort = "31330"
    user = "root"
    passwd = "root"

    if inference:
        query = f"SELECT DISTINCT * FROM {symbol}_Hour ORDER BY Timestamp DESC LIMIT 600;"
        connect_timeout = 5
    else:
        query = f"SELECT DISTINCT * FROM {symbol}_Hour WHERE Timestamp >= '2014-01-01 00:00:00';"
        connect_timeout = 45
    #print(f'{symbol}')

    hours = sql_read(dbHost, dbPort, user, passwd, symbol, connect_timeout, query)
    if type(hours) == str:
        raise KeyboardInterrupt

    hours.columns = ["Timestamp","Open","High","Low","Close"]
    hours = hours.drop_duplicates(subset='Timestamp')
    hours['Timestamp'] = pd.to_datetime(hours['Timestamp'])
    hours = hours.set_index('Timestamp').sort_index().resample('60T').nearest().astype(np.float32)
    

    query = f"SELECT DISTINCT Timestamp, Open, Close FROM {symbol}_Min WHERE Timestamp >= '{hours.index[0]}';"

    mins = sql_read(dbHost, dbPort, user, passwd, symbol, connect_timeout, query)
    if type(mins) == str:
        if mins == 'kbi':
            raise KeyboardInterrupt
        else:
            raise Exception    

    mins.columns = ["Timestamp","Open","Close"]
    mins = mins.drop_duplicates(subset='Timestamp')
    mins['Timestamp'] = pd.to_datetime(mins['Timestamp'])
    mins = mins.set_index('Timestamp').astype(np.float32).mean(axis=1).resample('60T').agg(['median', 'mean', 'std']).rename(columns=aggnames)

    temp = hours.loc[mins.index].merge(mins, left_index=True, right_index=True, how='inner').astype(np.float32).tz_localize('UTC')

    if inference:
        if temp.last('1D').isnull().sum().sum()>0:
            raise ValueError('NaN encountered in observation')
            """
            Logging code here
            """
            #return None
    if not inference:
        print(temp.shape)
    
    return temp

def readin_raw(db_location, symbol, timeperiod= '60T'):
  
  
  colnames = {'TimeStamp':'Timestamp', 'O':'Open', 'H':'High', 'L':'Low', 'C':'Close', 'V':'Volume'}
  aggnames = {'median':'mid', 'mean':'avg', 'std':'spread'}

  dbHost = db_location
  dbPort = "31330"
  user = "root"
  passwd = "root"


  query = f"SELECT DISTINCT * FROM {symbol}_Hour ORDER BY Timestamp DESC LIMIT 5;"
  connect_timeout = 2
    
  hours = sql_read(dbHost, dbPort, user, passwd, symbol, connect_timeout, query)
  if type(hours) == str:
    raise KeyboardInterrupt

  #hours = hours.rename(columns=colnames).drop_duplicates(subset='Timestamp')  
  hours = pd.DataFrame(hours, columns=["Timestamp","Open","High","Low","Close"]).drop_duplicates(subset='Timestamp')
  hours = hours.set_index('Timestamp').sort_index().resample('60T').nearest().astype(np.float32)

  query = f"SELECT Timestamp, Open, Close FROM {symbol}_Min WHERE Timestamp >= '{hours.index[0]}';"

  mins = sql_read(dbHost, dbPort, user, passwd, symbol, connect_timeout, query)
  if type(mins) == str:
    if mins == 'kbi':
      raise KeyboardInterrupt
    else:
      raise Exception
  #mins = mins.rename(columns=colnames).drop_duplicates(subset='Timestamp') 
  mins = pd.DataFrame(mins, columns=["Timestamp","Open","Close"]).drop_duplicates(subset='Timestamp')
  mins = mins.set_index('Timestamp').astype(np.float32).mean(axis=1).resample('60T').agg(['median', 'mean', 'std']).rename(columns=aggnames)

  temp = hours.loc[mins.index].merge(mins, left_index=True, right_index=True, how='inner').astype(np.float32).tz_localize('UTC')


  if temp.last(1).isnull().sum().sum()>0:
    raise ValueError('NaN encountered in observation')
    return None
  return temp.tail(1)

# @delayed
# def d_readin(db_location, symbol, inference=True, timeperiod= '60T'):
#   return readin(db_location, symbol, inference, timeperiod)

def wrap_function(df, function, high_col='High', low_col='Low', close_col='Close', volume_col=None):
  return function(df[high_col], df[low_col], df[close_col], volume=None)

def concat_frames(df, tis, symbol, dropcols):
  if len(dropcols) > 0:
    df.drop(dropcols, axis=1, inplace=True)
  df = pd.concat([df.diff()]+tis, axis=1)
  return df.add_prefix(f"{symbol}-").astype(np.float32)

def add_rolling(df, n, suffix):
  if suffix == 'roll':
    return df.add_suffix(f'-timediff_{n}_roll').rolling(n, min_periods=0).mean().astype(np.float32)
  elif suffix == 'ewm':
    return df.add_suffix(f'-timediff_{n}_ewm').ewm(span=n).mean().astype(np.float32)

def add_shifts(df, n):
  return df.shift(n).add_suffix(f'-timediff_{n}_shift')

def load_data(db_location,feat_map,inference = True):
    symbols = list(feat_map.keys())
    fn_lookup = get_function_lookup()
    suffixes = ['roll', 'ewm']
    
    dfs = [readin(db_location, symbol, inference) for symbol in symbols]
    tis = [[wrap_function(df, fn_lookup[function]) for function in feat_map[symbol]['t_i']] for df, symbol in zip(dfs, symbols)]
    job = [concat_frames(df, ti, symbol, feat_map[symbol]['main_drop']) for df, ti, symbol in zip(dfs, tis, symbols)]
    job = [[add_shifts(df, n) for n in feat_map[symbol]['shift']] for df, symbol in zip(job, symbols)]+[[add_rolling(df, n, s) for n, s in product(feat_map[symbol]['window'], suffixes)] for df, symbol in zip(job, symbols)]
    data = pd.concat(list(chain(*job)), axis=1)
    
    return data

def gen_train_dset(
  key_pair, # 'C:/Users/alexs/Downloads/fx_data/target_report/AUDUSD60T.csv',
  start_date,
  feat_map,
  norm_params,
  target_col= 'mid'
            ):
  """
  Returns a processed training dataset

  """
  
  bincols = ['volatility_bbhi', 'volatility_bbli', 'volatility_dchi', 'volatility_dcli',
          'volatility_kchi', 'volatility_kcli']
  
  
  ##### This will need to be rewritten for the database 
  
  backtest = readin(norm_params['data_loc'], key_pair, inference=False)
  
  norm_params['base_column'] = target_col
  norm_params['key_pair'] = key_pair
  
  X = load_data(norm_params['data_loc'], feat_map, inference=False).resample('60T').asfreq()
  if target_col in ['mid', 'avg']:
    tgt = backtest[target_col].diff().resample('60T').asfreq().shift(-1).rename('target')
  else:
    tgt = norm_params['custom_feature']['in_tx'](backtest).resample('60T').asfreq().shift(-1).rename('target')
  X = pd.concat([X, tgt], axis=1)[start_date:]
  X = X.dropna()
  y = X['target']
  norm_params['target_scale'] = y.abs().quantile(norm_params['target_norm_quantile'])
  y = y.div(norm_params['target_scale']).clip(
    lower=-norm_params['target_clip_range'], upper=norm_params['target_clip_range'])
  X.drop('target', axis=1, inplace=True)
  
  log_exceptions = X.columns[[(item[1] in bincols) for item in X.columns.str.split('-')]].tolist()

  # lambda x: np.sign(x)*np.log(np.abs(x)+1) is the symmetric log transform, which helps to flatten
  # out the Cauchy/double exponential distribution inherent to a lot of our data.
  
  X = pd.concat([X.drop(norm_params['lin_exceptions'], axis=1).add_suffix("-lin"),
                X.drop(norm_params['log_exceptions']+log_exceptions, axis=1).apply(
                  lambda x: np.sign(x)*np.log(np.abs(x)+1)).add_suffix('-log').astype(np.float32),
                ],
                axis=1, copy=False).astype(np.float32)

  norm = X.abs().quantile(norm_params['data_norm_quantile'])
  norm[norm==0] = 1
  norm_params['scale'] = norm
  X = X.div(norm_params['scale']).clip(lower=-norm_params['data_clip_range'], upper=norm_params['data_clip_range'])

  assert X.isnull().sum().sum() == 0
  
  embed_uniques = []
  
  if norm_params['timecols']['month_start']:
    X['TMSTMP-is_month_start-timediff_0_shift-lin'] = X.index.is_month_start.astype(np.float32)
    
  if norm_params['timecols']['month_end']:
    X['TMSTMP-is_month_end-timediff_0_shift-lin'] = X.index.is_month_end.astype(np.float32)
    
  if norm_params['timecols']['quarter_start']:
    X['TMSTMP-is_quarter_start-timediff_0_shift-lin'] = X.index.is_quarter_start.astype(np.float32)
    
  if norm_params['timecols']['quarter_end']:
    X['TMSTMP-is_quarter_end-timediff_0_shift-lin'] = X.index.is_quarter_end.astype(np.float32)
    
  if norm_params['timecols']['weekday']:
    weekdays = X.index.dayofweek.unique().tolist()
    weekday_dict= {i:n for n, i in enumerate(weekdays)}
    X['TMSTMP-weekday-timediff_0_shift-lin'] = X.index.dayofweek.astype(np.float32).map(
      weekday_dict).astype(np.float32)
    embed_uniques += [X['TMSTMP-weekday-timediff_0_shift-lin'].nunique()]
    
  if norm_params['timecols']['hour']:
    X['TMSTMP-hour-timediff_0_shift-lin'] = X.index.hour.astype(np.float32)
    embed_uniques += [X['TMSTMP-hour-timediff_0_shift-lin'].nunique()]

  embed_sizes = [min(30, round(item/2)) for item in embed_uniques]
  
  norm_params['column_order'] = X.columns
  norm_params['nn_params']['embed_uniques'] = embed_uniques
  norm_params['nn_params']['embed_sizes'] = embed_sizes
  if norm_params['timecols']['weekday']:
    norm_params['nn_params']['embed_weekdays'] = weekday_dict
  norm_params['nn_params']['n_in'] = X.shape[1]

  return X.astype(np.float32), y, backtest, norm_params

def gen_inference_dset(inference_file_loc, feat_map, norm_params):
    bincols = ['volatility_bbhi', 'volatility_bbli', 'volatility_dchi', 'volatility_dcli',
          'volatility_kchi', 'volatility_kcli']
    try:
        X = load_data(norm_params['data_loc'], feat_map)
        X = X.tail()
        log_exceptions = X.columns[[(item[1] in bincols) for item in X.columns.str.split('-')]].tolist()
        X = pd.concat([X.drop(norm_params['lin_exceptions'], axis=1).add_suffix("-lin"),
                      X.drop(norm_params['log_exceptions']+log_exceptions, axis=1).apply(
                        lambda x: np.sign(x)*np.log(np.abs(x)+1)).add_suffix('-log').astype(np.float32),
                      ],
                      axis=1, copy=False).astype(np.float32)
        X = X.div(norm_params['scale']).clip(lower=-norm_params['data_clip_range'], upper=norm_params['data_clip_range'])
        assert X.isnull().sum().sum() == 0

        if norm_params['timecols']['month_start']:
            X['TMSTMP-is_month_start-timediff_0_shift-lin'] = X.index.is_month_start.astype(np.float32)
        if norm_params['timecols']['month_end']:
            X['TMSTMP-is_month_end-timediff_0_shift-lin'] = X.index.is_month_end.astype(np.float32)
        if norm_params['timecols']['quarter_start']:
            X['TMSTMP-is_quarter_start-timediff_0_shift-lin'] = X.index.is_quarter_start.astype(np.float32)
        if norm_params['timecols']['quarter_end']:
            X['TMSTMP-is_quarter_end-timediff_0_shift-lin'] = X.index.is_quarter_end.astype(np.float32)
        if norm_params['timecols']['weekday']:
            weekdays = X.index.dayofweek.unique().tolist()
            X['TMSTMP-weekday-timediff_0_shift-lin'] = X.index.dayofweek.astype(np.float32).map(norm_params['nn_params']['embed_weekdays']).astype(np.float32)
        if norm_params['timecols']['hour']:
            X['TMSTMP-hour-timediff_0_shift-lin'] = X.index.hour.astype(np.float32)
        assert X.shape[1] == norm_params['nn_params']['n_in']

        return X[norm_params['column_order']].astype(np.float32).tail(1)
    except Exception as e:
        print('Error encountered.')
        print(e)
        """
        LOGGING CODE HERE
        """
        return 'Error'