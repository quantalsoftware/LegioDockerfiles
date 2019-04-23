import pickle
import os
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.utils.data as data_utils

#from aitrader_model import FXModel, gen_new_pred
#from sklearn.metrics import r2_score


"""
Suggested initial params:
base_symbols = ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDCAD', 'NZDUSD', 'USDCHF', 'USDJPY']
ta_cols = list(get_all_ta_cols()) # imported from the main helper file
fx_ta_cols = [col for col in ta_cols if (col[:6]!='volume')&(col not in ['momentum_mfi'])]
lags = [12, 24, 48, 72]
steps = 6
"""


def gen_init_feat_map(base_symbols, ta_cols, n_shift, windows):
  if type(n_shift) == int:
    shifts = range(n_shift)
  elif type(n_shift) == float:
    shifts = range(round(n_shift))
  elif type(n_shift) == list:
    shifts = n_shift
  else:
    raise ValueError("Numeric or list of numeric only")
  return {symbol:{'t_i':ta_cols,
                  'main_drop': [],
                  'shift':[n for n in shifts],
                  'window':windows}
  for symbol in base_symbols}

def gen_init_norm_params(data_loc): # 'localhost', "ec2-54-252-172-185.ap-southeast-2.compute.amazonaws.com",
  return {'custom_feature': {},
          'data_loc': data_loc,
          'data_clip_range': 2.5,
          'data_norm_quantile': 0.999,
          'decision_threshold': 2.5,
          'target_clip_range': 1.1,
          'target_norm_quantile': 0.995,
          'lin_exceptions': [],
          'log_exceptions': [],
          'timecols':{
            'month_start': True,
            'month_end': True,
            'quarter_start': True,
            'quarter_end': True,
            'weekday': True,
            'hour': True
          },
          'nn_params':{
            'n_hidden':512,
            'l1_coef': 0,
            'l1w_coef': 0,
            'cov_coef': 0,
            'learnrate': 1e-2,
            'lrmin': 3e-3,
            'iters': 5,
            'epochs': 5
          }
         }

def gen_backtest_markers():
  years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
  quarters = [('01', '03', 'Q3'), ('04', '06', 'Q4'), ('07', '09', 'Q1'), ('10', '12', 'Q2')]
  train_blocks = []
  test_blocks = []
  quarter_marker = []
  for i in range(len(years)-4):
    for j in range(len(quarters)):
      start_year = years[i]
      end_year = years[i+2]
      val_year = years[i+3]
      start_month = quarters[j][0]
      if j == 0:
        end_year = years[i+2]
        val_year = years[i+3]
        end_month = quarters[3][1]
      else:
        end_year = val_year = years[i+3]
        end_month = quarters[j-1][1]
      val_month = quarters[j][1]
      train_chunk = (start_year+'-'+start_month, end_year+'-'+end_month)
      test_chunk = (val_year+'-'+start_month, val_year+'-'+val_month)
      train_blocks.append(train_chunk)
      test_blocks.append(test_chunk)
      qtr = quarters[j][2]
      if qtr in ['Q3', 'Q4']: 
        quarter_marker.append(years[i+2]+qtr)
      else:
        quarter_marker.append(years[i+3]+qtr)
  return train_blocks, test_blocks, quarter_marker

def cov(x):
  """
  This is used to calculate a custom covariance penalty I like to throw into a lot of models.
  It seems to help the model learn from heterogenous or imbalanced data, and I've used it in
  some bioinformatics models to train powerful models from a few hundred observations. In theory,
  it decouples the "smearing" tendency of dropout from optimal results, forcing each unit to be
  both useful and unique, thus decorrelating errors and providing something of an ensembling effect.
  Works best when combined with a mild L1 penalty (which locks some elements of the vectors to 0).
  
  When I get some spare time I will build a custom Adam wrapper that implements AdamW for custom
  regularisers; for now we eat the test performance hit.
    
    x: weights tensor to be regularised.
  """
  fact = 1.0 / (x.size(1) - 1)
  m = x - torch.mean(x, dim=1, keepdim=True)
  mt = m.t()
  return fact * m.matmul(mt).mean()

def train(model, train_loader, optimiser, sched, norm_params_nn, USE_CUDA):
  """
  Wraps the training loop, applies regularisation and learning rate annealing (within epoch).
  Uses a combination of L1 and L2 loss to yield a robust estimator of the function (L1) and to
  enforce symmetry (L2). The L2 is naturally downweighted by the squaring term in most cases; leading
  the L1 to dominate estimation except in the extreme tails.
  
    model: instantiated Pytorch model
    train_loader: loader with the training dataset
    optimiser: the optim class paired with the model params
    sched: scheduler parameterised with the optimiser
    USE_CUDA: flag for GPU acceleration
    
  """
  model.train()
  run_loss = 0
  norm = 0
  first = True
  len_dl = len(train_loader.dataset)
  for n, (data, target) in enumerate(train_loader):
    if data.size()[0]<2: #safety for the batch norm as it depends on n>1 obs
      continue
    if USE_CUDA:
      data = data.cuda(); target = target.cuda()
    optimiser.zero_grad()
    output = model(data)
    loss = F.l1_loss(output, target)
#     loss += F.mse_loss(output, target)
    run_loss += loss.item()
    loss += norm_params_nn['l1_coef'] * torch.norm(model.betas, 1)
    if (norm_params_nn['cov_coef'] != 0)&(norm_params_nn['l1w_coef'] != 0):
      for name, W in model.named_parameters():
        if ('lin.weight' in name):
          loss += norm_params_nn['cov_coef'] * cov(W)
          loss += norm_params_nn['l1w_coef'] * torch.norm(W, 1)
    norm +=1
    loss.backward()
    optimiser.step()
    sched.step()

  train_loss = run_loss / norm  
  return run_loss / norm

def test(model, test_loader, USE_CUDA=True):
  """
  Test validation on .eval() mode. Returns absolute loss and R2 for loss-scale agnostic
  predictive assessment.
  
    model: instantiated Pytorch model
    test_loader: loader with the test/validation dataset
    USE_CUDA: flag for GPU acceleration
  """
  model.eval()
  test_loss = 0
  correct = 0
  target_list = []
  output_list = []
  with torch.no_grad():
    for data, target in test_loader:
      target_list += target.numpy().tolist()
      if USE_CUDA:
        data, target = data.cuda(), target.float().cuda()
      output = model(data)
      output_list += output.cpu().numpy().tolist()
      test_loss += F.l1_loss(output, target, reduction= 'sum')
  test_loss /= len(test_loader.dataset)
  o = np.array(output_list)
  t = np.array(target_list)
  r2d2 = r2_score(t, o) # ignore this naming convention; I think I was drunk and it amused me
  return test_loss, r2d2

def train_model(model, X_train, y_train, X_test, y_test, norm_params):
  """
  Main model training function; wrapped so as to allow for "for loop" training and 
  generation of backtests. Test data is not used for early stopping; rather it's used
  to compare how R2 varies with backtest performance.
  
    model: instantiated Pytorch model
    X_train, y_train, X_test, y_test: input data
    is_check: plot loss; suppressed for model validation loops/backtest generation
    LR: initial learning rate
    lrmin: minimum learning rate that the annealer will go down to
    iters: number of batch size annealing cycles to utilise
  """
  
  # Dataloaders defined in function to ensure cleanup
  train_data = data_utils.TensorDataset(torch.from_numpy(X_train.values.astype(np.float32)),
                                     torch.from_numpy(y_train.values.astype(np.float32)))
  test_data = data_utils.TensorDataset(torch.from_numpy(X_test.values.astype(np.float32)),
                                     torch.from_numpy(y_test.values.astype(np.float32)))
  USE_CUDA = True
  if USE_CUDA:
    model.cuda()
  test_dl = data_utils.DataLoader(test_data, batch_size=2048, shuffle=False, num_workers = 6)
  train_log = []
  test_log = []
  pseudoval_log = []
  r2_best = 0
  best_loss=100
  best_round = 0
  best_epoch = 0
  beta_out = None
  model_params = None
  optimiser = torch.optim.Adam(model.parameters(), lr=norm_params['nn_params']['learnrate'],
                              weight_decay= 0)
  print('', end='')
  for n in range(norm_params['nn_params']['iters']):
    for epoch in range(norm_params['nn_params']['epochs']):
      print("\rRound {} : Epoch {}   ---   Best loss: {:.6f} Best R2: {:.4f} @ {}:{}".format(
        n+1, epoch+1, best_loss, r2_best, best_round+1, best_epoch+1),
            end='')
      bs = 2**min(7+epoch,12) # batch rate annealing here
      train_dl = data_utils.DataLoader(train_data, batch_size=bs, shuffle=True, num_workers = 6)
      sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, len(train_dl.dataset),
                                                         norm_params['nn_params']['lrmin'])
      train_loss = train(model, train_dl, optimiser, sched, norm_params['nn_params'], USE_CUDA)
      test_loss, r2d2 = test(model, test_dl, USE_CUDA)
      pseudoval_loss, _ = test(model, train_dl, USE_CUDA)
      train_log.append(train_loss)
      test_log.append(test_loss)
      pseudoval_log.append(pseudoval_loss)
      if test_loss < best_loss:
        best_round = n
        best_epoch = epoch
        r2_best = r2d2
        best_loss = test_loss.item()
        model_params = model.state_dict().copy()
        beta_out = pd.Series(model.betas.detach().cpu().numpy(), index= X_train.columns).abs()
      
  print("\rBest loss: {:.5f}  Best R2: {:.5f} @ {}:{}".format(best_loss, r2_best, best_round+1, best_epoch+1))
  return model.state_dict(), beta_out, best_loss, r2_best, train_log, test_log, pseudoval_log, model_params

def gen_backtest(model_params, backtest, X_test, y_test, norm_params, plot=True, key_thresh=2.5, default_slip= 0.3):
  """
  Takes test data, raw target data and several descriptive labels and outputs a backtest datafile,
  ready for analysis.
  
    model: instantiated Pytorch model
    target: the target dataframe consisting high fidelity data
    X_test, y_test: input data
    pred_index: original index for y_test, used to reindex predictions and to subset target df
    thresh1: rescaling constant for y
    key_pair: label indicating target FX cross
    qm: quarter marker. Indicates which quarter of what financial year is currently being processed
    r2d2: R2 score, recorded for comparison. Again, apologies for Drunk Alex's sense of humor.
  """

  pred_dl = data_utils.DataLoader(
  data_utils.TensorDataset(
    torch.from_numpy(X_test.values.astype(np.float32)),
    torch.from_numpy(y_test.values.astype(np.float32))
  ),batch_size=2048, shuffle=False, num_workers = 6)
  model = FXModel(n_hidden= norm_params['nn_params']['n_hidden'],
                  n_in= norm_params['nn_params']['n_in'],
                  x_uniques= norm_params['nn_params']['embed_uniques'],
                  x_sizes= norm_params['nn_params']['embed_sizes'])
  model.load_state_dict(model_params)
  model.eval()
  collect = []
  with torch.no_grad():
    for dset, _ in pred_dl:
      collect+= model(dset).numpy().tolist()
  preds = pd.Series(collect, index=y_test.index, name='outputs')
  
  backtest['outputs'] = preds
  
  backtest.loc[(backtest['Open'].diff()==0)&(backtest['Close'].diff()==0), ['Open', 'Close']] = np.nan
  
  backtest['outputs_scaled'] = preds.mul(norm_params['target_scale'])
  if norm_params['base_column'] in ['mid', 'avg']:
    backtest['predicted_mid'] = backtest[norm_params['base_column']] + backtest['outputs_scaled']
    backtest['expected_delta'] = (backtest['predicted_mid'].shift(1) - backtest['Open']).fillna(0)
  else:
    backtest['expected_delta'] = norm_params['custom_feature']['out_tx'](backtest)
  backtest['actual_delta'] = backtest['Close'].sub(backtest['Open']).fillna(0)
  backtest['sign_test'] = backtest['expected_delta'].apply(np.sign) *\
  backtest['actual_delta'].apply(np.sign)
  if 'JPY' in norm_params['key_pair']:
    backtest[['expected_delta', 'actual_delta']] /=100
  backtest_est = backtest['actual_delta'].mul(np.sign(backtest['expected_delta'])).copy()

  backtest_est -= 0.0001*(0.4+default_slip)
  backtest_est.loc[backtest['expected_delta'].abs()<0.0001*key_thresh] = 0
  print("Estimated backtest value @ {} thresh, 0.3 slip: {:.6f} --- p-q: {:.4f}".format(
    key_thresh, backtest_est.sum(),backtest.sign_test[backtest_est!=0].mean()))
  if plot:
    plt.figure(figsize=(5,5))
    plt.scatter(preds, y_test, s=2, alpha=0.5)
    plt.plot([-1.2, 1.2], [-1.2, 1.2], c='k', alpha=0.5)
    plt.axhline(0, c='r', alpha=0.5)
    plt.axvline(0, c='r', alpha=0.5)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.show()
    fig, ax = plt.subplots(1, 1, figsize= (20, 5))
    for thresh in [1.5, 2.0, 2.5]:
      for slip in [0.1, 0.3, 0.5]:
        backtest_est2 = backtest['actual_delta'].mul(np.sign(backtest['expected_delta'])).copy()
        signtest = backtest['actual_delta'].apply(np.sign).mul(np.sign(backtest['expected_delta']))
        backtest_est2 -= 0.0001*(0.4+slip)
        backtest_est2.loc[backtest['expected_delta'].abs()<0.0001*thresh] = 0
        backtest_est2.rename("{} - {} - {:.4f}".format(
          slip, thresh, signtest[backtest_est2!=0].mean())).cumsum().plot(ax=ax)
    ax.legend(loc=2)
    ax.axhline(0, alpha=0.5, c='k')
    plt.show()
  return backtest, backtest_est.sum()
  
  
def train_model_eda(X, y, norm_params, backtest= None, tts=0.8, n_repeats=3, plot=True):
  if n_repeats > 10:
    print("Reducing number of repeats")
    n_repeats = 10
    
  norm_params_nn = norm_params['nn_params']
  breakpoint = int(X.shape[0]*tts)
  X_train, X_test = X[:breakpoint], X[breakpoint:]
  y_train, y_test = y[:breakpoint], y[breakpoint:]
  betas_log = []
  backtests_log = []
  losses_log = []
  best_loss_log = []
  best_r2_log = []
  profit_log = []

  for n in range(n_repeats):
    model = FXModel(n_hidden= norm_params_nn['n_hidden'],
                    n_in= norm_params_nn['n_in'],
                    x_uniques= norm_params_nn['embed_uniques'],
                    x_sizes= norm_params_nn['embed_sizes'])
    out = train_model(model, X_train, y_train, X_test, y_test, norm_params)
    model_params, beta_out, best_loss, r2_best, train_log, test_log, pseudoval_log, best_model = out
    betas_log.append(beta_out)
    losses_log.append((test_log, pseudoval_log))
    best_loss_log.append(best_loss)
    best_r2_log.append(r2_best)
    if backtest is not None:
      if model_params is not None:
        backtest, profit = gen_backtest(model_params, backtest[X_test.index[0]:X_test.index[-1]].resample('1H').asfreq().copy(),
                                X_test, y_test, norm_params, plot)
        backtests_log.append(backtest)
        profit_log.append(profit)
  if plot:
    plt.figure(figsize=(12, 7))
    plt.title('Losses over time')
    for n, (test_log, pseudoval_log) in enumerate(losses_log):
      colour = f'C{n}'
      plt.plot(np.log(np.array(pseudoval_log)), c=colour, marker='o', alpha = 0.6)
      plt.plot(np.log(np.array(test_log)), c=colour, marker='x')
    plt.show() 
  return betas_log, backtests_log, best_loss_log, best_r2_log, profit_log
      
def sweep_backtest_thresh(backtest_df, lower=1, upper=5, n_checks = 101,
                          slip_amt= [0.3, 0.5], plot=True):
  out_list = []
  if plot:
    fig, ax = plt.subplots(1, len(slip_amt), figsize= ((4*len(slip_amt))+1, 4))
  for n, slip in enumerate(slip_amt):
    thresh_list = np.linspace(lower, upper, n_checks).tolist()
    profit_list = []
    
    for thresh in thresh_list:
      pred_std = backtest_df['expected_delta'].std()
      backtest_est = backtest_df['actual_delta'].mul(np.sign(backtest_df['expected_delta'])).copy()
      backtest_est -= 0.0001*(0.4+slip)
      backtest_est.loc[backtest_df['expected_delta'].abs()<0.0001*thresh] = 0
      profit_list.append(backtest_est.sum()*100)
    if plot:
      ax[n].scatter(thresh_list, profit_list)
      ax[n].set_title("Slippage: {}".format(slip))
      ax[n].axhline(0, c='k', alpha=0.5)
    out_list.append((thresh_list, profit_list, pred_std))
  if plot:
    plt.show()
  return out_list

def train_model_final(X, y, norm_params, plot=True):
  model = FXModel(n_hidden= norm_params['nn_params']['n_hidden'],
                  n_in= norm_params['nn_params']['n_in'],
                  x_uniques= norm_params['nn_params']['embed_uniques'],
                  x_sizes= norm_params['nn_params']['embed_sizes'])

  train_data = data_utils.TensorDataset(torch.from_numpy(X.values.astype(np.float32)),
                                       torch.from_numpy(y.values.astype(np.float32)))

  USE_CUDA = True
  if USE_CUDA:
    model.cuda()
  train_log = []
  pseudoval_log = []
  best_loss=100
  model_params_best = None
  optimiser = torch.optim.Adam(model.parameters(), lr=norm_params['nn_params']['learnrate'],
                              weight_decay= 0)
  print('', end='')
  for n in range(norm_params['nn_params']['iters']):
    for epoch in range(norm_params['nn_params']['epochs']):
      print("\rRound {} : Epoch {}   ---   Best loss: {:.6f}".format(
        n+1, epoch+1, best_loss),
            end='')
      bs = 2**min(7+epoch,12) # batch rate annealing here
      train_dl = data_utils.DataLoader(train_data, batch_size=bs, shuffle=True, num_workers = 6)
      sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, len(train_dl.dataset),
                                                         norm_params['nn_params']['lrmin'])
      train_loss = train(model, train_dl, optimiser, sched, norm_params['nn_params'], USE_CUDA)
      pseudoval_loss, _ = test(model, train_dl, USE_CUDA)
      train_log.append(train_loss)
      pseudoval_log.append(pseudoval_loss)
      if pseudoval_loss < best_loss:
        best_round = n
        best_epoch = epoch
        best_loss = pseudoval_loss.item()
        model_params_best = model.state_dict().copy()
  print("\rBest loss: {:.5f} @ {}:{}".format(best_loss, best_round+1, best_epoch+1))
  if plot:
    pv = np.log(np.array(pseudoval_log))
    plt.figure(figsize= (20, 5))
    plt.title('Loss over run')
    plt.plot(np.log(np.array(train_log)), marker='x', label='Train loss')
    plt.plot(pv, marker='o', label='Pseudoval loss')
    plt.axhline(pv.min(), c='r', alpha= 0.5)
    plt.legend()
    plt.show()
  return model_params_best, model.state_dict()

def save_model_state(path_to_dir, feat_map, norm_params, model_state_dict):
  assert path_to_dir[-1] == '/'
  fn_1 = 'feature_map'
  fn_2 = 'norm_params'
  fn_3 = 'model_params.pt'
  os.makedirs(path_to_dir, exist_ok=True)
  existing = any(
    [os.path.exists(path_to_dir+fn_1),
     os.path.exists(path_to_dir+fn_2),
     os.path.exists(path_to_dir+fn_3)]
  )
  if not existing:
    check = 'y'
  
  while existing:
    try:
      check = str(input("Overwrite existing files? y/n:"))
    except:
      raise ValueError('y or n only')
    if check in ['y', 'n']:
      break
    else:
      raise ValueError('y or n only')
  
  if (check == 'y'):
    norm_params['nn_params']['state_dict_path'] = path_to_dir+fn_3
    torch.save(model_state_dict, path_to_dir+fn_3)
    with open(path_to_dir+fn_1, 'wb') as file:
      pickle.dump(feat_map, file)
    with open(path_to_dir+fn_2, 'wb') as file:
      pickle.dump(norm_params, file)
    print('Model saved.')
  else:
    print('Model not saved.')

def sim_backtest_new_obs(backtest, prediction, timestamp, norm_params):
  backtest = backtest[timestamp:].resample('1H').mean()
  pred = (prediction*norm_params['target_scale']) + backtest[norm_params['base_column']][0]
  estimated_diff = pred - backtest['Open'][1]
  actual_diff = backtest['Close'][1] - backtest['Open'][1]
  out = actual_diff * np.sign(estimated_diff)
  out -= 0.0001*0.7
  if abs(estimated_diff) < 0.0001*norm_params['decision_threshold']:
    out = 0
  return out

def sweep_average_return_thresh(backtest_df, lower=1, upper=5, n_checks = 101,
                          slip_amt= [0.3, 0.5], plot=True):
  out_list = []
  if plot:
    fig, ax = plt.subplots(1, len(slip_amt), figsize= ((4*len(slip_amt))+1, 4))
  for n, slip in enumerate(slip_amt):
    thresh_list = np.linspace(lower, upper, n_checks).tolist()
    profit_list = []
    
    for thresh in thresh_list:
      pred_std = backtest_df['expected_delta'].std()
      backtest_est = backtest_df['actual_delta'].mul(np.sign(backtest_df['expected_delta'])).copy()
      backtest_est -= 0.0001*(0.4+slip)
      backtest_est.loc[backtest_df['expected_delta'].abs()<0.0001*thresh] = 0
      profit_list.append(backtest_est[backtest_est!=0].mean()*10000)
    if plot:
      ax[n].scatter(thresh_list, profit_list)
      ax[n].set_title("Slippage: {}".format(slip))
      ax[n].axhline(0, c='k', alpha=0.5)
    out_list.append((thresh_list, profit_list, pred_std))
  if plot:
    plt.show()
  return out_list