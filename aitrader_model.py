import pickle
import torch.nn as nn

from aitrader_datagen import gen_inference_dset, readin_raw
from torch import cat, from_numpy, load, ones

class NNLayer(nn.Module):
  def __init__(self, n_in, n_out, droprate=0.5):
    super(NNLayer, self).__init__()
    self.lin = nn.Linear(n_in, n_out)
    self.bn1 = nn.BatchNorm1d(n_out)
    self.act = nn.PReLU()
    self.drop = nn.Dropout(droprate)

  def forward(self, x):
    return self.drop(self.bn1(self.act(self.lin(x))))

class FXModel(nn.Module):
  def __init__(self, n_hidden,
               n_in,
               x_uniques,
               x_sizes
              ):
    super(FXModel, self).__init__()
    embedding_cols = len(x_sizes)
    self.e_c = embedding_cols
    self.betas = nn.Parameter(ones(n_in)).float()
    self.em1 = nn.Embedding(x_uniques[0], x_sizes[0])
    self.em2 = nn.Embedding(x_uniques[1], x_sizes[1])
    self.h1 = NNLayer(n_in + sum(x_sizes) - embedding_cols, n_hidden)
    self.h2 = NNLayer(n_hidden, n_hidden)
    self.h3 = NNLayer(n_hidden, n_hidden)
    self.out = nn.Linear(n_hidden, 1)

  def forward(self, x):
    em1 = self.em1(x[:,-2].long())
    em2 = self.em2(x[:,-1].long())
    x = x[:,:-self.e_c] * self.betas[:-self.e_c]
    x = cat((x, em1, em2), 1)
    x = self.h1(x)
    x = self.h2(x)
    x = self.h3(x)
    return self.out(x).view(-1)

def load_all_the_things(model_pickle_savepath):
  """
  model_pickle_savepath denotes the location of the folder that contains the model files, norm params, etc.
  If set to none, it defaults to root.
  """
  
  if model_pickle_savepath is None:
    model_pickle_savepath = '/'
  
  fn_1 = 'feature_map'
  fn_2 = 'norm_params'
  
  with open(model_pickle_savepath+fn_1, 'rb') as file:
    feat_map = pickle.load(file)
    
  with open(model_pickle_savepath+fn_2, 'rb') as file:
    norm_params = pickle.load(file)
  
  model = FXModel(n_hidden= norm_params['nn_params']['n_hidden'],
                  n_in= norm_params['nn_params']['n_in'],
                  x_uniques= norm_params['nn_params']['embed_uniques'],
                  x_sizes= norm_params['nn_params']['embed_sizes'])
  model.load_state_dict(load(norm_params['nn_params']['state_dict_path']))
  model.eval()
  return feat_map, norm_params, model

def gen_new_pred(inference_output_numpy, model):
  return model(from_numpy(inference_output_numpy)).item()

def get_delta(model_savepath=None):

  new_obs = gen_inference_dset(model_savepath, feat_map, norm_params)
  value = gen_new_pred(new_obs.values, model) * norm_params['target_scale']
  if type(value) == str:
    if value == 'Error':
      return value
    else:
      raise ValueError('Abnormal value returned in get_delta()')
  
  last_row = readin_raw(norm_params['data_loc'], norm_params['key_pair'])
  last_price = last_row[norm_params['base_column']][0]
  current_price = last_row['Close'][0]
#   current_price = get_current_or_open_price_from_ib() if we want current price we need an alternate function here.
  expected_change = (last_price+value) - current_price
  return expected_change

def order_decision(thresh, expected_change):
  # Expected change is delta, both inputs should be single floats
  if abs(expected_change) < thresh:
    return 'WAIT'
  elif expect_change > 0:
    return 'BUY'
  elif expected_change < 0:
    return 'SELL'

  