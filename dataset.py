import numpy as np
import torch
from torch.utils.data import Dataset
import os


class TrajDataset(Dataset):
  def __init__(self, num_points=16, data_dir='traj_data/', data_session='data', mode='train'):
    data_fn = os.path.join(data_dir + data_session + '.npz')
    data = np.load(data_fn)['data']

    num_data = data.shape[0]

    np.random.seed(0)

    # split the dataset into training and test 
    test_idx = np.random.choice(num_data, num_data//5, replace=False).tolist()
    train_idx = list(set(range(num_data)) - set(test_idx))

    self.mode = mode
    if mode is 'train':
      self.data = data[train_idx,:].astype(np.float32)
    elif mode is 'test':
      self.data = data[test_idx,:].astype(np.float32)

  def __getitem__(self, idx):
    if self.mode is 'train':
      return self.data[idx,:], self.data[idx+1,:]
    elif self.mode is 'test':
      return self.data[idx,:], self.data[idx+1,:]
  
  def __len__(self):
    return self.data.shape[0]-1

