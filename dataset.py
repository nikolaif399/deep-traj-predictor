import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pickle
from matplotlib import pyplot as plt

class PedestrianDataset(Dataset):
  def __init__(self, data_path, mode='train'):
    
    with open(data_path,'rb') as f:
      data = pickle.load(f)
    
    num_data_files = len(data)
    print(type(data))
    print(len(data))
    print(type(data[0]))
    print(data[0].shape)

    pedestrian_tracks = [dict() for i in range(num_data_files)] # list of dicts, index by file, then by ped index

    for i in range(num_data_files):
      traj = data[i]
      total_steps = traj.shape[1]
      
      last_ped_id = -1
      #for j in range(total_steps):
        #cur_ped_id = traj[1,j]
        #print(cur_ped_id)
      plt.plot(traj[2,:])
      plt.show()
      print(total_steps)
      

    num_data = data.shape[0]
    np.random.seed(0)

    # split the dataset into training and test 

    # NOT IMPLEMENTED YET! USING ALL DATA FOR TRAINING!

  def __getitem__(self, idx):
    if self.mode is 'train':
      return self.data[idx,:], self.data[idx,:]
    elif self.mode is 'test':
      raise NotImplementedError
      #return self.data[idx,0], self.data[idx,:]
  
  def __len__(self):
    return self.data.shape[0]-1

