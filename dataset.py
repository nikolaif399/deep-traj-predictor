import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from matplotlib import pyplot as plt
from toolkit.loaders.loader_eth import load_eth
import numpy as np
import math

class PadSequence:
  def __call__(self, batch):
    # Let's assume that each element in "batch" is a tuple (data, label).
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    
    # Get each sequence and pad it
    sequences = [x[0] for x in sorted_batch]
    sequences_padded = pad_sequence(sequences, batch_first=True)
      
    # Also need to store the length of each sequence
    # This is later needed in order to unpad the sequences
    lengths = torch.Tensor([len(x) for x in sequences])

    labels = list(map(lambda x: x[1], sorted_batch))
    labels_padded = pad_sequence(labels, batch_first=True)
    
    return sequences_padded, lengths, labels_padded

class PedestrianDataset(Dataset):
  def __init__(self, data_path, train_test_split=0.8,mode='train'):
    self.mode = mode
    
    # Load dataset using ETH loader
    traj_dataset = load_eth(data_path)
    self.trajs = traj_dataset.get_trajectories().head()
    
    agent_ids_total = set(self.trajs['agent_id'])
    all_agent_ids = np.array(list(agent_ids_total))
    
    np.random.shuffle(all_agent_ids)
    last_train_idx = math.floor(train_test_split*all_agent_ids.size)

    self.mode = mode
    
    if (self.mode == 'train'):
      self.agent_ids = all_agent_ids[:last_train_idx]
    elif (self.mode == 'test'):
      self.agent_ids = all_agent_ids[last_train_idx:]
    else:
      raise Exception("Dataset mode {} not recognized!".format(self.mode))
    
    print("Loaded {} sequences for mode {}".format(self.agent_ids.size, self.mode))

  def __getitem__(self, idx):
    segment = self.trajs.loc[self.trajs['agent_id'] == self.agent_ids[idx]]
    xs = np.array(segment['pos_x']).reshape(-1,1)
    ys = np.array(segment['pos_y']).reshape(-1,1)
    vxs = np.array(segment['vel_x']).reshape(-1,1)
    vys = np.array(segment['vel_y']).reshape(-1,1)
    seq = torch.tensor(np.hstack((xs,ys,vxs,vys))).float()
      
    # Split into input and label
    if (self.mode == 'train'):
      return seq[:-1,:], seq[1:,:]
    elif (self.mode == 'test'):
      # Return first two segments as input, and all but first two as label
      return seq[:2,:],seq[2:,:]

  
  def __len__(self):
    return self.agent_ids.size
