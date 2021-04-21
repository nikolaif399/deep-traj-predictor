import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from matplotlib import pyplot as plt
from toolkit.loaders.loader_eth import load_eth
import numpy as np

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

def collate_fn_pad(list_pairs_seq_target):
  seqs = [seq for seq, target in list_pairs_seq_target]
  targets = [target for seq, target in list_pairs_seq_target]
  seqs_padded_batched = pad_sequence(seqs)   # will pad at beginning of sequences
  targets_batched = torch.stack(targets)
  assert seqs_padded_batched.shape[1] == len(targets_batched)
  return seqs_padded_batched, targets_batched

class PedestrianDataset(Dataset):
  def __init__(self, data_path, mode='train'):
    self.mode = mode
    
    # Load dataset using ETH loader
    traj_dataset = load_eth(data_path)
    self.trajs = traj_dataset.get_trajectories().head()
    self.agent_ids = list(set(self.trajs['agent_id']))

    # NOTE: Test mode not implemented yet, using all data for training

  def __getitem__(self, idx):
    if self.mode is 'train':
      segment = self.trajs.loc[self.trajs['agent_id'] == self.agent_ids[idx]]
      xs = np.array(segment['pos_x']).reshape(-1,1)
      ys = np.array(segment['pos_y']).reshape(-1,1)
      vxs = np.array(segment['vel_x']).reshape(-1,1)
      vys = np.array(segment['vel_y']).reshape(-1,1)
      seq = torch.tensor(np.hstack((xs,ys,vxs,vys)))
      # Split into input and label
      return seq[:-1,:], seq[1:,:]
    
    elif self.mode is 'test':
      raise NotImplementedError
  
  def __len__(self):
    return len(self.agent_ids)
