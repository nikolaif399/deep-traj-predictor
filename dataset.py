import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from matplotlib import pyplot as plt
from toolkit.loaders.loader_eth import load_eth
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
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
    def __init__(self, data_path, train_test_split=0.8, sample_increase=1, mode='train'):
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
            raise Exception(
                "Dataset mode {} not recognized!".format(self.mode))

        print("Loaded {} sequences for mode {}".format(
            self.agent_ids.size, self.mode))

        self.prepareData(sample_increase)
        print("Data finished preprocessing")

    def prepareData(self, sample_increase):
        self.data = dict()
        for agent_id in self.agent_ids:
            segment = self.trajs.loc[self.trajs['agent_id'] == agent_id]
            xs = np.array(segment['pos_x']).reshape(-1, 1)
            ys = np.array(segment['pos_y']).reshape(-1, 1)
            vxs = np.array(segment['vel_x']).reshape(-1, 1)
            vys = np.array(segment['vel_y']).reshape(-1, 1)
            
            # If doing data augmentation, spline and resample data
            if (sample_increase > 1):
                xs1 = np.empty((sample_increase*(xs.size-1),1))
                ys1 = np.empty((sample_increase*(xs.size-1),1))
                vxs1 = np.empty((sample_increase*(xs.size-1),1))
                vys1 = np.empty((sample_increase*(xs.size-1),1))
                for i in range(xs.size-1):
                    xs1[i*sample_increase:i*sample_increase+sample_increase] = np.linspace(xs[i],xs[i+1],sample_increase).reshape(sample_increase,1)
                    ys1[i*sample_increase:i*sample_increase+sample_increase] = np.linspace(ys[i],ys[i+1],sample_increase).reshape(sample_increase,1)
                    vxs1[i*sample_increase:i*sample_increase+sample_increase] = np.linspace(vxs[i],vxs[i+1],sample_increase).reshape(sample_increase,1)
                    vys1[i*sample_increase:i*sample_increase+sample_increase] = np.linspace(vys[i],vys[i+1],sample_increase).reshape(sample_increase,1)

                spl = UnivariateSpline(xs1, ys1)
                spl.set_smoothing_factor(100)
                
                xs2 = xs1
                ys2 = spl(xs2)
                vxs2 = np.diff(xs2,axis=0) * 0.4 / sample_increase
                vys2 = np.diff(ys2,axis=0) * 0.4 / sample_increase
                vxs2 = np.append(vxs2,vxs2[-1]).reshape(-1,1)
                vys2 = np.append(vys2,vys2[-1]).reshape(-1,1)
                
                fig,ax = plt.subplots()
                ax.quiver(xs, ys, vxs, vys, color='r')
                ax.quiver(xs2, ys2, vxs2, vys2, color='b')
                ax.set_title("Position Trajectory")
                ax.set_xlabel("X")
                ax.set_xlabel("Y")
                plt.show()
                
                xs = xs2
                ys = ys2
                vxs = vxs2
                vys = vys2
                
            self.data[agent_id] = torch.tensor(np.hstack((xs, ys, vxs, vys))).float() 

    def __getitem__(self, idx):
        """
        segment = self.trajs.loc[self.trajs['agent_id'] == self.agent_ids[idx]]
        xs = np.array(segment['pos_x']).reshape(-1, 1)
        ys = np.array(segment['pos_y']).reshape(-1, 1)
        vxs = np.array(segment['vel_x']).reshape(-1, 1)
        vys = np.array(segment['vel_y']).reshape(-1, 1)

        seq = torch.tensor(np.hstack((xs, ys, vxs, vys))).float()
        """
        
        seq = self.data[self.agent_ids[idx]]
        
        # Split into input and label
        if (self.mode == 'train'):
            return seq[:-1, :], seq[1:, :]
        elif (self.mode == 'test'):
            # Return first two segments as input, and all but first two as label
            return seq[:2, :], seq[2:, :]

    def __len__(self):
        return self.agent_ids.size
