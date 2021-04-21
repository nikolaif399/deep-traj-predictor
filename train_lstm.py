import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dataset import PedestrianDataset, PadSequence
from lstm import TrajLSTM

def main():
  # check if cuda available
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  device = 'cpu'
  print("Using device %s" % device)

  # set random seed to 0
  np.random.seed(0)
  torch.manual_seed(0)

  # define dataset and dataloader
  datapath = "ETH/seq_hotel/obsmat.txt"
  train_dataset = PedestrianDataset(mode='train',data_path=datapath)
  #test_dataset = PedestrianDataset(mode='test',data_path=datapath)
  train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, collate_fn=PadSequence(), num_workers=12)
  #test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=12)

  train = True

  # define your LSTM loss function here
  loss_func = nn.MSELoss()

  if (train):
    # hyper-parameters
    num_epochs = 200
    lr = 0.001
    input_size = 4 # x, y, dx, dy
    hidden_size = 128
    num_layers = 2
    dropout = 0.0

    model = TrajLSTM(
      input_size=input_size, 
      hidden_size=hidden_size, 
      num_layers=num_layers, 
      dropout=dropout,
      device=device
    ).to(device)

    # define optimizer for lstm model
    optim = Adam(model.parameters(), lr=lr)

    train_losses = []
    for epoch in range(num_epochs):
      for n_batch, (in_batch, in_batch_len, label) in enumerate(train_loader):
        
        print(in_batch.shape)
        print(in_batch_len.shape)
        print(label.shape)
        
        in_batch, in_batch_len, label = in_batch.to(device), in_batch_len.to(device),label.to(device)

        # train LSTM
        out = model(in_batch)
        
        # calculate LSTM loss
        loss = loss_func(out,label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        # print loss while training

        if (n_batch + 1) % 5 == 0:
          print("Epoch: [{}/{}], Batch: {}, Loss: {}".format(
              epoch, num_epochs, n_batch, loss.item()))
        train_losses.append(loss)

    # save trained LSTM model
    torch.save(model, "lstm_model.pt")
  
  else: # load model from file
    model = torch.load("lstm_model.pt")

  # test trained LSTM model
  l1_err, l2_err = 0, 0
  l1_loss = nn.L1Loss()
  l2_loss = nn.MSELoss()
  model.eval()
  with torch.no_grad():
    for n_batch, (in_batch, label) in enumerate(test_loader):
      in_batch, label = in_batch.to(device), label.to(device)

      pred = model.test(in_batch, future=99)

      l1_err += l1_loss(pred, label).item()
      l2_err += l2_loss(pred, label).item()

  print("Test L1 error:", l1_err)
  print("Test L2 error:", l2_err)

  # visualize the prediction comparing to the ground truth
  if device is 'cpu':
    pred = pred.detach().numpy()[0,:,:]
    label = label.detach().numpy()[0,:,:]
  else:
    pred = pred.detach().cpu().numpy()[0,:,:]
    label = label.detach().cpu().numpy()[0,:,:]

  print(label[0,:])
  print(pred[0,:])

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  
  ax.plot3D(pred[:,0],pred[:,1],pred[:,2],label='pred')
  ax.plot3D(label[:,0],label[:,1],label[:,2],label='label')
  ax.legend()
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.set(ylim=(-2,2))
  plt.show()

if __name__ == "__main__":
  main()

