import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import TrajDataset
from lstm import TrajLSTM

def main():
  # check if cuda available
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  print("Using device %s" % device)
  
  # set random seed to 0
  np.random.seed(0)
  torch.manual_seed(0)

  # define dataset and dataloader
  datasession = 'data_2021-04-06_17:00:05_r1000_n50'
  train_dataset = TrajDataset(mode='train',data_session=datasession)
  test_dataset = TrajDataset(mode='test',data_session=datasession)
  train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=12)
  test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=12)

  # hyper-parameters
  num_epochs = 100
  lr = 0.001
  input_size = 7 # x, y, z, dx, dy, dz, sensor distance
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

  # define your LSTM loss function here
  loss_func = nn.MSELoss()

  # define optimizer for lstm model
  optim = Adam(model.parameters(), lr=lr)

  train_losses = []
  for epoch in range(num_epochs):
    for n_batch, (in_batch, label) in enumerate(train_loader):
        
      in_batch, label = in_batch.to(device), label.to(device)

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
  #torch.save(model, "lstm_model.pt")
  return

  # test trained LSTM model
  l1_err, l2_err = 0, 0
  l1_loss = nn.L1Loss()
  l2_loss = nn.MSELoss()
  model.eval()
  with torch.no_grad():
    for n_batch, (in_batch, label) in enumerate(test_loader):
      in_batch, label = in_batch.to(device), label.to(device)
      pred = model.test(in_batch)

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

  r = []
  num_points = 17
  interval = 1./num_points
  x = int(num_points/2)
  for j in range(-x,x+1):
    r.append(interval*j)

  from matplotlib import pyplot as plt

  fig, axs = plt.subplots(1, 2)

  for i in range(1, len(pred)):
    c = (i/(num_points+1), 1-i/(num_points+1), 0.5)
    axs[0].plot(pred[i], r, label='t = %s' %(i), c=c)
  axs[0].set_xlabel('velocity [m/s]')
  axs[0].set_ylabel('r [m]')
  axs[0].set_title('Prediction')
  axs[0].legend(bbox_to_anchor=(1,1),fontsize='x-small')

  for i in range(1, len(label)):
    c = (i/(num_points+1), 1-i/(num_points+1), 0.5)
    axs[1].plot(label[i], r, label='t = %s' %(i), c=c)
  axs[1].set_xlabel('velocity [m/s]')
  axs[1].set_ylabel('r [m]')
  axs[1].set_title('Ground Truth')
  axs[1].legend(bbox_to_anchor=(1,1),fontsize='x-small')

  plt.figure()
  plt.plot(train_losses)
  plt.title("Training Loss")
  plt.xlabel("Epoch")
  plt.ylabel("MSE Loss")

  plt.show()


if __name__ == "__main__":
  main()

