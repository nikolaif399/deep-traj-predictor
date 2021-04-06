import torch
import torch.nn as nn
from torch.autograd import Variable


class TrajLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, dropout, device):
    super(TrajLSTM, self).__init__()
    
    self.num_layers = num_layers
    self.hiddens = hidden_size
    self.input_size = input_size
    self.device = device

    self.cells = []
    self.cells.append(nn.LSTMCell(input_size,hidden_size).to(device))
    for i in range(num_layers-1):
      self.cells.append(nn.LSTMCell(hidden_size,hidden_size).to(device))

    self.linear = nn.Linear(hidden_size,input_size).to(device)

  # forward pass through LSTM layer
  def forward(self, x, future=0):
    '''
    what is this 19? Length in LSTM?
    input: x of dim (batch_size, 19, input_size)
    '''
    # define your feedforward pass
    outputs = []

    b_size = x.size(0)

    h_t = torch.zeros(b_size, self.hiddens, dtype=torch.float, device=self.device)
    c_t = torch.zeros(b_size, self.hiddens, dtype=torch.float, device=self.device)
    h_t2 = torch.zeros(b_size, self.hiddens, dtype=torch.float, device=self.device)
    c_t2 = torch.zeros(b_size, self.hiddens, dtype=torch.float, device=self.device)

    for x_t in x.split(1, dim=1):
      x_t = torch.reshape(x_t, (x_t.size(0),-1,))
      h_t, c_t = self.cells[0](x_t, (h_t, c_t))
      h_t2, c_t2 = self.cells[1](h_t, (h_t2, c_t2))
      output = self.linear(h_t2)
      outputs += [output]

    for i in range(future):
      h_t, c_t = self.cells[0](output, (h_t, c_t))
      h_t2, c_t2 = self.cells[1](h_t, (h_t2, c_t2))
      output = self.linear(h_t2)
      outputs += [output]

    outputs = torch.stack(outputs).permute(1,0,2)
    return outputs

  # forward pass through LSTM layer for testing
  def test(self, x, future=10):
    '''
    input: x of dim (batch_size, input_size)
    '''
    x = torch.reshape(x, (x.size(0),1,x.size(1)))
    return self.forward(x,future)