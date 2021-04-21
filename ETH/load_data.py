from toolkit.loaders.loader_eth import load_eth
from matplotlib import pyplot as plt
import numpy as np

traj_dataset = load_eth("seq_hotel/obsmat.txt")
trajs = traj_dataset.get_trajectories().head()

agent_ids = list(set(trajs['agent_id']))

print(len(agent_ids))
for i in range(len(agent_ids)):
  segment = trajs.loc[trajs['agent_id'] == agent_ids[i]]
  #print(segment)
  xs = np.array(segment['pos_x'])
  ys = np.array(segment['pos_y'])
  vxs = np.array(segment['vel_x'])
  vys = np.array(segment['vel_y'])

  plt.plot(xs,ys)

plt.show()