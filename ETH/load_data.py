from toolkit.loaders.loader_eth import load_eth
# fixme: replace OPENTRAJ_ROOT with the address to root folder of OpenTraj
from matplotlib import pyplot as plt

traj_dataset = load_eth("seq_hotel/obsmat.txt")

trajs = traj_dataset.get_trajectories().head()

agent_ids = list(set(trajs['agent_id']))

for i in range(len(agent_ids)):
  segment = trajs.loc[trajs['agent_id'] == agent_ids[i]]
  xs = segment['pos_x']
  ys = segment['pos_y']
  plt.plot(xs,ys)

plt.show()