# deep-traj-predictor

Datasets are stored as compressed numpy arrays of size (num_points,num_states+num_sensors).

To load and inspect:
```python
run data = np.load(filename)['data']```
