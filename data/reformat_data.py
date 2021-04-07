import numpy as np
import os
from glob import glob
import pickle

PATH="."
EXT="*.csv"
all_csv_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, EXT))]


num_files = len(all_csv_files)

datasets = []

for i in range(num_files):
  a = np.loadtxt(open(all_csv_files[i], "rb"), delimiter=",", skiprows=0)  
  datasets.append(a)

pickle.dump(datasets,open("dataset.pkl","wb"))