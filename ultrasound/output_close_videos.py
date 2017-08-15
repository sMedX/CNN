
import numpy as np
import pickle

[sorted_keys, distance_matrix] = pickle.load(open("distances_0.pickle", "rb"))
distance_matrix = np.zeros((n, n))
