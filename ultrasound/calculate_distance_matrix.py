
import numpy as np
import pickle
import sklearn.cluster
from collections import defaultdict

print("loading distances...")
[sorted_keys0, distances0] = pickle.load(open("distances_0.pickle", "rb"))
[sorted_keys1, distances1] = pickle.load(open("distances_1.pickle", "rb"))
[sorted_keys2, distances2] = pickle.load(open("distances_2.pickle", "rb"))
[sorted_keys3, distances3] = pickle.load(open("distances_3.pickle", "rb"))

print("merging distances...")
distances = defaultdict(list)

for dm in [distances0, distances1, distances2, distances3]:
    for k, v in dm.items():
        distances[k].extend(v)

print("building distance matrix...")
n = len(sorted_keys0)
distance_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        distance_matrix[i, j] = np.mean(distances[(i, j)])

pickle.dump([sorted_keys0, distance_matrix], open("distance_matrix.pickle", "wb"))
