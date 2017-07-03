
import os
import sys
import gflags
import numpy as np
import pickle
import sklearn.cluster
import scipy.spatial

FLAGS = gflags.FLAGS
gflags.DEFINE_string("vectors", "vectors.pickle", "")
FLAGS(sys.argv)

filename_to_video = lambda s: os.path.basename(s).split("_")[0]

vectors = pickle.load(open(FLAGS.vectors, "rb"))
keys = list(vectors.keys())
X = np.array(list(vectors.values()))
print("%d keys, %s vectors loaded" % (len(keys), str(X.shape)))

key_filenames = [filename_to_video(k) for k in keys]
sorted_keys = sorted(set(key_filenames))
key_indices = {v: k for k, v in enumerate(list(sorted_keys))}
print("%d videos" % len(key_indices))

indexed_keys = np.array([key_indices[k] for k in key_filenames])

print("building trees")
trees = [scipy.spatial.KDTree(X[indexed_keys == k]) for k in range(len(key_indices))]

distance_matrix = np.zeros((len(key_indices), len(key_indices)))

for i, x in enumerate(X):
    print(i)

    for j, t in enumerate(trees):
        distances, indices = t.query(x, k = 20, p = 2)

        distance = np.mean(distances)

        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

pickle.dump([sorted_keys, distance_matrix], open("distance_matrix.pickle", "wb"))
