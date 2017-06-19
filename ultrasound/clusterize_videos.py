
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

#keys = keys[0:1000]
#X = X[0:1000]

keys = [filename_to_video(k) for k in keys]
sorted_keys = sorted(set(keys))
key_indices = {v: k for k, v in enumerate(list(sorted_keys))}
print("%d videos" % len(key_indices))

tree = scipy.spatial.KDTree(X)
print("tree has been built")

distance_matrix = np.zeros((len(key_indices), len(key_indices)))

for i, x in enumerate(X):
    print(i)

    distances, indices = tree.query(x, k = 10, p = 2, distance_upper_bound = 1.0e+4)

    index0 = key_indices[keys[i]]

    for d, j in zip(distances, indices):
        if not np.isinf(d):
            index1 = key_indices[keys[j]]

            distance_matrix[index0, index1] += 1
            distance_matrix[index1, index0] += 1

distance_matrix /= float(X.shape[0])

pickle.dump([sorted_keys, distance_matrix], open("clusterized.pickle", "wb"))
