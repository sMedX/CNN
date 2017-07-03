
import os
import sys
import gflags
import numpy as np
import pickle
from datetime import datetime
from collections import defaultdict
import sklearn.cluster
import scipy.spatial

FLAGS = gflags.FLAGS
gflags.DEFINE_string("vectors", "vectors.pickle", "")
gflags.DEFINE_integer("shard", 0, "")
gflags.DEFINE_integer("num_shards", 4, "")
FLAGS(sys.argv)

filename_to_video = lambda s: os.path.basename(s).split("_")[0]

vectors = pickle.load(open(FLAGS.vectors, "rb"))
keys = list(vectors.keys())
X = np.array(list(vectors.values()))
print("%d keys, %s vectors loaded" % (len(keys), str(X.shape)))

#X = X[0:1000]
#keys = keys[0:1000]

key_filenames = [filename_to_video(k) for k in keys]
sorted_keys = sorted(set(key_filenames))
key_indices = {v: k for k, v in enumerate(list(sorted_keys))}
print("%d videos" % len(key_indices))

indexed_keys = np.array([key_indices[k] for k in key_filenames])

print("building trees")
trees = [scipy.spatial.KDTree(X[indexed_keys == k]) for k in range(len(key_indices))]

distances = defaultdict(list)

start_time = datetime.now()

for i, x in enumerate(X):
    if i % FLAGS.num_shards != FLAGS.shard: continue

    time_passed = float((datetime.now() - start_time).total_seconds())
    eta = (time_passed / (i + 1)) * (X.shape[0] - i) / 60 / 60
    print("%8d\t%3.2f%%\teta = %.2f hours" % (i, 100. * i / float(X.shape[0]), eta))

    i = indexed_keys[i]

    for j, t in enumerate(trees):
        distance, indices = t.query(x, k = 1, p = 2)

        distances[(i, j)].append(distance)

pickle.dump([sorted_keys, distances], open("distances_%d.pickle" % FLAGS.shard, "wb"))
