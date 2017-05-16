#! /usr/bin/python3

import os
import sys
import json
import codecs
import scipy.misc

label_to_index = {}
all_labels = []

for fp in sys.argv[1:]:
    with open(fp, "rb") as f:
        reader = codecs.getreader("utf-8")
        annotations = json.load(reader(f))

    for filename, labels in annotations.items():
        image = scipy.misc.imread(os.path.join(os.path.dirname(fp), filename))
        (h, w, _) = image.shape

        with open(os.path.join(os.path.dirname(fp), os.path.splitext(filename)[0] + ".txt"), "w") as f:
            for l, t, r, b, label in labels:
                # if label not in label_to_index:
                #     all_labels.append(label)
                #     label_to_index[label] = len(label_to_index)
                # label = label_to_index[label]

                # l = l // 4
                # t = t // 4
                # r = r // 4
                # b = b // 4
                # if t < 0: t = 0
                # if l < 0: l = 0
                # if r > w - 1: r = w - 1
                # if b > h - 1: b = h - 1
                # print(t, b, l, r, image.shape)
                # scipy.misc.imsave("/large/home/test/%s_%s.jpg" % (label, filename), image[t:b, l:r, :])
                # continue

                label = 0

                if r < l: r, l = l, r
                if b < t: t, b = b, t

                if r - l < 10: continue
                if b - t < 10: continue

                l = float(max(l // 4, 0)) / w
                t = float(max(t // 4, 0)) / h
                r = float(min(r // 4, w)) / w
                b = float(min(b // 4, h)) / h

                x = (l + r) / 2.
                y = (b + t) / 2

                print(x, y, r - l, b - t, label, file = f)

for k in all_labels:
    print(k)
